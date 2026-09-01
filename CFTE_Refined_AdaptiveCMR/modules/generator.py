import torch
from torch import nn
import torch.nn.functional as F
from modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d, Hourglass
from modules.dense_motion import DenseMotionNetwork
from modules.util import AntiAliasInterpolation2d, make_coordinate_grid
from .GDN import GDN
import math
from modules.flowwarp import *


class ConditionalMotionRefinement(nn.Module):
    """
    One directional conditional motion refinement module.

    Following the advisor's suggested order, CMR1 uses the past motion field as
    the given condition and refines only the future motion field. Then CMR2 uses
    the CMR1-refined future motion field as the given condition and refines only
    the past motion field. The given field is never modified inside this module;
    only the target field receives a residual correction.
    """

    def __init__(self, hidden_channels=64, max_delta=1.0):
        super(ConditionalMotionRefinement, self).__init__()
        self.max_delta = float(max_delta)
        # Input: given_deformed(3), target_deformed(3), abs difference(3),
        # given_motion(2), target_motion(2) = 13 channels.
        self.net = nn.Sequential(
            nn.Conv2d(13, hidden_channels, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 2, kernel_size=3, padding=1)
        )

        # Start exactly from the input target motion field.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, given_deformed, target_deformed, given_deformation, target_deformation):
        # deformation_*: B,H,W,2.  deformed_*: B,3,h,w.
        b, h, w, _ = target_deformation.shape
        given_deformed_small = F.interpolate(given_deformed, size=(h, w), mode='bilinear', align_corners=False)
        target_deformed_small = F.interpolate(target_deformed, size=(h, w), mode='bilinear', align_corners=False)
        given_motion = given_deformation.permute(0, 3, 1, 2)
        target_motion = target_deformation.permute(0, 3, 1, 2)

        cmr_input = torch.cat([
            given_deformed_small,
            target_deformed_small,
            torch.abs(given_deformed_small - target_deformed_small),
            given_motion,
            target_motion
        ], dim=1)

        delta = torch.tanh(self.net(cmr_input)) * self.max_delta
        delta = delta.permute(0, 2, 3, 1)
        refined_target = target_deformation + delta
        return refined_target, delta

class OcclusionAwareGenerator(nn.Module):
    """
    Generator that given source image and and keypoints try to transform image according to movement trajectories
    induced by keypoints. Generator follows Johnson architecture.
    """

    def __init__(self, num_channels, num_kp, block_expansion, max_features, num_down_blocks,
                 num_bottleneck_blocks, estimate_occlusion_map=False, dense_motion_params=None, estimate_jacobian=False):
        super(OcclusionAwareGenerator, self).__init__()


        self.temperature =0.1

        # `adaptive_correction_params` belongs to the CMR/adaptive wrapper, not to
        # DenseMotionNetwork.  Remove it before instantiating the original motion
        # module; otherwise DenseMotionNetwork receives an unexpected keyword.
        dense_motion_params = dict(dense_motion_params) if dense_motion_params is not None else None
        adaptive_correction_params = {}
        if dense_motion_params is not None:
            adaptive_correction_params = dense_motion_params.pop('adaptive_correction_params', {}) or {}
            self.dense_motion_network = DenseMotionNetwork(num_kp=num_kp, num_channels=num_channels,
                                                           estimate_occlusion_map=estimate_occlusion_map,
                                                           **dense_motion_params)
        else:
            self.dense_motion_network = None

        # DISTS-guided adaptive correction.  CMR1/CMR2 still provide the initial
        # motion fields; the adaptive loop then compares the superposed deformed
        # image with the target using DISTS and updates the motion fields along the
        # negative DISTS gradient.
        self.adaptive_params = adaptive_correction_params
        self.use_adaptive_correction = bool(adaptive_correction_params.get('enabled', False))
        self.adaptive_max_iterations = int(adaptive_correction_params.get('max_iterations', 0))
        self.adaptive_error_threshold = float(adaptive_correction_params.get('error_threshold', 0.0))
        self.adaptive_target_improvement = float(adaptive_correction_params.get('target_improvement', 0.0))
        self.adaptive_step_size = float(adaptive_correction_params.get('step_size', 0.005))
        self.adaptive_max_delta = float(adaptive_correction_params.get('max_delta', 0.02))
        self.adaptive_gradient_normalize = bool(adaptive_correction_params.get('gradient_normalize', True))
        self.adaptive_smooth_update = bool(adaptive_correction_params.get('smooth_update', True))
        self.adaptive_smooth_kernel = int(adaptive_correction_params.get('smooth_kernel', 3))
        self.adaptive_accept_only_if_improved = bool(adaptive_correction_params.get('accept_only_if_improved', True))
        self.adaptive_backtracking_steps = int(adaptive_correction_params.get('backtracking_steps', 4))
        self.adaptive_backtracking_factor = float(adaptive_correction_params.get('backtracking_factor', 0.5))
        self.adaptive_min_improvement = float(adaptive_correction_params.get('min_improvement', 1e-6))
        self.adaptive_dists_scale = float(adaptive_correction_params.get('dists_scale', 1.0))
        self.adaptive_dists_resize = bool(adaptive_correction_params.get('dists_resize', True))
        self.adaptive_detach_target = bool(adaptive_correction_params.get('detach_target', True))
        self.adaptive_eps = float(adaptive_correction_params.get('eps', 1e-6))

        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        up_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.bottleneck = torch.nn.Sequential()
        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))

        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        self.estimate_occlusion_map = estimate_occlusion_map
        self.num_channels = num_channels

        # CMR1, CMR2, and post-superposition occlusion branch for the two-reference model.
        # To stay as close as possible to the original single-frame CFTE logic, the
        # fused deformed image first goes through its own post-superposition
        # hourglass, and the occlusion map is predicted from that motion-aware
        # hourglass feature.  We keep PyTorch default random initialization here
        # instead of forcing the occlusion map to start nearly white.
        encoded_features = min(max_features, block_expansion * (2 ** num_down_blocks))
        self.cmr1 = ConditionalMotionRefinement(hidden_channels=64, max_delta=1.0)
        self.cmr2 = ConditionalMotionRefinement(hidden_channels=64, max_delta=1.0)
        self.post_superposition_hourglass = Hourglass(
            block_expansion=dense_motion_params['block_expansion'],
            in_features=num_channels + 1,
            max_features=dense_motion_params['max_features'],
            num_blocks=dense_motion_params['num_blocks']
        )
        self.post_superposition_occlusion = nn.Conv2d(
            self.post_superposition_hourglass.out_filters, 1, kernel_size=(7, 7), padding=(3, 3)
        )

    def deform_input(self, inp, deformation):
        _, h_old, w_old, _ = deformation.shape
        _, _, h, w = inp.shape
        if h_old != h or w_old != w:
            deformation = deformation.permute(0, 3, 1, 2)
            deformation = F.interpolate(deformation, size=(h, w), mode='bilinear')
            deformation = deformation.permute(0, 2, 3, 1)
        return warp(inp, deformation) #F.grid_sample(inp, deformation)  #########

    def _decode_deformed_feature(self, deformed_feature, occlusion_map):
        """Decode an already motion-compensated feature map with a given occlusion map."""
        out = deformed_feature
        if out.shape[2] != occlusion_map.shape[2] or out.shape[3] != occlusion_map.shape[3]:
            occlusion_map = F.interpolate(occlusion_map, size=out.shape[2:], mode='bilinear')
        out = out * occlusion_map

        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)
        out = self.final(out)
        out = F.sigmoid(out)
        return out

    def _compose_superposed_state(self, source_past, source_future, generated_past, generated_future,
                                  deformation_past, deformation_future, weight_past, weight_future):
        """Warp both branches with the supplied motion fields and build the superposed state."""
        deformed_feature_past = self.deform_input(generated_past['encoded_feature'], deformation_past)
        deformed_feature_future = self.deform_input(generated_future['encoded_feature'], deformation_future)
        deformed_past = self.deform_input(source_past, deformation_past)
        deformed_future = self.deform_input(source_future, deformation_future)

        fused_deformed_feature = weight_past * deformed_feature_past + weight_future * deformed_feature_future
        fused_deformed = weight_past * deformed_past + weight_future * deformed_future
        fused_deformation = weight_past * deformation_past + weight_future * deformation_future

        return deformed_past, deformed_future, fused_deformed_feature, fused_deformed, fused_deformation

    def _build_post_superposition_occlusion(self, fused_deformed, generated_past, generated_future,
                                            weight_past, weight_future):
        """Predict a single occlusion map after deformed-image superposition."""
        fused_heatmap_representation = (
            weight_past * generated_past['heatmap_representation'] +
            weight_future * generated_future['heatmap_representation']
        )
        hg_h, hg_w = fused_heatmap_representation.shape[2:]
        fused_deformed_small = F.interpolate(
            fused_deformed, size=(hg_h, hg_w), mode='bilinear', align_corners=False
        )
        hourglass_heatmap = fused_heatmap_representation.unsqueeze(1).view(
            fused_heatmap_representation.shape[0], 1, -1, hg_h, hg_w
        )
        hourglass_deformed = fused_deformed_small.unsqueeze(1).view(
            fused_deformed_small.shape[0], 1, -1, hg_h, hg_w
        )
        hourglass_input = torch.cat([hourglass_heatmap, hourglass_deformed], dim=2)
        hourglass_input = hourglass_input.view(fused_deformed_small.shape[0], -1, hg_h, hg_w)
        post_superposition_hourglass_feature = self.post_superposition_hourglass(hourglass_input)
        occlusion_map = torch.sigmoid(self.post_superposition_occlusion(post_superposition_hourglass_feature))
        return occlusion_map, post_superposition_hourglass_feature

    def _prepare_dists_pair(self, current, target):
        """Prepare superposed deformed image and target for DISTS comparison."""
        if target.shape[2:] != current.shape[2:]:
            target = F.interpolate(target, size=current.shape[2:], mode='bilinear', align_corners=False)
        if self.adaptive_detach_target:
            target = target.detach()
        current = current.clamp(0.0, 1.0)
        target = target.clamp(0.0, 1.0)
        if self.adaptive_dists_scale != 1.0:
            h = max(16, int(current.shape[2] * self.adaptive_dists_scale))
            w = max(16, int(current.shape[3] * self.adaptive_dists_scale))
            current = F.interpolate(current, size=(h, w), mode='bilinear', align_corners=False)
            target = F.interpolate(target, size=(h, w), mode='bilinear', align_corners=False)
        return current, target

    def _dists_error(self, fused_deformed, target_image, dists_model):
        """Differentiable DISTS error between the superposed deformed image and target."""
        current, target = self._prepare_dists_pair(fused_deformed, target_image)
        return dists_model(target, current, as_loss=True, resize=self.adaptive_dists_resize)

    def _normalise_gradient(self, grad_tensor):
        if not self.adaptive_gradient_normalize:
            return grad_tensor
        denom = grad_tensor.abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(self.adaptive_eps)
        return grad_tensor / denom

    def _smooth_motion_delta(self, delta):
        """Optionally smooth a B,H,W,2 update field to avoid high-frequency flow noise."""
        if not self.adaptive_smooth_update or self.adaptive_smooth_kernel <= 1:
            return delta
        kernel = self.adaptive_smooth_kernel
        if kernel % 2 == 0:
            kernel += 1
        delta_chw = delta.permute(0, 3, 1, 2)
        delta_chw = F.avg_pool2d(delta_chw, kernel_size=kernel, stride=1, padding=kernel // 2)
        return delta_chw.permute(0, 2, 3, 1)

    def _make_batch_metric(self, value, batch_size, reference):
        """Convert a scalar tensor/value into a 1-D batch-shaped tensor for DataParallel."""
        if not torch.is_tensor(value):
            value = reference.new_tensor(float(value))
        return value.reshape(1).expand(batch_size)

    def _make_curve_metric(self, values, batch_size, reference):
        """
        Convert a list of scalar DISTS tensors into a fixed-length [B, T] curve.

        T is max_iterations + 1: index 0 is the CMR1/CMR2 initial DISTS and
        indices 1..T-1 are accepted adaptive updates.  If the adaptive loop
        stops early, the remaining entries are padded with the final DISTS.
        This makes the saved epoch curve directly show whether convergence
        happened before the configured maximum number of iterations.
        """
        target_len = max(1, self.adaptive_max_iterations + 1)
        if not values:
            values = [reference.new_tensor(0.0)]

        curve = []
        for value in values:
            if not torch.is_tensor(value):
                value = reference.new_tensor(float(value))
            curve.append(value.detach().reshape(1))

        if len(curve) < target_len:
            curve.extend([curve[-1]] * (target_len - len(curve)))
        elif len(curve) > target_len:
            curve = curve[:target_len]

        curve = torch.cat(curve, dim=0)
        return curve.reshape(1, target_len).expand(batch_size, target_len)

    def _adaptive_dists_gradient_refinement(self, source_past, source_future, generated_past, generated_future,
                                            deformation_past, deformation_future, target_image, dists_model,
                                            weight_past, weight_future):
        """Refine CMR1/CMR2 motion fields by minimizing DISTS(superposed_deformed, target).

        The update is performed directly on the motion fields:
            M^{k+1} = M^k - step_size * normalize(d DISTS / d M^k).
        Optional backtracking accepts an update only when the DISTS error decreases.
        """
        if (not self.use_adaptive_correction or self.adaptive_max_iterations <= 0 or
                target_image is None or dists_model is None):
            empty = deformation_past.new_zeros(deformation_past.shape[0])
            return deformation_past, deformation_future, {
                'initial_error': empty,
                'final_error': empty,
                'iterations': empty,
                'delta_mean': empty,
                'delta_past': torch.zeros_like(deformation_past),
                'delta_future': torch.zeros_like(deformation_future),
                'sequence_mean': empty,
                'dists_curve': self._make_curve_metric([empty.mean()], deformation_past.shape[0], deformation_past),
                'curve_length': empty,
            }

        batch_size = deformation_past.shape[0]
        current_past = deformation_past
        current_future = deformation_future
        total_delta_past = torch.zeros_like(current_past)
        total_delta_future = torch.zeros_like(current_future)
        delta_means = []
        accepted_iterations = 0
        sequence_errors = []
        dists_curve = []

        # Initial DISTS error on the superposed deformed image, not MAE.
        _, _, _, fused_deformed, _ = self._compose_superposed_state(
            source_past, source_future, generated_past, generated_future,
            current_past, current_future, weight_past, weight_future)
        current_error = self._dists_error(fused_deformed, target_image, dists_model)
        initial_error = current_error
        sequence_errors.append(current_error)
        dists_curve.append(current_error)

        for _ in range(self.adaptive_max_iterations):
            if self.adaptive_error_threshold > 0 and current_error.detach().item() <= self.adaptive_error_threshold:
                break
            if self.adaptive_target_improvement > 0:
                current_improvement = initial_error.detach().item() - current_error.detach().item()
                if current_improvement >= self.adaptive_target_improvement:
                    break

            grad_past, grad_future = torch.autograd.grad(
                current_error, [current_past, current_future],
                retain_graph=True, create_graph=False, allow_unused=False)
            grad_past = self._normalise_gradient(grad_past).detach()
            grad_future = self._normalise_gradient(grad_future).detach()

            base_delta_past = (-self.adaptive_step_size * grad_past).clamp(
                min=-self.adaptive_max_delta, max=self.adaptive_max_delta)
            base_delta_future = (-self.adaptive_step_size * grad_future).clamp(
                min=-self.adaptive_max_delta, max=self.adaptive_max_delta)
            base_delta_past = self._smooth_motion_delta(base_delta_past)
            base_delta_future = self._smooth_motion_delta(base_delta_future)

            accepted = False
            best_error = None
            best_past = None
            best_future = None
            best_delta_past = None
            best_delta_future = None

            num_trials = max(1, self.adaptive_backtracking_steps + 1)
            for trial in range(num_trials):
                scale = self.adaptive_backtracking_factor ** trial if self.adaptive_accept_only_if_improved else 1.0
                delta_past = scale * base_delta_past
                delta_future = scale * base_delta_future
                candidate_past = current_past + delta_past
                candidate_future = current_future + delta_future

                _, _, _, candidate_fused_deformed, _ = self._compose_superposed_state(
                    source_past, source_future, generated_past, generated_future,
                    candidate_past, candidate_future, weight_past, weight_future)
                candidate_error = self._dists_error(candidate_fused_deformed, target_image, dists_model)

                if (not self.adaptive_accept_only_if_improved or
                        candidate_error.detach().item() <= current_error.detach().item() - self.adaptive_min_improvement):
                    accepted = True
                    best_error = candidate_error
                    best_past = candidate_past
                    best_future = candidate_future
                    best_delta_past = delta_past
                    best_delta_future = delta_future
                    break

            if not accepted:
                break

            current_past = best_past
            current_future = best_future
            current_error = best_error
            sequence_errors.append(current_error)
            dists_curve.append(current_error)
            total_delta_past = total_delta_past + best_delta_past.detach()
            total_delta_future = total_delta_future + best_delta_future.detach()
            delta_means.append(0.5 * (best_delta_past.abs().mean() + best_delta_future.abs().mean()))
            accepted_iterations += 1

        if delta_means:
            delta_mean = torch.stack(delta_means).mean()
        else:
            delta_mean = deformation_past.new_tensor(0.0)
        sequence_mean = torch.stack(sequence_errors).mean() if sequence_errors else current_error

        stats = {
            'initial_error': self._make_batch_metric(initial_error.detach(), batch_size, deformation_past),
            'final_error': self._make_batch_metric(current_error, batch_size, deformation_past),
            'iterations': self._make_batch_metric(float(accepted_iterations), batch_size, deformation_past),
            'delta_mean': self._make_batch_metric(delta_mean, batch_size, deformation_past),
            'delta_past': total_delta_past,
            'delta_future': total_delta_future,
            'sequence_mean': self._make_batch_metric(sequence_mean, batch_size, deformation_past),
            'dists_curve': self._make_curve_metric(dists_curve, batch_size, deformation_past),
            'curve_length': self._make_batch_metric(float(len(dists_curve)), batch_size, deformation_past),
        }
        return current_past, current_future, stats

    def forward_superposed_after_deform(self, source_past, source_future, generated_past, generated_future,
                                        weight_past=0.5, weight_future=0.5,
                                        target_image=None, dists_model=None):
        """
        CMR1 + CMR2 + DISTS-guided adaptive motion correction.

        1. Obtain raw past/future deformations from the two CFTE branches.
        2. CMR1: use past as the condition and refine only the future motion field.
        3. CMR2: use the CMR1-refined future as the condition and refine only the past motion field.
        4. Use the CMR1/CMR2 results as the initial fields for adaptive correction.
        5. Compare the superposed deformed image with the target using DISTS, compute the
           DISTS gradient with respect to the two motion fields, and iteratively update them.
        6. Re-warp features/images with the final adaptive fields, superpose them, predict one
           post-superposition occlusion map, and decode the final image.
        """
        deformation_past_raw = generated_past['deformation']
        deformation_future_raw = generated_future['deformation']
        deformed_past_raw = generated_past['deformed']
        deformed_future_raw = generated_future['deformed']

        # CMR1: past motion is the given condition; refine future motion only.
        deformation_future_cmr1, delta_future_cmr1 = self.cmr1(
            deformed_past_raw, deformed_future_raw, deformation_past_raw, deformation_future_raw)
        deformed_future_cmr1 = self.deform_input(source_future, deformation_future_cmr1)

        # CMR2: refined future motion is now the given condition; refine past motion only.
        deformation_past_cmr2, delta_past_cmr2 = self.cmr2(
            deformed_future_cmr1, deformed_past_raw, deformation_future_cmr1, deformation_past_raw)

        # Adaptive refinement starts from the CMR fields.  The error is DISTS between
        # the superposed deformed image and the target, not MAE.
        deformation_past_adaptive, deformation_future_adaptive, adaptive_stats = \
            self._adaptive_dists_gradient_refinement(
                source_past, source_future, generated_past, generated_future,
                deformation_past_cmr2, deformation_future_cmr1, target_image, dists_model,
                weight_past, weight_future)

        deformed_past, deformed_future, fused_deformed_feature, fused_deformed, fused_deformation = \
            self._compose_superposed_state(
                source_past, source_future, generated_past, generated_future,
                deformation_past_adaptive, deformation_future_adaptive, weight_past, weight_future)

        occlusion_map, post_superposition_hourglass_feature = self._build_post_superposition_occlusion(
            fused_deformed, generated_past, generated_future, weight_past, weight_future)
        prediction = self._decode_deformed_feature(fused_deformed_feature, occlusion_map)

        return {
            'prediction': prediction,
            'deformed': fused_deformed,
            'deformed_past': deformed_past,
            'deformed_future': deformed_future,
            'deformed_future_cmr1': deformed_future_cmr1,
            'deformed_feature': fused_deformed_feature,
            'deformation': fused_deformation,
            'deformation_past': deformation_past_adaptive,
            'deformation_future': deformation_future_adaptive,
            'deformation_past_cmr2': deformation_past_cmr2,
            'deformation_future_cmr1': deformation_future_cmr1,
            'deformation_past_raw': deformation_past_raw,
            'deformation_future_raw': deformation_future_raw,
            'deformation_delta_future_cmr1': delta_future_cmr1,
            'deformation_delta_past_cmr2': delta_past_cmr2,
            'deformation_delta_past_adaptive': adaptive_stats['delta_past'],
            'deformation_delta_future_adaptive': adaptive_stats['delta_future'],
            # Backward-compatible aliases used by older logger/model code.
            'deformation_delta_past': delta_past_cmr2,
            'deformation_delta_future': delta_future_cmr1,
            'post_superposition_hourglass_feature': post_superposition_hourglass_feature,
            'occlusion_map': occlusion_map,
            'adaptive_dists_initial': adaptive_stats['initial_error'],
            'adaptive_dists_final': adaptive_stats['final_error'],
            'adaptive_dists_sequence_mean': adaptive_stats['sequence_mean'],
            'adaptive_dists_curve': adaptive_stats['dists_curve'],
            'adaptive_dists_curve_length': adaptive_stats['curve_length'],
            'adaptive_iterations': adaptive_stats['iterations'],
            'adaptive_delta_mean': adaptive_stats['delta_mean'],
        }

    def forward(self, source_image,heatmap_source,heatmap_driving):  
        
        # Encoding (downsampling) part
        out = self.first(source_image) 
        
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
        # Transforming feature representation according to deformation and occlusion
        output_dict = {}

        dense_motion = self.dense_motion_network(source_image=source_image,heatmap_source=heatmap_source,
                                                 heatmap_driving=heatmap_driving)


        occlusion_map = dense_motion['occlusion_map']  #64*64*1
        output_dict['occlusion_map'] = occlusion_map  
        
        deformation = dense_motion['deformation'] #64*64*2
        output_dict['deformation'] = deformation
        
        deformed_sparse_source = dense_motion['sparse_deformed'] #64*64*3
        output_dict['sparse_deformed'] = deformed_sparse_source
        
        sparse_motion = dense_motion['sparse_motion']  ###8*8*2
        output_dict['sparse_motion'] = sparse_motion
        output_dict['heatmap_representation'] = dense_motion.get('heatmap_representation')

        
        output_dict["deformed"] = self.deform_input(source_image, deformation)   #### deformation 64*64 interpolate 256*256

        # Keep branch features for the two-reference CMR/fusion path.
        output_dict["encoded_feature"] = out
        out = self.deform_input(out, deformation)
        output_dict["deformed_feature"] = out

#         ###for a comparison about dense flow
#         out_dense = self.bottleneck(out)
#         for i in range(len(self.up_blocks)):
#             out_dense = self.up_blocks[i](out_dense)
#         out_dense = self.final(out_dense)
#         out_dense = F.sigmoid(out_dense)        
#         output_dict["deformed"] = out_dense  

        ###real-part
        if out.shape[2] != occlusion_map.shape[2] or out.shape[3] != occlusion_map.shape[3]:
            occlusion_map = F.interpolate(occlusion_map, size=out.shape[2:], mode='bilinear')
        out = out * occlusion_map

        # Decoding part
        out = self.bottleneck(out)
        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)
        out = self.final(out)
        out = F.sigmoid(out)

        output_dict["prediction"] = out

        return output_dict
