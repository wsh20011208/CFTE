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

        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetwork(num_kp=num_kp, num_channels=num_channels,
                                                           estimate_occlusion_map=estimate_occlusion_map,
                                                           **dense_motion_params)
        else:
            self.dense_motion_network = None

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

    def forward_superposed_after_deform(self, source_past, source_future, generated_past, generated_future,
                                        weight_past=0.5, weight_future=0.5):
        """
        Advisor-directed CMR1 + CMR2 deformation-level superposition path.

        1. Take the raw past/future deformations after the standard motion modules.
        2. CMR1: given the past motion field, refine only the future motion field.
        3. CMR2: given the CMR1-refined future motion field, refine only the past motion field.
        4. Re-warp the branch features/images with the two refined motion fields.
        5. Superpose the refined deformed features/images.
        6. Predict one occlusion map after the superposition.
        7. Decode the fused deformed feature to the final image.
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

        deformed_feature_past = self.deform_input(generated_past['encoded_feature'], deformation_past_cmr2)
        deformed_feature_future = self.deform_input(generated_future['encoded_feature'], deformation_future_cmr1)

        deformed_past = self.deform_input(source_past, deformation_past_cmr2)
        deformed_future = self.deform_input(source_future, deformation_future_cmr1)

        fused_deformed_feature = weight_past * deformed_feature_past + weight_future * deformed_feature_future
        fused_deformed = weight_past * deformed_past + weight_future * deformed_future
        fused_deformation = weight_past * deformation_past_cmr2 + weight_future * deformation_future_cmr1

        # Build a post-superposition hourglass feature using the same single-frame
        # CFTE idea: [heatmap_representation, deformed_image] -> hourglass -> occlusion.
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
        prediction = self._decode_deformed_feature(fused_deformed_feature, occlusion_map)

        return {
            'prediction': prediction,
            'deformed': fused_deformed,
            'deformed_past': deformed_past,
            'deformed_future': deformed_future,
            'deformed_future_cmr1': deformed_future_cmr1,
            'deformed_feature': fused_deformed_feature,
            'deformation': fused_deformation,
            'deformation_past': deformation_past_cmr2,
            'deformation_future': deformation_future_cmr1,
            'deformation_past_raw': deformation_past_raw,
            'deformation_future_raw': deformation_future_raw,
            'deformation_future_cmr1': deformation_future_cmr1,
            'deformation_delta_future_cmr1': delta_future_cmr1,
            'deformation_delta_past_cmr2': delta_past_cmr2,
            # Backward-compatible aliases used by older logger/model code.
            'deformation_delta_past': delta_past_cmr2,
            'deformation_delta_future': delta_future_cmr1,
            'post_superposition_hourglass_feature': post_superposition_hourglass_feature,
            'occlusion_map': occlusion_map,
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
