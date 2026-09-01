# -*- coding: utf-8 -*-
from torch import nn
import torch
from modules.util import *
import numpy as np
from torch.autograd import grad
from .GDN import GDN
import math
from modules.vggloss import *
from modules.dists import *


class GeneratorFullModel(torch.nn.Module):
    """
    Two-branch CFTE generator model for temporal superposition.

    Branch 1 (past-CFTE):   past reference frame   -> target/current frame
    Branch 2 (future-CFTE): future reference frame -> target/current frame

    The two branch motion fields are refined with CMR1/CMR2 and then superposed
    at the deformation-feature level before one post-superposition occlusion map
    and the final prediction are generated.

    In the shared-weight version, the past/future module references may point to the
    same kp_detector / videocompressor / generator objects. The two forward passes
    still produce branch-specific motion, occlusion, and prediction tensors.
    """

    def __init__(self, kp_extractor_past, generator_past, kp_extractor_future, generator_future,
                 discriminator, videocompressor_past, videocompressor_future, train_params):
        super(GeneratorFullModel, self).__init__()
        self.kp_extractor_past = kp_extractor_past
        self.generator_past = generator_past
        self.kp_extractor_future = kp_extractor_future
        self.generator_future = generator_future
        self.discriminator = discriminator
        self.videocompressor_past = videocompressor_past
        self.videocompressor_future = videocompressor_future

        self.train_params = train_params
        self.scale_factor = train_params['scale_factor']
        self.scales = train_params['scales']
        self.temperature = train_params['temperature']
        self.out_channels = train_params['num_kp']
        self.disc_scales = self.discriminator.scales

        self.superposition_weight_past = train_params.get('superposition_weight_past', 0.5)
        self.superposition_weight_future = train_params.get('superposition_weight_future', 0.5)

        self.down = AntiAliasInterpolation2d(generator_past.num_channels, self.scale_factor)

        self.pyramid = ImagePyramide(self.scales, generator_past.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

        self.vgg = Vgg19()
        if torch.cuda.is_available():
            self.vgg = self.vgg.cuda()

        self.dists = DISTS()
        if torch.cuda.is_available():
            self.dists = self.dists.cuda()

    def _run_cfte_branch(self, reference_image, driving_image, kp_extractor, generator, videocompressor):
        """Run one CFTE branch from one reference frame to the current target frame."""
        heatmap_reference = kp_extractor(reference_image)
        heatmap_driving = kp_extractor(driving_image)
        total_bits, quant_driving = videocompressor(heatmap_driving, heatmap_reference)
        generated = generator(reference_image, heatmap_source=heatmap_reference, heatmap_driving=quant_driving)
        generated.update({'heatmap_source': heatmap_reference, 'heatmap_driving': quant_driving})
        return total_bits, generated

    def forward(self, x, lambda_var):
        # Backward compatibility: if an old dataloader only provides `source`, use it as the past reference.
        source_past = x['source_past'] if 'source_past' in x else x['source']
        source_future = x['source_future']
        driving = x['driving']

        bs, _, width, height = driving.shape
        lamdaloss = lambda_var

        if torch.cuda.is_available():
            lambda_var = torch.tensor(lambda_var).cuda()

        total_bits_past, generated_past = self._run_cfte_branch(
            source_past, driving, self.kp_extractor_past, self.generator_past, self.videocompressor_past)
        total_bits_future, generated_future = self._run_cfte_branch(
            source_future, driving, self.kp_extractor_future, self.generator_future, self.videocompressor_future)

        generated = {}
        generated['prediction_past'] = generated_past['prediction']
        generated['prediction_future'] = generated_future['prediction']

        # Advisor-directed path: refine the motion fields after branch deformation with CMR1/CMR2,
        # superpose the refined deformed representation, then predict one
        # post-superposition occlusion map and decode the final image.
        if hasattr(self.generator_past, 'forward_superposed_after_deform'):
            fused_generated = self.generator_past.forward_superposed_after_deform(
                source_past, source_future, generated_past, generated_future,
                self.superposition_weight_past, self.superposition_weight_future)
            generated.update(fused_generated)
        else:
            # Fallback for old generators.
            generated['prediction'] = (self.superposition_weight_past * generated_past['prediction'] +
                                       self.superposition_weight_future * generated_future['prediction'])
            for key in ['deformed', 'occlusion_map']:
                if key in generated_past and key in generated_future:
                    generated[key] = (self.superposition_weight_past * generated_past[key] +
                                      self.superposition_weight_future * generated_future[key])

        # Keep branch tensors for diagnostics.  If CMR1/CMR2 produced refined tensors,
        # they remain in generated; otherwise use the raw branch outputs.
        for key in ['sparse_deformed']:
            if key in generated_past and key in generated_future:
                generated[key] = (self.superposition_weight_past * generated_past[key] +
                                  self.superposition_weight_future * generated_future[key])
                generated[key + '_past'] = generated_past[key]
                generated[key + '_future'] = generated_future[key]

        # Keep branch-wise deformed images for diagnostics; the occlusion map is now
        # a single post-superposition map and should not be duplicated into past/future aliases.
        for key in ['deformed']:
            if key in generated_past and key in generated_future:
                generated.setdefault(key + '_past', generated_past[key])
                generated.setdefault(key + '_future', generated_future[key])

        for key in ['sparse_motion', 'deformation']:
            if key in generated_past:
                generated.setdefault(key, generated_past[key])
                generated.setdefault(key + '_past', generated_past[key])
            if key in generated_future:
                generated.setdefault(key + '_future', generated_future[key])

        generated['heatmap_source_past'] = generated_past['heatmap_source']
        generated['heatmap_driving_past'] = generated_past['heatmap_driving']
        generated['heatmap_source_future'] = generated_future['heatmap_source']
        generated['heatmap_driving_future'] = generated_future['heatmap_driving']

        loss_values = {}

        # Diagnostic metrics for the single post-superposition occlusion map.
        # These are logged only and ignored by back-propagation because train.py
        # filters out keys beginning with `metric_`.
        if 'occlusion_map' in generated:
            occ_fused = generated['occlusion_map'].detach()
            loss_values['metric_occ_fused_min'] = occ_fused.min()
            loss_values['metric_occ_fused_mean'] = occ_fused.mean()
            loss_values['metric_occ_fused_max'] = occ_fused.max()

        pyramide_real = self.pyramid(driving)
        pyramide_generated = self.pyramid(generated['prediction'])

        driving_image_downsample = self.down(driving)
        pyramide_real_downsample = self.pyramid(driving_image_downsample)
        sparse_deformed_generated = generated['sparse_deformed']
        sparse_pyramide_generated = self.pyramid(sparse_deformed_generated)

        # RD terms.  Both compact-feature residual streams have to be transmitted.
        # Keep metrics separate from loss terms; train.py ignores keys beginning with `metric_`.
        bpp_past = total_bits_past / (bs * width * height)
        bpp_future = total_bits_future / (bs * width * height)
        bpp_mv = bpp_past + bpp_future
        loss_values['metric_lambda'] = lambda_var.detach() if torch.is_tensor(lambda_var) else torch.tensor(lambda_var, device=driving.device)
        loss_values['metric_bpp_past'] = bpp_past.detach()
        loss_values['metric_bpp_future'] = bpp_future.detach()
        loss_values['metric_bpp'] = bpp_mv.detach()

        # Differentiable DISTS loss on the final superposed prediction.
        # The previous implementation converted tensors to numpy, which broke gradients.
        prediction_final = generated['prediction'].clamp(0, 1)
        target_image = driving.clamp(0, 1)
        dists = self.dists(target_image, prediction_final, as_loss=True)
        loss_values['metric_dists'] = dists.detach()

        # Actual RD optimization term used for back-propagation.
        rdloss = lamdaloss * bpp_mv + dists
        loss_values['loss_rdloss'] = rdloss

        # Branch-level supervision: keep both branch outputs close to the target before 0.5/0.5 superposition.
        # This directly addresses the previous color/brightness imbalance between pred_past and pred_future.
        branch_dists_weight = self.loss_weights.get('branch_dists', self.train_params.get('branch_dists_weight', 0.25))
        if branch_dists_weight != 0:
            dists_past = self.dists(target_image, generated['prediction_past'].clamp(0, 1), as_loss=True)
            dists_future = self.dists(target_image, generated['prediction_future'].clamp(0, 1), as_loss=True)
            loss_values['metric_dists_past'] = dists_past.detach()
            loss_values['metric_dists_future'] = dists_future.detach()
            loss_values['loss_branch_dists'] = branch_dists_weight * (dists_past + dists_future)

        # Perceptual Loss -- Initial / motion-guided 64x64 branch
        if sum(self.loss_weights['perceptual_initial']) != 0:
            value_total = 0
            for scale in [1, 0.5, 0.25]:
                x_vgg = self.vgg(sparse_pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real_downsample['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual_initial']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual_initial'][i] * value
                loss_values['perceptual_64INITIAL'] = value_total

        # Perceptual Loss -- Final superposed 256x256 prediction
        if sum(self.loss_weights['perceptual_final']) != 0:
            value_total = 0
            for scale in self.scales:
                x_vgg = self.vgg(pyramide_generated['prediction_' + str(scale)])
                y_vgg = self.vgg(pyramide_real['prediction_' + str(scale)])

                for i, weight in enumerate(self.loss_weights['perceptual_final']):
                    value = torch.abs(x_vgg[i] - y_vgg[i].detach()).mean()
                    value_total += self.loss_weights['perceptual_final'][i] * value
                loss_values['perceptual_256FINAL'] = value_total

        # GAN Loss on the final superposed prediction.
        if self.loss_weights['generator_gan'] != 0:
            discriminator_maps_generated = self.discriminator(pyramide_generated)
            discriminator_maps_real = self.discriminator(pyramide_real)

            value_total = 0
            for scale in self.disc_scales:
                key = 'prediction_map_%s' % scale
                value = ((1 - discriminator_maps_generated[key]) ** 2).mean()
                value_total += self.loss_weights['generator_gan'] * value
            loss_values['gen_gan'] = value_total

            if sum(self.loss_weights['feature_matching']) != 0:
                value_total = 0
                for scale in self.disc_scales:
                    key = 'feature_maps_%s' % scale
                    for i, (a, b) in enumerate(zip(discriminator_maps_real[key], discriminator_maps_generated[key])):
                        if self.loss_weights['feature_matching'][i] == 0:
                            continue
                        value = torch.abs(a - b).mean()
                        value_total += self.loss_weights['feature_matching'][i] * value
                    loss_values['feature_matching'] = value_total

        return loss_values, generated


class DiscriminatorFullModel(torch.nn.Module):
    """
    Discriminator wrapper. The discriminator sees the final 0.5/0.5 superposed prediction.
    """

    def __init__(self, kp_extractor_past, generator_past, kp_extractor_future, generator_future,
                 discriminator, videocompressor_past, videocompressor_future, train_params):
        super(DiscriminatorFullModel, self).__init__()
        self.kp_extractor_past = kp_extractor_past
        self.generator_past = generator_past
        self.kp_extractor_future = kp_extractor_future
        self.generator_future = generator_future
        self.discriminator = discriminator
        self.videocompressor_past = videocompressor_past
        self.videocompressor_future = videocompressor_future

        self.train_params = train_params
        self.scales = self.discriminator.scales
        self.pyramid = ImagePyramide(self.scales, generator_past.num_channels)
        if torch.cuda.is_available():
            self.pyramid = self.pyramid.cuda()

        self.loss_weights = train_params['loss_weights']

    def forward(self, x, generated):
        pyramide_real = self.pyramid(x['driving'])
        pyramide_generated = self.pyramid(generated['prediction'].detach())

        discriminator_maps_generated = self.discriminator(pyramide_generated)
        discriminator_maps_real = self.discriminator(pyramide_real)

        loss_values = {}
        value_total = 0
        for scale in self.scales:
            key = 'prediction_map_%s' % scale
            value = (1 - discriminator_maps_real[key]) ** 2 + discriminator_maps_generated[key] ** 2
            value_total += self.loss_weights['discriminator_gan'] * value.mean()
        loss_values['disc_gan'] = value_total

        return loss_values
