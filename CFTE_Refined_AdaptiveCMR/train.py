# -*- coding: utf-8 -*-
from itertools import chain
from tqdm import trange
import torch

from torch.utils.data import DataLoader

from logger import Logger
from modules.model import GeneratorFullModel, DiscriminatorFullModel

from torch.optim.lr_scheduler import MultiStepLR

from sync_batchnorm import DataParallelWithCallback

from frames_dataset import DatasetRepeater


def _unique_parameters(*modules):
    """Return parameters once, even when past/future modules share the same object."""
    seen = set()
    params = []
    for module in modules:
        if module is None:
            continue
        for parameter in module.parameters():
            if id(parameter) not in seen:
                params.append(parameter)
                seen.add(id(parameter))
    return params


def _unique_modules(*modules):
    seen = set()
    unique = []
    for module in modules:
        if module is None:
            continue
        if id(module) not in seen:
            unique.append(module)
            seen.add(id(module))
    return unique


def _split_main_aux_parameters(*modules):
    """Split entropy bottleneck quantile parameters from normal RD parameters.

    CompressAI entropy bottlenecks use auxiliary `quantiles` parameters. These
    should be optimized by the auxiliary entropy-bottleneck loss, while the normal
    RD likelihood path optimizes the remaining parameters. This avoids stepping
    the same videocompressor parameters twice.
    """
    main_params, aux_params = [], []
    seen_main, seen_aux = set(), set()
    for module in _unique_modules(*modules):
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            if 'quantiles' in name:
                if id(parameter) not in seen_aux:
                    aux_params.append(parameter)
                    seen_aux.add(id(parameter))
            else:
                if id(parameter) not in seen_main:
                    main_params.append(parameter)
                    seen_main.add(id(parameter))
    return main_params, aux_params


def _loss_values_for_backward(losses):
    """Separate real optimization terms from logged metrics.

    Entries prefixed by `metric_` are written to the log but are not optimized.
    This avoids double-counting bpp, lambda, and diagnostic DISTS values when
    `loss_rdloss` already contains the actual RD objective.
    """
    return [value.mean() for key, value in losses.items() if not key.startswith('metric_')]

def make_generated_dataparallel_safe(obj, batch_size):
    """Convert nested scalar tensors in `generated` to batch-shaped tensors.

    PyTorch DataParallel recursively scatters dict/list inputs. A 0-D tensor has
    no batch dimension and can trigger `chunk expects at least a 1-dimensional tensor`.
    """
    import torch
    if torch.is_tensor(obj):
        if obj.dim() == 0:
            return obj.reshape(1).expand(batch_size)
        return obj
    if isinstance(obj, dict):
        return {k: make_generated_dataparallel_safe(v, batch_size) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_generated_dataparallel_safe(v, batch_size) for v in obj]
    if isinstance(obj, tuple):
        return tuple(make_generated_dataparallel_safe(v, batch_size) for v in obj)
    return obj


def train(config, generator_past, generator_future, discriminator,
          kp_detector_past, kp_detector_future,
          videocompressor_past, videocompressor_future,
          checkpoint, log_dir, dataset, device_ids):
    train_params = config['train_params']

    rdlambdas = config['train_params']['loss_weights']['rdlambda']

    # Past and future branches may share the same modules. Use unique parameters
    # so shared weights are not inserted twice into the same optimizer.
    optimizer_generator = torch.optim.Adam(
        _unique_parameters(generator_past, generator_future), lr=train_params['lr_generator'], betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(
        discriminator.parameters(), lr=train_params['lr_discriminator'], betas=(0.5, 0.999))
    optimizer_kp_detector = torch.optim.Adam(
        _unique_parameters(kp_detector_past, kp_detector_future), lr=train_params['lr_kp_detector'], betas=(0.5, 0.999))
    videocompressor_main_params, videocompressor_aux_params = _split_main_aux_parameters(
        videocompressor_past, videocompressor_future)
    optimizer_videocompressor = torch.optim.Adam(
        videocompressor_main_params, lr=train_params['lr_videocompressor'], betas=(0.5, 0.999))

    optimizer_aux = None
    if len(videocompressor_aux_params) > 0:
        optimizer_aux = torch.optim.Adam(
            videocompressor_aux_params, lr=train_params['lr_videocompressor'], betas=(0.5, 0.999))

    if checkpoint is not None:
        start_epoch = Logger.load_cpk(
            checkpoint,
            generator=generator_past,
            generator_future=generator_future,
            discriminator=discriminator,
            kp_detector=kp_detector_past,
            kp_detector_future=kp_detector_future,
            videocompressor=videocompressor_past,
            videocompressor_future=videocompressor_future,
            optimizer_generator=optimizer_generator,
            optimizer_discriminator=optimizer_discriminator,
            optimizer_kp_detector=None if train_params['lr_kp_detector'] == 0 else optimizer_kp_detector,
            optimizer_videocompressor=optimizer_videocompressor)
    else:
        start_epoch = 0

    scheduler_generator = MultiStepLR(optimizer_generator, train_params['epoch_milestones'], gamma=0.1,
                                      last_epoch=start_epoch - 1)
    scheduler_discriminator = MultiStepLR(optimizer_discriminator, train_params['epoch_milestones'], gamma=0.1,
                                          last_epoch=start_epoch - 1)
    scheduler_kp_detector = MultiStepLR(optimizer_kp_detector, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=-1 + start_epoch * (train_params['lr_kp_detector'] != 0))
    scheduler_videocompressor = MultiStepLR(optimizer_videocompressor, train_params['epoch_milestones'], gamma=0.1,
                                            last_epoch=start_epoch - 1)
    scheduler_aux = None
    if optimizer_aux is not None:
        scheduler_aux = MultiStepLR(optimizer_aux, train_params['epoch_milestones'], gamma=0.1, last_epoch=-1)

    if 'num_repeats' in train_params and train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])
    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=6, drop_last=True)

    generator_full = GeneratorFullModel(kp_detector_past, generator_past,
                                        kp_detector_future, generator_future,
                                        discriminator,
                                        videocompressor_past, videocompressor_future,
                                        train_params)
    discriminator_full = DiscriminatorFullModel(kp_detector_past, generator_past,
                                                kp_detector_future, generator_future,
                                                discriminator,
                                                videocompressor_past, videocompressor_future,
                                                train_params)

    # if torch.cuda.is_available():
    #     generator_full = DataParallelWithCallback(generator_full, device_ids=device_ids).cuda()
    #     discriminator_full = DataParallelWithCallback(discriminator_full, device_ids=device_ids).cuda()

    if torch.cuda.is_available():
        generator_full = torch.nn.DataParallel(generator_full, device_ids=device_ids).to(device_ids[0])
        discriminator_full = torch.nn.DataParallel(discriminator_full, device_ids=device_ids).to(device_ids[0])

    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            for x in dataloader:
                optimizer_generator.zero_grad()
                optimizer_kp_detector.zero_grad()
                optimizer_videocompressor.zero_grad()
                if optimizer_aux is not None:
                    optimizer_aux.zero_grad()

                lambda_var = rdlambdas
                print("lambda_var")
                print(lambda_var)

                losses_generator, generated = generator_full(x, lambda_var)

                loss_values = _loss_values_for_backward(losses_generator)
                loss = sum(loss_values)
                loss.backward(retain_graph=True)

                if optimizer_aux is not None:
                    aux_loss = sum(module.entropy_bottleneck.loss()
                                   for module in _unique_modules(videocompressor_past, videocompressor_future))
                    print(aux_loss)
                    aux_loss.backward(retain_graph=True)

                optimizer_generator.step()
                optimizer_kp_detector.step()
                optimizer_videocompressor.step()
                if optimizer_aux is not None:
                    optimizer_aux.step()

                if train_params['loss_weights']['generator_gan'] != 0:
                    optimizer_discriminator.zero_grad()
                    generated_for_discriminator = make_generated_dataparallel_safe(generated, x['driving'].shape[0])
                    losses_discriminator = discriminator_full(x, generated_for_discriminator)
                    loss_values = _loss_values_for_backward(losses_discriminator)
                    loss = sum(loss_values)

                    loss.backward(retain_graph=True)
                    optimizer_discriminator.step()
                    optimizer_discriminator.zero_grad()
                else:
                    losses_discriminator = {}

                losses_generator.update(losses_discriminator)
                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                logger.log_iter(losses=losses)
                print(losses)
                print()

            scheduler_generator.step()
            scheduler_discriminator.step()
            scheduler_kp_detector.step()
            scheduler_videocompressor.step()
            if scheduler_aux is not None:
                scheduler_aux.step()

            # Save both shared and branch-specific aliases for compatibility.
            logger.log_epoch(epoch, {'generator': generator_past,
                                     'generator_shared': generator_past,
                                     'generator_past': generator_past,
                                     'generator_future': generator_future,
                                     'discriminator': discriminator,
                                     'kp_detector': kp_detector_past,
                                     'kp_detector_shared': kp_detector_past,
                                     'kp_detector_past': kp_detector_past,
                                     'kp_detector_future': kp_detector_future,
                                     'videocompressor': videocompressor_past,
                                     'videocompressor_shared': videocompressor_past,
                                     'videocompressor_past': videocompressor_past,
                                     'videocompressor_future': videocompressor_future,
                                     'optimizer_generator': optimizer_generator,
                                     'optimizer_discriminator': optimizer_discriminator,
                                     'optimizer_kp_detector': optimizer_kp_detector,
                                     'optimizer_videocompressor': optimizer_videocompressor}, inp=x, out=generated)
