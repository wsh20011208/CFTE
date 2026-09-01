import numpy as np
import torch
import torch.nn.functional as F
import imageio
from torch import nn

import os
from skimage.draw import disk

import matplotlib.pyplot as plt
import collections
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2
except ImportError:
    cv2 = None
from modules.util import make_coordinate_grid
from flowvisual import *


class Logger:
    def __init__(self, log_dir, checkpoint_freq=100, visualizer_params=None, zfill_num=8, log_file_name='log.txt'):

        self.loss_list = []
        self.cpk_dir = log_dir
        self.visualizations_dir = os.path.join(log_dir, 'train-vis')
        if not os.path.exists(self.visualizations_dir):
            os.makedirs(self.visualizations_dir)
        self.log_file = open(os.path.join(log_dir, log_file_name), 'a')
        self.zfill_num = zfill_num
        self.visualizer = Visualizer(**visualizer_params)
        self.checkpoint_freq = checkpoint_freq
        self.epoch = 0
        self.best_loss = float('inf')
        self.names = None

    def log_scores(self, loss_names):
        loss_mean = np.array(self.loss_list).mean(axis=0)

        loss_string = "; ".join(["%s - %.5f" % (name, value) for name, value in zip(loss_names, loss_mean)])
        loss_string = str(self.epoch).zfill(self.zfill_num) + ") " + loss_string

        print(loss_string, file=self.log_file)
        self.loss_list = []
        self.log_file.flush()

    def visualize_rec(self, inp, out):
        source = inp['source_past'] if 'source_past' in inp else inp['source']
        source_future = inp['source_future'] if 'source_future' in inp else None
        image = self.visualizer.visualize(inp['driving'], source, out, source_future=source_future)
        imageio.imsave(os.path.join(self.visualizations_dir, "%s-rec.png" % str(self.epoch).zfill(self.zfill_num)), image)

    def save_cpk(self, emergent=False):
        cpk = {k: v.state_dict() for k, v in self.models.items()}
        cpk['epoch'] = self.epoch
        cpk_path = os.path.join(self.cpk_dir, '%s-checkpoint.pth.tar' % str(self.epoch).zfill(self.zfill_num)) 
        if not (os.path.exists(cpk_path) and emergent):
            torch.save(cpk, cpk_path)

    @staticmethod
    def load_cpk(checkpoint_path, generator=None, discriminator=None, kp_detector=None, videocompressor=None,
                 generator_future=None, kp_detector_future=None, videocompressor_future=None,
                 optimizer_generator=None, optimizer_discriminator=None, optimizer_kp_detector=None,
                 optimizer_videocompressor=None):
        checkpoint = torch.load(checkpoint_path)

        def _load_module(module, preferred_key, fallback_key=None, message='module'):
            if module is None:
                return
            key = preferred_key if preferred_key in checkpoint else fallback_key
            if key is None or key not in checkpoint:
                print('No %s in the state-dict. It will be randomly initialized' % message)
                return
            try:
                module.load_state_dict(checkpoint[key])
            except Exception as e:
                # New CMR checkpoints add a few generator parameters.  Allow loading
                # older single/shared CFTE checkpoints into the unchanged parts and
                # randomly initialize only the newly added modules.
                try:
                    incompatible = module.load_state_dict(checkpoint[key], strict=False)
                    print('Loaded %s from key %s with strict=False. Missing keys: %s; unexpected keys: %s' %
                          (message, key, incompatible.missing_keys, incompatible.unexpected_keys))
                except Exception:
                    print('Could not load %s from key %s: %s' % (message, key, str(e)))

        # New checkpoints contain *_past and *_future. Old single-CFTE checkpoints only contain
        # generator / kp_detector / videocompressor; in that case both branches are initialized
        # from the same pretrained single-reference CFTE weights.
        _load_module(generator, 'generator_past', 'generator', 'generator_past')
        _load_module(generator_future, 'generator_future', 'generator', 'generator_future')
        _load_module(kp_detector, 'kp_detector_past', 'kp_detector', 'kp_detector_past')
        _load_module(kp_detector_future, 'kp_detector_future', 'kp_detector', 'kp_detector_future')
        _load_module(videocompressor, 'videocompressor_past', 'videocompressor', 'videocompressor_past')
        _load_module(videocompressor_future, 'videocompressor_future', 'videocompressor', 'videocompressor_future')
        _load_module(discriminator, 'discriminator', None, 'discriminator')

        if optimizer_generator is not None and 'optimizer_generator' in checkpoint:
            try:
                optimizer_generator.load_state_dict(checkpoint['optimizer_generator'])
            except Exception as e:
                print('Generator optimizer is not compatible with this checkpoint and will be reinitialized: %s' % str(e))
        if optimizer_discriminator is not None and 'optimizer_discriminator' in checkpoint:
            try:
                optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator'])
            except Exception as e:
                print('Discriminator optimizer is not compatible with this checkpoint and will be reinitialized: %s' % str(e))
        if optimizer_kp_detector is not None and 'optimizer_kp_detector' in checkpoint:
            try:
                optimizer_kp_detector.load_state_dict(checkpoint['optimizer_kp_detector'])
            except Exception as e:
                print('KP detector optimizer is not compatible with this checkpoint and will be reinitialized: %s' % str(e))
        if optimizer_videocompressor is not None and 'optimizer_videocompressor' in checkpoint:
            try:
                optimizer_videocompressor.load_state_dict(checkpoint['optimizer_videocompressor'])
            except Exception as e:
                print('Videocompressor optimizer is not compatible with this checkpoint and will be reinitialized: %s' % str(e))
        return checkpoint['epoch']

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if 'models' in self.__dict__:
            self.save_cpk()
        self.log_file.close()

    def log_iter(self, losses):
        losses = collections.OrderedDict(losses.items())
        if self.names is None:
            self.names = list(losses.keys())
        self.loss_list.append(list(losses.values()))

    def log_epoch(self, epoch, models, inp, out):
        self.epoch = epoch
        self.models = models
        if (self.epoch + 1) % self.checkpoint_freq == 0:
            self.save_cpk()
        self.log_scores(self.names)
        self.visualize_rec(inp, out)


class Visualizer:
    def __init__(self, kp_size=5, draw_border=False, colormap='gist_rainbow',
                 show_headers=True, header_font_size=20, header_height=38,
                 row_label_width=150):
        self.kp_size = kp_size
        self.draw_border = draw_border
        self.colormap = plt.get_cmap(colormap)
        self.show_headers = show_headers
        self.header_font_size = header_font_size
        self.header_height = header_height
        self.row_label_width = row_label_width

        # Resolve the paper-style bold serif font once and reuse the exact same
        # font file for the top header, Past/Future row labels, and CMR labels.
        # The priority order is identical to the NoRefine logger.
        self._paper_font_path = self._resolve_paper_font_path()
        self._paper_font_cache = {}

    @staticmethod
    def _resolve_paper_font_path():
        """Resolve the same bold serif font family used by NoRefine."""
        font_candidates = [
            '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf',
            '/usr/share/fonts/truetype/msttcorefonts/timesbd.ttf',
            '/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf',
            '/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf',
            '/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf',
        ]
        for font_path in font_candidates:
            if os.path.exists(font_path):
                try:
                    ImageFont.truetype(font_path, 12)
                    return font_path
                except OSError:
                    pass

        raise RuntimeError(
            'No supported bold serif paper font was found. '
            'Install Times New Roman, Liberation Serif, or DejaVu Serif.'
        )

    def _load_paper_font(self, font_size=None):
        """Load one cached font object from the resolved NoRefine font file."""
        if font_size is None:
            font_size = self.header_font_size
        font_size = int(font_size)

        if font_size not in self._paper_font_cache:
            self._paper_font_cache[font_size] = ImageFont.truetype(
                self._paper_font_path,
                font_size,
            )

        return self._paper_font_cache[font_size]

    def _draw_centered_text(self, draw, box, text, font):
        left, top, right, bottom = box
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = left + (right - left - text_width) / 2 - bbox[0]
        y = top + (bottom - top - text_height) / 2 - bbox[1]
        draw.text((x, y), text, fill=(0, 0, 0), font=font)

    def _add_column_headers(self, image, labels, column_widths):
        """Add one header row only, matching the single-frame CFTE layout."""
        if not self.show_headers or not labels:
            return image

        header = Image.new('RGB', (image.shape[1], self.header_height),
                           color=(255, 255, 255))
        draw = ImageDraw.Draw(header)
        font = self._load_paper_font(self.header_font_size)

        x_offset = 0
        for label, width in zip(labels, column_widths):
            self._draw_centered_text(
                draw,
                (x_offset, 0, x_offset + width, self.header_height),
                label,
                font,
            )
            x_offset += width

        return np.concatenate([np.asarray(header), image], axis=0)

    @staticmethod
    def _ordinal(number):
        if 10 <= number % 100 <= 20:
            suffix = 'th'
        else:
            suffix = {1: 'st', 2: 'nd', 3: 'rd'}.get(number % 10, 'th')
        return '%d%s' % (number, suffix)

    def _make_row_label(self, text, row_height):
        """Create the left-most Past/Future row label cell."""
        cell = Image.new('RGB', (self.row_label_width, row_height),
                         color=(255, 255, 255))
        draw = ImageDraw.Draw(cell)
        self._draw_centered_text(
            draw,
            (0, 0, self.row_label_width, row_height),
            text,
            self._load_paper_font(self.header_font_size),
        )
        return np.asarray(cell, dtype=np.float32) / 255.0

    def _to_numpy_image_batch(self, tensor):
        """Convert BCHW tensor in [0,1] to NHWC numpy float in [0,1]."""
        array = tensor.detach().cpu().numpy()
        if array.ndim == 4 and array.shape[1] in (1, 3):
            array = np.transpose(array, [0, 2, 3, 1])
        if array.ndim == 4 and array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        return np.clip(array, 0.0, 1.0)

    def _resize_batch(self, images, size_hw, mode='nearest'):
        """Resize NHWC images to H,W using the existing display conversion."""
        images = np.asarray(images)
        if images.ndim != 4:
            raise ValueError('Expected NHWC batch, got shape %s' % (images.shape,))
        tensor = torch.from_numpy(np.transpose(images, [0, 3, 1, 2])).float()
        if mode == 'nearest':
            tensor = F.interpolate(tensor, size=size_hw, mode='nearest')
        else:
            tensor = F.interpolate(tensor, size=size_hw, mode=mode, align_corners=False)
        out = tensor.detach().cpu().numpy()
        out = np.transpose(out, [0, 2, 3, 1])
        return np.clip(out, 0.0, 1.0)

    def _blank_like(self, batch_size, height, width):
        return np.zeros((batch_size, height, width, 3), dtype=np.float32)

    def _tensor_or_blank(self, out, key, batch_size, height, width, is_occlusion=False):
        if key not in out:
            return self._blank_like(batch_size, height, width)
        tensor = out[key]
        if is_occlusion:
            tensor = tensor.detach().cpu().repeat(1, 3, 1, 1)
            array = np.transpose(tensor.numpy(), [0, 2, 3, 1])
            return self._resize_batch(array, (height, width))
        array = self._to_numpy_image_batch(tensor)
        if array.shape[1] != height or array.shape[2] != width:
            array = self._resize_batch(array, (height, width))
        return array

    def _flow_or_blank(self, out, key, batch_size, height, width):
        if key not in out:
            return self._blank_like(batch_size, height, width)
        flow_tensor = out[key].detach().cpu().numpy()
        flows = []
        for batch in range(flow_tensor.shape[0]):
            flow = flow_tensor[batch]
            if flow.ndim == 4:
                flow = flow.reshape(flow.shape[-3], flow.shape[-2], flow.shape[-1])
            flow_img = flow_to_image(flow).astype(np.float32) / 255.0
            flows.append(flow_img)
        flows = np.asarray(flows, dtype=np.float32)
        if flows.shape[1] != height or flows.shape[2] != width:
            flows = self._resize_batch(flows, (height, width))
        return np.clip(flows, 0.0, 1.0)

    def _image_tile(self, batch, idx):
        image = np.copy(batch[idx])
        if self.draw_border:
            image[:, [0, -1], :] = 1.0
            image[[0, -1], :, :] = 1.0
        return image

    def _annotate_tile(self, image, text):
        """Add a compact bold-serif label without changing the flow data."""
        image_uint8 = (255 * np.clip(image, 0.0, 1.0)).astype(np.uint8)
        tile = Image.fromarray(image_uint8)
        draw = ImageDraw.Draw(tile)
        font_size = max(12, min(18, self.header_font_size - 2))

        font = self._load_paper_font(font_size)

        bbox = draw.textbbox((0, 0), text, font=font)
        pad_x, pad_y = 5, 3
        box = (
            3,
            3,
            3 + (bbox[2] - bbox[0]) + 2 * pad_x,
            3 + (bbox[3] - bbox[1]) + 2 * pad_y,
        )
        draw.rectangle(box, fill=(255, 255, 255))
        draw.text(
            (box[0] + pad_x - bbox[0], box[1] + pad_y - bbox[1]),
            text,
            fill=(0, 0, 0),
            font=font,
        )
        return np.asarray(tile, dtype=np.float32) / 255.0

    def visualize(self, driving, source, out, source_future=None):
        """Create a single-CFTE-style table for the two-reference Refined FixedCMR model.

        The column titles appear once at the top.  The first column contains only
        the row identifiers: Past 1st, Future 1st, ..., Past 10th, Future 10th.
        No text is inserted between the image tiles.  The Dense Flow sequence is
        shown as initial -> CMR residual -> final.  In the shared CMR column,
        the Future row is explicitly marked CMR1 and the Past row is explicitly
        marked CMR2.  Deformed always uses the final refined branch motion field.
        """
        past_ref = self._to_numpy_image_batch(source)
        target = self._to_numpy_image_batch(driving)
        if source_future is None:
            source_future = source
        future_ref = self._to_numpy_image_batch(source_future)

        batch_size, height, width = target.shape[:3]

        visuals = {
            'past_ref': past_ref,
            'future_ref': future_ref,
            'target': target,
            'sparse_flow_past': self._flow_or_blank(
                out, 'sparse_motion_past', batch_size, height, width),
            'sparse_flow_future': self._flow_or_blank(
                out, 'sparse_motion_future', batch_size, height, width),
            'sparse_def_past': self._tensor_or_blank(
                out, 'sparse_deformed_past', batch_size, height, width),
            'sparse_def_future': self._tensor_or_blank(
                out, 'sparse_deformed_future', batch_size, height, width),
            # Dense-motion stages used by FixedCMR.
            #
            # CMR1 refines only the future branch; the past branch is unchanged
            # during this stage. CMR2 then refines only the past branch; the
            # future branch remains the CMR1 result. Showing the unchanged branch
            # at each stage keeps the two rows temporally aligned and makes the
            # complete refinement sequence explicit.
            'initial_dense_past': self._flow_or_blank(
                out, 'deformation_past_raw', batch_size, height, width),
            'initial_dense_future': self._flow_or_blank(
                out, 'deformation_future_raw', batch_size, height, width),

            # Compact CMR column shows the learned residual corrections,
            # not the already-refined final dense flows:
            #   Future row: CMR1 residual correction for the future branch.
            #   Past row:   CMR2 residual correction for the past branch.
            'cmr_flow_past': self._flow_or_blank(
                out, 'deformation_delta_past_cmr2',
                batch_size, height, width),
            'cmr_flow_future': self._flow_or_blank(
                out, 'deformation_delta_future_cmr1',
                batch_size, height, width),

            'final_dense_past': self._flow_or_blank(
                out, 'deformation_past', batch_size, height, width),
            'final_dense_future': self._flow_or_blank(
                out, 'deformation_future', batch_size, height, width),

            # These deformed images are explicitly the final branch outputs:
            # past warped by the CMR2-refined field and future warped by the
            # CMR1-refined field.
            'deformed_past': self._tensor_or_blank(
                out, 'deformed_past', batch_size, height, width),
            'deformed_future': self._tensor_or_blank(
                out, 'deformed_future', batch_size, height, width),
            'superposition': self._tensor_or_blank(
                out, 'deformed', batch_size, height, width),
            'occ_fused': self._tensor_or_blank(
                out, 'occlusion_map', batch_size, height, width, is_occlusion=True),
            'final': self._tensor_or_blank(
                out, 'prediction', batch_size, height, width),
        }

        rows = []
        for idx in range(batch_size):
            ordinal = self._ordinal(idx + 1)
            branch_rows = [
                (
                    'Past %s' % ordinal,
                    [
                        visuals['past_ref'],
                        visuals['target'],
                        visuals['sparse_flow_past'],
                        visuals['sparse_def_past'],
                        visuals['initial_dense_past'],
                        visuals['cmr_flow_past'],
                        visuals['final_dense_past'],
                        visuals['deformed_past'],
                        visuals['superposition'],
                        visuals['occ_fused'],
                        visuals['final'],
                        visuals['target'],
                    ],
                    'CMR2 (Past)',
                ),
                (
                    'Future %s' % ordinal,
                    [
                        visuals['future_ref'],
                        visuals['target'],
                        visuals['sparse_flow_future'],
                        visuals['sparse_def_future'],
                        visuals['initial_dense_future'],
                        visuals['cmr_flow_future'],
                        visuals['final_dense_future'],
                        visuals['deformed_future'],
                        visuals['superposition'],
                        visuals['occ_fused'],
                        visuals['final'],
                        visuals['target'],
                    ],
                    'CMR1 (Future)',
                ),
            ]

            for row_label, batches, cmr_annotation in branch_rows:
                tiles = []
                for column_idx, batch in enumerate(batches):
                    tile = self._image_tile(batch, idx)
                    # CMR is the sixth image column in each branch row.
                    if column_idx == 5:
                        tile = self._annotate_tile(tile, cmr_annotation)
                    tiles.append(tile)

                image_row = np.concatenate(tiles, axis=1)
                label_cell = self._make_row_label(row_label, height)
                rows.append(np.concatenate([label_cell, image_row], axis=1))

        image = np.concatenate(rows, axis=0)
        image = (255 * np.clip(image, 0.0, 1.0)).astype(np.uint8)

        labels = [
            'Sample', 'Source', 'Driving', 'Sparse Flow', 'Sparse Deformed',
            'Initial Dense Flow', 'CMR', 'Final Dense Flow', 'Deformed',
            'Superposition', 'Occlusion Map', 'Prediction', 'Target'
        ]
        column_widths = [self.row_label_width] + [width] * 12
        image = self._add_column_headers(image, labels, column_widths)
        return image
