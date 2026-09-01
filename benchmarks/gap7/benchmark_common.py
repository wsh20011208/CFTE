#!/usr/bin/env python3
import argparse
import csv
import importlib
import math
import os
import sys
from functools import lru_cache
from contextlib import contextmanager
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from skimage import io, img_as_ubyte
from tqdm import tqdm

PANEL_LAYOUT_VERSION = '2026-07-18-driving-v3-small-occ-title'


def set_seed(seed=1234):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@contextmanager
def project_import_context(project_dir):
    """Temporarily import modules from one CFTE project directory."""
    project_dir = str(Path(project_dir).expanduser().resolve())
    old_cwd = os.getcwd()
    old_path = list(sys.path)
    # Remove previously imported project-local modules that have identical names
    # across the four repositories.
    for name in list(sys.modules.keys()):
        if (name == 'logger' or name == 'frames_dataset' or name == 'augmentation' or
                name == 'flowvisual' or name == 'reconstruction' or name == 'animate' or
                name == 'train' or name == 'run' or name.startswith('modules') or
                name.startswith('sync_batchnorm')):
            sys.modules.pop(name, None)
    sys.path.insert(0, project_dir)
    os.chdir(project_dir)
    try:
        yield
    finally:
        os.chdir(old_cwd)
        sys.path = old_path


def resolve_eval_root(root_dir, split='test'):
    root = Path(root_dir).expanduser().resolve()
    if (root / split).is_dir():
        return root / split
    return root


def read_test_index(csv_path):
    with open(csv_path, newline='') as f:
        return list(csv.DictReader(f))


def read_video_with_project(project_dir, video_path, frame_shape):
    with project_import_context(project_dir):
        frames_dataset = importlib.import_module('frames_dataset')
        video = frames_dataset.read_video(str(video_path), frame_shape=tuple(frame_shape))
    # np array T,H,W,C float32 [0,1]
    return video.astype('float32')


def np_frame_to_tensor(frame_np, device):
    if frame_np.ndim == 2:
        frame_np = np.repeat(frame_np[..., None], 3, axis=2)
    if frame_np.shape[-1] == 4:
        frame_np = frame_np[..., :3]
    frame_np = np.clip(frame_np.astype('float32'), 0.0, 1.0)
    tensor = torch.from_numpy(frame_np.transpose(2, 0, 1)).unsqueeze(0).to(device)
    return tensor


def tensor_to_np_image(tensor):
    if isinstance(tensor, torch.Tensor):
        x = tensor.detach().float().cpu().clamp(0, 1)
        if x.dim() == 4:
            x = x[0]
        x = x.permute(1, 2, 0).numpy()
        return np.clip(x, 0.0, 1.0)
    return np.clip(tensor, 0.0, 1.0)


def tensor_scalar(x):
    if x is None:
        return float('nan')
    if isinstance(x, torch.Tensor):
        return float(x.detach().float().mean().cpu().item())
    try:
        return float(x)
    except Exception:
        return float('nan')


def safe_save_image(path, image_np):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image_np = np.clip(image_np, 0, 1)
    io.imsave(str(path), img_as_ubyte(image_np))


def make_panel(images, labels=None, pad=4):
    arrays = []
    for img in images:
        if img is None:
            continue
        arr = np.clip(img, 0, 1)
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        arrays.append(arr)
    if not arrays:
        return None
    h = max(a.shape[0] for a in arrays)
    padded = []
    for a in arrays:
        if a.shape[0] != h:
            # simple center padding if needed
            top = (h - a.shape[0]) // 2
            bottom = h - a.shape[0] - top
            a = np.pad(a, ((top, bottom), (0, 0), (0, 0)), mode='edge')
        padded.append(a)
        padded.append(np.ones((h, pad, 3), dtype='float32'))
    return np.concatenate(padded[:-1], axis=1)




def _to_rgb_float(image_np):
    """Convert HxW, HxWx1, HxWx3 or tensor-converted array to float RGB in [0, 1]."""
    arr = np.asarray(image_np)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.ndim == 3 and arr.shape[-1] > 3:
        arr = arr[..., :3]
    arr = arr.astype('float32')
    if arr.max() > 1.5:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _resize_rgb_float(image_np, cell_size):
    """Resize an RGB float image into a square white canvas without distortion."""
    try:
        from PIL import Image
        arr = _to_rgb_float(image_np)
        h, w = arr.shape[:2]
        scale = min(float(cell_size) / max(h, 1), float(cell_size) / max(w, 1))
        nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        im = Image.fromarray(img_as_ubyte(arr)).resize((nw, nh), Image.BILINEAR)
        canvas = Image.new('RGB', (cell_size, cell_size), (255, 255, 255))
        canvas.paste(im, ((cell_size - nw) // 2, (cell_size - nh) // 2))
        return np.asarray(canvas).astype('float32') / 255.0
    except Exception:
        arr = _to_rgb_float(image_np)
        # Fallback: simple skimage resize if PIL is unavailable.
        from skimage.transform import resize
        return resize(arr, (cell_size, cell_size), anti_aliasing=True, preserve_range=True).astype('float32')


@lru_cache(maxsize=32)
def _paper_serif_font(size=17, bold=True):
    """Load a thesis-style serif font without bundling any font file.

    Times New Roman is preferred when it is installed. Liberation Serif,
    Nimbus Roman, and DejaVu Serif are compatible server-side fallbacks.
    Set BENCHMARK_FONT or BENCHMARK_FONT_BOLD to override the search path.
    """
    from PIL import ImageFont

    env_key = 'BENCHMARK_FONT_BOLD' if bold else 'BENCHMARK_FONT'
    candidates = [os.environ.get(env_key), os.environ.get('BENCHMARK_FONT')]
    if bold:
        candidates += [
            '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman_Bold.ttf',
            '/usr/share/fonts/truetype/msttcorefonts/timesbd.ttf',
            '/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf',
            '/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf',
            '/usr/share/fonts/opentype/urw-base35/NimbusRoman-Bold.otf',
            '/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf',
        ]
    else:
        candidates += [
            '/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf',
            '/usr/share/fonts/truetype/msttcorefonts/times.ttf',
            '/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf',
            '/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf',
            '/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf',
            '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf',
        ]

    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _fit_paper_font(draw, text, max_width, start_size=17, min_size=10, bold=True):
    """Choose the largest thesis-style font that fits within max_width."""
    text = str(text)
    for size in range(int(start_size), int(min_size) - 1, -1):
        font = _paper_serif_font(size=size, bold=bold)
        bbox = draw.textbbox((0, 0), text, font=font)
        if bbox[2] - bbox[0] <= max_width:
            return font
    return _paper_serif_font(size=min_size, bold=bold)


def _draw_centered_paper_text(draw, box, text, start_size=17, min_size=10, bold=True):
    """Draw centered black serif text in (left, top, right, bottom)."""
    left, top, right, bottom = box
    text = str(text)
    font = _fit_paper_font(
        draw, text, max_width=max(1, right - left - 8),
        start_size=start_size, min_size=min_size, bold=bold)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = left + max(0, (right - left - text_w) // 2)
    y = top + max(0, (bottom - top - text_h) // 2) - bbox[1]
    draw.text((x, y), text, fill=(0, 0, 0), font=font)


def make_labeled_row(images, labels, cell_size=160, label_height=44, pad=4):
    """Make one horizontal sample row with thesis-style labels above each tile.

    Ordinary column headings use a slightly smaller thesis-style serif font.
    The Occlusion Map heading remains more prominent, with its diagnostic
    values placed on a smaller second line.
    """
    from PIL import Image, ImageDraw
    tiles = []
    for img, label in zip(images, labels):
        if img is None:
            continue
        rgb = _resize_rgb_float(img, cell_size)
        tile = Image.new('RGB', (cell_size, cell_size + label_height), (255, 255, 255))
        draw = ImageDraw.Draw(tile)
        label_text = str(label)[:96]
        if label_text.startswith('Occlusion Map'):
            parts = label_text.split('\n', 1)
            title_text = parts[0]
            metric_text = parts[1] if len(parts) > 1 else ''
            _draw_centered_paper_text(
                draw, (0, 0, cell_size, 25), title_text,
                start_size=12, min_size=8, bold=True)
            if metric_text:
                _draw_centered_paper_text(
                    draw, (0, 23, cell_size, label_height), metric_text,
                    start_size=10, min_size=8, bold=True)
        else:
            _draw_centered_paper_text(
                draw, (0, 0, cell_size, label_height), label_text,
                start_size=12, min_size=8, bold=True)
        tile.paste(Image.fromarray(img_as_ubyte(rgb)), (0, label_height))
        tiles.append(tile)
    if not tiles:
        return None
    width = sum(t.width for t in tiles) + pad * (len(tiles) - 1)
    height = max(t.height for t in tiles)
    out = Image.new('RGB', (width, height), (255, 255, 255))
    x = 0
    for t in tiles:
        out.paste(t, (x, 0))
        x += t.width + pad
    return np.asarray(out).astype('float32') / 255.0


def stack_labeled_rows(rows, title=None, pad=8):
    """Stack sample rows vertically with a thesis-style serif title."""
    if not rows:
        return None
    from PIL import Image, ImageDraw
    pil_rows = [Image.fromarray(img_as_ubyte(_to_rgb_float(r))) for r in rows if r is not None]
    if not pil_rows:
        return None
    width = max(r.width for r in pil_rows)
    title_height = 42 if title else 0
    height = title_height + sum(r.height for r in pil_rows) + pad * (len(pil_rows) - 1)
    out = Image.new('RGB', (width, height), (255, 255, 255))
    y = 0
    if title:
        draw = ImageDraw.Draw(out)
        _draw_centered_paper_text(
            draw, (0, 0, width, title_height), str(title),
            start_size=14, min_size=9, bold=True)
        y += title_height
    for r in pil_rows:
        out.paste(r, (0, y))
        y += r.height + pad
    return np.asarray(out).astype('float32') / 255.0




def display_model_name(model_name):
    """Return the thesis-facing method name without changing CSV identifiers."""
    key = str(model_name).strip().lower().replace('-', '_').replace(' ', '_')
    mapping = {
        'single': 'Single CFTE',
        'single_cfte': 'Single CFTE',
        'norefine': 'NoRefine',
        'no_refine': 'NoRefine',
        'fixedcmr': 'FixedCMR',
        'fixed_cmr': 'FixedCMR',
        'adaptivecmr': 'AdaptiveCMR',
        'adaptive_cmr': 'AdaptiveCMR',
    }
    return mapping.get(key, str(model_name))

def safe_video_id(name):
    return ''.join(c if c.isalnum() or c in ('-', '_', '.') else '_' for c in str(name))


def _first_tensor(out, keys):
    """Return the first tensor found under ``keys`` and its key name."""
    for key in keys:
        value = out.get(key)
        if isinstance(value, torch.Tensor):
            return value, key
    return None, None


def select_final_occlusion_and_feature(out, model_kind=None):
    """Select the final visibility mask and the feature tensor it gates.

    Current project versions expose these tensors as ``occlusion_map`` and
    ``deformed_feature``. The aliases below tolerate minor naming differences
    while preserving the Chapter-4 definition.
    """
    occlusion_keys = [
        'occlusion_map',
        'occlusion_map_final',
        'final_occlusion_map',
        'occlusion_map_sup',
        'occlusion_map_superposed',
        'post_superposition_occlusion_map',
    ]
    feature_keys = [
        'deformed_feature',
        'fused_deformed_feature',
        'deformed_feature_fused',
        'feature_superposed',
        'fused_feature',
        'feature_before_occlusion',
        'pre_occlusion_feature',
    ]
    occlusion, occlusion_key = _first_tensor(out, occlusion_keys)
    feature, feature_key = _first_tensor(out, feature_keys)
    return occlusion, feature, occlusion_key, feature_key


@torch.no_grad()
def compute_occlusion_suppression_diagnostics(
        out, model_kind, strong_suppression_threshold=0.1, epsilon=1e-8):
    """Compute the final visibility-mask diagnostics defined in Chapter 4.

    ``O`` is interpreted as a soft visibility mask: values close to one pass
    the warped feature, while values close to zero suppress it. Results are
    averaged across the batch; the formal evaluator uses batch size one.
    """
    nan = float('nan')
    result = {
        'occlusion_mean_visibility': nan,
        'occlusion_mean_suppression': nan,
        'occlusion_strong_suppression_ratio': nan,
        'occlusion_std': nan,
        'occlusion_feature_attenuation': nan,
        'occlusion_height': nan,
        'occlusion_width': nan,
        'occlusion_suppression_threshold': float(strong_suppression_threshold),
    }

    occlusion, feature, _, _ = select_final_occlusion_and_feature(out, model_kind)
    if occlusion is None:
        return result

    occ = occlusion.detach().float()
    if occ.dim() == 2:
        occ = occ.unsqueeze(0).unsqueeze(0)
    elif occ.dim() == 3:
        occ = occ.unsqueeze(1)
    if occ.dim() != 4 or occ.shape[1] != 1:
        return result

    occ = occ.clamp(0.0, 1.0)
    spatial_dims = tuple(range(1, occ.dim()))
    visibility_per_sample = occ.mean(dim=spatial_dims)
    suppression_per_sample = 1.0 - visibility_per_sample
    strong_ratio_per_sample = (occ < float(strong_suppression_threshold)).float().mean(
        dim=spatial_dims)
    std_per_sample = occ.flatten(start_dim=1).std(dim=1, unbiased=False)

    result.update({
        'occlusion_mean_visibility': tensor_scalar(visibility_per_sample),
        'occlusion_mean_suppression': tensor_scalar(suppression_per_sample),
        'occlusion_strong_suppression_ratio': tensor_scalar(strong_ratio_per_sample),
        'occlusion_std': tensor_scalar(std_per_sample),
        'occlusion_height': int(occ.shape[-2]),
        'occlusion_width': int(occ.shape[-1]),
    })

    if feature is None:
        return result

    feat = feature.detach().float()
    if feat.dim() == 3:
        feat = feat.unsqueeze(0)
    if feat.dim() != 4 or feat.shape[0] != occ.shape[0]:
        return result

    occ_for_feature = occ
    if occ_for_feature.shape[-2:] != feat.shape[-2:]:
        occ_for_feature = F.interpolate(
            occ_for_feature,
            size=feat.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )

    reduce_dims = tuple(range(1, feat.dim()))
    before_l1 = feat.abs().sum(dim=reduce_dims)
    after_l1 = (feat * occ_for_feature).abs().sum(dim=reduce_dims)
    attenuation_per_sample = 1.0 - after_l1 / (before_l1 + float(epsilon))
    result['occlusion_feature_attenuation'] = tensor_scalar(attenuation_per_sample)
    return result


def occlusion_to_rgb(out):
    occlusion, _, _, _ = select_final_occlusion_and_feature(out)
    if occlusion is None:
        return None
    occ = occlusion.detach().float().cpu()
    if occ.dim() == 4:
        occ_np = occ[0, 0].numpy()
    elif occ.dim() == 3:
        occ_np = occ[0].numpy()
    else:
        occ_np = occ.numpy()
    return np.repeat(np.clip(occ_np, 0, 1)[..., None], 3, axis=2)


def build_visual_sample_row(model_kind, model_name, past_np, future_np, target_np, pred, out,
                            past_idx, target_idx, future_idx, metrics=None,
                            cell_size=160):
    """Create a logger-style row for one sampled target frame."""
    pred_np = tensor_to_np_image(pred)
    metrics = metrics or {}

    def _metric_label(prefix, d_key, l_key):
        d_val = metrics.get(d_key, float('nan'))
        l_val = metrics.get(l_key, float('nan'))
        txt = f'{prefix} D={d_val:.3f}' if math.isfinite(d_val) else f'{prefix} D=nan'
        if math.isfinite(l_val):
            txt += f' L={l_val:.3f}'
        return txt

    # Final decoded prediction quality: prediction vs target.
    metric_txt = _metric_label('pred', 'dists', 'lpips')
    # Two-frame intermediate quality: superposed/deformed image vs target.
    superposed_metric_txt = _metric_label('superposed', 'superposed_dists', 'superposed_lpips')

    images = []
    labels = []
    if model_kind == 'single':
        images.append(past_np); labels.append(f'Past {past_idx}')
        # The target frame is also shown as the driving input for consistency
        # with the training/logger visualization, while the final column remains
        # the untouched target reference used for visual comparison.
        images.append(target_np); labels.append(f'Driving {target_idx}')
        if 'deformed' in out and isinstance(out['deformed'], torch.Tensor):
            images.append(tensor_to_np_image(out['deformed'])); labels.append('Deformed')
        occ = occlusion_to_rgb(out)
        if occ is not None:
            vis = metrics.get('occlusion_mean_visibility', float('nan'))
            sup = metrics.get('occlusion_mean_suppression', float('nan'))
            att = metrics.get('occlusion_feature_attenuation', float('nan'))
            metric_parts = []
            if math.isfinite(vis) and math.isfinite(sup):
                metric_parts.append(f'μ={vis:.3f}  s={sup:.3f}')
            if math.isfinite(att):
                metric_parts.append(f'a={att:.3f}')
            occ_label = 'Occlusion Map'
            if metric_parts:
                occ_label += '\n' + '  '.join(metric_parts)
            images.append(occ); labels.append(occ_label)
        images.append(pred_np); labels.append(metric_txt.replace('pred', 'Prediction', 1))
        images.append(target_np); labels.append(f'Target {target_idx}')
    else:
        images.append(past_np); labels.append(f'Past {past_idx}')
        images.append(target_np); labels.append(f'Driving {target_idx}')
        images.append(future_np); labels.append(f'Future {future_idx}')
        if 'deformed_past' in out and isinstance(out['deformed_past'], torch.Tensor):
            images.append(tensor_to_np_image(out['deformed_past'])); labels.append('Deformed Past')
        if 'deformed_future' in out and isinstance(out['deformed_future'], torch.Tensor):
            images.append(tensor_to_np_image(out['deformed_future'])); labels.append('Deformed Future')
        if 'deformed' in out and isinstance(out['deformed'], torch.Tensor):
            images.append(tensor_to_np_image(out['deformed'])); labels.append(
                superposed_metric_txt.replace('superposed', 'Superposition', 1))
        occ = occlusion_to_rgb(out)
        if occ is not None:
            vis = metrics.get('occlusion_mean_visibility', float('nan'))
            sup = metrics.get('occlusion_mean_suppression', float('nan'))
            att = metrics.get('occlusion_feature_attenuation', float('nan'))
            metric_parts = []
            if math.isfinite(vis) and math.isfinite(sup):
                metric_parts.append(f'μ={vis:.3f}  s={sup:.3f}')
            if math.isfinite(att):
                metric_parts.append(f'a={att:.3f}')
            occ_label = 'Occlusion Map'
            if metric_parts:
                occ_label += '\n' + '  '.join(metric_parts)
            images.append(occ); labels.append(occ_label)
        images.append(pred_np); labels.append(metric_txt.replace('pred', 'Prediction', 1))
        images.append(target_np); labels.append(f'Target {target_idx}')
    return make_labeled_row(images, labels, cell_size=cell_size)

def image_sharpness_laplacian_var(image_np):
    try:
        import cv2
        gray = np.dot(image_np[..., :3], [0.299, 0.587, 0.114]).astype('float32')
        return float(cv2.Laplacian(gray, cv2.CV_32F).var())
    except Exception:
        return float('nan')


def basic_frame_stats(prefix, image_np):
    arr = np.clip(image_np, 0, 1).astype('float32')
    luma = np.dot(arr[..., :3], [0.299, 0.587, 0.114])
    return {
        f'{prefix}_mean': float(arr.mean()),
        f'{prefix}_std': float(arr.std()),
        f'{prefix}_luma_mean': float(luma.mean()),
        f'{prefix}_luma_std': float(luma.std()),
        f'{prefix}_sharpness': image_sharpness_laplacian_var(arr),
    }



def import_project_metrics(project_dir):
    """Import the original CFTE evaluation metric implementations.

    The project file evaluate/multiMetric.py imports `utils` as a top-level
    module, so the evaluate/ directory must be temporarily added to sys.path.
    """
    project_dir = Path(project_dir).expanduser().resolve()
    eval_dir = project_dir / 'evaluate'
    sys.path.insert(0, str(eval_dir))
    try:
        multi_metric = importlib.import_module('evaluate.multiMetric')
    except Exception:
        # Some copies are not treated as a package; fall back to direct import.
        multi_metric = importlib.import_module('multiMetric')
    return multi_metric


def image_np_to_uint8_chw(image_np):
    """Convert HxWxC float [0,1] image to CxHxW uint8, matching original evaluator."""
    arr = _to_rgb_float(image_np)
    arr_u8 = np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)
    return arr_u8.transpose(2, 0, 1)


def tensor_to_uint8_chw(tensor):
    return image_np_to_uint8_chw(tensor_to_np_image(tensor))


def original_psnr_ssim_from_uint8_chw(img_a_chw, img_b_chw, multi_metric):
    """Original-project PSNR/SSIM: compute RGB channels separately, then average."""
    psnrs = []
    ssims = []
    for c in range(3):
        psnrs.append(float(multi_metric.cacl_psnr(img_a_chw[c], img_b_chw[c])))
        ssims.append(float(multi_metric.cacl_ssim(img_a_chw[c], img_b_chw[c])))
    return float(np.mean(psnrs)), float(np.mean(ssims))


def compute_pair_basic_metrics(pred_np, target_np, multi_metric=None):
    """Compute MAE/MSE plus original-project PSNR/SSIM for numpy images."""
    pred_np = np.clip(pred_np, 0, 1).astype('float32')
    target_np = np.clip(target_np, 0, 1).astype('float32')
    out = {}
    out['mae'] = float(np.mean(np.abs(pred_np - target_np)))
    out['mse'] = float(np.mean((pred_np - target_np) ** 2))
    if multi_metric is not None:
        try:
            pred_chw = image_np_to_uint8_chw(pred_np)
            target_chw = image_np_to_uint8_chw(target_np)
            psnr, ssim = original_psnr_ssim_from_uint8_chw(target_chw, pred_chw, multi_metric)
            out['psnr'] = psnr
            out['ssim'] = ssim
        except Exception as e:
            print(f'[WARN] original PSNR/SSIM failed: {e}')
            out['psnr'] = float('nan')
            out['ssim'] = float('nan')
    else:
        out['psnr'] = float('nan')
        out['ssim'] = float('nan')
    return out


class OriginalProjectMetricComputer:
    """Metric computer using the original CFTE project metric implementations.

    Uses evaluate/multiMetric.py for all four reported quality metrics:
      - DISTS
      - LPIPSvgg
      - PSNR via cacl_psnr
      - SSIM via cacl_ssim
    """
    def __init__(self, project_dir, device, compute_lpips=True, require_lpips=False,
                 external_dists_model=None):
        self.project_dir = Path(project_dir).expanduser().resolve()
        self.device = device
        self.compute_lpips = compute_lpips
        self.lpips_model = None
        self.lpips_error = None
        with project_import_context(self.project_dir):
            self.multi_metric = import_project_metrics(self.project_dir)
            # AdaptiveCMR already needs a differentiable DISTS model inside its
            # correction loop. Reuse that same frozen model for evaluation to
            # avoid keeping a second VGG16-based DISTS network on the GPU.
            self.dists = (external_dists_model if external_dists_model is not None
                          else self.multi_metric.DISTS().to(device).eval())
            self.dists.eval()
            for param in self.dists.parameters():
                param.requires_grad_(False)
            if compute_lpips:
                try:
                    self.lpips_model = self.multi_metric.LPIPSvgg().to(device).eval()
                    # LPIPSvgg stores learned weights in a Python list, not as buffers.
                    # Move them manually to avoid CPU/CUDA device mismatch.
                    self.lpips_model.weights = [(k, v.to(device)) for k, v in self.lpips_model.weights]
                    for param in self.lpips_model.parameters():
                        param.requires_grad_(False)
                    print('LPIPS enabled: original project LPIPSvgg')
                except Exception as e:
                    self.lpips_error = e
                    msg = (
                        f'Original-project LPIPSvgg unavailable; lpips column will be NaN: {e}\n'
                        'Check evaluate/weights/LPIPSvgg.pt and torchvision VGG16 weights/cache. '
                        'Use --require_lpips to stop immediately instead of writing NaN.'
                    )
                    if require_lpips:
                        raise RuntimeError(msg)
                    print(f'[WARN] {msg}')

    @torch.no_grad()
    def compute(self, pred, target):
        pred = pred.detach().float().clamp(0, 1)
        target = target.detach().float().clamp(0, 1)
        pred_np = tensor_to_np_image(pred)
        target_np = tensor_to_np_image(target)
        out = compute_pair_basic_metrics(pred_np, target_np, self.multi_metric)
        try:
            out['dists'] = tensor_scalar(self.dists(target, pred, as_loss=True))
        except Exception as e:
            print(f'[WARN] original DISTS failed: {e}')
            out['dists'] = float('nan')
        if self.lpips_model is not None:
            try:
                lp = self.lpips_model(target, pred, as_loss=True)
                out['lpips'] = tensor_scalar(lp)
            except Exception as e:
                print(f'[WARN] original LPIPSvgg failed: {e}')
                out['lpips'] = float('nan')
        else:
            out['lpips'] = float('nan')
        return out

    @torch.no_grad()
    def compute_dists_only(self, pred, target):
        pred = pred.detach().float().clamp(0, 1)
        target = target.detach().float().clamp(0, 1)
        try:
            return tensor_scalar(self.dists(target, pred, as_loss=True))
        except Exception as e:
            print(f'[WARN] original DISTS-only failed: {e}')
            return float('nan')

def load_config(config_path):
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_single_model(project_dir, config, checkpoint, device):
    with project_import_context(project_dir):
        OcclusionAwareGenerator = importlib.import_module('modules.generator').OcclusionAwareGenerator
        KPDetector = importlib.import_module('modules.keypoint_detector').KPDetector
        VideoCompressor = importlib.import_module('modules.RDloss').VideoCompressor
        Logger = importlib.import_module('logger').Logger

        generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                            **config['model_params']['common_params']).to(device)
        kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                                 **config['model_params']['common_params']).to(device)
        videocompressor = VideoCompressor(**config['model_params']['videocompressor_params']).to(device)
        Logger.load_cpk(checkpoint, generator=generator, kp_detector=kp_detector, videocompressor=videocompressor)
        generator.eval(); kp_detector.eval(); videocompressor.eval()
        return generator, kp_detector, videocompressor


def build_dual_model(project_dir, config, checkpoint, device):
    with project_import_context(project_dir):
        OcclusionAwareGenerator = importlib.import_module('modules.generator').OcclusionAwareGenerator
        KPDetector = importlib.import_module('modules.keypoint_detector').KPDetector
        VideoCompressor = importlib.import_module('modules.RDloss').VideoCompressor
        Logger = importlib.import_module('logger').Logger

        # Match run.py: one shared branch object used for both past and future.
        generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                            **config['model_params']['common_params']).to(device)
        kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                                 **config['model_params']['common_params']).to(device)
        videocompressor = VideoCompressor(**config['model_params']['videocompressor_params']).to(device)
        Logger.load_cpk(checkpoint,
                        generator=generator, generator_future=generator,
                        kp_detector=kp_detector, kp_detector_future=kp_detector,
                        videocompressor=videocompressor, videocompressor_future=videocompressor)
        generator.eval(); kp_detector.eval(); videocompressor.eval()
        return generator, kp_detector, videocompressor


@torch.no_grad()
def forward_single(model_pack, source, target):
    generator, kp_detector, videocompressor = model_pack
    heatmap_source = kp_detector(source)
    heatmap_target = kp_detector(target)
    total_bits, quant_target = videocompressor(heatmap_target, heatmap_source)
    out = generator(source, heatmap_source=heatmap_source, heatmap_driving=quant_target)

    # The current Single CFTE generator returns the final occlusion map but does
    # not expose the encoded feature immediately before masking. Reconstruct the
    # exact same pre-mask feature path for the Chapter-4 attenuation diagnostic:
    # first convolution -> down blocks -> warp by the returned dense deformation.
    if ('deformed_feature' not in out and
            isinstance(out.get('deformation'), torch.Tensor)):
        encoded_feature = generator.first(source)
        for down_block in generator.down_blocks:
            encoded_feature = down_block(encoded_feature)
        out['encoded_feature'] = encoded_feature
        out['deformed_feature'] = generator.deform_input(
            encoded_feature, out['deformation'])

    out['heatmap_source'] = heatmap_source
    out['heatmap_driving'] = quant_target
    bpp = total_bits / (source.shape[0] * source.shape[2] * source.shape[3])
    return out, bpp, {'bpp': bpp, 'bpp_past': bpp, 'bpp_future': torch.zeros_like(bpp)}


def forward_dual(model_pack, past, future, target, project_dir, use_adaptive=False, dists_for_adaptive=None,
                 weight_past=0.5, weight_future=0.5):
    generator, kp_detector, videocompressor = model_pack
    # The adaptive path needs gradients inside the inner loop. Do not wrap the whole
    # method in no_grad when use_adaptive=True.
    grad_context = torch.enable_grad() if use_adaptive else torch.no_grad()
    with grad_context:
        heatmap_past = kp_detector(past)
        heatmap_target_past = kp_detector(target)
        bits_past, quant_target_past = videocompressor(heatmap_target_past, heatmap_past)
        generated_past = generator(past, heatmap_source=heatmap_past, heatmap_driving=quant_target_past)
        generated_past.update({'heatmap_source': heatmap_past, 'heatmap_driving': quant_target_past})

        heatmap_future = kp_detector(future)
        heatmap_target_future = kp_detector(target)
        bits_future, quant_target_future = videocompressor(heatmap_target_future, heatmap_future)
        generated_future = generator(future, heatmap_source=heatmap_future, heatmap_driving=quant_target_future)
        generated_future.update({'heatmap_source': heatmap_future, 'heatmap_driving': quant_target_future})

        kwargs = {}
        if use_adaptive:
            kwargs = {'target_image': target, 'dists_model': dists_for_adaptive}
        if hasattr(generator, 'forward_superposed_after_deform'):
            out = generator.forward_superposed_after_deform(
                past, future, generated_past, generated_future, weight_past, weight_future, **kwargs)
        else:
            out = {
                'prediction': weight_past * generated_past['prediction'] + weight_future * generated_future['prediction']
            }
            if 'deformed' in generated_past and 'deformed' in generated_future:
                out['deformed'] = weight_past * generated_past['deformed'] + weight_future * generated_future['deformed']

    # Detach outputs for metric computation and saving.
    out['prediction_past'] = generated_past.get('prediction')
    out['prediction_future'] = generated_future.get('prediction')
    out.setdefault('deformed_past', generated_past.get('deformed'))
    out.setdefault('deformed_future', generated_future.get('deformed'))
    out['heatmap_source_past'] = heatmap_past
    out['heatmap_source_future'] = heatmap_future

    bpp_past = bits_past / (past.shape[0] * past.shape[2] * past.shape[3])
    bpp_future = bits_future / (future.shape[0] * future.shape[2] * future.shape[3])
    bpp = bpp_past + bpp_future
    return out, bpp, {'bpp': bpp, 'bpp_past': bpp_past, 'bpp_future': bpp_future}


def flatten_extra_outputs(out):
    """Export scalar model diagnostics while excluding large tensor fields."""
    result = {}
    scalar_keys = [
        'adaptive_dists_initial',
        'adaptive_dists_final',
        'adaptive_dists_sequence_mean',
        'adaptive_dists_curve_length',
        'adaptive_iterations',
        'adaptive_delta_mean',
    ]
    for key in scalar_keys:
        if key in out:
            result[key] = tensor_scalar(out[key])

    initial = result.get('adaptive_dists_initial')
    final = result.get('adaptive_dists_final')
    if initial is not None and final is not None:
        result['adaptive_dists_improvement'] = initial - final
    return result


def write_rows_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if len(rows) == 0:
        with path.open('w') as f:
            f.write('')
        return
    # Preserve a stable core order, then append any optional columns alphabetically.
    core = [
        'sample_id', 'video_id', 'video_rel_path', 'num_frames',
        'past_idx', 'target_idx', 'future_idx', 'gap_past', 'gap_future',
        'past_frame_name', 'target_frame_name', 'future_frame_name', 'model',
        # Final decoded prediction vs target.
        'dists', 'lpips', 'psnr', 'ssim',
        'final_dists', 'final_lpips', 'final_psnr', 'final_ssim',
        # Rate / bitrate.
        'bpp', 'bpp_total', 'bpp_past', 'bpp_future', 'rdloss',
        # Superposed/deformed intermediate vs target.
        'superposed_dists', 'superposed_lpips', 'superposed_psnr', 'superposed_ssim',
        # Final visibility-mask suppression diagnostics.
        'occlusion_mean_visibility', 'occlusion_mean_suppression',
        'occlusion_strong_suppression_ratio', 'occlusion_std',
        'occlusion_feature_attenuation',
        'occlusion_height', 'occlusion_width',
        'occlusion_suppression_threshold',
        # AdaptiveCMR convergence diagnostics.
        'adaptive_dists_initial', 'adaptive_dists_final',
        'adaptive_dists_improvement', 'adaptive_dists_sequence_mean',
        'adaptive_dists_curve_length', 'adaptive_iterations',
        'adaptive_delta_mean',
    ]
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
    fields = [k for k in core if k in all_keys] + sorted(k for k in all_keys if k not in core)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_csv(args, model_kind):
    set_seed(args.seed)
    project_dir = Path(args.project_dir).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    test_csv = Path(args.test_csv).expanduser().resolve()
    output_csv = Path(args.output_csv).expanduser().resolve()
    save_images = Path(args.save_images).expanduser().resolve() if args.save_images else None
    save_video_panels = Path(args.save_video_panels).expanduser().resolve() if getattr(args, 'save_video_panels', None) else None

    required_paths = {
        'project directory': project_dir,
        'config file': config_path,
        'checkpoint file': checkpoint_path,
        'test index CSV': test_csv,
    }
    for label, path in required_paths.items():
        exists = path.is_dir() if label == 'project directory' else path.is_file()
        if not exists:
            raise FileNotFoundError(f'Missing {label}: {path}')

    config = load_config(config_path)
    root_dir = Path(args.root_dir).expanduser().resolve() if args.root_dir else Path(config['dataset_params']['root_dir']).expanduser().resolve()
    eval_root = resolve_eval_root(root_dir, args.split)
    frame_shape = tuple(config['dataset_params'].get('frame_shape', [256, 256, 3]))
    if not eval_root.is_dir():
        raise FileNotFoundError(f'Missing evaluation dataset directory: {eval_root}')

    # These four uploaded projects call Tensor.cuda() inside dense_motion.py.
    # CPU evaluation is therefore not supported. Setting the current device is
    # also required when a nonzero logical GPU index is selected.
    if not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA is required by the current CFTE project code '
            '(dense_motion.py contains explicit .cuda() calls).')
    if args.device < 0 or args.device >= torch.cuda.device_count():
        raise ValueError(
            f'Invalid CUDA device {args.device}; available logical devices: '
            f'0..{torch.cuda.device_count() - 1}')
    torch.cuda.set_device(args.device)
    device = torch.device(f'cuda:{args.device}')
    print(f'Using device: {device} ({torch.cuda.get_device_name(args.device)})')
    print(f'Project: {project_dir}')
    print(f'Checkpoint: {checkpoint_path}')
    print(f'Test index: {test_csv}')
    print(f'Eval root: {eval_root}')

    if model_kind == 'single':
        model_pack = build_single_model(project_dir, config, str(checkpoint_path), device)
        model_name = args.model_name or 'single_cfte'
        adaptive_dists_model = None
    else:
        model_pack = build_dual_model(project_dir, config, str(checkpoint_path), device)
        model_name = args.model_name or model_kind
        adaptive_dists_model = None
        if model_kind == 'adaptive_cmr':
            with project_import_context(project_dir):
                DISTS = importlib.import_module('modules.dists').DISTS
                adaptive_dists_model = DISTS().to(device).eval()

    metric_computer = OriginalProjectMetricComputer(
        project_dir,
        device,
        compute_lpips=not args.no_lpips,
        require_lpips=args.require_lpips,
        external_dists_model=adaptive_dists_model,
    )
    rows = read_test_index(test_csv)
    out_rows = []
    # Keep only one decoded video in RAM. The deterministic CSV is grouped by
    # video, so retaining every previously decoded video would unnecessarily
    # grow memory throughout the full 444-video benchmark.
    cached_video_path = None
    cached_video = None

    panel_dir = None
    current_panel_video_id = None
    current_panel_rows = []
    if save_video_panels is not None:
        panel_dir = save_video_panels / model_name
        panel_dir.mkdir(parents=True, exist_ok=True)

    def flush_current_video_panel():
        nonlocal current_panel_video_id, current_panel_rows
        if panel_dir is None or current_panel_video_id is None or not current_panel_rows:
            current_panel_rows = []
            return
        title = (
            f'{display_model_name(model_name)} | video_id={current_panel_video_id} | '
            'each row = one sampled target frame'
        )
        panel = stack_labeled_rows(current_panel_rows, title=title)
        if panel is not None:
            safe_save_image(
                panel_dir / f'{safe_video_id(current_panel_video_id)}.png',
                panel,
            )
        current_panel_rows = []

    lambda_rd = float(config['train_params']['loss_weights'].get('rdlambda', args.rdlambda))
    occlusion_keys_reported = False

    for row in tqdm(rows, desc=f'Evaluating {model_name}'):
        video_rel_path = row['video_rel_path']
        video_path = eval_root / video_rel_path
        if cached_video_path != video_path:
            cached_video = read_video_with_project(project_dir, video_path, frame_shape)
            cached_video_path = video_path
        video = cached_video
        past_idx = int(row['past_idx'])
        target_idx = int(row['target_idx'])
        future_idx = int(row['future_idx'])
        if future_idx >= len(video) or target_idx >= len(video) or past_idx >= len(video):
            print(f'[WARN] Skip {row["sample_id"]}: index exceeds video length {len(video)}')
            continue

        past_np = video[past_idx]
        target_np = video[target_idx]
        future_np = video[future_idx]
        past = np_frame_to_tensor(past_np, device)
        target = np_frame_to_tensor(target_np, device)
        future = np_frame_to_tensor(future_np, device)

        if model_kind == 'single':
            with torch.no_grad():
                out, bpp, bpp_parts = forward_single(model_pack, past, target)
        else:
            out, bpp, bpp_parts = forward_dual(
                model_pack, past, future, target, project_dir,
                use_adaptive=(model_kind == 'adaptive_cmr'),
                dists_for_adaptive=adaptive_dists_model,
                weight_past=args.weight_past,
                weight_future=args.weight_future)

        pred = out['prediction'].detach().clamp(0, 1)

        # Final decoded prediction vs target.
        # These four quality metrics are computed with the original CFTE evaluation code
        # in evaluate/multiMetric.py: DISTS, LPIPSvgg, cacl_psnr, and cacl_ssim.
        metrics = metric_computer.compute(pred, target)
        for metric_name in ['dists', 'lpips', 'psnr', 'ssim']:
            if metric_name in metrics:
                metrics[f'final_{metric_name}'] = metrics[metric_name]

        bpp_val = tensor_scalar(bpp)
        metrics['bpp'] = bpp_val
        metrics['bpp_past'] = tensor_scalar(bpp_parts.get('bpp_past'))
        metrics['bpp_future'] = tensor_scalar(bpp_parts.get('bpp_future'))
        metrics['rdloss'] = lambda_rd * bpp_val + metrics.get('dists', float('nan'))
        # Explicit aliases for bitrate/rate analysis.
        metrics['bpp_total'] = bpp_val

        # Superimposed/deformed intermediate image vs target.
        # This is defined only for two-frame models.
        # Single CFTE has no two-frame superposition, so superposed_* columns are not written.
        # For NoRefine: raw two-frame superposed deformed image before the decoder.
        # For FixedCMR: CMR-refined superposed deformed image before the decoder.
        # For AdaptiveCMR: adaptive-corrected superposed deformed image after the iteration loop.
        # All superposed_* metrics are computed through evaluate/multiMetric.py.
        if model_kind != 'single' and 'deformed' in out and isinstance(out['deformed'], torch.Tensor):
            superposed = out['deformed'].detach().clamp(0, 1)
            superposed_metrics = metric_computer.compute(superposed, target)
            for metric_name in ['dists', 'lpips', 'psnr', 'ssim']:
                metrics[f'superposed_{metric_name}'] = superposed_metrics.get(metric_name, float('nan'))

        # Chapter-4 final visibility-mask diagnostics. Single CFTE uses the
        # branch-level mask and deformed feature; dual-reference methods use
        # the final post-superposition mask and fused deformed feature.
        if not occlusion_keys_reported:
            occ_tensor, feature_tensor, occ_key, feature_key = select_final_occlusion_and_feature(
                out, model_kind)
            print(
                'Occlusion diagnostics tensors: '
                f'mask={occ_key or "MISSING"}, '
                f'pre-decoder feature={feature_key or "MISSING"}'
            )
            if occ_tensor is None:
                print('[WARN] Final occlusion map was not found; all occlusion diagnostics will be NaN.')
            elif feature_tensor is None:
                print('[WARN] Pre-mask feature was not found; feature attenuation will be NaN.')
            occlusion_keys_reported = True

        metrics.update(compute_occlusion_suppression_diagnostics(
            out,
            model_kind=model_kind,
            strong_suppression_threshold=args.occlusion_suppression_threshold,
            epsilon=args.occlusion_epsilon,
        ))

        result = dict(row)
        result['model'] = model_name
        for k, v in metrics.items():
            result[k] = v
        result.update(flatten_extra_outputs(out))
        out_rows.append(result)

        # Optional: collect logger-style rows and write one visual panel per video ID.
        # This treats all sampled target frames from the same video as one visual batch.
        if save_video_panels is not None:
            vid = row['video_id']
            if current_panel_video_id is None:
                current_panel_video_id = vid
            elif vid != current_panel_video_id:
                flush_current_video_panel()
                current_panel_video_id = vid

            if (args.video_panel_max_samples <= 0 or
                    len(current_panel_rows) < args.video_panel_max_samples):
                sample_row = build_visual_sample_row(
                    model_kind=model_kind, model_name=model_name,
                    past_np=past_np, future_np=future_np, target_np=target_np,
                    pred=pred, out=out,
                    past_idx=past_idx, target_idx=target_idx,
                    future_idx=future_idx,
                    metrics=metrics, cell_size=args.video_panel_cell_size)
                if sample_row is not None:
                    current_panel_rows.append(sample_row)

        if save_images is not None and (args.save_all_images or len(out_rows) <= args.max_saved_images):
            sid = row['sample_id']
            vid = row['video_id']
            base = save_images / f'{sid}_{vid}_t{target_idx:06d}'
            base.mkdir(parents=True, exist_ok=True)
            safe_save_image(base / 'past.png', past_np)
            safe_save_image(base / 'target.png', target_np)
            if model_kind != 'single':
                safe_save_image(base / 'future.png', future_np)
            safe_save_image(base / f'prediction_{model_name}.png', tensor_to_np_image(pred))
            if 'deformed' in out and isinstance(out['deformed'], torch.Tensor):
                safe_save_image(base / 'superposed_deformed.png', tensor_to_np_image(out['deformed']))
            for key in ['deformed_past', 'deformed_future']:
                if key in out and isinstance(out[key], torch.Tensor):
                    safe_save_image(base / f'{key}.png', tensor_to_np_image(out[key]))
            occ_tensor, _, _, _ = select_final_occlusion_and_feature(out, model_kind)
            if isinstance(occ_tensor, torch.Tensor):
                occ = occ_tensor.detach().float().cpu()
                if occ.dim() == 4:
                    occ_np = occ[0, 0].numpy()
                    safe_save_image(base / 'occlusion_map.png', np.repeat(occ_np[..., None], 3, axis=2))
            panel_imgs = [past_np, target_np]
            if model_kind != 'single':
                panel_imgs.insert(1, future_np)
            panel_imgs.append(tensor_to_np_image(pred))
            panel = make_panel(panel_imgs)
            if panel is not None:
                safe_save_image(base / f'panel_{model_name}.png', panel)

    if save_video_panels is not None:
        flush_current_video_panel()
        print(f'Wrote streamed per-video visual panels to {panel_dir}')

    write_rows_csv(output_csv, out_rows)
    print(f'Wrote {len(out_rows)} rows to {output_csv}')


def add_common_eval_args(parser):
    parser.add_argument('--project_dir', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--test_csv', required=True)
    parser.add_argument('--output_csv', required=True)
    parser.add_argument('--root_dir', default=None, help='Dataset root. If omitted, uses config dataset_params.root_dir.')
    parser.add_argument('--split', default='test')
    parser.add_argument('--save_images', default=None, help='Optional directory for visual outputs.')
    parser.add_argument('--save_all_images', action='store_true')
    parser.add_argument('--max_saved_images', type=int, default=50)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--rdlambda', type=float, default=0.0157)
    parser.add_argument('--model_name', default=None)
    parser.add_argument('--no_lpips', action='store_true')
    parser.add_argument('--require_lpips', action='store_true',
                        help='Fail fast if official LPIPS cannot be loaded, instead of writing NaN.')
    parser.add_argument('--lpips_net', default='alex', choices=['alex', 'vgg', 'squeeze'],
                        help='Official LPIPS backbone. Default: alex.')
    parser.add_argument('--no_ms_ssim', action='store_true')
    parser.add_argument('--weight_past', type=float, default=0.5)
    parser.add_argument('--weight_future', type=float, default=0.5)
    parser.add_argument('--occlusion_suppression_threshold', type=float, default=0.1,
                        help='Mask values below this threshold count as strongly suppressed. Default: 0.1.')
    parser.add_argument('--occlusion_epsilon', type=float, default=1e-8,
                        help='Numerical stabilizer for relative feature attenuation. Default: 1e-8.')
    parser.add_argument('--save_video_panels', default=None,
                        help='Optional directory for logger-style per-video panels. Each video_id becomes one visual batch/panel.')
    parser.add_argument('--video_panel_max_samples', type=int, default=12,
                        help='Maximum sampled target frames to show per video panel. Use 0 to show all sampled frames; this can make very large images.')
    parser.add_argument('--video_panel_cell_size', type=int, default=160,
                        help='Image tile size in the per-video visual panel.')
    return parser
