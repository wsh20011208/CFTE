#!/usr/bin/env python3
"""
Create a deterministic frame index for the four CFTE variants.

Revised sampling rule:
    target_idx = stride, 2*stride, 3*stride, ...
    past_idx   = target_idx - gap
    future_idx = target_idx + gap

With the default gap=5 and stride=3, candidate target frames remain:
    3, 6, 9, 12, ...

After the symmetric boundary test, the first valid target for gap=5 is 6:
    past=1, target=6, future=11

A candidate target frame is kept only if both references exist:
    target_idx - gap >= 0
    target_idx + gap < num_frames

Therefore, even for single-frame CFTE, the shared CSV still contains future_idx
so that all four methods are evaluated on exactly the same target frames. The
single-frame evaluator ignores future_idx and only uses past_idx -> target_idx.

By default, all valid target frames are kept. This means no artificial cap:
every target that is a positive multiple of stride and has both past and future
references is included. Use --max_targets_per_video only for quick debugging.
"""
import argparse
import csv
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from skimage import io

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp'}
VIDEO_EXTS = {'.gif', '.mp4', '.mov'}


def _resolve_eval_root(root_dir: Path, split: str) -> Path:
    """If root_dir/split exists use it, otherwise use root_dir directly."""
    if split and (root_dir / split).is_dir():
        return root_dir / split
    return root_dir


def _list_videos(eval_root: Path):
    items = []
    for p in sorted(eval_root.iterdir()):
        if p.name.startswith('.'):
            continue
        if p.is_dir():
            frame_files = [q for q in p.iterdir() if q.suffix.lower() in IMAGE_EXTS]
            if len(frame_files) > 0:
                items.append(p)
        elif p.suffix.lower() in (IMAGE_EXTS | VIDEO_EXTS):
            items.append(p)
    return items


def _count_video_frames(video_path: Path) -> int:
    """Count frames without loading the entire video into memory when possible."""
    reader = imageio.get_reader(str(video_path))
    try:
        # ffmpeg reader normally supports count_frames(). This is much cheaper
        # than imageio.mimread(...), which loads all frames into RAM.
        try:
            n = reader.count_frames()
            if isinstance(n, (int, np.integer)) and n > 0:
                return int(n)
        except Exception:
            pass

        try:
            meta = reader.get_meta_data()
            n = meta.get('nframes', None)
            if n is not None and math.isfinite(float(n)) and int(n) > 0:
                return int(n)
        except Exception:
            pass

        count = 0
        for _ in reader:
            count += 1
        return count
    finally:
        try:
            reader.close()
        except Exception:
            pass


def _count_frames(video_path: Path, frame_shape=(256, 256, 3)) -> int:
    if video_path.is_dir():
        return len([p for p in sorted(video_path.iterdir()) if p.suffix.lower() in IMAGE_EXTS])

    suffix = video_path.suffix.lower()
    if suffix in VIDEO_EXTS:
        return _count_video_frames(video_path)

    if suffix in IMAGE_EXTS:
        image = io.imread(str(video_path))
        if image.ndim == 2:
            h, w = image.shape
        else:
            h, w = image.shape[:2]
        fh, fw = int(frame_shape[0]), int(frame_shape[1])
        if h == fh and w % fw == 0:
            return w // fw
        if w == fw and h % fh == 0:
            return h // fh
        return max(1, w // fw)

    raise ValueError(f'Unsupported video/file type: {video_path}')


def _frame_path_if_folder(video_path: Path, frame_idx: int):
    if not video_path.is_dir():
        return ''
    frames = sorted([p for p in video_path.iterdir() if p.suffix.lower() in IMAGE_EXTS])
    if 0 <= frame_idx < len(frames):
        return frames[frame_idx].name
    return ''


def _select_targets(indices, max_targets_per_video: int, strategy: str):
    if max_targets_per_video is None or max_targets_per_video <= 0:
        return indices
    if len(indices) <= max_targets_per_video:
        return indices

    if strategy == 'first':
        return indices[:max_targets_per_video]

    if strategy == 'uniform':
        chosen = np.linspace(0, len(indices) - 1, max_targets_per_video)
        chosen = sorted(set(int(round(x)) for x in chosen))
        return [indices[i] for i in chosen]

    raise ValueError(f'Unknown subsample strategy: {strategy}')


def _target_candidates(num_frames: int, gap: int, stride: int):
    # target frames must be exact positive multiples of stride: stride, 2*stride, ...
    # The thesis keeps stride=3 for every temporal gap: 3, 6, 9, 12, ...
    candidates = []
    t = stride
    while t < num_frames:
        if (t - gap) >= 0 and (t + gap) < num_frames:
            candidates.append(t)
        t += stride
    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', required=True,
                        help='Dataset root, e.g. ~/CFTE/vox/test or ~/CFTE/vox. If root/split exists, that split is used.')
    parser.add_argument('--output_csv', default='benchmark/test_triplets_gap5_stride3.csv')
    parser.add_argument('--split', default='test', help='Subdirectory to use if it exists under root_dir. Default: test')
    parser.add_argument('--gap', type=int, default=5, help='Symmetric reference-frame gap. Default: past=t-5 and future=t+5.')
    parser.add_argument('--stride', type=int, default=3, help='Target-frame multiple/stride. Default target frames: 3,6,9,12,...')
    parser.add_argument('--max_targets_per_video', type=int, default=0,
                        help='0 means use all valid target multiples. Positive N is only for quick debugging and keeps at most N per video.')
    parser.add_argument('--subsample_strategy', choices=['first', 'uniform'], default='first',
                        help='When max_targets_per_video is positive: first keeps the earliest valid stride-3 targets; uniform spreads N targets over the video.')
    parser.add_argument('--max_videos', type=int, default=0, help='0 means all videos.')
    parser.add_argument('--frame_shape', default='256,256,3')
    parser.add_argument('--progress_every', type=int, default=50, help='Print progress every N videos. 0 disables progress prints.')
    args = parser.parse_args()

    if args.gap <= 0:
        parser.error('--gap must be a positive integer')
    if args.stride <= 0:
        parser.error('--stride must be a positive integer')

    root_dir = Path(args.root_dir).expanduser().resolve()
    eval_root = _resolve_eval_root(root_dir, args.split)
    frame_shape = tuple(int(x) for x in args.frame_shape.split(','))

    videos = _list_videos(eval_root)
    if args.max_videos and args.max_videos > 0:
        videos = videos[:args.max_videos]

    out_path = Path(args.output_csv).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    sample_counter = 0
    skipped_no_valid = 0
    print(f'Indexing {len(videos)} videos from {eval_root}')
    print(f'Sampling rule: target={args.stride}, {2*args.stride}, {3*args.stride}, ...; past=target-{args.gap}; future=target+{args.gap}')
    if args.max_targets_per_video > 0:
        print(f'Max targets per video: {args.max_targets_per_video} ({args.subsample_strategy})')
    else:
        print('Max targets per video: unlimited; using all valid target multiples')

    for i, video_path in enumerate(videos, start=1):
        if args.progress_every and (i == 1 or i % args.progress_every == 0 or i == len(videos)):
            print(f'[{i}/{len(videos)}] {video_path.name}', flush=True)
        try:
            num_frames = _count_frames(video_path, frame_shape=frame_shape)
        except Exception as e:
            print(f'[WARN] Skip {video_path}: cannot count frames: {e}')
            continue

        candidates = _target_candidates(num_frames, gap=args.gap, stride=args.stride)
        candidates = _select_targets(candidates, args.max_targets_per_video, args.subsample_strategy)
        if not candidates:
            skipped_no_valid += 1
            print(f'[WARN] Skip {video_path.name}: only {num_frames} frames, no valid target for gap={args.gap}, stride={args.stride}')
            continue

        video_rel_path = str(video_path.relative_to(eval_root))
        video_id = video_path.stem if video_path.is_file() else video_path.name
        for target_idx in candidates:
            sample_counter += 1
            past_idx = target_idx - args.gap
            future_idx = target_idx + args.gap
            rows.append({
                'sample_id': f'{sample_counter:06d}',
                'video_id': video_id,
                'video_rel_path': video_rel_path,
                'num_frames': num_frames,
                'past_idx': past_idx,
                'target_idx': target_idx,
                'future_idx': future_idx,
                'gap_past': target_idx - past_idx,
                'gap_future': future_idx - target_idx,
                'target_multiple': args.stride,
                'past_frame_name': _frame_path_if_folder(video_path, past_idx),
                'target_frame_name': _frame_path_if_folder(video_path, target_idx),
                'future_frame_name': _frame_path_if_folder(video_path, future_idx),
            })

    fieldnames = [
        'sample_id', 'video_id', 'video_rel_path', 'num_frames',
        'past_idx', 'target_idx', 'future_idx', 'gap_past', 'gap_future', 'target_multiple',
        'past_frame_name', 'target_frame_name', 'future_frame_name'
    ]
    with out_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'Wrote {len(rows)} samples from {len(videos)} videos to {out_path}')
    if skipped_no_valid:
        print(f'Skipped {skipped_no_valid} videos with no valid target frame.')


if __name__ == '__main__':
    main()
