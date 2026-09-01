#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Per-video physical-bitstream GOP=4 pipeline.

For each test video, complete the WHOLE pipeline before moving to the next:
    1. Shared VTM anchors: I0, I4, I8, ...
       original -> VTM .bin -> VTM decode -> receiver cache
    2. Single CFTE compact residual bitstreams -> decode -> MP4
    3. NoRefine past/future bitstreams -> decode -> MP4
    4. FixedCMR past/future bitstreams -> decode -> MP4
    5. AdaptiveCMR past/future bitstreams -> decode -> MP4
       with true It supplied directly to decoder as oracle side information

Important:
- Shared anchors are transmitted only ONCE and reused by all four methods.
- Single / NoRefine / FixedCMR decoder never reads true inter-frame targets.
- AdaptiveCMR reads true It only for its DISTS-guided oracle update.
- Adaptive oracle It is NOT counted as transmitted bits.
- This is a dense GOP=4 qualitative codec demo using gap-3-trained checkpoints,
  not the thesis's strict symmetric gap-3 benchmark.
"""

import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch
from tqdm import tqdm

WORK = Path("/home/featurize/work")
TEST_DIR = WORK / "vox" / "test"
SHARED_ROOT = WORK / "shared_anchor_channel_gop4"
CHANNEL_ROOT = WORK / "physical_channel_gop4_fast"
RECON_ROOT = WORK / "codec_recon_gop4_fast"

# Import the physical-bitstream stages from this archive directory.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import prepare_shared_anchors_gop4 as anchor_stage
import encode_streams_gop4_fast as stream_stage
import decode_channel_gop4_fast as decode_stage


METHODS = ("single", "norefine", "fixed", "adaptive")


def clear_project_modules():
    """
    The four projects all use the package name `modules`.
    Remove only project-module cache entries before loading the next project.
    Already-instantiated model objects remain valid because their class/module
    objects are still referenced by the instances themselves.
    """
    exact = {
        "logger", "frames_dataset", "augmentation", "animate",
        "reconstruction", "train", "run", "flowvisual",
        "sync_batchnorm",
    }
    prefixes = ("modules.", "sync_batchnorm.")
    for name in list(sys.modules):
        if name == "modules" or name in exact or name.startswith(prefixes):
            sys.modules.pop(name, None)


def gpu_mem():
    if not torch.cuda.is_available():
        return "CUDA unavailable"
    alloc = torch.cuda.memory_allocated(0) / 1024**3
    reserv = torch.cuda.memory_reserved(0) / 1024**3
    return f"allocated={alloc:.2f} GB reserved={reserv:.2f} GB"


def load_all_models(device):
    """
    Load each method ONCE at startup.
    We use decode_stage.setup because it returns generator + kp + compressor,
    and the same kp/compressor objects are reused by the physical-stream encoder.
    """
    bundles = {}
    original_cwd = os.getcwd()

    for method in METHODS:
        print(f"\n===== LOAD {method} ONCE =====")
        clear_project_modules()

        # Each setup() prepends that project's directory to sys.path.
        # Record/restore cwd only; old sys.path entries are harmless because
        # `modules*` is explicitly cleared before each load.
        bundle = decode_stage.setup(method, device)
        generator, kp, vc, dists_model = bundle

        bundles[method] = {
            "generator": generator,
            "kp": kp,
            "vc": vc,
            "dists": dists_model,
        }

        gc.collect()
        torch.cuda.empty_cache()
        print(f"[GPU MEM after {method}] {gpu_mem()}")

    os.chdir(original_cwd)
    return bundles


def shared_manifest_for(video_path, qp):
    rel = video_path.relative_to(TEST_DIR)
    return SHARED_ROOT / f"qp{qp}" / rel.with_suffix("") / "manifest.json"


def method_manifest_for(method, video_path, qp):
    rel = video_path.relative_to(TEST_DIR)
    return (
        CHANNEL_ROOT / method / f"qp{qp}" /
        rel.with_suffix("") / "manifest.json"
    )


def output_path_for(method, video_path, qp):
    rel = video_path.relative_to(TEST_DIR)
    out_name = decode_stage.METHODS[method]["out_name"]
    return (
        RECON_ROOT / out_name / f"qp{qp}" / rel
    ).with_suffix(".mp4")


def process_one_video(video_idx, total, video_path, qp, vtm_workers,
                      bundles, device, overwrite):
    rel = video_path.relative_to(TEST_DIR)
    name = str(rel)
    t0 = time.time()

    print("\n" + "=" * 88)
    print(f"[VIDEO {video_idx}/{total}] {name}")
    print("=" * 88)

    # ------------------------------------------------------------
    # 1) Shared physical VTM anchor transmission ONCE
    # ------------------------------------------------------------
    ta = time.time()
    state, nanchors = anchor_stage.prepare_one(
        video_path, qp, vtm_workers, overwrite
    )
    anchor_manifest = shared_manifest_for(video_path, qp)
    if not anchor_manifest.exists():
        raise RuntimeError(f"Shared anchor manifest missing: {anchor_manifest}")

    shared = json.loads(anchor_manifest.read_text())
    print(
        f"[ANCHORS] {state}; n={nanchors}; "
        f"physical VTM payload={shared['payload_bytes'] * 8} bits; "
        f"time={time.time()-ta:.1f}s"
    )

    # ------------------------------------------------------------
    # 2/3) For THIS SAME VIDEO, finish each method completely.
    # ------------------------------------------------------------
    for method in METHODS:
        b = bundles[method]

        print(f"\n--- {method}: physical compact-stream ENCODE ---")
        te = time.time()
        enc_state = stream_stage.encode_one(
            method=method,
            video_path=video_path,
            qp=qp,
            kp=b["kp"],
            vc=b["vc"],
            device=device,
            overwrite=overwrite,
        )

        manifest_path = method_manifest_for(method, video_path, qp)
        if not manifest_path.exists():
            raise RuntimeError(
                f"{method} stream manifest missing: {manifest_path}"
            )

        m = json.loads(manifest_path.read_text())
        print(
            f"[{method} ENCODE] {enc_state}; "
            f"compact={m['method_stream_payload_bytes'] * 8} bits; "
            f"total visual payload={m['total_visual_payload_bits']} bits; "
            f"time={time.time()-te:.1f}s"
        )

        print(f"--- {method}: DECODER -> MP4 ---")
        td = time.time()
        out_path = output_path_for(method, video_path, qp)

        dec_state = decode_stage.decode_one(
            method=method,
            manifest_path=manifest_path,
            out_path=out_path,
            generator=b["generator"],
            kp=b["kp"],
            vc=b["vc"],
            dists_model=b["dists"],
            device=device,
            overwrite=overwrite,
        )

        if not out_path.exists():
            raise RuntimeError(f"Decoded MP4 missing: {out_path}")

        oracle_tag = " [It ORACLE]" if method == "adaptive" else ""
        print(
            f"[{method} DECODE]{oracle_tag} {dec_state}; "
            f"MP4={out_path}; "
            f"time={time.time()-td:.1f}s"
        )

        gc.collect()
        torch.cuda.empty_cache()

    print(
        f"\n[VIDEO COMPLETE {video_idx}/{total}] {name} "
        f"TOTAL={time.time()-t0:.1f}s"
    )
    print("[VIDEO COMPLETE] Four reconstructed MP4s are now available.")


def main():
    global anchor_stage

    ap = argparse.ArgumentParser()
    ap.add_argument("--qp", type=int, default=32)
    ap.add_argument("--vtm-workers", type=int, default=14)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument(
        "--max-videos", type=int, default=0,
        help="0 = all remaining test videos"
    )
    ap.add_argument(
        "--overwrite", action="store_true",
        help="rebuild already completed anchors/streams/videos"
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    print("=" * 88)
    print("PER-VIDEO physical-bitstream GOP=4 pipeline")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VTM anchor QP: {args.qp}")
    print(f"Parallel VTM workers per video: {args.vtm_workers}")
    print("Order: one video -> shared anchors -> 4 methods -> 4 MP4s -> next video")
    print("AdaptiveCMR: true It is decoder-side ORACLE, not counted as transmitted bits")
    print("=" * 88)

    # Initialize shared VTM paths once.
    (
        anchor_stage.ENC,
        anchor_stage.DEC,
        anchor_stage.LOWDELAY_CFG,
        anchor_stage.SEQ_CFG,
        anchor_stage.RGB_CFG,
    ) = anchor_stage.vtm_paths()

    # Load all four inference models once. This avoids re-reading ~800 MB
    # checkpoints for every video.
    bundles = load_all_models(device)
    print(f"\n[ALL MODELS READY] {gpu_mem()}")

    videos = sorted(TEST_DIR.rglob("*.mp4"))
    videos = videos[args.start_index:]
    if args.max_videos > 0:
        videos = videos[:args.max_videos]

    total = len(videos)
    if total == 0:
        print("No test videos selected.")
        return

    error_log = WORK / "per_video_codec_errors.log"
    failures = 0

    for local_idx, video_path in enumerate(videos, 1):
        absolute_idx = args.start_index + local_idx
        try:
            process_one_video(
                absolute_idx,
                args.start_index + total,
                video_path,
                args.qp,
                args.vtm_workers,
                bundles,
                device,
                args.overwrite,
            )
        except Exception as exc:
            failures += 1
            with open(error_log, "a", encoding="utf-8") as f:
                f.write(
                    f"\n===== VIDEO {absolute_idx}: {video_path} =====\n"
                )
                f.write(f"{type(exc).__name__}: {exc}\n")
                f.write(traceback.format_exc())
            print(f"\n[VIDEO ERROR] {video_path}: {exc}")
            print(f"[VIDEO ERROR] Full traceback appended to {error_log}")
            gc.collect()
            torch.cuda.empty_cache()

    print("\n" + "=" * 88)
    print(f"ALL SELECTED VIDEOS FINISHED. failures={failures}")
    print(f"Shared VTM channel: {SHARED_ROOT}")
    print(f"Compact bitstreams: {CHANNEL_ROOT}")
    print(f"Reconstructed MP4s: {RECON_ROOT}")
    if failures:
        print(f"Error log: {error_log}")
    print("=" * 88)


if __name__ == "__main__":
    main()
