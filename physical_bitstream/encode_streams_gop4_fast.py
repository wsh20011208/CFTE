#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 2: encode method-specific compact residual bitstreams.

Shared VTM anchors must already exist under:
  /home/featurize/work/shared_anchor_channel_gop4/qpXX/

No anchor is VTM-encoded again here.

Single:
    one physical EntropyBottleneck byte stream per inter frame.

NoRefine / FixedCMR / AdaptiveCMR:
    two physical EntropyBottleneck byte streams per inter frame
    (target-minus-past-anchor and target-minus-future-anchor).

Adaptive true It is NOT transmitted; it is decoder-side oracle information.
"""

import argparse
import gc
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from tqdm import tqdm


WORK = Path("/home/featurize/work")
TEST_DIR = WORK / "vox" / "test"
SHARED_ROOT = WORK / "shared_anchor_channel_gop4"
CHANNEL_ROOT = WORK / "physical_channel_gop4_fast"

METHODS = {
    "single": {
        "project": WORK / "CFTE",
        "checkpoint": WORK / "CFTE" / "00000199-checkpoint_singlecfte_d3.pth.tar",
    },
    "norefine": {
        "project": WORK / "CFTE_NoRefine",
        "checkpoint": WORK / "CFTE_NoRefine" / "00000199-checkpoint_norefine_d3.pth.tar",
    },
    "fixed": {
        "project": WORK / "CFTE_Refined_FixedCMR",
        "checkpoint": WORK / "CFTE_Refined_FixedCMR" / "00000199-checkpoint_fixedcmr_d3.pth.tar",
    },
    "adaptive": {
        "project": WORK / "CFTE_Refined_AdaptiveCMR",
        "checkpoint": WORK / "CFTE_Refined_AdaptiveCMR" / "00000199-checkpoint_adaptivecmr_d3.pth.tar",
    },
}


def torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def pick_state(checkpoint, keys, label):
    for key in keys:
        if key in checkpoint:
            return checkpoint[key], key
    raise KeyError(f"Missing {label}; tried {keys}")


def setup(method, device):
    spec = METHODS[method]
    project = spec["project"]
    config_path = project / "config" / "vox-256_gap3.yaml"

    os.chdir(project)
    sys.path.insert(0, str(project))

    from modules.keypoint_detector import KPDetector
    from modules.RDloss import VideoCompressor

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    kp = KPDetector(
        **config["model_params"]["kp_detector_params"],
        **config["model_params"]["common_params"],
    ).to(device)
    vc = VideoCompressor(
        **config["model_params"]["videocompressor_params"]
    ).to(device)

    cpk = torch_load(spec["checkpoint"], device)

    if method == "single":
        kp_state, kp_key = pick_state(cpk, ["kp_detector"], "kp_detector")
        vc_state, vc_key = pick_state(cpk, ["videocompressor"], "videocompressor")
    else:
        kp_state, kp_key = pick_state(
            cpk, ["kp_detector_past", "kp_detector", "kp_detector_future"], "kp_detector"
        )
        vc_state, vc_key = pick_state(
            cpk, ["videocompressor_past", "videocompressor", "videocompressor_future"],
            "videocompressor"
        )

    kp.load_state_dict(kp_state, strict=True)
    vc.load_state_dict(vc_state, strict=True)
    kp.eval()
    vc.eval()

    # Populate CDF tables for actual entropy coding.
    vc.entropy_bottleneck.update(force=True)

    print(f"[LOAD] method={method} epoch={cpk.get('epoch')}")
    print(f"[LOAD] kp={kp_key} videocompressor={vc_key}")
    return project, kp, vc


def read_video(path):
    reader = imageio.get_reader(str(path))
    try:
        frames = [np.asarray(f)[..., :3] for f in reader]
    finally:
        reader.close()
    if not frames:
        raise RuntimeError(f"No frames: {path}")
    arr = np.stack(frames, axis=0)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.5:
            arr = np.clip(np.rint(arr * 255), 0, 255).astype(np.uint8)
        else:
            arr = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    return arr


def to_tensor(frame, device):
    x = torch.from_numpy(frame.astype(np.float32) / 255.0)
    return x.permute(2, 0, 1).unsqueeze(0).to(device)


@torch.no_grad()
def kp_value(kp, frame, device):
    return kp(to_tensor(frame, device))["value"]


@torch.no_grad()
def compress_residual(vc, target_value, anchor_value):
    residual = target_value - anchor_value
    strings = vc.entropy_bottleneck.compress(residual)
    if len(strings) != 1:
        raise RuntimeError(f"Expected one entropy string, got {len(strings)}")
    return strings[0], list(residual.shape[-2:])


def save_bytes(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def bracketing(idx, anchors):
    past = max(a for a in anchors if a < idx)
    future = min(a for a in anchors if a > idx)
    return past, future


def encode_one(method, video_path, qp, kp, vc, device, overwrite):
    rel = video_path.relative_to(TEST_DIR)
    shared_dir = SHARED_ROOT / f"qp{qp}" / rel.with_suffix("")
    shared_manifest_path = shared_dir / "manifest.json"
    if not shared_manifest_path.exists():
        raise FileNotFoundError(f"Shared anchor manifest missing: {shared_manifest_path}")

    shared = json.loads(shared_manifest_path.read_text())
    if not shared.get("complete"):
        raise RuntimeError("Shared anchor stage incomplete")

    final_dir = CHANNEL_ROOT / method / f"qp{qp}" / rel.with_suffix("")
    manifest_path = final_dir / "manifest.json"
    if manifest_path.exists() and not overwrite:
        try:
            old = json.loads(manifest_path.read_text())
            if old.get("complete") is True:
                return "skipped"
        except Exception:
            pass

    tmp_dir = final_dir.with_name(final_dir.name + ".building")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        frames = read_video(video_path)
        n = len(frames)
        if n != int(shared["num_frames"]):
            raise RuntimeError("Video frame count differs from shared anchor manifest")

        anchors = [int(x) for x in shared["anchors"]]
        anchor_values = {}
        for a in anchors:
            rec = shared["anchor_records"][str(a)]
            frame = np.load(shared_dir / rec["receiver_cache"], allow_pickle=False)
            anchor_values[a] = kp_value(kp, frame, device)

        frame_records = {}
        spatial = None

        for idx in range(n):
            if idx in anchor_values:
                frame_records[str(idx)] = {"type": "anchor", "shared_anchor": idx}
                continue

            target = kp_value(kp, frames[idx], device)
            past, future = bracketing(idx, anchors)

            if method == "single":
                payload, shp = compress_residual(
                    vc, target, anchor_values[past]
                )
                spatial = shp
                rel_stream = Path("streams") / f"frame_{idx:06d}_single.bin"
                save_bytes(tmp_dir / rel_stream, payload)
                frame_records[str(idx)] = {
                    "type": "inter",
                    "past_anchor": past,
                    "future_anchor_boundary": future,
                    "single_stream": str(rel_stream),
                }
            else:
                payload_p, shp_p = compress_residual(
                    vc, target, anchor_values[past]
                )
                payload_f, shp_f = compress_residual(
                    vc, target, anchor_values[future]
                )
                if shp_p != shp_f:
                    raise RuntimeError("Past/future residual shapes differ")
                spatial = shp_p

                rp = Path("streams") / f"frame_{idx:06d}_past.bin"
                rf = Path("streams") / f"frame_{idx:06d}_future.bin"
                save_bytes(tmp_dir / rp, payload_p)
                save_bytes(tmp_dir / rf, payload_f)

                frame_records[str(idx)] = {
                    "type": "inter",
                    "past_anchor": past,
                    "future_anchor": future,
                    "past_stream": str(rp),
                    "future_stream": str(rf),
                    "oracle_target_at_decoder": method == "adaptive",
                }

        stream_bytes = sum(
            p.stat().st_size for p in (tmp_dir / "streams").glob("*.bin")
        ) if (tmp_dir / "streams").exists() else 0

        manifest = {
            "complete": True,
            "format_version": 2,
            "method": method,
            "source_video": str(rel),
            "qp": int(qp),
            "gop": 4,
            "num_frames": int(n),
            "residual_spatial_shape": spatial or [4, 4],
            "shared_anchor_manifest": str(shared_manifest_path),
            "shared_anchor_payload_bytes": int(shared["payload_bytes"]),
            "method_stream_payload_bytes": int(stream_bytes),
            "total_visual_payload_bytes": int(shared["payload_bytes"] + stream_bytes),
            "total_visual_payload_bits": int((shared["payload_bytes"] + stream_bytes) * 8),
            "frames": frame_records,
            "adaptive_oracle": {
                "enabled": method == "adaptive",
                "true_It_given_directly_to_decoder": method == "adaptive",
                "oracle_It_bits_counted": False,
            },
            "note": (
                "Shared anchor VTM bitstreams are transmitted once and reused. "
                "Dense GOP=4 demo uses gap-3-trained checkpoints."
            ),
        }
        (tmp_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        final_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(final_dir, ignore_errors=True)
        os.replace(tmp_dir, final_dir)
        return "encoded"

    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    finally:
        gc.collect()
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["single", "norefine", "fixed", "adaptive"])
    ap.add_argument("--qp", type=int, default=32)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda:0")

    _, kp, vc = setup(args.method, device)

    videos = sorted(TEST_DIR.rglob("*.mp4"))[args.start_index:]
    if args.max_videos > 0:
        videos = videos[:args.max_videos]

    error_log = CHANNEL_ROOT / f"encode_errors_{args.method}_qp{args.qp}.log"
    print(f"[GPU] {torch.cuda.get_device_name(0)}")
    print(f"[STREAM ENCODE] method={args.method} videos={len(videos)}")

    failures = 0
    for video in tqdm(videos, desc=f"streams-{args.method}"):
        try:
            encode_one(
                args.method, video, args.qp, kp, vc, device, args.overwrite
            )
        except Exception as e:
            failures += 1
            error_log.parent.mkdir(parents=True, exist_ok=True)
            with open(error_log, "a", encoding="utf-8") as f:
                f.write(f"\n===== {video} =====\n")
                f.write(f"{type(e).__name__}: {e}\n")
                f.write(traceback.format_exc())
            print(f"\n[STREAM ERROR] {video}: {e}")

    print(f"[DONE] stream encode method={args.method} failures={failures}")
    if failures:
        print(f"[ERROR LOG] {error_log}")


if __name__ == "__main__":
    main()
