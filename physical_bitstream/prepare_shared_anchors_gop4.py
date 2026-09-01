#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 1: prepare ONE shared physical anchor channel for all four methods.

Regular anchors: I0, I4, I8, ...
If the clip ends inside an incomplete GOP, the final frame is additionally
sent as a boundary anchor so every remaining inter frame has a future anchor.

Each anchor is:
  original RGB frame
    -> VTM10Enc (QP selected by --qp)
    -> physical .bin payload
    -> VTM10Dec
    -> receiver-side decoded anchor cache (.npy)

The four CFTE variants subsequently reuse these exact decoded anchors.
Thus the expensive VTM anchor transmission is performed only once.

Parallelism:
  independent single-frame VTM jobs are launched concurrently.
"""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm


WORK = Path("/home/featurize/work")
TEST_DIR = WORK / "vox" / "test"
CFTE_DIR = WORK / "CFTE"
SHARED_ROOT = WORK / "shared_anchor_channel_gop4"


def read_video(path):
    reader = imageio.get_reader(str(path))
    try:
        meta = reader.get_meta_data()
        fps = float(meta.get("fps", 25.0))
        frames = [np.asarray(f)[..., :3] for f in reader]
    finally:
        reader.close()

    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")

    arr = np.stack(frames, axis=0)
    if arr.dtype != np.uint8:
        if arr.max() <= 1.5:
            arr = np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)
        else:
            arr = np.clip(np.rint(arr), 0, 255).astype(np.uint8)
    return arr, fps


def anchor_indices(n):
    idx = list(range(0, n, 4))
    if not idx:
        return [0]
    # Boundary anchor for a trailing incomplete GOP.
    if idx[-1] != n - 1:
        idx.append(n - 1)
    return idx


def write_planar_rgb(path, frame):
    frame = np.asarray(frame, dtype=np.uint8)
    chw = frame.transpose(2, 0, 1)
    with open(path, "wb") as f:
        f.write(chw.tobytes(order="C"))


def read_planar_rgb(path, width, height):
    arr = np.fromfile(path, dtype=np.uint8)
    expected = 3 * width * height
    if arr.size != expected:
        raise RuntimeError(
            f"Unexpected VTM RGB size: got {arr.size}, expected {expected}"
        )
    return arr.reshape(3, height, width).transpose(1, 2, 0).copy()


def vtm_paths():
    enc = CFTE_DIR / "vtm" / "bin" / "VTM10Enc"
    dec = CFTE_DIR / "vtm" / "bin" / "VTM10Dec"
    lowdelay = CFTE_DIR / "vtm" / "cfg" / "encoder_lowdelay_vtm.cfg"
    seq = CFTE_DIR / "vtm" / "cfg" / "per-sequence" / "43.cfg"
    rgb = CFTE_DIR / "vtm" / "cfg" / "formatRGB.cfg"

    for p in (enc, dec, lowdelay, seq, rgb):
        if not p.exists():
            raise FileNotFoundError(f"Missing VTM component: {p}")

    enc.chmod(enc.stat().st_mode | 0o111)
    dec.chmod(dec.stat().st_mode | 0o111)
    return enc, dec, lowdelay, seq, rgb


ENC, DEC, LOWDELAY_CFG, SEQ_CFG, RGB_CFG = (None,) * 5


def encode_decode_anchor(frame, frame_idx, qp, video_tmp_dir):
    """One independent physical VTM anchor job."""
    h, w = frame.shape[:2]

    bin_dir = video_tmp_dir / "anchors"
    cache_dir = video_tmp_dir / "receiver_cache"
    bin_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    bin_path = bin_dir / f"frame_{frame_idx:06d}.bin"
    cache_path = cache_dir / f"frame_{frame_idx:06d}.npy"

    with tempfile.TemporaryDirectory(prefix=f"vtm_{frame_idx:06d}_") as td:
        td = Path(td)
        org = td / "org.rgb"
        enc_rec = td / "enc_rec.rgb"
        dec_rec = td / "dec_rec.rgb"
        enc_log = td / "enc.log"
        dec_log = td / "dec.log"

        write_planar_rgb(org, frame)

        enc_cmd = [
            str(ENC),
            "-c", str(LOWDELAY_CFG),
            "-c", str(SEQ_CFG),
            "-c", str(RGB_CFG),
            "-q", str(qp),
            "-i", str(org),
            "-wdt", str(w),
            "-hgt", str(h),
            "-o", str(enc_rec),
            "-b", str(bin_path),
        ]
        with open(enc_log, "wb") as f:
            subprocess.run(
                enc_cmd, cwd=CFTE_DIR, stdout=f,
                stderr=subprocess.STDOUT, check=True
            )

        dec_cmd = [
            str(DEC),
            "-b", str(bin_path),
            "-o", str(dec_rec),
            "--OutputColourSpaceConvert=GBRtoRGB",
        ]
        with open(dec_log, "wb") as f:
            subprocess.run(
                dec_cmd, cwd=CFTE_DIR, stdout=f,
                stderr=subprocess.STDOUT, check=True
            )

        rec = read_planar_rgb(dec_rec, w, h)
        np.save(cache_path, rec, allow_pickle=False)

    return {
        "frame": int(frame_idx),
        "bin": str(Path("anchors") / bin_path.name),
        "receiver_cache": str(Path("receiver_cache") / cache_path.name),
        "bytes": int(bin_path.stat().st_size),
    }


def prepare_one(video_path, qp, workers, overwrite):
    rel = video_path.relative_to(TEST_DIR)
    final_dir = SHARED_ROOT / f"qp{qp}" / rel.with_suffix("")
    manifest_path = final_dir / "manifest.json"

    if manifest_path.exists() and not overwrite:
        try:
            m = json.loads(manifest_path.read_text())
            if m.get("complete") is True:
                return "skipped", m.get("num_anchors", 0)
        except Exception:
            pass

    tmp_dir = final_dir.with_name(final_dir.name + ".building")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        frames, fps = read_video(video_path)
        n, h, w, c = frames.shape
        if c != 3:
            raise RuntimeError(f"Expected RGB video, got {frames.shape}")

        anchors = anchor_indices(n)
        records = []

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(
                    encode_decode_anchor,
                    frames[a].copy(), a, qp, tmp_dir
                ): a
                for a in anchors
            }
            for fut in as_completed(futures):
                records.append(fut.result())

        records.sort(key=lambda r: r["frame"])
        payload_bytes = sum(r["bytes"] for r in records)

        manifest = {
            "complete": True,
            "format_version": 2,
            "channel_model": "error-free digital channel",
            "anchor_codec": "VTM10 low-delay config, one independently coded frame per anchor",
            "source_video": str(rel),
            "qp": int(qp),
            "gop": 4,
            "fps": float(fps),
            "num_frames": int(n),
            "width": int(w),
            "height": int(h),
            "anchors": anchors,
            "num_anchors": len(anchors),
            "anchor_records": {str(r["frame"]): r for r in records},
            "payload_bytes": int(payload_bytes),
            "payload_bits": int(payload_bytes * 8),
            "receiver_cache_note": (
                "receiver_cache/*.npy is the exact VTM-decoded reconstruction "
                "derived from the transmitted anchor .bin; cache bytes are not "
                "counted as channel payload."
            ),
        }
        (tmp_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        final_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(final_dir, ignore_errors=True)
        os.replace(tmp_dir, final_dir)
        return "prepared", len(anchors)

    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def main():
    global ENC, DEC, LOWDELAY_CFG, SEQ_CFG, RGB_CFG

    ap = argparse.ArgumentParser()
    ap.add_argument("--qp", type=int, default=32)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    ENC, DEC, LOWDELAY_CFG, SEQ_CFG, RGB_CFG = vtm_paths()

    videos = sorted(TEST_DIR.rglob("*.mp4"))[args.start_index:]
    if args.max_videos > 0:
        videos = videos[:args.max_videos]

    root = SHARED_ROOT / f"qp{args.qp}"
    root.mkdir(parents=True, exist_ok=True)
    error_log = SHARED_ROOT / f"anchor_errors_qp{args.qp}.log"

    print(f"[SHARED ANCHORS] videos={len(videos)} qp={args.qp} workers={args.workers}")
    print(f"[SHARED ROOT] {root}")
    print("[IMPORTANT] VTM anchors are generated ONCE and reused by all four methods.")

    failures = 0
    pbar = tqdm(videos, desc="shared-vtm-anchors")
    for i, video in enumerate(pbar, 1):
        try:
            state, na = prepare_one(
                video, args.qp, args.workers, args.overwrite
            )
            pbar.set_postfix_str(f"anchors={na} {video.name[:22]}")
        except Exception as e:
            failures += 1
            with open(error_log, "a", encoding="utf-8") as f:
                f.write(f"\n===== {video} =====\n")
                f.write(f"{type(e).__name__}: {e}\n")
                f.write(traceback.format_exc())
            print(f"\n[ANCHOR ERROR] {video}: {e}")

    print(f"[DONE] shared anchors failures={failures}")
    if failures:
        print(f"[ERROR LOG] {error_log}")


if __name__ == "__main__":
    main()
