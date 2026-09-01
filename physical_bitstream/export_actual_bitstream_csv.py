#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aggregate the completed GOP=4 physical-bitstream experiment into ONE CSV.

Reads actual .bin file sizes from disk (not only manifest-reported values) for:
  1) shared VTM anchor bitstreams
  2) method-specific EntropyBottleneck compact-residual bitstreams

The output contains:
  - one row per test video per method
  - four final dataset-level SUMMARY rows (one per method)

AdaptiveCMR:
  true It is decoder-side oracle information and its bits are NOT counted.

Default output:
  /home/featurize/work/actual_bitstream_rates_qp32.csv
"""

import argparse
import csv
import json
import math
from pathlib import Path

WORK = Path("/home/featurize/work")
CHANNEL_ROOT = WORK / "physical_channel_gop4_fast"

METHODS = ("single", "norefine", "fixed", "adaptive")
DISPLAY_NAMES = {
    "single": "Single CFTE",
    "norefine": "NoRefine",
    "fixed": "FixedCMR",
    "adaptive": "AdaptiveCMR",
}


def sum_bin_bytes(folder: Path) -> tuple[int, int]:
    files = sorted(folder.glob("*.bin")) if folder.exists() else []
    return sum(p.stat().st_size for p in files), len(files)


def safe_ratio(a, b):
    if b in (0, None) or not math.isfinite(float(b)):
        return ""
    return a / b


def load_rows(qp: int):
    rows = []
    expected_videos = None

    for method in METHODS:
        root = CHANNEL_ROOT / method / f"qp{qp}"
        manifests = sorted(root.rglob("manifest.json"))
        print(f"[{DISPLAY_NAMES[method]}] manifests found: {len(manifests)}")

        if expected_videos is None:
            expected_videos = len(manifests)

        for mp in manifests:
            m = json.loads(mp.read_text(encoding="utf-8"))
            if not m.get("complete", False):
                print(f"[WARN] incomplete manifest skipped: {mp}")
                continue

            source_video = m["source_video"]
            shared_manifest_path = Path(m["shared_anchor_manifest"])
            if not shared_manifest_path.exists():
                raise FileNotFoundError(
                    f"Shared anchor manifest missing for {source_video}: "
                    f"{shared_manifest_path}"
                )

            s = json.loads(shared_manifest_path.read_text(encoding="utf-8"))
            if not s.get("complete", False):
                raise RuntimeError(
                    f"Shared anchor manifest incomplete: {shared_manifest_path}"
                )

            width = int(s["width"])
            height = int(s["height"])
            nframes = int(s["num_frames"])
            fps = float(s["fps"])
            anchors = [int(x) for x in s["anchors"]]
            nanchors = len(anchors)
            ninter = nframes - nanchors

            if fps <= 0:
                raise RuntimeError(f"Invalid fps={fps} for {source_video}")

            duration_s = nframes / fps
            total_sequence_pixels = nframes * width * height
            total_inter_pixels = ninter * width * height

            # ---------------------------------------------------------
            # ACTUAL PHYSICAL BIT COUNTS FROM THE .bin FILES ON DISK
            # ---------------------------------------------------------
            shared_dir = shared_manifest_path.parent
            anchor_bytes, anchor_file_count = sum_bin_bytes(
                shared_dir / "anchors"
            )

            method_dir = mp.parent
            compact_bytes, compact_file_count = sum_bin_bytes(
                method_dir / "streams"
            )

            anchor_bits = anchor_bytes * 8
            compact_bits = compact_bytes * 8
            total_bits = anchor_bits + compact_bits

            # Manifest cross-checks.
            manifest_anchor_bytes = int(m["shared_anchor_payload_bytes"])
            manifest_compact_bytes = int(m["method_stream_payload_bytes"])
            manifest_total_bits = int(m["total_visual_payload_bits"])

            shared_manifest_anchor_bytes = int(s["payload_bytes"])

            anchor_match = (
                anchor_bytes == manifest_anchor_bytes
                == shared_manifest_anchor_bytes
            )
            compact_match = compact_bytes == manifest_compact_bytes
            total_match = total_bits == manifest_total_bits
            all_match = anchor_match and compact_match and total_match

            if not all_match:
                raise RuntimeError(
                    "\nPhysical-bitstream size mismatch detected!\n"
                    f"video={source_video}\n"
                    f"method={method}\n"
                    f"actual anchor bytes={anchor_bytes}, "
                    f"method manifest={manifest_anchor_bytes}, "
                    f"shared manifest={shared_manifest_anchor_bytes}\n"
                    f"actual compact bytes={compact_bytes}, "
                    f"manifest={manifest_compact_bytes}\n"
                    f"actual total bits={total_bits}, "
                    f"manifest={manifest_total_bits}"
                )

            row = {
                "record_type": "video",
                "method": DISPLAY_NAMES[method],
                "method_key": method,
                "video": source_video,
                "qp": qp,
                "width": width,
                "height": height,
                "fps": fps,
                "num_frames": nframes,
                "num_anchors": nanchors,
                "num_inter_frames": ninter,
                "duration_s": duration_s,
                "anchor_bin_files": anchor_file_count,
                "compact_bin_files": compact_file_count,
                "anchor_bits": anchor_bits,
                "compact_bits": compact_bits,
                "total_bits": total_bits,

                # Sequence-normalized actual BPP:
                # all transmitted bits / all source pixels in the video.
                "anchor_bpp_sequence": (
                    anchor_bits / total_sequence_pixels
                    if total_sequence_pixels else 0.0
                ),
                "compact_bpp_sequence": (
                    compact_bits / total_sequence_pixels
                    if total_sequence_pixels else 0.0
                ),
                "total_bpp_sequence": (
                    total_bits / total_sequence_pixels
                    if total_sequence_pixels else 0.0
                ),

                # Compact BPP normalized only by generated inter frames.
                # This is the closest physical-stream analogue to an
                # inter-frame compact-feature BPP.
                "compact_bpp_inter_only": (
                    compact_bits / total_inter_pixels
                    if total_inter_pixels else 0.0
                ),

                "anchor_kbps": anchor_bits / duration_s / 1000.0,
                "compact_kbps": compact_bits / duration_s / 1000.0,
                "total_kbps": total_bits / duration_s / 1000.0,

                "compact_rate_vs_single": "",
                "total_rate_vs_single": "",

                "adaptive_oracle_It_at_decoder": method == "adaptive",
                "oracle_It_bits_counted": False,
                "physical_bin_size_crosscheck": "PASS",
            }
            rows.append(row)

    return rows


def add_per_video_ratios(rows):
    singles = {
        r["video"]: r
        for r in rows
        if r["record_type"] == "video" and r["method_key"] == "single"
    }

    for r in rows:
        if r["record_type"] != "video":
            continue
        s = singles.get(r["video"])
        if not s:
            continue

        r["compact_rate_vs_single"] = safe_ratio(
            r["compact_bits"], s["compact_bits"]
        )
        r["total_rate_vs_single"] = safe_ratio(
            r["total_bits"], s["total_bits"]
        )


def summary_rows(rows, qp):
    out = []

    by_method = {
        method: [
            r for r in rows
            if r["record_type"] == "video" and r["method_key"] == method
        ]
        for method in METHODS
    }

    pooled = {}

    for method in METHODS:
        rs = by_method[method]
        if not rs:
            continue

        total_frames = sum(r["num_frames"] for r in rs)
        total_anchors = sum(r["num_anchors"] for r in rs)
        total_inter = sum(r["num_inter_frames"] for r in rs)
        total_duration = sum(r["duration_s"] for r in rs)

        anchor_bits = sum(r["anchor_bits"] for r in rs)
        compact_bits = sum(r["compact_bits"] for r in rs)
        total_bits = anchor_bits + compact_bits

        total_pixels = sum(
            r["num_frames"] * r["width"] * r["height"] for r in rs
        )
        inter_pixels = sum(
            r["num_inter_frames"] * r["width"] * r["height"] for r in rs
        )

        widths = {r["width"] for r in rs}
        heights = {r["height"] for r in rs}
        fps_values = {round(float(r["fps"]), 6) for r in rs}

        sr = {
            "record_type": "SUMMARY",
            "method": DISPLAY_NAMES[method],
            "method_key": method,
            "video": "ALL_TEST_VIDEOS",
            "qp": qp,
            "width": next(iter(widths)) if len(widths) == 1 else "mixed",
            "height": next(iter(heights)) if len(heights) == 1 else "mixed",
            "fps": next(iter(fps_values)) if len(fps_values) == 1 else "mixed",
            "num_frames": total_frames,
            "num_anchors": total_anchors,
            "num_inter_frames": total_inter,
            "duration_s": total_duration,
            "anchor_bin_files": sum(r["anchor_bin_files"] for r in rs),
            "compact_bin_files": sum(r["compact_bin_files"] for r in rs),
            "anchor_bits": anchor_bits,
            "compact_bits": compact_bits,
            "total_bits": total_bits,
            "anchor_bpp_sequence": anchor_bits / total_pixels,
            "compact_bpp_sequence": compact_bits / total_pixels,
            "total_bpp_sequence": total_bits / total_pixels,
            "compact_bpp_inter_only": (
                compact_bits / inter_pixels if inter_pixels else 0.0
            ),
            "anchor_kbps": anchor_bits / total_duration / 1000.0,
            "compact_kbps": compact_bits / total_duration / 1000.0,
            "total_kbps": total_bits / total_duration / 1000.0,
            "compact_rate_vs_single": "",
            "total_rate_vs_single": "",
            "adaptive_oracle_It_at_decoder": method == "adaptive",
            "oracle_It_bits_counted": False,
            "physical_bin_size_crosscheck": (
                "PASS" if all(
                    r["physical_bin_size_crosscheck"] == "PASS" for r in rs
                ) else "FAIL"
            ),
        }

        pooled[method] = sr
        out.append(sr)

    single = pooled.get("single")
    if single:
        for r in out:
            r["compact_rate_vs_single"] = safe_ratio(
                r["compact_bits"], single["compact_bits"]
            )
            r["total_rate_vs_single"] = safe_ratio(
                r["total_bits"], single["total_bits"]
            )

    return out


def write_csv(rows, output_path):
    fieldnames = [
        "record_type",
        "method",
        "method_key",
        "video",
        "qp",
        "width",
        "height",
        "fps",
        "num_frames",
        "num_anchors",
        "num_inter_frames",
        "duration_s",
        "anchor_bin_files",
        "compact_bin_files",
        "anchor_bits",
        "compact_bits",
        "total_bits",
        "anchor_bpp_sequence",
        "compact_bpp_sequence",
        "total_bpp_sequence",
        "compact_bpp_inter_only",
        "anchor_kbps",
        "compact_kbps",
        "total_kbps",
        "compact_rate_vs_single",
        "total_rate_vs_single",
        "adaptive_oracle_It_at_decoder",
        "oracle_It_bits_counted",
        "physical_bin_size_crosscheck",
    ]

    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Put the four thesis-ready summary rows first.
        summaries = [r for r in rows if r["record_type"] == "SUMMARY"]
        videos = [r for r in rows if r["record_type"] == "video"]

        writer.writerows(summaries)
        writer.writerows(videos)


def print_summary(summaries):
    print("\n" + "=" * 105)
    print("DATASET-LEVEL ACTUAL PHYSICAL BITSTREAM SUMMARY")
    print("=" * 105)
    header = (
        f"{'Method':<14}"
        f"{'Anchor BPP':>13}"
        f"{'Compact BPP':>14}"
        f"{'Total BPP':>12}"
        f"{'Inter compact BPP':>19}"
        f"{'Total kbps':>13}"
        f"{'Compact/Single':>16}"
    )
    print(header)
    print("-" * 105)

    for r in summaries:
        ratio = r["compact_rate_vs_single"]
        ratio_str = f"{ratio:.3f}x" if isinstance(ratio, (int, float)) else ""
        print(
            f"{r['method']:<14}"
            f"{r['anchor_bpp_sequence']:>13.6f}"
            f"{r['compact_bpp_sequence']:>14.6f}"
            f"{r['total_bpp_sequence']:>12.6f}"
            f"{r['compact_bpp_inter_only']:>19.6f}"
            f"{r['total_kbps']:>13.3f}"
            f"{ratio_str:>16}"
        )

    print("=" * 105)
    print(
        "NOTE: AdaptiveCMR true It is decoder-side oracle information and "
        "is NOT counted in transmitted bits."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qp", type=int, default=32)
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV output path",
    )
    args = ap.parse_args()

    output = args.output or (
        WORK / f"actual_bitstream_rates_qp{args.qp}.csv"
    )

    rows = load_rows(args.qp)
    if not rows:
        raise RuntimeError(
            f"No completed manifests found under {CHANNEL_ROOT} for QP={args.qp}"
        )

    add_per_video_ratios(rows)
    summaries = summary_rows(rows, args.qp)
    all_rows = summaries + rows
    write_csv(all_rows, output)

    print_summary(summaries)

    video_rows = [r for r in rows if r["record_type"] == "video"]
    print(f"\nPer-video method rows: {len(video_rows)}")
    print(f"Summary rows: {len(summaries)}")
    print(f"CSV written to:\n{output}")
    print("\nThe first four CSV rows are the thesis-ready dataset summaries.")


if __name__ == "__main__":
    main()
