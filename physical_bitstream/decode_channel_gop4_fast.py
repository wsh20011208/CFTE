#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 3: decode videos from:
  - the shared receiver-side VTM-decoded anchor cache, which was produced
    from the actual transmitted VTM .bin bitstreams in Stage 1
  - the method-specific physical EntropyBottleneck byte streams

Single / NoRefine / FixedCMR do not read true inter target frames.

AdaptiveCMR additionally reads true It from vox/test as explicitly requested
oracle side information. Those oracle frames are NOT counted as transmitted bits.
"""

import argparse
import gc
import json
import os
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
CHANNEL_ROOT = WORK / "physical_channel_gop4_fast"
RECON_ROOT = WORK / "codec_recon_gop4_fast"

METHODS = {
    "single": {
        "project": WORK / "CFTE",
        "checkpoint": WORK / "CFTE" / "00000199-checkpoint_singlecfte_d3.pth.tar",
        "out_name": "single_cfte",
    },
    "norefine": {
        "project": WORK / "CFTE_NoRefine",
        "checkpoint": WORK / "CFTE_NoRefine" / "00000199-checkpoint_norefine_d3.pth.tar",
        "out_name": "norefine",
    },
    "fixed": {
        "project": WORK / "CFTE_Refined_FixedCMR",
        "checkpoint": WORK / "CFTE_Refined_FixedCMR" / "00000199-checkpoint_fixedcmr_d3.pth.tar",
        "out_name": "fixedcmr",
    },
    "adaptive": {
        "project": WORK / "CFTE_Refined_AdaptiveCMR",
        "checkpoint": WORK / "CFTE_Refined_AdaptiveCMR" / "00000199-checkpoint_adaptivecmr_d3.pth.tar",
        "out_name": "adaptivecmr_oracle",
    },
}


def torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def pick_state(cpk, keys, label):
    for k in keys:
        if k in cpk:
            return cpk[k], k
    raise KeyError(f"Missing {label}; tried {keys}")


def setup(method, device):
    spec = METHODS[method]
    project = spec["project"]
    os.chdir(project)
    sys.path.insert(0, str(project))

    from modules.generator import OcclusionAwareGenerator
    from modules.keypoint_detector import KPDetector
    from modules.RDloss import VideoCompressor

    with open(project / "config" / "vox-256_gap3.yaml", "r") as f:
        config = yaml.safe_load(f)

    generator = OcclusionAwareGenerator(
        **config["model_params"]["generator_params"],
        **config["model_params"]["common_params"],
    ).to(device)
    kp = KPDetector(
        **config["model_params"]["kp_detector_params"],
        **config["model_params"]["common_params"],
    ).to(device)
    vc = VideoCompressor(
        **config["model_params"]["videocompressor_params"]
    ).to(device)

    cpk = torch_load(spec["checkpoint"], device)

    if method == "single":
        gs, gk = pick_state(cpk, ["generator"], "generator")
        ks, kk = pick_state(cpk, ["kp_detector"], "kp_detector")
        vs, vk = pick_state(cpk, ["videocompressor"], "videocompressor")
    else:
        gs, gk = pick_state(
            cpk, ["generator_past", "generator", "generator_future"], "generator"
        )
        ks, kk = pick_state(
            cpk, ["kp_detector_past", "kp_detector", "kp_detector_future"], "kp_detector"
        )
        vs, vk = pick_state(
            cpk, ["videocompressor_past", "videocompressor", "videocompressor_future"],
            "videocompressor"
        )

    generator.load_state_dict(gs, strict=True)
    kp.load_state_dict(ks, strict=True)
    vc.load_state_dict(vs, strict=True)

    generator.eval()
    kp.eval()
    vc.eval()
    vc.entropy_bottleneck.update(force=True)

    dists_model = None
    if method == "adaptive":
        from modules.dists import DISTS
        dists_model = DISTS().to(device).eval()

    print(f"[LOAD] method={method} epoch={cpk.get('epoch')}")
    print(f"[LOAD] generator={gk} kp={kk} vc={vk}")
    return generator, kp, vc, dists_model


def to_tensor(frame, device):
    x = torch.from_numpy(frame.astype(np.float32) / 255.0)
    return x.permute(2, 0, 1).unsqueeze(0).to(device)


def tensor_to_u8(x):
    arr = x.detach().float().cpu().clamp(0, 1)[0]
    return np.clip(
        np.rint(arr.permute(1, 2, 0).numpy() * 255.0), 0, 255
    ).astype(np.uint8)


def read_oracle_video(path):
    reader = imageio.get_reader(str(path))
    try:
        frames = [np.asarray(f)[..., :3] for f in reader]
    finally:
        reader.close()
    return np.stack(frames, axis=0).astype(np.uint8)


def decompress_target(vc, stream_path, spatial, source_kp):
    payload = Path(stream_path).read_bytes()
    residual_hat = vc.entropy_bottleneck.decompress(
        [payload], tuple(int(v) for v in spatial)
    )
    return {"value": residual_hat + source_kp["value"]}


@torch.no_grad()
def decode_single(generator, kp, vc, source, stream, spatial):
    kp_source = kp(source)
    target_hat = decompress_target(vc, stream, spatial, kp_source)
    out = generator(
        source,
        heatmap_source=kp_source,
        heatmap_driving=target_hat,
    )
    return out["prediction"].detach().clamp(0, 1)


def decode_dual(method, generator, kp, vc, dists_model,
                past, future, stream_p, stream_f, spatial, oracle_target=None):
    ctx = torch.enable_grad() if method == "adaptive" else torch.no_grad()

    with ctx:
        kp_p = kp(past)
        kp_f = kp(future)

        target_p = decompress_target(vc, stream_p, spatial, kp_p)
        target_f = decompress_target(vc, stream_f, spatial, kp_f)

        out_p = generator(
            past,
            heatmap_source=kp_p,
            heatmap_driving=target_p,
        )
        out_f = generator(
            future,
            heatmap_source=kp_f,
            heatmap_driving=target_f,
        )

        out_p.update({
            "heatmap_source": kp_p,
            "heatmap_driving": target_p,
        })
        out_f.update({
            "heatmap_source": kp_f,
            "heatmap_driving": target_f,
        })

        kwargs = {}
        if method == "adaptive":
            if oracle_target is None:
                raise RuntimeError("Adaptive oracle It missing")
            kwargs = {
                "target_image": oracle_target,
                "dists_model": dists_model,
            }

        out = generator.forward_superposed_after_deform(
            past, future, out_p, out_f, 0.5, 0.5, **kwargs
        )
        return out["prediction"].detach().clamp(0, 1)


def make_writer(path, fps):
    path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(
        str(path),
        fps=float(fps),
        codec="libx264",
        ffmpeg_params=["-preset", "fast", "-crf", "17", "-pix_fmt", "yuv420p"],
        macro_block_size=None,
    )


def decode_one(method, manifest_path, out_path,
               generator, kp, vc, dists_model, device, overwrite):
    if out_path.exists() and not overwrite:
        return "skipped"

    m = json.loads(manifest_path.read_text())
    shared_path = Path(m["shared_anchor_manifest"])
    shared = json.loads(shared_path.read_text())
    shared_dir = shared_path.parent
    channel_dir = manifest_path.parent

    n = int(m["num_frames"])
    fps = float(shared["fps"])
    spatial = m["residual_spatial_shape"]

    anchors = {}
    for a in shared["anchors"]:
        rec = shared["anchor_records"][str(a)]
        frame = np.load(shared_dir / rec["receiver_cache"], allow_pickle=False)
        anchors[int(a)] = to_tensor(frame, device)

    oracle_frames = None
    if method == "adaptive":
        oracle_frames = read_oracle_video(TEST_DIR / m["source_video"])
        if len(oracle_frames) != n:
            raise RuntimeError("Oracle frame count mismatch")

    tmp_out = out_path.with_name(out_path.stem + ".tmp.mp4")
    tmp_out.unlink(missing_ok=True)
    writer = make_writer(tmp_out, fps)

    try:
        for idx in range(n):
            r = m["frames"][str(idx)]

            if r["type"] == "anchor":
                writer.append_data(tensor_to_u8(anchors[idx]))
                continue

            if method == "single":
                past = int(r["past_anchor"])
                pred = decode_single(
                    generator, kp, vc, anchors[past],
                    channel_dir / r["single_stream"], spatial
                )
            else:
                past = int(r["past_anchor"])
                future = int(r["future_anchor"])
                oracle = None
                if method == "adaptive":
                    oracle = to_tensor(oracle_frames[idx], device)

                pred = decode_dual(
                    method, generator, kp, vc, dists_model,
                    anchors[past], anchors[future],
                    channel_dir / r["past_stream"],
                    channel_dir / r["future_stream"],
                    spatial, oracle_target=oracle
                )
                if oracle is not None:
                    del oracle

            writer.append_data(tensor_to_u8(pred))
            del pred

    finally:
        writer.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(tmp_out, out_path)

    del anchors
    if oracle_frames is not None:
        del oracle_frames
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

    generator, kp, vc, dists_model = setup(args.method, device)

    channel_root = CHANNEL_ROOT / args.method / f"qp{args.qp}"
    manifests = sorted(channel_root.rglob("manifest.json"))[args.start_index:]
    if args.max_videos > 0:
        manifests = manifests[:args.max_videos]

    out_root = (
        RECON_ROOT / METHODS[args.method]["out_name"] / f"qp{args.qp}"
    )
    error_log = RECON_ROOT / f"decode_errors_{args.method}_qp{args.qp}.log"

    print(f"[GPU] {torch.cuda.get_device_name(0)}")
    print(f"[DECODE] method={args.method} videos={len(manifests)}")
    if args.method == "adaptive":
        print("[ORACLE] TRUE It IS GIVEN DIRECTLY TO THE ADAPTIVECMR DECODER")
        print("[ORACLE] It IS NOT INCLUDED IN TRANSMITTED-BIT COUNTS")

    failures = 0
    for mp in tqdm(manifests, desc=f"decode-{args.method}"):
        m = json.loads(mp.read_text())
        out_path = (out_root / Path(m["source_video"])).with_suffix(".mp4")
        try:
            decode_one(
                args.method, mp, out_path,
                generator, kp, vc, dists_model, device, args.overwrite
            )
        except Exception as e:
            failures += 1
            error_log.parent.mkdir(parents=True, exist_ok=True)
            with open(error_log, "a", encoding="utf-8") as f:
                f.write(f"\n===== {mp} =====\n")
                f.write(f"{type(e).__name__}: {e}\n")
                f.write(traceback.format_exc())
            print(f"\n[DECODE ERROR] {mp}: {e}")

    print(f"[DONE] decode method={args.method} failures={failures}")
    if failures:
        print(f"[ERROR LOG] {error_log}")


if __name__ == "__main__":
    main()
