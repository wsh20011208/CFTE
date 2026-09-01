#!/usr/bin/env python3
"""Static/server preflight for the four CFTE benchmark projects.

This checks the exact interfaces used by the benchmark before a long evaluation:
- required Python packages and CUDA
- project/config/checkpoint files
- checkpoint key compatibility
- final occlusion/pre-mask feature output interfaces
- DISTS/LPIPS weight files
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import sys
from pathlib import Path

import torch
import yaml

from benchmark_common import project_import_context


METHOD_SPECS = {
    "single": {
        "dual": False,
        "required_checkpoint_groups": [
            ("generator",),
            ("kp_detector",),
            ("videocompressor",),
        ],
    },
    "norefine": {
        "dual": True,
        "required_checkpoint_groups": [
            ("generator_past", "generator"),
            ("generator_future", "generator"),
            ("kp_detector_past", "kp_detector"),
            ("kp_detector_future", "kp_detector"),
            ("videocompressor_past", "videocompressor"),
            ("videocompressor_future", "videocompressor"),
        ],
    },
    "fixedcmr": {
        "dual": True,
        "required_checkpoint_groups": [
            ("generator_past", "generator"),
            ("generator_future", "generator"),
            ("kp_detector_past", "kp_detector"),
            ("kp_detector_future", "kp_detector"),
            ("videocompressor_past", "videocompressor"),
            ("videocompressor_future", "videocompressor"),
        ],
    },
    "adaptive": {
        "dual": True,
        "adaptive": True,
        "required_checkpoint_groups": [
            ("generator_past", "generator"),
            ("generator_future", "generator"),
            ("kp_detector_past", "kp_detector"),
            ("kp_detector_future", "kp_detector"),
            ("videocompressor_past", "videocompressor"),
            ("videocompressor_future", "videocompressor"),
        ],
    },
}


def check_dependencies() -> None:
    modules = [
        "cv2", "compressai", "imageio", "numpy", "pandas", "scipy",
        "skimage", "torch", "torchvision", "tqdm", "yaml",
    ]
    failures = []
    for name in modules:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "unknown")
            print(f"[OK] import {name}: {version}")
        except Exception as exc:
            failures.append(f"{name}: {exc}")
            print(f"[FAIL] import {name}: {exc}")
    if failures:
        raise RuntimeError("Missing/incompatible dependencies:\n  " + "\n  ".join(failures))

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. The uploaded CFTE projects contain explicit "
            "Tensor.cuda() calls in dense_motion.py and cannot run on CPU."
        )
    print(f"[OK] CUDA devices: {torch.cuda.device_count()}")
    for idx in range(torch.cuda.device_count()):
        print(f"     cuda:{idx}: {torch.cuda.get_device_name(idx)}")


def load_checkpoint_keys(path: Path) -> set[str]:
    checkpoint = torch.load(str(path), map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint is not a dictionary: {path}")
    keys = set(checkpoint.keys())
    del checkpoint
    return keys


def check_method(method: str, project_dir: Path, config_path: Path,
                 checkpoint_path: Path) -> None:
    spec = METHOD_SPECS[method]
    print(f"\n===== {method} =====")

    for label, path, is_dir in [
        ("project", project_dir, True),
        ("config", config_path, False),
        ("checkpoint", checkpoint_path, False),
    ]:
        ok = path.is_dir() if is_dir else path.is_file()
        if not ok:
            raise FileNotFoundError(f"Missing {label}: {path}")
        print(f"[OK] {label}: {path}")

    config = yaml.safe_load(config_path.read_text())
    required_config = [
        ("model_params", "generator_params"),
        ("model_params", "kp_detector_params"),
        ("model_params", "videocompressor_params"),
        ("model_params", "common_params"),
        ("dataset_params", "frame_shape"),
    ]
    for first, second in required_config:
        if first not in config or second not in config[first]:
            raise KeyError(f"{config_path}: missing {first}.{second}")
    print("[OK] configuration sections")

    for weight_path in [
        project_dir / "modules" / "DISTS.pt",
        project_dir / "evaluate" / "weights" / "DISTS.pt",
        project_dir / "evaluate" / "weights" / "LPIPSvgg.pt",
    ]:
        if not weight_path.is_file():
            raise FileNotFoundError(f"Missing metric weight file: {weight_path}")
    print("[OK] local DISTS and LPIPS weight files")

    with project_import_context(project_dir):
        generator_module = importlib.import_module("modules.generator")
        kp_module = importlib.import_module("modules.keypoint_detector")
        rd_module = importlib.import_module("modules.RDloss")
        logger_module = importlib.import_module("logger")

        generator_cls = generator_module.OcclusionAwareGenerator
        kp_cls = kp_module.KPDetector
        vc_cls = rd_module.VideoCompressor
        logger_cls = logger_module.Logger

        print(f"[OK] generator: {generator_cls.__name__}")
        print(f"[OK] keypoint detector: {kp_cls.__name__}")
        print(f"[OK] video compressor: {vc_cls.__name__}")

        load_signature = inspect.signature(logger_cls.load_cpk)
        print(f"[OK] Logger.load_cpk{load_signature}")

        forward_signature = inspect.signature(generator_cls.forward)
        required_forward = {"source_image", "heatmap_source", "heatmap_driving"}
        if not required_forward.issubset(forward_signature.parameters):
            raise TypeError(
                f"Unexpected generator.forward signature: {forward_signature}"
            )
        print(f"[OK] generator.forward{forward_signature}")

        if spec["dual"]:
            if not hasattr(generator_cls, "forward_superposed_after_deform"):
                raise AttributeError(
                    f"{method}: generator lacks forward_superposed_after_deform"
                )
            super_signature = inspect.signature(
                generator_cls.forward_superposed_after_deform
            )
            required = {
                "source_past", "source_future", "generated_past",
                "generated_future", "weight_past", "weight_future",
            }
            if spec.get("adaptive"):
                required |= {"target_image", "dists_model"}
            if not required.issubset(super_signature.parameters):
                raise TypeError(
                    f"Unexpected superposition signature: {super_signature}"
                )
            print(
                "[OK] generator.forward_superposed_after_deform"
                f"{super_signature}"
            )

    keys = load_checkpoint_keys(checkpoint_path)
    print(f"[OK] checkpoint keys ({len(keys)}): {sorted(keys)}")
    missing_groups = [
        group for group in spec["required_checkpoint_groups"]
        if not any(key in keys for key in group)
    ]
    if missing_groups:
        raise KeyError(
            f"{method}: checkpoint lacks compatible key groups: {missing_groups}"
        )
    print("[OK] checkpoint keys are compatible with the project loader")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single_dir", required=True)
    parser.add_argument("--single_checkpoint", required=True)
    parser.add_argument("--norefine_dir", required=True)
    parser.add_argument("--norefine_checkpoint", required=True)
    parser.add_argument("--fixedcmr_dir", required=True)
    parser.add_argument("--fixedcmr_checkpoint", required=True)
    parser.add_argument("--adaptive_dir", required=True)
    parser.add_argument("--adaptive_checkpoint", required=True)
    args = parser.parse_args()

    check_dependencies()

    entries = [
        ("single", Path(args.single_dir), Path(args.single_checkpoint)),
        ("norefine", Path(args.norefine_dir), Path(args.norefine_checkpoint)),
        ("fixedcmr", Path(args.fixedcmr_dir), Path(args.fixedcmr_checkpoint)),
        ("adaptive", Path(args.adaptive_dir), Path(args.adaptive_checkpoint)),
    ]

    for method, project, checkpoint in entries:
        project = project.expanduser().resolve()
        checkpoint = checkpoint.expanduser().resolve()
        check_method(
            method,
            project,
            project / "config" / "vox-256_gap7.yaml",
            checkpoint,
        )

    print("\nAll four projects passed the static compatibility preflight.")
    print(
        "Note: the uploaded project code uses explicit .cuda() calls; run with "
        "a CUDA device and set torch.cuda.set_device through the benchmark."
    )


if __name__ == "__main__":
    main()
