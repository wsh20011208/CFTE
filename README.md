# CFTE Multi-Reference Thesis Archive

Source-code archive for the thesis **Superposition Models Using Multiple Reference Frames for Enhanced Generative Video Quality**.

## Methods

- **Single CFTE** — single-reference baseline
- **NoRefine** — two shared-weight CFTE branches with direct equal-weight superposition
- **FixedCMR** — two sequential learned Conditional Motion Refinement modules before final superposition
- **AdaptiveCMR** — FixedCMR initialization followed by target-assisted DISTS-guided iterative motion correction

## Repository layout

```text
CFTE/
CFTE_NoRefine/
CFTE_Refined_FixedCMR/
CFTE_Refined_AdaptiveCMR/
benchmarks/
  gap3/
  gap5/
  gap7/
physical_bitstream/
```

## Exact controlled configurations

The reported controlled experiments use:

| Method | Gap 3/5/7 batch size | Epochs | Repetitions |
|---|---:|---:|---:|
| Single CFTE | 10 | 200 | 10 |
| NoRefine | 10 | 200 | 10 |
| FixedCMR | 10 | 200 | 10 |
| AdaptiveCMR | 8 | 200 | 10 |

Each project contains `config/vox-256_gap3.yaml`, `vox-256_gap5.yaml`, and `vox-256_gap7.yaml`.

`config/vox-256.yaml` is included only as a **gap-3 compatibility alias** for legacy scripts. Formal gap-specific reproduction should always use the explicitly named gap configuration.

## Formal benchmark

The three benchmark directories are gap-specific. Their `run_four_models_example.sh` wrappers use the corresponding gap configuration and require the corresponding final epoch-199 checkpoint.

Matched target counts in the reported results:

- gap 3: 15,023
- gap 5: 14,191
- gap 7: 13,627
- total: 42,841

The formal benchmark reports DISTS, LPIPS, PSNR, SSIM and likelihood-estimated BPP.

## Supplementary physical-bitstream experiment

See `physical_bitstream/README.md`.

This experiment is a dense GOP-4 implementation-level demonstration using gap-3-trained checkpoints, VTM anchor coding at QP 32 and real entropy-coded compact-residual byte streams. It is not a replacement for the strict symmetric gap-3/5/7 benchmark.

**AdaptiveCMR caveat:** the true target frame is supplied directly to the decoder as DISTS oracle information. The bitrate required to provide that target is excluded; AdaptiveCMR must therefore not be interpreted as a deployable target-free codec.

## Data

The VoxCeleb-derived data are not redistributed in this repository. Configure the benchmark/data paths for your own processed dataset.

## Checkpoints

Large final checkpoints should be distributed through a GitHub Release rather than normal Git history.

## Environment

The project snapshots contain `environment.yml`. Use the archived environment as the software reference for reproduction.

## License and upstream code

Retain the original project license files and respect the licenses of CFTE, VTM, CompressAI and all third-party dependencies.
