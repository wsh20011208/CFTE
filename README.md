# CFTE Multi-Reference Thesis Archive

Source-code archive for the thesis **Superposition Models Using Multiple Reference Frames for Enhanced Generative Video Quality**.

## Methods

- **Single CFTE** — single-reference baseline
- **NoRefine** — two shared-weight CFTE branches with direct equal-weight superposition
- **FixedCMR** — two sequential learned Conditional Motion Refinement modules before final superposition
- **AdaptiveCMR** — FixedCMR initialization followed by target-assisted DISTS-guided iterative motion correction

## Repository layout

```text
CFTE_Single/
CFTE_NoRefine/
CFTE_Refined_FixedCMR/
CFTE_Refined_AdaptiveCMR/
benchmarks/
  gap3/
  gap5/
  gap7/
data/
  train/
  test/
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

Each project contains:

- `config/vox-256_gap3.yaml`
- `config/vox-256_gap5.yaml`
- `config/vox-256_gap7.yaml`

`config/vox-256.yaml` is included only as a **gap-3 compatibility alias** for legacy scripts. Formal gap-specific reproduction should use the explicitly named gap configuration.

## Formal benchmark

The three benchmark directories are gap-specific. Their `run_four_models_example.sh` wrappers use the corresponding gap configuration and expect the corresponding final epoch-199 checkpoint.

Matched target counts in the reported results:

- gap 3: 15,023
- gap 5: 14,191
- gap 7: 13,627
- total: 42,841

The formal benchmark reports:

- DISTS
- LPIPS
- PSNR
- SSIM
- likelihood-estimated BPP

## Supplementary physical-bitstream experiment

See `physical_bitstream/README.md`.

This experiment is a dense GOP-4 implementation-level demonstration using gap-3-trained checkpoints, VTM anchor coding at QP 32, and actual entropy-coded compact-residual byte streams. It is supplementary to the strict symmetric gap-3/5/7 benchmark rather than a replacement for it.

The physical-bitstream pipeline uses:

- shared VTM-reconstructed anchors
- one compact residual stream per inter frame for **Single CFTE**
- two compact residual streams per inter frame for **NoRefine**, **FixedCMR**, and **AdaptiveCMR**
- an error-free file-based digital channel

**AdaptiveCMR caveat:** the true target frame is supplied directly to the decoder as DISTS oracle information. The bitrate required to provide that oracle target is excluded from the reported transmitted-bit counts. AdaptiveCMR should therefore not be interpreted as a deployable target-free codec.

### Authoritative codec path for the thesis

The legacy `Encoder.py` and `Decoder.py` files retained inside the four
model directories originate from the upstream CFTE implementation and are
preserved for reference and compatibility.

They are **not used to generate the physical-bitstream results reported in
this thesis**.

The authoritative encode/decode path for the thesis supplementary
physical-bitstream experiment is the dense GOP-4 pipeline under:

`physical_bitstream/`

The reported GOP-4 experiment uses:

- `prepare_shared_anchors_gop4.py` for shared VTM anchor preparation;
- `encode_streams_gop4_fast.py` for compact-residual entropy coding;
- `decode_channel_gop4_fast.py` for decoder-side reconstruction;
- `run_per_video_codec_gop4.py` / `run_per_video_codec_gop4.sh` for
  per-video orchestration.

The experiment uses gap-3-trained checkpoints, shared VTM-reconstructed
anchors at QP 32, and actual entropy-coded compact-residual byte streams.

The legacy per-model `Encoder.py` / `Decoder.py` path should therefore not
be used to reproduce the physical-bitstream measurements reported in the thesis.

## Reconstruction outputs

The generated GOP-4 physical-channel and reconstruction outputs are included in the repository, while packaged archives are additionally provided through the GitHub Releases. The reconstructed outputs of the supplementary dense GOP-4 physical-bitstream experiment are distributed through the GitHub Release:

**[`v1.0.0-thesis` — Thesis Final Archive and Reconstruction Outputs](https://github.com/wsh20011208/CFTE/releases/tag/v1.0.0-thesis)**

The release contains:

```text
codec_recon_gop4_fast.zip
```

with **1,776 reconstructed MP4 files** in total:

- 444 Single CFTE reconstructions
- 444 NoRefine reconstructions
- 444 FixedCMR reconstructions
- 444 AdaptiveCMR oracle reconstructions

All four methods use the same 444-video test set, and the supplementary experiment uses VTM anchor coding at QP 32.

The original VoxCeleb source videos are **not redistributed**.

The released MP4 files are reconstruction/visualization outputs. Their MP4 container sizes are **not** used as transmitted bitrate in the thesis.

## Checkpoints

The `v1.0.0-thesis` release provides the final epoch-199 gap-3 checkpoints for all four methods:

- Single CFTE
- NoRefine
- FixedCMR
- AdaptiveCMR

Gap-5 and gap-7 pretrained checkpoints are not distributed with this archive.

The repository contains the complete source code, gap-specific training configurations, benchmark scripts, and supplementary physical-bitstream implementation.

To reproduce a trained model for gap 3, 5, or 7:

1. select the corresponding `vox-256_gap*.yaml` configuration;
2. train the desired method using that configuration;
3. retain the final epoch-199 checkpoint;
4. run the matching benchmark under `benchmarks/gap3/`, `benchmarks/gap5/`, or `benchmarks/gap7/`.

The supplementary dense GOP-4 physical-bitstream experiment uses **gap-3-trained checkpoints**.

## Data

The repository includes the preprocessed 256x256 train and test videos used by the thesis experiments:

```text
data/
├── train/
└── test/
```

## Environment

The archived project snapshots contain `environment.yml`. Use the archived environment specification as the software reference for reproduction.

## License and upstream code

Retain the original project license files and respect the licenses of CFTE, VTM, CompressAI, VoxCeleb, and all third-party dependencies.
