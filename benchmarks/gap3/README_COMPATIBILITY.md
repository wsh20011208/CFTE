# Four-method benchmark compatibility notes

This package was checked against these uploaded projects:

- `CFTE`
- `CFTE_NoRefine`
- `CFTE_Refined_FixedCMR`
- `CFTE_Refined_AdaptiveCMR`

## Verified interfaces

- Single CFTE exposes `occlusion_map` but not its pre-mask feature. The
  benchmark now reconstructs the exact feature path
  `first -> down_blocks -> deform_input` before computing feature attenuation.
- NoRefine, FixedCMR, and AdaptiveCMR expose the final post-superposition
  `occlusion_map` and `deformed_feature`.
- AdaptiveCMR accepts `target_image` and `dists_model` in
  `forward_superposed_after_deform`.
- Dual checkpoints support the shared/branch aliases used by the project
  `Logger.load_cpk`.

## Important runtime constraints

1. Activate the original `cfte39` environment first.
2. CUDA is mandatory. The project `dense_motion.py` files contain explicit
   `.cuda()` calls.
3. The benchmark calls `torch.cuda.set_device(DEVICE)` before inference, so a
   selected logical CUDA device is used consistently.
4. VGG16 pretrained weights must already exist in the PyTorch cache or be
   downloadable. The project-local DISTS/LPIPS scalar weights are included in
   each project.
5. The evaluator keeps only one decoded video in RAM and streams per-video
   panels to disk, avoiding unbounded RAM growth.
6. Sample images and video panels are disabled by default in the full runner.

## Server usage

Place the four extracted projects at:

```text
/home/featurize/work/CFTE
/home/featurize/work/CFTE_NoRefine
/home/featurize/work/CFTE_Refined_FixedCMR
/home/featurize/work/CFTE_Refined_AdaptiveCMR
```

Place this `benchmark` directory under `/home/featurize/work/`, then run:

```bash
conda activate cfte39
cd /home/featurize/work
bash benchmark/run_four_models_example.sh
```

The runner automatically selects the most recently modified checkpoint under
each project. To force exact checkpoints:

```bash
SINGLE_CKPT=/path/to/single.pth.tar \
NOREFINE_CKPT=/path/to/norefine.pth.tar \
FIXEDCMR_CKPT=/path/to/fixed.pth.tar \
ADAPTIVE_CKPT=/path/to/adaptive.pth.tar \
bash benchmark/run_four_models_example.sh
```

For a short initial check, edit `GAPS=(3 5 7)` temporarily or run one evaluator
against a CSV generated with `--max_videos 1 --max_targets_per_video 1`.
