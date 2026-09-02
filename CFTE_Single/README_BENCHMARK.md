# CFTE Four-Model Benchmark Scripts

This package evaluates four CFTE variants with one shared deterministic frame-index CSV:

1. Single-frame CFTE: `past -> target`
2. Two-frame NoRefine CFTE: `past + future -> target`
3. Two-frame FixedCMR CFTE: `past + future -> target`
4. Two-frame AdaptiveCMR CFTE: `past + future -> target`

## Frame sampling rule: use all valid multiples of 3

For every video, target frames are exact multiples of 3 by default:

```text
target_idx = 3, 6, 9, 12, ...
past_idx   = target_idx - 3
future_idx = target_idx + 3
```

A target frame is kept only if both references exist:

```text
past_idx >= 0 and future_idx < num_frames
```

So if the last target is a multiple of 3 but `target_idx + 3` does not exist, that target is discarded. This is intentional because the same CSV is shared by all four methods. The single-frame evaluator ignores `future_idx`, but the CSV still stores it so that single-frame and two-frame models are evaluated on exactly the same target frames.

By default, the script uses **all possible valid target frames**. Do **not** pass `--max_targets_per_video` for the formal comparison. Use `--max_targets_per_video` only for quick debugging.

## Install metrics packages

To avoid `LPIPS=NaN`, install official LPIPS in the active conda environment:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cfte39
python -m pip install lpips pytorch-msssim
```

You can force the evaluator to stop if LPIPS is unavailable:

```bash
--require_lpips
```

## Generate the shared CSV from test videos

For test videos stored in `~/CFTE/vox/test`:

```bash
cd ~/CFTE
mkdir -p benchmark_results_gap3_stride3/tables
mkdir -p benchmark_results_gap3_stride3/video_panels
mkdir -p benchmark_results_gap3_stride3/predictions

rm -f benchmark_results_gap3_stride3/tables/test_triplets_gap3_stride3.csv

python benchmark/make_test_triplets_gap3_stride3.py   --root_dir ~/CFTE/vox/test   --output_csv benchmark_results_gap3_stride3/tables/test_triplets_gap3_stride3.csv
```

If you truly want to index training videos instead, use:

```bash
python benchmark/make_test_triplets_gap3_stride3.py   --root_dir ~/CFTE/vox/train   --output_csv benchmark_results_gap3_stride3/tables/train_triplets_gap3_stride3.csv
```

Check the selected target frames:

```bash
python - <<'PY'
import pandas as pd
p='benchmark_results_gap3_stride3/tables/test_triplets_gap3_stride3.csv'
df=pd.read_csv(p)
print(df[['video_id','num_frames','past_idx','target_idx','future_idx']].head(80))
print('all targets multiples of 3:', (df['target_idx'] % 3 == 0).all())
print('all past available:', (df['past_idx'] >= 0).all())
print('all future available:', (df['future_idx'] < df['num_frames']).all())
print('samples:', len(df), 'videos:', df['video_id'].nunique())
PY
```

## Run one model, saving metrics and per-video visual panels

Single-frame CFTE example:

```bash
python benchmark/test_single_cfte.py   --project_dir ~/CFTE   --config ~/CFTE/config/vox-256.yaml   --checkpoint ~/CFTE/checkpoint_new/L4/vox-256/00000099-checkpoint.pth.tar   --root_dir ~/CFTE/vox/test   --test_csv benchmark_results_gap3_stride3/tables/test_triplets_gap3_stride3.csv   --output_csv benchmark_results_gap3_stride3/tables/results_single_cfte.csv   --save_video_panels benchmark_results_gap3_stride3/video_panels   --video_panel_max_samples 20   --require_lpips   --device 0
```

`--video_panel_max_samples 20` only limits how many rows are displayed in each per-video PNG. It does not affect the CSV metrics. If you want the panel to show every sampled target frame, use:

```bash
--video_panel_max_samples 0
```

Be careful: this can create very tall images for long videos.

For the two-frame scripts, use the same CSV and replace the script/project/checkpoint accordingly:

```text
benchmark/test_norefine_twoframe.py
benchmark/test_fixedcmr_twoframe.py
benchmark/test_adaptivecmr_twoframe.py
```

## Merge results

After all four CSVs are produced:

```bash
python benchmark/merge_results.py   --tables_dir benchmark_results_gap3_stride3/tables   --output_dir benchmark_results_gap3_stride3/tables
```
