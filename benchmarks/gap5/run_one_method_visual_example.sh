#!/usr/bin/env bash
set -euo pipefail

# Example: run only one method and generate per-video logger-style panels.
# Edit METHOD and paths before running.
METHOD="single"   # single | norefine | fixedcmr | adaptive
ROOT_DIR="/home/jovyan/CFTE/vox"
OUT_DIR="benchmark_results_gap3_stride3"
DEVICE=0

SINGLE_DIR="/home/jovyan/CFTE"
NOREFINE_DIR="/home/jovyan/CFTE_superposition/CFTE_NoRefine"
FIXEDCMR_DIR="/home/jovyan/CFTE_superposition/CFTE_Refined_FixedCMR"
ADAPTIVE_DIR="/home/jovyan/CFTE_superposition/CFTE_Refined_AdaptiveCMR"

SINGLE_CKPT="$SINGLE_DIR/checkpoint_new/vox-256/00000999-checkpoint.pth.tar"
NOREFINE_CKPT="$NOREFINE_DIR/checkpoint_new/vox-256/00000999-checkpoint.pth.tar"
FIXEDCMR_CKPT="$FIXEDCMR_DIR/checkpoint_new/vox-256/00000999-checkpoint.pth.tar"
ADAPTIVE_CKPT="$ADAPTIVE_DIR/checkpoint_new/vox-256/00000999-checkpoint.pth.tar"

mkdir -p "$OUT_DIR/tables" "$OUT_DIR/video_panels" "$OUT_DIR/predictions"

if [ ! -f "$OUT_DIR/tables/test_triplets_gap3_stride3.csv" ]; then
  python benchmark/make_test_triplets_gap3_stride3.py \
    --root_dir "$ROOT_DIR" \
    --gap 3 \
    --stride 3 \
    --output_csv "$OUT_DIR/tables/test_triplets_gap3_stride3.csv"
fi

case "$METHOD" in
  single)
    python benchmark/test_single_cfte.py \
      --project_dir "$SINGLE_DIR" \
      --config "$SINGLE_DIR/config/vox-256.yaml" \
      --checkpoint "$SINGLE_CKPT" \
      --root_dir "$ROOT_DIR" \
      --test_csv "$OUT_DIR/tables/test_triplets_gap3_stride3.csv" \
      --output_csv "$OUT_DIR/tables/results_single_cfte.csv" \
      --save_images "$OUT_DIR/predictions/single_cfte" \
      --save_video_panels "$OUT_DIR/video_panels" \
      --video_panel_max_samples 12 \
      --device "$DEVICE"
    ;;
  norefine)
    python benchmark/test_norefine_twoframe.py \
      --project_dir "$NOREFINE_DIR" \
      --config "$NOREFINE_DIR/config/vox-256.yaml" \
      --checkpoint "$NOREFINE_CKPT" \
      --root_dir "$ROOT_DIR" \
      --test_csv "$OUT_DIR/tables/test_triplets_gap3_stride3.csv" \
      --output_csv "$OUT_DIR/tables/results_norefine.csv" \
      --save_images "$OUT_DIR/predictions/norefine" \
      --save_video_panels "$OUT_DIR/video_panels" \
      --video_panel_max_samples 12 \
      --device "$DEVICE"
    ;;
  fixedcmr)
    python benchmark/test_fixedcmr_twoframe.py \
      --project_dir "$FIXEDCMR_DIR" \
      --config "$FIXEDCMR_DIR/config/vox-256.yaml" \
      --checkpoint "$FIXEDCMR_CKPT" \
      --root_dir "$ROOT_DIR" \
      --test_csv "$OUT_DIR/tables/test_triplets_gap3_stride3.csv" \
      --output_csv "$OUT_DIR/tables/results_fixedcmr.csv" \
      --save_images "$OUT_DIR/predictions/fixedcmr" \
      --save_video_panels "$OUT_DIR/video_panels" \
      --video_panel_max_samples 12 \
      --device "$DEVICE"
    ;;
  adaptive)
    python benchmark/test_adaptivecmr_twoframe.py \
      --project_dir "$ADAPTIVE_DIR" \
      --config "$ADAPTIVE_DIR/config/vox-256.yaml" \
      --checkpoint "$ADAPTIVE_CKPT" \
      --root_dir "$ROOT_DIR" \
      --test_csv "$OUT_DIR/tables/test_triplets_gap3_stride3.csv" \
      --output_csv "$OUT_DIR/tables/results_adaptivecmr.csv" \
      --save_images "$OUT_DIR/predictions/adaptivecmr" \
      --save_video_panels "$OUT_DIR/video_panels" \
      --video_panel_max_samples 12 \
      --device "$DEVICE"
    ;;
  *)
    echo "Unknown METHOD=$METHOD" >&2
    exit 1
    ;;
esac
