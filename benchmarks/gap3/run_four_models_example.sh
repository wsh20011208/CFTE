#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

GAP=3
STRIDE="${STRIDE:-3}"
DEVICE="${DEVICE:-0}"
ROOT_DIR="${ROOT_DIR:-$REPO_ROOT/data}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/benchmark_results/gap3}"

SINGLE_DIR="${SINGLE_DIR:-$REPO_ROOT/CFTE_Single}"
NOREFINE_DIR="${NOREFINE_DIR:-$REPO_ROOT/CFTE_NoRefine}"
FIXEDCMR_DIR="${FIXEDCMR_DIR:-$REPO_ROOT/CFTE_Refined_FixedCMR}"
ADAPTIVE_DIR="${ADAPTIVE_DIR:-$REPO_ROOT/CFTE_Refined_AdaptiveCMR}"

SAVE_SAMPLE_IMAGES="${SAVE_SAMPLE_IMAGES:-0}"
SAVE_VIDEO_PANELS="${SAVE_VIDEO_PANELS:-0}"
MAX_SAVED_IMAGES="${MAX_SAVED_IMAGES:-50}"
VIDEO_PANEL_MAX_SAMPLES="${VIDEO_PANEL_MAX_SAMPLES:-12}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

gap_final_checkpoint() {
  local project_dir="$1"
  local search_root="$project_dir/checkpoint_d${GAP}"
  local -a found=()

  if [[ -d "$search_root" ]]; then
    mapfile -t found < <(
      find "$search_root" -type f -name '00000199-checkpoint.pth.tar' -print | sort
    )
  fi

  if [[ "${#found[@]}" -eq 0 ]]; then
    echo "No gap-${GAP} epoch-199 checkpoint found under: $search_root" >&2
    echo "Either place the final checkpoint there or set the corresponding *_CKPT environment variable." >&2
    exit 1
  fi

  if [[ "${#found[@]}" -ne 1 ]]; then
    echo "Multiple gap-${GAP} epoch-199 checkpoints found under: $search_root" >&2
    printf '  %s\n' "${found[@]}" >&2
    echo "Set the corresponding *_CKPT environment variable explicitly." >&2
    exit 1
  fi

  printf '%s\n' "${found[0]}"
}

SINGLE_CKPT="${SINGLE_CKPT:-$(gap_final_checkpoint "$SINGLE_DIR")}"
NOREFINE_CKPT="${NOREFINE_CKPT:-$(gap_final_checkpoint "$NOREFINE_DIR")}"
FIXEDCMR_CKPT="${FIXEDCMR_CKPT:-$(gap_final_checkpoint "$FIXEDCMR_DIR")}"
ADAPTIVE_CKPT="${ADAPTIVE_CKPT:-$(gap_final_checkpoint "$ADAPTIVE_DIR")}"

CONFIG_NAME="vox-256_gap${GAP}.yaml"

echo "Gap:           $GAP"
echo "Dataset:       $ROOT_DIR"
echo "Output:        $OUT_ROOT"
echo "CUDA device:   $DEVICE"
echo "Single CKPT:   $SINGLE_CKPT"
echo "NoRefine CKPT: $NOREFINE_CKPT"
echo "Fixed CKPT:    $FIXEDCMR_CKPT"
echo "Adaptive CKPT: $ADAPTIVE_CKPT"

"$PYTHON_BIN" "$SCRIPT_DIR/preflight_check.py" \
  --single_dir "$SINGLE_DIR" \
  --single_checkpoint "$SINGLE_CKPT" \
  --norefine_dir "$NOREFINE_DIR" \
  --norefine_checkpoint "$NOREFINE_CKPT" \
  --fixedcmr_dir "$FIXEDCMR_DIR" \
  --fixedcmr_checkpoint "$FIXEDCMR_CKPT" \
  --adaptive_dir "$ADAPTIVE_DIR" \
  --adaptive_checkpoint "$ADAPTIVE_CKPT"

TABLE_DIR="$OUT_ROOT/tables"
INDEX_CSV="$TABLE_DIR/test_triplets_gap${GAP}_stride${STRIDE}.csv"
mkdir -p "$TABLE_DIR"

"$PYTHON_BIN" "$SCRIPT_DIR/make_test_triplets_gap3_stride3.py" \
  --root_dir "$ROOT_DIR" \
  --gap "$GAP" \
  --stride "$STRIDE" \
  --output_csv "$INDEX_CSV"

run_method() {
  local script_name="$1"
  local project_dir="$2"
  local checkpoint="$3"
  local output_csv="$4"
  local image_dir="$5"
  local panel_dir="$6"
  shift 6

  local args=(
    "$PYTHON_BIN" "$SCRIPT_DIR/$script_name"
    --project_dir "$project_dir"
    --config "$project_dir/config/$CONFIG_NAME"
    --checkpoint "$checkpoint"
    --root_dir "$ROOT_DIR"
    --test_csv "$INDEX_CSV"
    --output_csv "$output_csv"
    --device "$DEVICE"
    --require_lpips
    "$@"
  )

  if [[ "$SAVE_SAMPLE_IMAGES" == "1" ]]; then
    args+=(--save_images "$image_dir" --max_saved_images "$MAX_SAVED_IMAGES")
  fi

  if [[ "$SAVE_VIDEO_PANELS" == "1" ]]; then
    args+=(
      --save_video_panels "$panel_dir"
      --video_panel_max_samples "$VIDEO_PANEL_MAX_SAMPLES"
    )
  fi

  "${args[@]}"
}

run_method test_single_cfte.py \
  "$SINGLE_DIR" "$SINGLE_CKPT" \
  "$TABLE_DIR/results_single_cfte.csv" \
  "$OUT_ROOT/predictions/single_cfte" "$OUT_ROOT/video_panels"

run_method test_norefine_twoframe.py \
  "$NOREFINE_DIR" "$NOREFINE_CKPT" \
  "$TABLE_DIR/results_norefine.csv" \
  "$OUT_ROOT/predictions/norefine" "$OUT_ROOT/video_panels"

run_method test_fixedcmr_twoframe.py \
  "$FIXEDCMR_DIR" "$FIXEDCMR_CKPT" \
  "$TABLE_DIR/results_fixedcmr.csv" \
  "$OUT_ROOT/predictions/fixedcmr" "$OUT_ROOT/video_panels"

run_method test_adaptivecmr_twoframe.py \
  "$ADAPTIVE_DIR" "$ADAPTIVE_CKPT" \
  "$TABLE_DIR/results_adaptivecmr.csv" \
  "$OUT_ROOT/predictions/adaptivecmr" "$OUT_ROOT/video_panels"

"$PYTHON_BIN" "$SCRIPT_DIR/merge_results.py" \
  --inputs \
    "$TABLE_DIR/results_single_cfte.csv" \
    "$TABLE_DIR/results_norefine.csv" \
    "$TABLE_DIR/results_fixedcmr.csv" \
    "$TABLE_DIR/results_adaptivecmr.csv" \
  --output_dir "$TABLE_DIR"

"$PYTHON_BIN" "$SCRIPT_DIR/4Method_Analysis_with_BPP_Superposed.py" \
  --single "$TABLE_DIR/results_single_cfte.csv" \
  --norefine "$TABLE_DIR/results_norefine.csv" \
  --fixedcmr "$TABLE_DIR/results_fixedcmr.csv" \
  --adaptive "$TABLE_DIR/results_adaptivecmr.csv" \
  --outdir "$OUT_ROOT/four_method_summary"

echo "Gap-${GAP} benchmark completed: $OUT_ROOT"
