#!/usr/bin/env bash
set -euo pipefail

# Example: run one gap-7 method and generate the same CSV/images/video-panel
# outputs as the original evaluator.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

METHOD="${METHOD:-single}"   # single | norefine | fixedcmr | adaptive
ROOT_DIR="${ROOT_DIR:-/home/jovyan/CFTE/vox}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/../benchmark_results_gap7_stride3}"
DEVICE="${DEVICE:-0}"

SINGLE_DIR="${SINGLE_DIR:-/home/jovyan/CFTE}"
NOREFINE_DIR="${NOREFINE_DIR:-/home/jovyan/CFTE_superposition/CFTE_NoRefine}"
FIXEDCMR_DIR="${FIXEDCMR_DIR:-/home/jovyan/CFTE_superposition/CFTE_Refined_FixedCMR}"
ADAPTIVE_DIR="${ADAPTIVE_DIR:-/home/jovyan/CFTE_superposition/CFTE_Refined_AdaptiveCMR}"

gap7_final_checkpoint() {
  local project_dir="$1"
  local search_root="$project_dir/checkpoint_d7"
  local -a found=()
  if [[ -d "$search_root" ]]; then
    mapfile -t found < <(
      find "$search_root" -type f -name '00000199-checkpoint.pth.tar' -print | sort
    )
  fi
  if [[ "${#found[@]}" -eq 0 ]]; then
    echo "No gap-7 epoch-199 checkpoint found under: $search_root" >&2
    exit 1
  fi
  if [[ "${#found[@]}" -ne 1 ]]; then
    echo "Multiple gap-7 epoch-199 checkpoints found under: $search_root" >&2
    printf '  %s\n' "${found[@]}" >&2
    echo "Set the corresponding *_CKPT environment variable explicitly." >&2
    exit 1
  fi
  printf '%s\n' "${found[0]}"
}

SINGLE_CKPT="${SINGLE_CKPT:-$(gap7_final_checkpoint "$SINGLE_DIR")}"
NOREFINE_CKPT="${NOREFINE_CKPT:-$(gap7_final_checkpoint "$NOREFINE_DIR")}"
FIXEDCMR_CKPT="${FIXEDCMR_CKPT:-$(gap7_final_checkpoint "$FIXEDCMR_DIR")}"
ADAPTIVE_CKPT="${ADAPTIVE_CKPT:-$(gap7_final_checkpoint "$ADAPTIVE_DIR")}"

INDEX_CSV="$OUT_DIR/tables/test_triplets_gap7_stride3.csv"
mkdir -p "$OUT_DIR/tables" "$OUT_DIR/video_panels" "$OUT_DIR/predictions"

if [[ ! -f "$INDEX_CSV" ]]; then
  "$PYTHON_BIN" "$SCRIPT_DIR/make_test_triplets_gap7_stride3.py" \
    --root_dir "$ROOT_DIR" \
    --gap 7 \
    --stride 3 \
    --output_csv "$INDEX_CSV"
fi

run_eval() {
  local script="$1"
  local project="$2"
  local checkpoint="$3"
  local result_csv="$4"
  local image_dir="$5"

  "$PYTHON_BIN" "$SCRIPT_DIR/$script" \
    --project_dir "$project" \
    --config "$project/config/vox-256_gap7.yaml" \
    --checkpoint "$checkpoint" \
    --root_dir "$ROOT_DIR" \
    --test_csv "$INDEX_CSV" \
    --output_csv "$OUT_DIR/tables/$result_csv" \
    --save_images "$OUT_DIR/predictions/$image_dir" \
    --save_video_panels "$OUT_DIR/video_panels" \
    --video_panel_max_samples 12 \
    --device "$DEVICE"
}

case "$METHOD" in
  single)
    run_eval test_single_cfte.py "$SINGLE_DIR" "$SINGLE_CKPT" results_single_cfte.csv single_cfte
    ;;
  norefine)
    run_eval test_norefine_twoframe.py "$NOREFINE_DIR" "$NOREFINE_CKPT" results_norefine.csv norefine
    ;;
  fixedcmr)
    run_eval test_fixedcmr_twoframe.py "$FIXEDCMR_DIR" "$FIXEDCMR_CKPT" results_fixedcmr.csv fixedcmr
    ;;
  adaptive)
    run_eval test_adaptivecmr_twoframe.py "$ADAPTIVE_DIR" "$ADAPTIVE_CKPT" results_adaptivecmr.csv adaptivecmr
    ;;
  *)
    echo "Unknown METHOD=$METHOD" >&2
    exit 1
    ;;
esac
