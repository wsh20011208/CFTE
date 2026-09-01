#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${CFTE_WORK_ROOT:-/home/featurize/work}"

QP="${QP:-32}"
VTM_WORKERS="${VTM_WORKERS:-14}"

echo "Running PER-VIDEO physical codec pipeline"
echo "QP=${QP}"
echo "VTM_WORKERS=${VTM_WORKERS}"
echo "Each test video will produce all four MP4s before the next video starts."
echo ""

python -u "$SCRIPT_DIR/run_per_video_codec_gop4.py" \
    --qp "${QP}" \
    --vtm-workers "${VTM_WORKERS}"
