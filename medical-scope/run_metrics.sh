#!/bin/bash
# Compute SCOPE-Medical run metrics from output folders (*_convo.txt + *_results.jsonl).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-/home/hyang/mediQ}"
PYTHON="${PYTHON:-/home/hyang/miniconda3/envs/scope/bin/python}"
METRICS_SCRIPT="${METRICS_SCRIPT:-$REPO_DIR/scripts/compute_scope_medical_metrics.py}"

OUTPUT_DIR="/home/hyang/mediQ/medical-scope/output/med-feature-reward"
MAX_QUESTIONS="${MAX_QUESTIONS:-10}"
TEMP="${TEMP:-0.8}"
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-5.0}"

# Pass a single run folder, or default to all subfolders under OUTPUT_DIR.
INPUT="${1:-$OUTPUT_DIR}"

"$PYTHON" "$METRICS_SCRIPT" \
  "$INPUT" \
  --max-questions "$MAX_QUESTIONS" \
  --temperature "$TEMP" \
  --confidence-threshold "$CONFIDENCE_THRESHOLD"
