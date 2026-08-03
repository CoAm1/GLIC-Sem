#!/usr/bin/env bash
set -euo pipefail

# Required inputs are environment variables so source control never embeds
# machine credentials, model files, datasets, or generated results.
: "${TEACHER_DIR:?set TEACHER_DIR to the 60-frame SAM1/OpenCLIP teacher directory}"
: "${PCA_BASIS_DIR:?set PCA_BASIS_DIR to the fixed PCA128 basis directory}"
: "${QUERY_FEATURES:?set QUERY_FEATURES to pca_text_queries_v2 JSON}"
: "${PROMPT_GROUPS:?set PROMPT_GROUPS to frozen MCD prompt-group JSON}"
: "${REFERENCE_DIR:?set REFERENCE_DIR to the sparse MCD projection directory}"
: "${ALIGNMENT_CSV:?set ALIGNMENT_CSV to frame/label-scan alignment CSV}"
: "${OUTPUT_DIR:?set OUTPUT_DIR to a new experiment directory}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

args=(
  "${PROJECT_ROOT}/scripts/semantic/evaluate_semantic_error_decomposition.py"
  --teacher-dir "${TEACHER_DIR}"
  --basis-dir "${PCA_BASIS_DIR}"
  --query-features "${QUERY_FEATURES}"
  --prompt-groups "${PROMPT_GROUPS}"
  --reference-dir "${REFERENCE_DIR}"
  --alignment-csv "${ALIGNMENT_CSV}"
  --output-dir "${OUTPUT_DIR}"
  --start-frame 1800
  --end-frame 1859
  --keyframe-period 5
  --keyframe-offset 4
  --min-reference-confidence 0.35
  --image-height 480
  --image-width 640
  --aggregation max
  --bootstrap-replicates "${BOOTSTRAP_REPLICATES:-2000}"
  --bootstrap-block-size "${BOOTSTRAP_BLOCK_SIZE:-3}"
  --bootstrap-seed "${BOOTSTRAP_SEED:-3407}"
)

if [[ -n "${STAGE3_SCORE_ROOT:-}" || -n "${STAGE3_ALPHA_ROOT:-}" ]]; then
  : "${STAGE3_SCORE_ROOT:?set both STAGE3_SCORE_ROOT and STAGE3_ALPHA_ROOT}"
  : "${STAGE3_ALPHA_ROOT:?set both STAGE3_SCORE_ROOT and STAGE3_ALPHA_ROOT}"
  args+=(
    --stage3-score-root "${STAGE3_SCORE_ROOT}"
    --stage3-alpha-root "${STAGE3_ALPHA_ROOT}"
  )
fi

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  args+=(--preflight-only)
fi

exec "${PYTHON_BIN}" "${args[@]}"
