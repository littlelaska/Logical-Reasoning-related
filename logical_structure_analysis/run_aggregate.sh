#!/usr/bin/env bash
set -euo pipefail

DEMO_METHOD="embed"   # logical | embed | bm25 | random

ROOT="${DEMO_METHOD}_results"
OUT_PREFIX="statistic_files/${DEMO_METHOD}_summary"

python3 aggregate_results.py \
  --root "${ROOT}" \
  --out_prefix "${OUT_PREFIX}"
