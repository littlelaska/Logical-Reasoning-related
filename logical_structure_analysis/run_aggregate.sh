#!/usr/bin/env bash
set -euo pipefail

ROOT="logical_results"
OUT_PREFIX="logical_summary"

python3 aggregate_results.py \
  --root "${ROOT}" \
  --out_prefix "${OUT_PREFIX}"
