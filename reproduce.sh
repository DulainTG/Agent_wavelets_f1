#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$ROOT_DIR/outputs"
export PYTHONPATH="$ROOT_DIR/src"

python -m scattering_repro.workflow --outputs "$OUTPUT_DIR"
