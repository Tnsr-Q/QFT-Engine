#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p results

echo "Running verification suite with 2 hour safety timeout..."
timeout 7200 pytest tests/ -v --junitxml=results/results.xml

echo "Results written to results/results.xml"
