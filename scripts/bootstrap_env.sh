#!/usr/bin/env bash
set -euo pipefail
python -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev]'
echo "Environment ready. Install GPU/model extras separately after checking the target CUDA/server setup."
