#!/bin/bash
# Installs dependencies, then type-checks and builds a distribution.

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." || exit 1
source "./scripts/common.sh"
set -e

pip install -r requirements.txt
mypy
pip install build
python -m build
