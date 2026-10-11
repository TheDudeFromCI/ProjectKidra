#!/bin/bash

set -e

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

[ ! -d .venv ] && python -m venv .venv
[ -f .venv/bin/activate ] && source .venv/bin/activate
