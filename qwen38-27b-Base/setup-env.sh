#!/usr/bin/env bash
# Create the project environment; never update an existing environment implicitly.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BIN="${CONDA_BIN:-conda}"
if [[ "${1:-}" == --help ]]; then
  printf 'Usage: setup-env.sh [--dry-run]\nCreates the environment defined in this project. CONDA_BIN can override conda.\n'
  exit 0
fi
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --dry-run ) ]]; then
  printf 'Usage: setup-env.sh [--dry-run]\n' >&2
  exit 2
fi
cd -- "$SCRIPT_DIR"
exec "$CONDA_BIN" env create --file "$SCRIPT_DIR/environment.yml" "$@"
