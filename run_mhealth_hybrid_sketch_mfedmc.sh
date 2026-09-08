#!/usr/bin/env bash
set -euo pipefail
exec "$(cd "$(dirname "$0")" && pwd)/HybridSketchMFedMC/run_mhealth.sh" "$@"
