#!/usr/bin/env bash
set -euo pipefail
exec "$(cd "$(dirname "$0")" && pwd)/HybridSketchMFedMC_v2/run_mhealth.sh" "$@"
