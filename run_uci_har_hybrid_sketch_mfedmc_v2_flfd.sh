#!/usr/bin/env bash
set -euo pipefail
exec "$(cd "$(dirname "$0")" && pwd)/HybridSketchMFedMC_v2_flfd/run_uci_har.sh" "$@"
