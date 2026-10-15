#!/usr/bin/env bash
set -euo pipefail
exec "$(cd "$(dirname "$0")" && pwd)/FLASH/torch_port/run_mhealth.sh" "$@"
