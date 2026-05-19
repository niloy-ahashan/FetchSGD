#!/usr/bin/env python3
"""
ActionSense ``S00_preprocessed.hdf5`` → FedMultiModal ``data.npz``.

Thin wrapper around :mod:`prepare_actionsense_allstreams_mm` with defaults:
  • ``--hdf5_path`` → ``<repo>/MFedMC/ActionSense/data/S00_preprocessed.hdf5``
  • ``--out_dir``   → ``<repo>/datasets/actionsense_s00_mm``

Pass extra CLI flags as for ``prepare_actionsense_allstreams_mm.py`` (they override
or extend the defaults).

Usage
-----
  python prepare_actionsense_s00_mm.py
  python prepare_actionsense_s00_mm.py --test_size 0.2 --out_dir /tmp/out
"""

from __future__ import annotations

import os
import sys

from prepare_actionsense_allstreams_mm import main as prepare_main


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main(argv: list[str] | None = None) -> None:
    root = _repo_root()
    default_hdf5 = os.path.join(
        root, "MFedMC", "ActionSense", "data", "S00_preprocessed.hdf5"
    )
    default_out = os.path.join(root, "datasets", "actionsense_s00_mm")
    argv = list(sys.argv[1:] if argv is None else argv)

    def _take(flag: str) -> bool:
        return flag in argv

    if not _take("--hdf5_path"):
        argv = ["--hdf5_path", default_hdf5] + argv
    if not _take("--out_dir"):
        argv = ["--out_dir", default_out] + argv

    prepare_main(argv)


if __name__ == "__main__":
    main()
