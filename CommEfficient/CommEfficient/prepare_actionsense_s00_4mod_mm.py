#!/usr/bin/env python3
"""
Wrapper: ``S00_preprocessed.hdf5`` → ``datasets/actionsense_s00_4mod_mm``.

Delegates to :mod:`prepare_actionsense_4mod_mm`. Override paths via CLI.

Usage
-----
  python prepare_actionsense_s00_4mod_mm.py
  python prepare_actionsense_s00_4mod_mm.py --out_dir /tmp/out
"""

from __future__ import annotations

import os
import sys

from prepare_actionsense_4mod_mm import main as prepare_main


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def main(argv: list[str] | None = None) -> None:
    root = _repo_root()
    default_hdf5 = os.path.join(root, "datasets", "S00_preprocessed.hdf5")
    if not os.path.isfile(default_hdf5):
        alt = os.path.join(
            root, "MFedMC", "ActionSense", "data", "S00_preprocessed.hdf5"
        )
        if os.path.isfile(alt):
            default_hdf5 = alt
    default_out = os.path.join(root, "datasets", "actionsense_s00_4mod_mm")
    argv = list(sys.argv[1:] if argv is None else argv)

    def _has(flag: str) -> bool:
        return flag in argv

    if not _has("--hdf5_path"):
        argv = ["--hdf5_path", default_hdf5] + argv
    if not _has("--out_dir"):
        argv = ["--out_dir", default_out] + argv

    prepare_main(argv)


if __name__ == "__main__":
    main()
