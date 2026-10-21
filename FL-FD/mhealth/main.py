"""FL-FD (Qi et al., Information Fusion 2023) on mHealth, next to FLASH / Hybrid v2 / MFedMC v2.

Runs FL-FD/uci_har/main.py unchanged (FL-FD algorithm from FL-FD/fixed/, GAF inputs, byte accounting,
logging); only two of its module globals are swapped:
  * load_uci_har_mm -> HybridSketchMFedMC_v2's load_mhealth_mm (same signature and return), so the
    10-client Dirichlet(0.1) split, labels and test set are the ones Hybrid v2 uses on mHealth
    (Acc 177 / Gyro 118 / Mag 118 / ECG 43 feature vectors, 12 classes)
  * HERE -> FL-FD/mhealth/, so results/ and result_updated/ are written here, not under uci_har/
With 4 modalities the input is 4 x 70 x 70 and FLFDNet's first conv takes 4 channels (handled there).
"""

import importlib.util
import os
import sys

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

_spec = importlib.util.spec_from_file_location("flfd_uci_har_main", os.path.join(ROOT, "FL-FD", "uci_har", "main.py"))
flfd_uci = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(flfd_uci)  # also puts HybridSketchMFedMC_v2 on sys.path

from dataset import load_mhealth_mm  # HybridSketchMFedMC_v2

flfd_uci.load_uci_har_mm = load_mhealth_mm
flfd_uci.HERE = HERE


if __name__ == "__main__":
    flfd_uci.main()
