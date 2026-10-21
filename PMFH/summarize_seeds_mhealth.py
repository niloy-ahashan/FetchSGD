"""``summarize_seeds.py`` for the mHealth runs of ``main_mhealth_pmfh.py``.

Same columns and output as ``summarize_seeds.py``; only adds the mHealth prefixes
(PMFHfhG = IMU | ECG grouping, PMFHfhN = one encoder per modality), e.g.
  venv/bin/python PMFH/summarize_seeds_mhealth.py --prefix PMFHfhN \
    --dataset mhealth_mm_random_group --seeds 42 43 44 45 46
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import summarize_seeds

summarize_seeds.METHODS.update({
    "PMFHfhG": "PMFH (fed. head, IMU | ECG)",
    "PMFHfhN": "PMFH-N (fed. head, 4 encoders)",
})

if __name__ == "__main__":
    summarize_seeds.main()
