"""Sweep the bath drive omega at a fixed central-spin drive Omega = 1.

Plots the bath and global QFI for ``J`` versus omega; see ``sweep_helpers``.
"""

from pathlib import Path
import sys
T = [1.8, 2, 2.2, 2.4, 2.6]

try:
    from CRB.sweeps.sweep_helpers import BATH_DRIVE, DriveSweepConfig, main
except ModuleNotFoundError:  # Allow: python CRB/sweeps/omega_bath_sweep.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from CRB.sweeps.sweep_helpers import BATH_DRIVE, DriveSweepConfig, main

# DEFAULTS = DriveSweepConfig(fixed_drive=4, sweep_min=0.0, sweep_max=5.0)


if __name__ == "__main__":
    for t in T:
        DEFAULTS = DriveSweepConfig(fixed_drive=2.0, interrogation_time=t, sweep_min=0.0, sweep_max=5.0)
        main(BATH_DRIVE, __file__, DEFAULTS, description=__doc__)

    # main(BATH_DRIVE, __file__, DEFAULTS, description=__doc__)
