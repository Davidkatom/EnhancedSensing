"""Sweep the central-spin drive Omega at a fixed bath drive omega = 1.

Plots the bath and global QFI for ``J`` versus Omega; see ``sweep_helpers``.
"""

from pathlib import Path
import sys

T = [1.8, 2, 2.2, 2.4, 2.6]

try:
    from CRB.sweeps.sweep_helpers import CENTRAL_DRIVE, DriveSweepConfig, main
except ModuleNotFoundError:  # Allow: python CRB/sweeps/Omega_sweep.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from CRB.sweeps.sweep_helpers import CENTRAL_DRIVE, DriveSweepConfig, main

# DEFAULTS = DriveSweepConfig(fixed_drive=1.0,interrogation_time=2, sweep_min=0.0, sweep_max=5.0)


if __name__ == "__main__":
    for t in T:
        DEFAULTS = DriveSweepConfig(fixed_drive=1.0, interrogation_time=t, sweep_min=0.0, sweep_max=5.0)
        main(CENTRAL_DRIVE, __file__, DEFAULTS, description=__doc__)
