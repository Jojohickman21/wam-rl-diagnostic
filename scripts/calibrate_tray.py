"""Check that the A7 tray object slides at the friction angle the env draws for it.

    python scripts/calibrate_tray.py

For each seed the plate is tilted with joint 5 to a fraction of the drawn slide angle and held
for 2 s. Below the angle the cylinder should not move; above it, it should leave the plate.
"""
import numpy as np

from wam.checks import hold_tray_at
from wam.close_cabinet import CloseCabinetEnv

FRACTIONS = (0.8, 0.95, 1.05, 1.2)


def main():
    env = CloseCabinetEnv(tray_object=True)
    print("slide angle  " + "  ".join(f"{f:>5.2f}x" for f in FRACTIONS))
    for seed in range(10):
        moved = [hold_tray_at(env, seed, f) for f in FRACTIONS]
        cells = ["  off " if np.isinf(m) else f"{m:4.1f}mm" for m in moved]
        print(f"{np.degrees(env.tilt_max):8.1f} deg  " + "  ".join(cells))


if __name__ == "__main__":
    main()
