"""Simulation-only direct mapping reference for recorded input comparisons."""

import numpy as np

from gear_sonic.utils.teleop.arms_only import ArmsOnlyTargets


class DirectDemoTargets(ArmsOnlyTargets):
    """Fixed calibration, no workspace/IK projection; tracking holds still apply.

    This is a comparison reference, not a byte-for-byte copy of arms-only-v1.
    In particular it does not reproduce v1's workspace-triggered freeze.
    """

    MAX_EXCURSION_M = np.inf

    def update(self, sample, now=None, measured_q=None):
        if self.target is None:
            return super().update(sample, now, measured_q)
        origin = self.controller_origin.copy()
        rotation = self.controller_rotation
        result = super().update(sample, now, measured_q)
        # Never change the calibrated reference due to floating point residuals.
        self.controller_origin = origin
        self.controller_rotation = rotation
        return result
