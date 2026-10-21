"""Controller gestures shared by live teleoperation and demonstration replay."""

import numpy as np


def generate_finger_data(hand, trigger, grip, profile="legacy"):
    if profile not in ("legacy", "rtc", "enhanced"):
        raise ValueError(f"Unknown gripper profile: {profile}")
    fingertips = np.zeros((25, 4, 4))
    fingertips[4, 0, 3] = 1.0
    if profile == "enhanced":
        # Select index / ring / middle closure from trigger/grip combinations.
        if trigger > 0.5 and grip > 0.5:
            fingertips[14, 0, 3] = 1.0
        elif trigger > 0.5:
            fingertips[9, 0, 3] = 1.0
        elif grip > 0.5:
            fingertips[19, 0, 3] = 1.0
    elif trigger > 0.5:
        fingertips[14, 0, 3] = 1.0
    return fingertips
