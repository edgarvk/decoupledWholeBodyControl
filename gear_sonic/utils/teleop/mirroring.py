"""Shared mirror transforms. Raw headset/controller packets are never modified."""

import numpy as np
from scipy.spatial.transform import Rotation as R

ARM_GLOBAL_POSE_PAIRS = ((13, 14), (16, 17), (18, 19), (20, 21), (22, 23))


def mirror_controller_targets(positions, rotations):
    """Exchange controllers and reflect robot Y, preserving forward/up motion.

    Called before anchoring to measured wrists, so asymmetric initial robot poses
    do not jump when mirror mode starts. Rotations use S R S, never a reflection
    matrix passed directly as a quaternion.
    """
    reflection = np.diag([1.0, -1.0, 1.0])
    return (
        np.asarray(positions)[[1, 0]] @ reflection,
        R.from_matrix(reflection @ rotations.as_matrix()[[1, 0]] @ reflection),
    )


def route_hand_inputs(lt, lg, rt, rg, mirror=False):
    return (rt, rg, lt, lg) if mirror else (lt, lg, rt, rg)


def mirror_body_rotations(global_rots):
    """Preserve the intern's torso-relative legacy SMPL arm reflection."""
    reflection = np.diag([-1.0, 1.0, 1.0])
    root = global_rots[0]
    relative = [root.inv() * rotation for rotation in global_rots]
    result = relative.copy()
    for left, right in ARM_GLOBAL_POSE_PAIRS:
        result[left] = R.from_matrix(reflection @ relative[right].as_matrix() @ reflection)
        result[right] = R.from_matrix(reflection @ relative[left].as_matrix() @ reflection)
    return [root * rotation for rotation in result]


def mirror_vr_wrist_targets(vr_pose):
    """Intern's legacy VR convention reflects X; direct arms-only uses robot Y."""
    original = np.asarray(vr_pose)
    if original.shape != (3, 7):
        raise ValueError(f"Expected VR targets with shape (3, 7), got {original.shape}")
    reflection = np.diag([-1.0, 1.0, 1.0])
    result = original.copy()
    result[:2, :3] = original[[1, 0], :3] @ reflection
    rotations = R.from_quat(original[[1, 0], 3:], scalar_first=True)
    result[:2, 3:] = R.from_matrix(reflection @ rotations.as_matrix() @ reflection).as_quat(scalar_first=True)
    return result


def mirror_pose_message_outputs(data):
    output = dict(data)
    pose = np.concatenate(
        [
            np.asarray(data["vr_position"]).reshape(3, 3),
            np.asarray(data["vr_orientation"]).reshape(3, 4),
        ],
        axis=1,
    )
    pose = mirror_vr_wrist_targets(pose)
    output["vr_position"] = pose[:, :3].reshape(-1)
    output["vr_orientation"] = pose[:, 3:].reshape(-1)
    for left, right in (("left_trigger", "right_trigger"), ("left_grip", "right_grip")):
        output[left], output[right] = data[right], data[left]
    return output


def mirror_flat_vr_targets(position, orientation):
    if position is None or orientation is None:
        return position, orientation
    pose = np.concatenate([np.asarray(position).reshape(3, 3), np.asarray(orientation).reshape(3, 4)], axis=1)
    pose = mirror_vr_wrist_targets(pose)
    return pose[:, :3].reshape(-1).tolist(), pose[:, 3:].reshape(-1).tolist()


def swap_lr_flat_targets(values, stride):
    if values is None:
        return None
    result = np.asarray(values).reshape(-1).copy()
    result[:stride], result[stride : 2 * stride] = result[stride : 2 * stride].copy(), result[:stride].copy()
    return result.tolist()
