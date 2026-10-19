"""G1 kinematic target projection. Its joint solution never commands motors.

The TELEOP policy only observes the projected wrists. Elbow-down is a preference
of the feasibility calculation, not a constraint on SONIC's resulting posture.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


class G1ArmTargetFilter:
    ITERATIONS = 6
    JOINT_MARGIN_RAD = 0.08
    MIN_ELBOW_BEND_RAD = 0.25
    MAX_TARGET_STEP_M = 0.02
    MAX_TARGET_STEP_RAD = np.deg2rad(5)

    def __init__(self, robot_model=None, max_gravity_nm=8.0):
        import pinocchio as pin

        from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
        from gear_sonic.utils.teleop.vis.vr3pt_pose_visualizer import G1_KEY_FRAME_OFFSETS

        if not np.isfinite(max_gravity_nm) or max_gravity_nm <= 0:
            raise ValueError("Estimated gravity budget must be positive and finite")
        self.pin = pin
        self.robot = robot_model if robot_model is not None else instantiate_g1_robot_model()
        self.model = self.robot.pinocchio_wrapper.model
        # Separate data buffers keep monitoring from corrupting solver state.
        self.data = self.model.createData()
        self.max_gravity_nm = max_gravity_nm
        self.body_indices = np.asarray(self.robot.get_body_actuated_joint_indices())
        self.arms = []
        for side in ("left", "right"):
            names = [
                f"{side}_{joint}_joint"
                for joint in (
                    "shoulder_pitch",
                    "shoulder_roll",
                    "shoulder_yaw",
                    "elbow",
                    "wrist_roll",
                    "wrist_pitch",
                    "wrist_yaw",
                )
            ]
            indices = np.array([self.robot.dof_index(name) for name in names])
            lower = self.robot.lower_joint_limits[indices] + self.JOINT_MARGIN_RAD
            upper = self.robot.upper_joint_limits[indices] - self.JOINT_MARGIN_RAD
            lower[3] = max(lower[3], self.MIN_ELBOW_BEND_RAD)
            preferred = np.array([0.15, 0.30 if side == "left" else -0.30, 0.0, 0.65, 0.0, 0.0, 0.0])
            self.arms.append(
                {
                    "indices": indices,
                    "lower": lower,
                    "upper": upper,
                    "preferred": preferred,
                    "wrist": self.model.getFrameId(f"{side}_wrist_yaw_link"),
                    "elbow": self.model.getFrameId(f"{side}_elbow_link"),
                    "shoulder": self.model.getFrameId(f"{side}_shoulder_pitch_link"),
                    "offset": G1_KEY_FRAME_OFFSETS[f"{side}_wrist"].copy(),
                }
            )
        self.q = None
        self.status = ["not calibrated", "not calibrated"]
        self.solution_diagnostics = []

    def reset(self, measured_q, target):
        # Reanchor preserves outgoing wrists and the existing projection state.
        # Only initial calibration seeds the solver from actual measured joints.
        if self.q is None:
            self.q = self.robot.get_configuration_from_actuated_joints(
                body_actuated_joint_values=np.asarray(measured_q)
            )
        self.status = ["calibrated", "calibrated"]

    def _frames(self, q, jacobians=False):
        if jacobians:
            self.pin.computeJointJacobians(self.model, self.data, q)
        else:
            self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)

    def _pose(self, arm):
        placement = self.data.oMf[arm["wrist"]]
        return np.r_[
            placement.translation + placement.rotation @ arm["offset"],
            R.from_matrix(placement.rotation).as_quat(scalar_first=True),
        ]

    def _jacobian(self, arm):
        jac = self.pin.getFrameJacobian(self.model, self.data, arm["wrist"], self.pin.LOCAL_WORLD_ALIGNED)[
            :, arm["indices"]
        ].copy()
        offset = self.data.oMf[arm["wrist"]].rotation @ arm["offset"]
        # Velocity of the offset manipulation point: v + omega cross offset.
        jac[:3] += np.cross(jac[3:].T, offset).T
        return jac

    def _gravity(self, q):
        # Unloaded URDF gravity estimate: neither payload nor dynamic/contact torque.
        return self.pin.computeGeneralizedGravity(self.model, self.data, q).copy()

    def _arm_metrics(self, q, arm):
        self._frames(q, jacobians=True)
        values = q[arm["indices"]]
        margin = np.minimum(values - arm["lower"], arm["upper"] - values)
        joint_margin = np.minimum(
            values - self.robot.lower_joint_limits[arm["indices"]],
            self.robot.upper_joint_limits[arm["indices"]] - values,
        )
        elbow_above_shoulder = (
            self.data.oMf[arm["elbow"]].translation[2] - self.data.oMf[arm["shoulder"]].translation[2]
        )
        singular_values = np.linalg.svd(self._jacobian(arm)[:3], compute_uv=False)
        return {
            "min_comfort_margin_rad": float(np.min(margin)),
            "min_joint_limit_margin_rad": float(np.min(joint_margin)),
            "elbow_above_shoulder_m": float(elbow_above_shoulder),
            "position_jacobian_sigma_min": float(singular_values[-1]),
            "gravity_nm": self._gravity(q)[arm["indices"]].tolist(),
        }

    def project(self, requested, previous):
        if self.q is None:
            raise RuntimeError("Measured-joint calibration must precede target projection")
        output = previous.copy()
        self.status = []
        self.solution_diagnostics = []
        for row, arm in enumerate(self.arms):
            indices = arm["indices"]
            start = self.q.copy()
            start_metrics = self._arm_metrics(start, arm)
            same_position = np.linalg.norm(requested[row, :3] - previous[row, :3]) < 1e-10
            same_rotation = (
                R.from_quat(requested[row, 3:], scalar_first=True)
                * R.from_quat(previous[row, 3:], scalar_first=True).inv()
            ).magnitude() < 1e-10
            if same_position and same_rotation:
                # Initial calibration and head-only motion must not initiate a
                # posture change merely because the IK preference is different.
                self.status.append("tracking")
                self.solution_diagnostics.append(start_metrics)
                continue
            q = start.copy()
            q[indices] = np.clip(q[indices], arm["lower"], arm["upper"])
            desired_rot = R.from_quat(requested[row, 3:], scalar_first=True)
            # Six bounded DLS iterations; translation has priority over orientation.
            for _ in range(self.ITERATIONS):
                self._frames(q, jacobians=True)
                pose = self._pose(arm)
                error = np.r_[
                    requested[row, :3] - pose[:3],
                    0.10 * (desired_rot * R.from_quat(pose[3:], scalar_first=True).inv()).as_rotvec(),
                ]
                jac = self._jacobian(arm)
                jac[3:] *= 0.10
                inverse = jac.T @ np.linalg.solve(jac @ jac.T + 0.025**2 * np.eye(6), np.eye(6))
                preference = 0.08 * (arm["preferred"] - q[indices])
                # Lower the elbow in the redundant direction when it is elevated.
                height = self.data.oMf[arm["elbow"]].translation[2] - self.data.oMf[arm["shoulder"]].translation[2]
                if height > -0.08:
                    elbow_jac = self.pin.getFrameJacobian(
                        self.model, self.data, arm["elbow"], self.pin.LOCAL_WORLD_ALIGNED
                    )[2, indices]
                    preference -= 0.3 * (height + 0.08) * elbow_jac
                delta = inverse @ error + (np.eye(7) - inverse @ jac) @ preference
                delta *= min(1.0, 0.12 / max(np.max(np.abs(delta)), 1e-12))
                # Scale the whole step, preserving its descent direction. Independent
                # clipping of joint increments can invert the Cartesian motion.
                current_cost = float(error @ error)
                for line_search in range(6):
                    trial = q.copy()
                    trial[indices] = np.clip(q[indices] + delta * 0.5**line_search, arm["lower"], arm["upper"])
                    self._frames(trial)
                    trial_pose = self._pose(arm)
                    trial_error = np.r_[
                        requested[row, :3] - trial_pose[:3],
                        0.10 * (desired_rot * R.from_quat(trial_pose[3:], scalar_first=True).inv()).as_rotvec(),
                    ]
                    if float(trial_error @ trial_error) <= current_cost + 1e-8:
                        q = trial
                        break
                else:
                    break
            # Bound output motion, including when initial posture lies outside the
            # comfort envelope. No snap to the preferred/default arm posture.
            alpha = 1.0
            accepted = None
            reason = "kinematic projection"
            baseline_load = np.abs(np.asarray(start_metrics["gravity_nm"]))
            for _ in range(10):
                trial = start.copy()
                trial[indices] += alpha * (q[indices] - start[indices])
                self._frames(trial)
                pose = self._pose(arm)
                step_m = np.linalg.norm(pose[:3] - previous[row, :3])
                step_r = (
                    R.from_quat(pose[3:], scalar_first=True)
                    * R.from_quat(previous[row, 3:], scalar_first=True).inv()
                ).magnitude()
                metrics = self._arm_metrics(trial, arm)
                load = np.abs(np.asarray(metrics["gravity_nm"]))
                # Grandfather a starting posture only while improving it; calibration
                # remains continuous even if actual arms start outside comfort bounds.
                load_ok = np.all(load <= np.maximum(self.max_gravity_nm, baseline_load) + 1e-6)
                elbow_ok = (
                    metrics["elbow_above_shoulder_m"] <= max(0.02, start_metrics["elbow_above_shoulder_m"]) + 1e-6
                )
                if (
                    step_m <= self.MAX_TARGET_STEP_M
                    and step_r <= self.MAX_TARGET_STEP_RAD
                    and load_ok
                    and elbow_ok
                ):
                    accepted = trial, pose, metrics
                    break
                reason = (
                    "gravity budget" if not load_ok else "elbow height" if not elbow_ok else "output step limit"
                )
                alpha *= 0.5
            if accepted is not None:
                self.q, output[row], metrics = accepted
                position_error = np.linalg.norm(output[row, :3] - requested[row, :3])
                rotation_error = (desired_rot * R.from_quat(output[row, 3:], scalar_first=True).inv()).magnitude()
                if position_error > 0.005 or rotation_error > np.deg2rad(5):
                    boundary = "joint_margin" if metrics["min_comfort_margin_rad"] < 1e-3 else "pose_fit"
                    self.status.append(f"projected:{boundary}")
                else:
                    self.status.append("tracking")
            else:
                metrics = start_metrics
                self.status.append(f"limited: {reason}")
            self.solution_diagnostics.append(metrics)
        # Torso reference is untouched; only projected Cartesian targets leave here.
        return output

    def observe(self, measured_q, target):
        q = self.robot.get_configuration_from_actuated_joints(body_actuated_joint_values=measured_q)
        self._frames(q)
        actual = np.stack([self._pose(arm) for arm in self.arms])
        result = {
            "actual_wrist_pose": actual.tolist(),
            "wrist_error_m": np.linalg.norm(target[:2, :3] - actual[:, :3], axis=1).tolist(),
            "wrist_error_rad": (
                R.from_quat(target[:2, 3:], scalar_first=True)
                * R.from_quat(actual[:, 3:], scalar_first=True).inv()
            )
            .magnitude()
            .tolist(),
            "measured_arm_q": [q[arm["indices"]].tolist() for arm in self.arms],
            "measured_arm_metrics": [self._arm_metrics(q, arm) for arm in self.arms],
        }
        return result
