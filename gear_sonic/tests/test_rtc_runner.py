"""Exercise the actual VLA main loop with delayed RPC and controller fakes."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import queue
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model
from gear_sonic.utils.inference import gr00t_adapter, rtc
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import HEADER_SIZE


class TestRTCRunner(unittest.TestCase):
    def run_loop(self, enabled):
        path = Path(__file__).parents[1] / "scripts/run_vla_inference.py"
        spec = importlib.util.spec_from_file_location("_rtc_test_runner", path)
        runner = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = runner
        spec.loader.exec_module(runner)
        robot = instantiate_g1_robot_model(waist_location="lower_and_upper_body")
        clock = SimpleNamespace(value=100.0)
        clock.monotonic = lambda: clock.value
        clock.sleep = lambda seconds: setattr(clock, "value", clock.value + seconds)
        sent, calls, pending, worker_args = [], [], [], []

        def config(keys, delta):
            return SimpleNamespace(modality_keys=keys, delta_indices=delta)

        dims = dict(
            gr00t_adapter.JOINT_DIMS,
            left_wrist_eef_9d=9,
            right_wrist_eef_9d=9,
            navigate_command=3,
            base_height_command=1,
        )
        configs = {
            "video": config(["ego_view"], [-20, 0]),
            "state": config(list(gr00t_adapter.JOINT_DIMS) + ["left_wrist_eef_9d", "right_wrist_eef_9d"], [0]),
            "language": config(["annotation.human.task_description"], [0]),
            "action": config(list(dims), list(range(40))),
        }

        class Client:
            def __init__(self, **kwargs):
                pass

            def ping(self):
                return True

            def get_modality_config(self):
                return configs

            def call_endpoint(self, endpoint, **kwargs):
                assert endpoint == "get_capabilities"
                return {"rtc": {"version": 1}}

            def get_action(self, observation, options=None):
                assert observation["video"]["ego_view"].shape == (1, 2, 8, 8, 3)
                actions = {key: np.zeros((1, 40, dim), np.float32) for key, dim in dims.items()}
                info = {}
                if options is not None:
                    request = options["rtc"]
                    assert set(request["prefix_actions"]) == set(dims)
                    frozen = request["frozen_steps"]
                    for key in actions:
                        actions[key][:, :frozen] = request["prefix_actions"][key][:, :frozen]
                    info = {
                        "rtc": {
                            "version": 1,
                            "frozen_steps": frozen,
                            "overlap_steps": next(iter(request["prefix_actions"].values())).shape[1],
                        }
                    }
                calls.append(options)
                return actions, info

        class Camera:
            def __init__(self, **kwargs):
                self.count = 0

            def read(self):
                if pending and clock.value >= pending[0][1]:
                    request, _ = pending.pop()
                    observation = worker_args[4]()
                    if isinstance(request, rtc.RTCRequest):
                        if observation is not None:
                            action, info = worker_args[5](observation, request)
                            worker_args[1].put((request, action, info))
                        else:
                            worker_args[1].put((request, None, {"error": "waiting"}))
                    elif observation is not None:
                        worker_args[1].put((worker_args[5](observation), clock.value - 0.12))
                self.count += 1
                return {
                    "images": {"ego_view": np.zeros((8, 8, 3), np.uint8)},
                    "timestamps": {"ego_view": self.count},
                }

            def close(self):
                pass

        class State:
            def __init__(self, **kwargs):
                pass

            def get_msg(self):
                return {
                    "body_q": robot.get_body_actuated_joints(robot.default_body_pose),
                    "left_hand_q": np.zeros(7),
                    "right_hand_q": np.zeros(7),
                    "base_quat": np.array([1, 0, 0, 0]),
                }

            def close(self):
                pass

        class Keyboard:
            def __init__(self, **kwargs):
                self.count = 0
                self.changed = False

            def read_msg(self):
                self.count += 1
                if self.count > 180:
                    raise KeyboardInterrupt
                if self.count in (1, 2, 3):
                    return {1: "k", 2: "i", 3: "p"}[self.count]
                if enabled and not self.changed and pending and pending[0][0].prefix is not None:
                    self.changed = True
                    return "prompt:raise your arms again"
                return None

            def close(self):
                pass

        class Socket:
            def bind(self, address):
                pass

            def send(self, message):
                sent.append(message)

            def close(self):
                pass

        class Context:
            def socket(self, kind):
                return Socket()

            def term(self):
                pass

        class Worker:
            def __init__(self, target, args, **kwargs):
                worker_args[:] = args

            def start(self):
                pass

            def join(self, **kwargs):
                pass

        original_queue = queue.Queue

        class WorkQueue(original_queue):
            def put_nowait(self, item):
                if item is None or isinstance(item, rtc.RTCRequest):
                    if pending:
                        if item is None:
                            raise queue.Full
                        raise AssertionError("Concurrent RTC policy requests")
                    pending.append((item, clock.value + 0.12))
                else:
                    super().put_nowait(item)

        output = io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch("gr00t.policy.server_client.PolicyClient", Client))
            for target, attr, value in (
                (runner, "instantiate_g1_robot_model", lambda **kw: robot),
                (runner, "ComposedCameraClientSensor", Camera),
                (runner, "ZMQStateSubscriber", State),
                (runner, "ZMQKeyboardSubscriber", Keyboard),
                (runner, "time", clock),
                (gr00t_adapter, "time", clock),
                (rtc, "time", clock),
                (runner.zmq, "Context", Context),
                (runner.threading, "Thread", Worker),
                (runner.queue, "Queue", WorkQueue),
            ):
                stack.enter_context(patch.object(target, attr, value))
            with contextlib.redirect_stdout(output):
                runner.main(runner.InferenceConfig(rtc=enabled))
        self.assertTrue(calls)
        self.assertTrue(any(msg.startswith(b"planner") for msg in sent))
        self.assertFalse(any(msg.startswith(b"pose") for msg in sent))
        for message in sent:
            if message.startswith(b"planner"):
                header = json.loads(message[7 : 7 + HEADER_SIZE].rstrip(b"\0"))
                self.assertEqual(
                    next(f for f in header["fields"] if f["name"] == "upper_body_position")["shape"], [17]
                )
        return calls, output.getvalue()

    def test_rtc_runs_multiple_conditioned_chunks_and_rejects_old_prompt(self):
        calls, output = self.run_loop(True)
        self.assertGreater(sum(call is not None for call in calls), 2)
        self.assertIn("RTC chunk:", output)
        self.assertIn("earlier prompt/control generation", output)

    def test_previous_non_rtc_loop_still_runs(self):
        calls, output = self.run_loop(False)
        self.assertTrue(all(call is None for call in calls))
        self.assertIn("New action chunk", output)


if __name__ == "__main__":
    unittest.main()
