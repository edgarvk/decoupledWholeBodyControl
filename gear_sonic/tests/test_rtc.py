"""RTC control timeline tests without a robot, GPU or network server."""

import unittest

import numpy as np

from gear_sonic.utils.inference.rtc import RTCScheduler


def trajectory(start, horizon=8):
    return {"joint": np.arange(start, start + horizon, dtype=np.float32).reshape(1, horizon, 1)}


def response(request, horizon=8):
    action = trajectory(request.start_step, horizon)
    info = {}
    if request.prefix is not None:
        for key in action:
            action[key][:, : request.frozen_steps] = request.prefix[key][:, : request.frozen_steps]
        info = {
            "rtc": {
                "version": 1,
                "overlap_steps": next(iter(request.prefix.values())).shape[1],
                "frozen_steps": request.frozen_steps,
            }
        }
    return action, info


class TestRTC(unittest.TestCase):
    def scheduler(self):
        return RTCScheduler(8, rate=50, overlap_steps=4, initial_delay=0.02, safety_steps=1)

    def loaded(self):
        rtc = self.scheduler()
        request = rtc.start_request(now=0)
        self.assertIsNone(request.options)
        self.assertTrue(rtc.finish(request, trajectory(0), now=1.5))
        return rtc

    def test_initial_model_latency_does_not_skip_first_actions(self):
        rtc = self.loaded()
        plan, index = rtc.current()
        self.assertEqual(index, 0)
        self.assertEqual(plan["joint"][0, index, 0], 0)

    def test_prefix_is_remaining_absolute_actions_at_request_time(self):
        rtc = self.loaded()
        for _ in range(4):
            rtc.advance()
        request = rtc.start_request(now=2)
        np.testing.assert_array_equal(request.options["rtc"]["prefix_actions"]["joint"][0, :, 0], [4, 5, 6, 7])
        rtc.advance()
        np.testing.assert_array_equal(request.prefix["joint"][0, :, 0], [4, 5, 6, 7])

    def test_delayed_result_uses_control_step_once(self):
        rtc = self.loaded()
        for _ in range(4):
            rtc.advance()
        request = rtc.start_request(now=2)
        rtc.advance()
        rtc.advance()
        action, info = response(request)
        self.assertTrue(rtc.finish(request, action, info, now=2.04))
        plan, index = rtc.current()
        self.assertEqual(index, 2)
        self.assertEqual(plan["joint"][0, index, 0], 6)

    def test_many_chunks_neither_repeat_nor_skip_committed_steps(self):
        rtc = self.loaded()
        pending = None
        for step in range(100):
            if pending is not None and step == pending.start_step + 2:
                self.assertTrue(rtc.finish(pending, *response(pending), now=step / 50))
                pending = None
            if pending is None and rtc.can_request():
                pending = rtc.start_request(now=step / 50)
            plan, index = rtc.current()
            self.assertEqual(plan["joint"][0, index, 0], step)
            rtc.advance()

    def test_missing_server_ack_rejects_unconditioned_prediction(self):
        rtc = self.loaded()
        for _ in range(4):
            rtc.advance()
        request = rtc.start_request(now=2)
        self.assertFalse(rtc.finish(request, trajectory(4), {}, now=2.1))
        self.assertIn("acknowledge", rtc.last_status)
        self.assertEqual(rtc.current()[0]["joint"][0, 4, 0], 4)

    def test_modified_frozen_actions_are_rejected(self):
        rtc = self.loaded()
        for _ in range(4):
            rtc.advance()
        request = rtc.start_request(now=2)
        action, info = response(request)
        action["joint"][0, 0, 0] = -10
        self.assertFalse(rtc.finish(request, action, info, now=2.1))
        self.assertIn("committed", rtc.last_status)

    def test_reset_discards_inflight_old_prompt_and_prevents_concurrent_rpc(self):
        rtc = self.loaded()
        for _ in range(4):
            rtc.advance()
        old = rtc.start_request(now=2)
        rtc.reset()
        self.assertIsNone(rtc.current())
        self.assertFalse(rtc.can_request())
        self.assertFalse(rtc.finish(old, *response(old), now=2.1))
        self.assertTrue(rtc.can_request())
        self.assertIsNone(rtc.start_request(now=3).prefix)

    def test_failure_releases_request_and_can_retry(self):
        rtc = self.scheduler()
        request = rtc.start_request(now=0)
        self.assertFalse(rtc.finish(request, None, {"error": "timeout"}, now=1))
        self.assertIn("timeout", rtc.last_status)
        self.assertTrue(rtc.can_request())

    def test_exhaustion_holds_instead_of_repeating_chunk(self):
        rtc = self.loaded()
        for _ in range(8):
            rtc.advance()
        self.assertIsNone(rtc.current())
        with self.assertRaises(RuntimeError):
            rtc.advance()
        self.assertIsNone(rtc.start_request(now=2).prefix)

    def test_latency_updates_freeze_window(self):
        rtc = self.loaded()
        for _ in range(4):
            rtc.advance()
        request = rtc.start_request(now=2)
        for _ in range(4):
            rtc.advance()
        self.assertTrue(rtc.finish(request, *response(request), now=2.12))
        self.assertGreater(rtc.freeze_steps, request.frozen_steps)
        self.assertEqual(rtc.last_overrun_steps, 2)
        self.assertLess(rtc.freeze_steps, rtc.horizon)

    def test_bad_configuration_and_nonfinite_result(self):
        with self.assertRaises(ValueError):
            RTCScheduler(1)
        with self.assertRaises(ValueError):
            RTCScheduler(40, ramp_rate=float("nan"))
        rtc = self.scheduler()
        request = rtc.start_request(now=0)
        action = trajectory(0)
        action["joint"][0, 2, 0] = np.nan
        self.assertFalse(rtc.finish(request, action, now=1))


if __name__ == "__main__":
    unittest.main()
