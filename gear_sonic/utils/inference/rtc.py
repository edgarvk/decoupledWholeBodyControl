"""Control-step timeline for asynchronous real-time chunking.

Only the main control thread mutates this timeline. A worker receives a copied
absolute prefix and returns a trajectory anchored at that request's start step.
Wall time estimates the freeze window; actual published steps select the result
index, so network latency is never applied twice.
"""

from collections import deque
from dataclasses import dataclass
import math
import time

import numpy as np


@dataclass(frozen=True)
class RTCRequest:
    sequence: int
    generation: int
    start_step: int
    started_at: float
    prefix: dict | None
    frozen_steps: int
    ramp_rate: float

    @property
    def options(self):
        if self.prefix is None:
            return None
        return {
            "rtc": {
                "version": 1,
                "prefix_actions": self.prefix,
                "frozen_steps": self.frozen_steps,
                "ramp_rate": self.ramp_rate,
            }
        }


class RTCScheduler:
    def __init__(self, horizon, rate=50.0, overlap_steps=20, initial_delay=0.2, safety_steps=2, ramp_rate=5.0):
        if horizon < 2 or rate <= 0 or overlap_steps < 1 or initial_delay < 0 or safety_steps < 0:
            raise ValueError("Invalid RTC horizon, rate, overlap, delay or safety margin")
        if not all(math.isfinite(v) for v in (rate, initial_delay, ramp_rate)) or ramp_rate <= 0:
            raise ValueError("RTC rates must be finite and ramp_rate positive")
        self.horizon = horizon
        self.rate = rate
        self.overlap_steps = min(overlap_steps, horizon - 1)
        self.initial_delay = initial_delay
        self.safety_steps = safety_steps
        self.ramp_rate = ramp_rate
        self._latencies = deque(maxlen=8)
        self._plan = None
        self._cursor = 0
        self._step = 0
        self._generation = 0
        self._sequence = 0
        self._inflight = None
        self.last_status = "idle"
        self.last_latency = 0.0
        self.last_overrun_steps = 0

    @property
    def freeze_steps(self):
        delay = max(self._latencies) if self._latencies else self.initial_delay
        return min(self.horizon - 1, math.ceil(delay * self.rate) + self.safety_steps)

    @property
    def remaining(self):
        return 0 if self._plan is None else max(0, self.horizon - self._cursor)

    def can_request(self):
        overlap = max(self.overlap_steps, self.freeze_steps)
        return self._inflight is None and self.remaining <= overlap

    def start_request(self, now=None):
        if not self.can_request():
            raise RuntimeError("RTC request is not due or another request is in flight")
        self._sequence += 1
        prefix = None
        if self.remaining:
            prefix = {key: value[:, self._cursor :].copy() for key, value in self._plan.items()}
        request = RTCRequest(
            self._sequence,
            self._generation,
            self._step,
            time.monotonic() if now is None else now,
            prefix,
            min(self.remaining, self.freeze_steps),
            self.ramp_rate,
        )
        self._inflight = request
        return request

    def finish(self, request, action, info=None, now=None):
        if self._inflight is None or self._inflight.sequence != request.sequence:
            self.last_status = "discarded unknown request"
            return False
        self._inflight = None
        if request.generation != self._generation:
            self.last_status = "discarded response from an earlier prompt/control generation"
            return False
        self.last_latency = max(0.0, (time.monotonic() if now is None else now) - request.started_at)
        if action is None:
            self.last_status = f"inference failed: {(info or {}).get('error', 'no action')}"
            return False
        if not isinstance(action, dict) or not action:
            self.last_status = "discarded empty RTC trajectory"
            return False
        for key, value in action.items():
            if (
                not isinstance(value, np.ndarray)
                or value.ndim != 3
                or value.shape[:2] != (1, self.horizon)
                or not np.isfinite(value).all()
            ):
                self.last_status = f"discarded invalid RTC action {key}"
                return False
        elapsed_steps = self._step - request.start_step
        if request.prefix is not None:
            self._latencies.append(self.last_latency)
            overlap = next(iter(request.prefix.values())).shape[1]
            ack = (info or {}).get("rtc", {})
            if (
                ack.get("version") != 1
                or ack.get("overlap_steps") != overlap
                or ack.get("frozen_steps") != request.frozen_steps
            ):
                self.last_status = "server did not acknowledge the requested RTC conditioning"
                return False
            if set(action) != set(request.prefix) or any(
                not np.array_equal(action[key][:, : request.frozen_steps], prefix[:, : request.frozen_steps])
                for key, prefix in request.prefix.items()
            ):
                self.last_status = "discarded RTC trajectory that changed committed physical actions"
                return False
            self.last_overrun_steps = max(0, elapsed_steps - request.frozen_steps)
        else:
            self.last_overrun_steps = 0
        if not 0 <= elapsed_steps < self.horizon:
            self.last_status = "discarded expired RTC response"
            return False
        self._plan = {key: value.copy() for key, value in action.items()}
        self._cursor = elapsed_steps
        self.last_status = "accepted"
        return True

    def current(self):
        return (self._plan, self._cursor) if self.remaining else None

    def advance(self):
        """Commit exactly one action *after* the controller message was sent."""
        if not self.remaining:
            raise RuntimeError("Cannot advance an exhausted RTC trajectory")
        self._step += 1
        self._cursor += 1

    def reset(self):
        """Invalidate queued replies on prompt, pause, stop or sensor discontinuity.

        Keep the in-flight token until its worker returns: ZMQ REQ sockets must
        never issue another request concurrently, even across a reset.
        """
        self._generation += 1
        self._plan = None
        self._cursor = 0
        self.last_status = "reset"
