"""Compile the real binding against a fake service; no headset/robot connection."""

import ctypes
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sysconfig
import time

import pytest


@pytest.fixture(scope="module")
def native_sdk(tmp_path_factory):
    pybind11 = pytest.importorskip("pybind11")
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("A C++ compiler is needed for the native packet replay test")
    root = Path(__file__).resolve().parents[2]
    sdk = root / "external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64"
    directory = tmp_path_factory.mktemp("xrt-fake-service")
    stub = directory / "fake_service.cpp"
    stub.write_text("""
#include <cstdint>
#include <cstring>
#include "PXREARobotSDK.h"
static pfPXREAClientCallback callback;
extern "C" int PXREAInit(void*, pfPXREAClientCallback cb, unsigned) { callback = cb; return 0; }
extern "C" int PXREADeinit() { return 0; }
extern "C" int PXREADeviceControlJson(const char*, const char*) { return 0; }
extern "C" int PXREASendBytesToDevice(const char*, const char*, unsigned) { return 0; }
extern "C" void inject(const char* text, int type) {
    PXREADevStateJson data{};
    std::strncpy(data.stateJson, text, sizeof(data.stateJson) - 1);
    callback(nullptr, static_cast<PXREAClientCallbackType>(type), 0, &data);
}
""")
    output = directory / ("xrobotoolkit_sdk" + sysconfig.get_config_var("EXT_SUFFIX"))
    command = [
        compiler,
        "-shared",
        "-std=c++14",
        "-fPIC",
        "-DPXREACLIENTSDK_EXPORT=",
        "-I" + pybind11.get_include(),
        "-I" + sysconfig.get_path("include"),
        "-I" + str(sdk / "include"),
        str(sdk / "bindings/py_bindings.cpp"),
        str(stub),
        "-o",
        str(output),
    ]
    if sysconfig.get_platform().startswith("macosx"):
        command += ["-undefined", "dynamic_lookup"]
    subprocess.run(command, check=True, capture_output=True, text=True)
    spec = importlib.util.spec_from_file_location("xrobotoolkit_sdk", output)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    lib = ctypes.CDLL(str(output))
    lib.inject.argtypes = [ctypes.c_char_p, ctypes.c_int]
    module.init()  # calls ONLY the stub above

    def inject(packet=None, event=1 << 25):
        lib.inject(json.dumps({"value": json.dumps(packet)}).encode(), event)

    yield module, inject
    module.close()


def test_native_snapshot_missing_controller_cannot_reuse_old_pose(native_sdk):
    sdk, inject = native_sdk
    # Snapshot publication happens independently of Hand/Body/ankle processing.
    packet = {
        "timeStampNs": 123,
        "Controller": {"left": {"pose": "1,2,3,0,0,0,1"}, "right": {"pose": "4,5,6,0,0,0,1"}},
        "Head": {"pose": "0,1.6,0,0,0,0,1"},
    }
    inject(packet)
    result = json.loads(sdk.get_controller_snapshot())
    assert result["Controller"] == packet["Controller"]
    assert result["timeStampNs"] == 123
    assert 0 <= result["receive_age_s"] < 1
    del packet["Controller"]["left"]
    packet["timeStampNs"] = 124
    inject(packet)
    result = json.loads(sdk.get_controller_snapshot())
    assert "left" not in result["Controller"]
    assert result["timeStampNs"] == 124
    time.sleep(0.01)
    assert json.loads(sdk.get_controller_snapshot())["receive_age_s"] >= 0.01
    inject(event=1 << 3)  # server disconnect invalidates immediately
    assert "Controller" not in json.loads(sdk.get_controller_snapshot())


def test_deploy_publishes_measured_joints_before_start_only_when_fresh(tmp_path):
    """Compile/replay the actual pre-switch Control block without TensorRT/DDS."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler required")
    root = Path(__file__).resolve().parents[2]
    source = (root / "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp").read_text()
    prefix = source.split("    void Control() {", 1)[1].split("      switch (program_state_)", 1)[0]
    harness = (
        r"""
#include <array>
#include <cassert>
#include <chrono>
#include <memory>
#include <vector>
using Clock = std::chrono::steady_clock;
struct Motor {
    double value;
    double q() const { return value; }
    double dq() const { return .12; }
    double tau_est() const { return 1.4; }
    std::array<int, 2> temperature() const { return {30, 29}; }
};
struct State {
    std::array<Motor, 29> motors;
    const auto& motor_state() const { return motors; }
};
struct Data { std::shared_ptr<State> data; Clock::time_point timestamp; };
struct Buffer { Data value; Data GetDataWithTime() const { return value; } };
struct Output {
    int calls = 0;
    std::array<double, 29> joints;
    void publish_calibration_state(const std::array<double, 29>& q, double age,
                                  const std::array<double, 58>& temperatures,
                                  const std::array<double, 29>& torques,
                                  const std::array<double, 29>& velocities) {
        for (int i = 0; i < 29; ++i) {
            assert(temperatures[i * 2] == 30 && temperatures[i * 2 + 1] == 29);
            assert(torques[i] == 1.4 && velocities[i] == .12);
        }
        assert(age >= 0 && age <= .05);
        joints = q; ++calls;
    }
};
struct Controller {
    enum class ProgramState { INIT, WAIT_FOR_CONTROL, CONTROL };
    ProgramState program_state_ = ProgramState::WAIT_FOR_CONTROL;
    struct { bool stop = false; } operator_state;
    Buffer low_state_buffer_;
    std::vector<std::shared_ptr<Output>> output_interfaces_;
    static constexpr std::chrono::milliseconds LOW_STATE_LATE_THRESHOLD{50};
    void Control() {
"""
        + prefix
        + r"""
    }
};
int main() {
    Controller c;
    auto out = std::make_shared<Output>();
    c.output_interfaces_.push_back(out);
    c.Control(); assert(out->calls == 0); // no measured state
    auto state = std::make_shared<State>();
    for (int i = 0; i < 29; ++i) state->motors[i].value = .01 * i;
    c.low_state_buffer_.value = {state, Clock::now()};
    c.Control(); assert(out->calls == 1); // still WAIT_FOR_CONTROL, no start command
    for (int i = 0; i < 29; ++i) assert(out->joints[i] == .01 * i);
    c.program_state_ = Controller::ProgramState::INIT;
    c.Control(); assert(out->calls == 1);
    c.program_state_ = Controller::ProgramState::CONTROL;
    c.Control(); assert(out->calls == 2);
    c.low_state_buffer_.value.timestamp -= std::chrono::seconds(1);
    c.Control(); assert(out->calls == 2);
    c.low_state_buffer_.value.timestamp = Clock::now();
    c.operator_state.stop = true;
    c.Control(); assert(out->calls == 2);
}
"""
    )
    path = tmp_path / "calibration.cpp"
    path.write_text(harness)
    binary = tmp_path / "calibration_test"
    subprocess.run([compiler, "-std=c++17", str(path), "-o", str(binary)], check=True, capture_output=True)
    subprocess.run([str(binary)], check=True)
