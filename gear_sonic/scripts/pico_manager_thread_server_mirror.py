"""Compatibility entry point for the intern's mirrored teleoperation command."""

import runpy
import sys

if __name__ == "__main__":
    if "--mirror" not in sys.argv:
        sys.argv.append("--mirror")
    if not any(arg == "--gripper-profile" or arg.startswith("--gripper-profile=") for arg in sys.argv):
        sys.argv.extend(["--gripper-profile", "intern"])
    runpy.run_module("gear_sonic.scripts.pico_manager_thread_server", run_name="__main__")
