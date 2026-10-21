#!/usr/bin/env bash
# Source after ROS/CUDA environment setup. Idempotent and independent of cwd.

teleop_configure_unitree_dds() {
    local deploy_root="$1" architecture="$2" library_dir entry library
    case "$architecture" in
        x86_64|aarch64) ;;
        arm64) architecture="aarch64" ;;
        *) echo "Unsupported Unitree DDS architecture: $architecture" >&2; return 1 ;;
    esac
    library_dir="$deploy_root/thirdparty/unitree_sdk2/thirdparty/lib/$architecture"
    for library in libddsc.so.0 libddscxx.so.0; do
        if [[ ! -r "$library_dir/$library" ]]; then
            echo "Missing bundled Unitree DDS library: $library_dir/$library" >&2
            return 1
        fi
        if [[ "$(LC_ALL=C head -c 7 "$library_dir/$library")" == "version" ]]; then
            echo "Unitree DDS library is an LFS pointer; restore LFS assets: $library_dir/$library" >&2
            return 1
        fi
    done
    local -a old_entries
    IFS=: read -r -a old_entries <<< "${LD_LIBRARY_PATH:-}"
    local selected_path="$library_dir"
    for entry in "${old_entries[@]}"; do
        if [[ -n "$entry" && "$entry" != "$library_dir" ]]; then
            selected_path+=":$entry"
        fi
    done
    export UNITREE_DDS_LIBRARY_DIR="$library_dir"
    export LD_LIBRARY_PATH="$selected_path"
    echo "Unitree matched DDS libraries: $library_dir"
}

if [[ "$(uname -s)" == "Linux" ]]; then
    teleop_configure_unitree_dds \
        "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" "$(uname -m)"
fi
