# Set up the AM62D TVM 12.01 toolchain env for tidlrunner-cli.
#
# SOURCE this file (don't execute it) — the vars must be live in your shell BEFORE
# Python starts, because TVM dlopen's libvx_tidl_rt.so from LD_LIBRARY_PATH.
#
# It exports the vars set_env.py honors when already present (it only fills absent
# ones), so the wheel-bundled x86 TIDL tools are used instead of the
# tools/tidl_tools_package download.
#
# ARM64_GCC_PATH / CGT7X_ROOT resolve to the toolchains inside this repo's
# tools/tidl_tools_package/bin/ (downloaded by setup_runner_pc.sh).
# Override by exporting either var before sourcing.
#
# Prereqs: the dedicated venv (tidlrunner-am62d) with the RC x86 TVM wheel active,
# and tools/tidl_tools_package/bin/ populated (run setup_runner_pc.sh once).
# Full setup: tidlrunner/docs/setup_am62d.md.
# Usage:
#   pyenv activate tidlrunner-am62d
#   source devices/am62d_env.sh
#   tidlrunner-cli compile  --config_path data/configs/samples/models/audio/speech_enhancement/voicebank_demand_16k/gcrn_g16_fixed_4sec_tvmrt_config.yaml
#   tidlrunner-cli evaluate --config_path "$CFG" --target_device AM62D

# Refuse to run as an executed script — it would set vars in a throwaway subshell.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo "Error: source this script, don't execute it:  source ${0##*/}"
    exit 1
fi

# Repo root, resolved from this script's location (works from any cwd).
_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Resolve the installed tvm package dir (works with whichever venv is active).
_TVM_DIR="$(python -c 'import tvm, os; print(os.path.dirname(tvm.__file__))' 2>/dev/null)"
if [ -z "$_TVM_DIR" ]; then
    echo "Error: could not locate the tvm package. Activate the tidlrunner-am62d venv first."
    return 1
fi

# Bundled x86 TIDL tools for AM62D live inside the RC wheel.
export TIDL_TOOLS_PATH="$_TVM_DIR/3rdparty/x86_tidl_tools/AM62D"
if [ ! -d "$TIDL_TOOLS_PATH" ]; then
    echo "Error: bundled TIDL tools not found at $TIDL_TOOLS_PATH"
    echo "       (is the AM62D 12.01 RC TVM wheel installed?)"
    return 1
fi

export LD_LIBRARY_PATH="$TIDL_TOOLS_PATH:${LD_LIBRARY_PATH:-}"

# Cross-toolchains from the repo's tidl_tools_package download (setup_runner_pc.sh).
_TOOLS_BIN="$_REPO_ROOT/tools/tidl_tools_package/bin"
export ARM64_GCC_PATH="${ARM64_GCC_PATH:-$_TOOLS_BIN/arm-gnu-toolchain-15.2.rel1-x86_64-aarch64-none-linux-gnu}"
export CGT7X_ROOT="${CGT7X_ROOT:-$_TOOLS_BIN/ti-cgt-c7000_5.0.0.LTS}"
if [ ! -d "$ARM64_GCC_PATH" ] || [ ! -d "$CGT7X_ROOT" ]; then
    echo "Error: cross-toolchains not found under $_TOOLS_BIN"
    echo "       CGT7X_ROOT (ti-cgt-c7000_5.0.0.LTS) comes from ./setup_runner_pc.sh."
    echo "       ARM64_GCC_PATH (arm-gnu-toolchain-15.2) is installed by ./devices/setup_am62d.sh"
    echo "       (setup_runner_pc.sh fetches only 13.2) — run that setup first,"
    echo "       or export ARM64_GCC_PATH/CGT7X_ROOT to existing installs before sourcing."
    echo "       Full setup: tidlrunner/docs/setup_am62d.md."
    return 1
fi

# SOC identifier read by the RC wheel's TIDL compile path.
export SOC="${SOC:-am62d}"

unset _TVM_DIR _REPO_ROOT _TOOLS_BIN

echo "INFO: TIDL_TOOLS_PATH = $TIDL_TOOLS_PATH"
echo "INFO: ARM64_GCC_PATH  = $ARM64_GCC_PATH"
echo "INFO: CGT7X_ROOT      = $CGT7X_ROOT"
echo "INFO: SOC             = $SOC"
echo "INFO: LD_LIBRARY_PATH = $LD_LIBRARY_PATH"
echo "INFO: AM62D toolchain env is set — run tidlrunner-cli directly in this shell."
