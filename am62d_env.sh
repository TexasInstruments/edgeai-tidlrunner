# Set up the AM62D TVM 12.01 RC toolchain env for tidlrunner-cli (feasibility trial).
#
# SOURCE this file (don't execute it) — the vars must be live in your shell BEFORE
# Python starts, because TVM dlopen's libvx_tidl_rt.so from LD_LIBRARY_PATH.
#
# It exports the vars set_env.py honors when already present (it only fills absent
# ones), so the wheel-bundled x86 TIDL tools are used instead of the
# tools/tidl_tools_package download.
#
# Prereqs: the dedicated venv (tidlrunner-am62d) with the RC x86 TVM wheel active.
# Usage:
#   pyenv activate tidlrunner-am62d
#   source am62d_env.sh
#   tidlrunner-cli compile  --config_path data/configs/.../gcrn_..._am62d_tvmrt_config.yaml
#   tidlrunner-cli evaluate --config_path "$CFG" --target_device AM62D

# Refuse to run as an executed script — it would set vars in a throwaway subshell.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    echo "Error: source this script, don't execute it:  source ${0##*/}"
    exit 1
fi

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

# External host toolchains used by the RC reference (verified present on this host).
export ARM64_GCC_PATH="${ARM64_GCC_PATH:-/home/jroh/ti/arm/arm-gnu-toolchain-15.2.rel1-x86_64-aarch64-none-linux-gnu}"
export CGT7X_ROOT="${CGT7X_ROOT:-/home/jroh/ti/c7x/ti-cgt-c7000_5.0.0.LTS}"

# SOC identifier read by the RC wheel's TIDL compile path.
export SOC="${SOC:-am62d}"

unset _TVM_DIR

echo "INFO: TIDL_TOOLS_PATH = $TIDL_TOOLS_PATH"
echo "INFO: ARM64_GCC_PATH  = $ARM64_GCC_PATH"
echo "INFO: CGT7X_ROOT      = $CGT7X_ROOT"
echo "INFO: SOC             = $SOC"
echo "INFO: LD_LIBRARY_PATH = $LD_LIBRARY_PATH"
echo "INFO: AM62D toolchain env is set — run tidlrunner-cli directly in this shell."
