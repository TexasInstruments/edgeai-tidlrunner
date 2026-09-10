#!/usr/bin/env bash
#
# One-shot AM62D setup — the single-script equivalent of setup_runner_pc.sh for
# the AM62D audio flow. Run it AFTER activating the dedicated venv:
#
#   pyenv virtualenv 3.10 tidlrunner-am62d   # once
#   pyenv activate tidlrunner-am62d
#   ./devices/setup_am62d.sh
#   source devices/am62d_env.sh              # per shell, see that file
#
# It installs the ARM GCC 15.2 cross-toolchain into tools/tidl_tools_package/bin/
# (setup_runner_pc.sh only fetches 13.2), the RC x86 TVM wheel (force-reinstall so
# it wins over any resolved tvm), tidlrunner[pc,audio] + tools, and
# tidl_onnx_model_optimizer (which this flow needs but the standard
# tidlrunner-tools-download — bypassed here — normally provides). It does NOT
# create the venv, run setup_runner_pc.sh, or source am62d_env.sh: those are
# prereq / per-shell runtime env, kept separate by design.
#
# Full setup: tidlrunner/docs/setup_am62d.md.

set -e

# Internal-artifactory RC wheel (x86, cp310). Override with AM62D_TVM_WHEEL.
AM62D_TVM_WHEEL="${AM62D_TVM_WHEEL:-https://artifactory.itg.ti.com/artifactory/generic-epd-sdto-codegen-local/tvm/c7x/am62d/12_1/tvm-0.18.0-0git6acc98882-cp310-cp310-linux_x86_64.whl}"

# Ref for tidl_onnx_model_optimizer — match the RC wheel's bundled tools version.
TIDL_OPT_REF="${TIDL_OPT_REF:-11_02_16_00}"

# Repo root, resolved from this script's location (works from any cwd).
_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Guard 1: a venv must be active — installing into system Python (or the standard
# tidlrunner venv) would clobber its TVM with the RC git build.
if python -c "import sys; sys.exit(0 if sys.prefix == sys.base_prefix else 1)"; then
    echo "Error: no virtualenv active. Create and activate the dedicated venv first:"
    echo "       pyenv virtualenv 3.10 tidlrunner-am62d && pyenv activate tidlrunner-am62d"
    exit 1
fi
_VENV_NAME="${PYENV_VERSION:-$(basename "${VIRTUAL_ENV:-}")}"
if [ -n "$_VENV_NAME" ] && [[ "$_VENV_NAME" != *am62d* ]]; then
    echo "WARNING: active venv '$_VENV_NAME' does not look like the AM62D venv;"
    echo "         expected something containing 'am62d' (e.g. tidlrunner-am62d)."
fi

# Guard 2: toolchain prereq — setup_runner_pc.sh must have populated the bin dir
# (am62d_env.sh points ARM64_GCC_PATH/CGT7X_ROOT into it).
_TOOLS_BIN="$_REPO_ROOT/tools/tidl_tools_package/bin"
if [ ! -d "$_TOOLS_BIN" ]; then
    echo "Error: $_TOOLS_BIN not found."
    echo "       Run ./setup_runner_pc.sh once first (downloads the cross-toolchains)."
    exit 1
fi

# ARM GCC 15.2 cross-toolchain — am62d_env.sh points ARM64_GCC_PATH here, but
# setup_runner_pc.sh only fetches 13.2. Install 15.2 into the shared bin/ dir.
# ponytail: temporary — fold into tools/tidl_tools_package/download.py once it
# supports 15.2, then drop this block.
_GCC_NAME="arm-gnu-toolchain-15.2.rel1-x86_64-aarch64-none-linux-gnu"
ARM_GCC_15_2_URL="${ARM_GCC_15_2_URL:-https://developer.arm.com/-/media/Files/downloads/gnu/15.2.rel1/binrel/${_GCC_NAME}.tar.xz}"
if [ -d "$_TOOLS_BIN/$_GCC_NAME" ]; then
    echo "INFO: ARM GCC 15.2 already present, skipping download."
else
    echo "INFO: downloading ARM GCC 15.2 cross-toolchain..."
    curl -L --fail -o "$_TOOLS_BIN/$_GCC_NAME.tar.xz" "$ARM_GCC_15_2_URL"
    echo "INFO: extracting into $_TOOLS_BIN ..."
    tar -xf "$_TOOLS_BIN/$_GCC_NAME.tar.xz" -C "$_TOOLS_BIN"
    rm -f "$_TOOLS_BIN/$_GCC_NAME.tar.xz"
fi

echo "INFO: installing RC x86 TVM wheel (force-reinstall)..."
echo "INFO:   $AM62D_TVM_WHEEL"
pip install --force-reinstall --upgrade "$AM62D_TVM_WHEEL"

echo "INFO: installing tidlrunner[pc,audio] + tools..."
pip install -e "$_REPO_ROOT/tidlrunner[pc,audio]"
pip install -e "$_REPO_ROOT/tools"

echo "INFO: installing tidl_onnx_model_optimizer (git, ref $TIDL_OPT_REF)..."
pip install "tidl_onnx_model_optimizer@git+https://github.com/TexasInstruments/edgeai-tidl-tools.git@${TIDL_OPT_REF}#subdirectory=model-tools/tidl-onnx-model-optimizer"

echo "INFO: completed AM62D setup."
echo "INFO: next, per shell:  source devices/am62d_env.sh"
