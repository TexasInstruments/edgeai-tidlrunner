#!/usr/bin/env bash
#
# One-shot AM62D setup — standalone, no other setup script needed. Run it
# AFTER activating the dedicated venv:
#
#   pyenv virtualenv 3.10 tidlrunner-am62d   # once
#   pyenv activate tidlrunner-am62d
#   ./devices/setup_am62d.sh
#   source devices/am62d_env.sh              # per shell, see that file
#
# It installs both cross-toolchains this flow needs into
# tools/tidl_tools_package/bin/ — ARM GCC 15.2 and C7x CGT 5.0.0.LTS, each
# skipped if already present — the RC x86 TVM wheel (force-reinstall so it wins
# over any resolved tvm), tidlrunner[pc,audio] + tools + onnxruntime, and
# tidl_onnx_model_optimizer (which this flow needs but the standard
# tidlrunner-tools-download — bypassed here — normally provides). It does NOT
# create the venv or source am62d_env.sh: those are prereq / per-shell runtime
# env, kept separate by design.
#
# Full setup: tidlrunner/docs/setup_am62d.md.

set -e

# Internal-artifactory RC wheel (x86, cp310). Override with AM62D_TVM_WHEEL.
AM62D_TVM_WHEEL="${AM62D_TVM_WHEEL:-https://artifactory.itg.ti.com/artifactory/generic-epd-sdto-codegen-local/tvm/c7x/am62d/12_1/tvm-0.18.0-0git6acc98882-cp310-cp310-linux_x86_64.whl}"

# Ref for tidl_onnx_model_optimizer — match the RC wheel's bundled tools version.
TIDL_OPT_REF="${TIDL_OPT_REF:-11_02_16_00}"

# Repo root, resolved from this script's location (works from any cwd).
_REPO_ROOT="$(CDPATH= cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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

_TOOLS_BIN="$_REPO_ROOT/tools/tidl_tools_package/bin"
mkdir -p "$_TOOLS_BIN"

# ARM GCC 15.2 cross-toolchain — am62d_env.sh points ARM64_GCC_PATH here.
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

# C7x CGT compiler — CGT7X_ROOT in am62d_env.sh. Version/URL mirror
# download.py:404 / :785 (the standard flow's source of truth); it only fetches
# this on the 11.2.x/11.2 paths, so don't rely on setup_runner_pc.sh for it.
_CGT_VER="${C7X_CGT_VERSION:-5.0.0.LTS}"
_CGT_NAME="ti-cgt-c7000_${_CGT_VER}"
_CGT_FILE="ti_cgt_c7000_${_CGT_VER}_linux-x64_installer.bin"
C7X_CGT_URL="${C7X_CGT_URL:-https://dr-download.ti.com/software-development/ide-configuration-compiler-or-debugger/MD-707zYe3Rik/${_CGT_VER}/${_CGT_FILE}}"
if [ -d "$_TOOLS_BIN/$_CGT_NAME" ]; then
    echo "INFO: C7x CGT $_CGT_VER already present, skipping download."
else
    echo "INFO: downloading C7x CGT $_CGT_VER ..."
    curl -L --fail -o "$_TOOLS_BIN/$_CGT_FILE" "$C7X_CGT_URL"
    echo "INFO: running the unattended installer into $_TOOLS_BIN ..."
    chmod +x "$_TOOLS_BIN/$_CGT_FILE"
    "$_TOOLS_BIN/$_CGT_FILE" --mode unattended --prefix "$_TOOLS_BIN"
    rm -f "$_TOOLS_BIN/$_CGT_FILE"
    # The installer picks its own dirname — fail loudly if it isn't what am62d_env.sh expects.
    [ -d "$_TOOLS_BIN/$_CGT_NAME" ] || {
        echo "Error: installer did not produce $_TOOLS_BIN/$_CGT_NAME"; exit 1; }
fi

echo "INFO: installing RC x86 TVM wheel (force-reinstall)..."
echo "INFO:   $AM62D_TVM_WHEEL"
pip install --force-reinstall --upgrade "$AM62D_TVM_WHEEL"

echo "INFO: installing tidlrunner[pc,audio] + tools + onnxruntime..."
# Plain onnxruntime (CPU): tvmrt_wrapper.py's _get_{input,output}_details use it
# only to read onnx model I/O shapes via CPUExecutionProvider, not for TIDL
# inference — no need for the TI-modified onnxruntime-tidl the standard flow uses.
pip install -e "$_REPO_ROOT/tidlrunner[pc,audio]" onnxruntime
pip install -e "$_REPO_ROOT/tools"

echo "INFO: installing tidl_onnx_model_optimizer (git, ref $TIDL_OPT_REF)..."
pip install "tidl_onnx_model_optimizer@git+https://github.com/TexasInstruments/edgeai-tidl-tools.git@${TIDL_OPT_REF}#subdirectory=model-tools/tidl-onnx-model-optimizer"

echo "INFO: completed AM62D setup."
echo "INFO: next, per shell:  source devices/am62d_env.sh"
