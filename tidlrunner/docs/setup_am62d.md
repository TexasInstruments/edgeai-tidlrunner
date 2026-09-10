# Setup for AM62D (TVM RC toolchain)

AM62D is a device dedicated to audio DL models. Its compile flow differs from the standard [setup](./setup.md): it uses a **TVM wheel** whose bundled `x86_tidl_tools/AM62D` replace the usual `tools/tidl_tools_package` download, and it offloads to the C7x via the TVM runtime (tvmrt).

Because the RC wheel is `tvm==0.18.0` (a special git build) it would clobber the standard TVM in the main `tidlrunner` venv. Keep the AM62D flow in its own dedicated venv, **`tidlrunner-am62d`**.

For the audio models this device runs (and how to run their pipelines), see [audio_models_and_datasets.md](./audio_models_and_datasets.md).

## 1. Prereq: standard PC setup

Run the standard PC setup once so `tools/tidl_tools_package/bin/` is populated:

```bash
pyenv activate tidlrunner   # the standard venv from setup.md
./setup_runner_pc.sh
```

Run this in your **standard `tidlrunner` venv** (see [setup.md](./setup.md)) — `setup_runner_pc.sh` runs `pip install`, so it needs a venv active, and it must not be the `tidlrunner-am62d` venv created below (it would pull in the standard TVM). For the AM62D flow, the only thing this step produces that matters is the shared repo download under `tools/tidl_tools_package/bin/`; that directory is shared across venvs, so this only needs to happen once.

This step provides **`ti-cgt-c7000_5.0.0.LTS`** (the version `devices/am62d_env.sh` points `CGT7X_ROOT` at). It does **not** install the ARM GCC **15.2** toolchain this flow needs (`download.py` fetches 13.2) — step 3's `setup_am62d.sh` downloads 15.2 into the same `bin/` directory for you.

## 2. Create and activate the dedicated venv

```bash
pyenv virtualenv 3.10 tidlrunner-am62d
pyenv activate tidlrunner-am62d
```

## 3. Run the AM62D setup script

With the venv active, one script installs the ARM GCC 15.2 cross-toolchain (into `tools/tidl_tools_package/bin/`, skipped if already present), the RC TVM wheel, `tidlrunner[pc,audio]` + `tools`, and `tidl_onnx_model_optimizer`:

```bash
./devices/setup_am62d.sh
```

Notes:

- The RC wheel is hosted internally on artifactory (the x86 `cp310` build for PC compilation). Override the URL with `AM62D_TVM_WHEEL=<url> ./devices/setup_am62d.sh` if it moves. The matching `...cp314-...linux_aarch64.whl` is the **on-device (EVM) runtime** wheel — not needed for PC compilation.
- `tidl_onnx_model_optimizer` is installed here because `surgery.py` imports it unconditionally for `.onnx` models, and this flow bypasses the standard `tidlrunner-tools-download` that normally provides it.

## 4. Set the env (once per shell) and run

```bash
source devices/am62d_env.sh
```

`am62d_env.sh` **must be sourced, not executed** — `LD_LIBRARY_PATH` has to be live before Python starts, because TVM `dlopen`s `libvx_tidl_rt.so`. It exports `TIDL_TOOLS_PATH` (the wheel-bundled AM62D tools), `LD_LIBRARY_PATH`, `ARM64_GCC_PATH`, `CGT7X_ROOT`, and `SOC=am62d`, then prints an `INFO:` block confirming each.

```bash
tidlrunner-cli compile \
  --config_path data/configs/samples/models/audio/speech_enhancement/voicebank_demand_16k/gcrn_g16_fixed_4sec_tvmrt_config.yaml

tidlrunner-cli evaluate \
  --config_path data/configs/samples/models/audio/speech_enhancement/voicebank_demand_16k/gcrn_g16_fixed_4sec_tvmrt_config.yaml \
  --target_device AM62D
```
