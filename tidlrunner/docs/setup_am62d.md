# Setup for AM62D (TVM toolchain)

AM62D is a device dedicated to audio DL models. Its compile flow differs from the standard [setup](./setup.md): it uses a **TVM wheel** whose bundled `x86_tidl_tools/AM62D` replace the usual `tools/tidl_tools_package` download, and it offloads to the C7x via the TVM runtime (tvmrt).

Because the TVM wheel is `tvm==0.18.0` (a special git build) it would clobber the standard TVM in the main `tidlrunner` venv. Keep the AM62D flow in its own dedicated venv, **`tidlrunner-am62d`**, standalone from any standard `tidlrunner` setup.

For the audio models this device runs (and how to run their pipelines), see [audio_models_and_datasets.md](./audio_models_and_datasets.md).

## 1. Create and activate the dedicated venv

```bash
pyenv virtualenv 3.10 tidlrunner-am62d
pyenv activate tidlrunner-am62d
```

## 2. Run the AM62D setup script

With the venv active, one script installs the ARM GCC 15.2 and C7x CGT 5.0.0.LTS cross-toolchains (into `tools/tidl_tools_package/bin/`, shared with the standard flow and skipped if already present), the RC TVM wheel, `tidlrunner[pc,audio]` + `tools` + `onnxruntime`, and `tidl_onnx_model_optimizer`:

```bash
./devices/setup_am62d.sh
```

Notes:

- `tidl_onnx_model_optimizer` is installed here because `surgery.py` imports it unconditionally for `.onnx` models, and this flow bypasses the standard `tidlrunner-tools-download` that normally provides it.
- Plain `onnxruntime` (CPU) is installed because `tvmrt_wrapper.py` imports it to read onnx model I/O shapes via `CPUExecutionProvider`; it is not declared as a dependency anywhere else in this flow. This is not the TI-modified `onnxruntime-tidl` the standard flow uses — that's only needed for TIDL-accelerated inference, which this code path doesn't do.
- **Build prereqs.** The script does no `apt` install. Wheels cover everything on Python 3.10 x86_64, but if a source build fails: `sudo apt-get install -y cmake libffi-dev libjpeg-dev zlib1g-dev protobuf-compiler`.

## 3. Set the env (once per shell) and run

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
