# Audio Models and Datasets (AM62D)

This guide covers how to run audio model pipelines in **edgeai-tidlrunner** on **AM62D** — sound classification and speech enhancement. AM62D compiles through the **TVM runtime** (`session_name: tvmrt`), offloading supported layers to the C7x DSP; these pipelines use the `*_tvmrt_config.yaml` configs.

## Supported Models

| Model | Task | Dataset | Input Shape | TIDL |
| --- | --- | --- | --- | --- |
| VGGish11 | Sound classification | UrbanSound8K | `(1, 1, 64, 126)` | Yes (FP32→INT8, TVM rt) |
| YAMNet | Sound classification | UrbanSound8K | `(1, 1, 96, 64)` | Yes (FP32→INT8, TVM rt) |
| GTCRN | Speech enhancement | VoiceBank-DEMAND-16k | `(1, 257, T, 2)` | No (ARM CPU) |
| GCRN | Speech enhancement | VoiceBank-DEMAND-16k | `(1, 2, 401, 161)` | Yes (INT16, AM62D) |

## Expected Accuracy

Speech-enhancement metrics below are the **ARM CPU FP32 reference** values (`model_info.metric_reference` in each config); GCRN's on-device 16-bit TIDL accuracy is a feasibility trial and may differ.

| Model | Hardware | Dataset (samples) | Metric | Value |
| --- | --- | --- | --- | --- |
| VGGish11 | TIDL INT8 | UrbanSound8K fold-10 (837) | top1 / top5 / f1_macro | 77.3% / 94.3% / 78.5% |
| YAMNet | TIDL INT8 | UrbanSound8K fold-10 (837) | top1 / top5 / f1_macro | 54.5% / 89.6% / 54.6% |
| GTCRN | CPU FP32 | VoiceBank-DEMAND-16k test (824) | PESQ / STOI / SI-SDR | 2.508 / 0.915 / 15.7 dB |
| GCRN | CPU FP32 | VoiceBank-DEMAND-16k test (824) | PESQ / STOI / SI-SDR | 2.246 / 0.922 / 17.7 dB |

## Prerequisites

The AM62D flow uses a dedicated venv and the RC TVM toolchain. Complete the [AM62D setup](setup_am62d.md) first — it creates the **`tidlrunner-am62d`** venv and `devices/setup_am62d.sh` installs `tidlrunner[pc,audio]` (with the audio extras: `librosa`, `soundfile`, `scipy`, `pesq`, `pystoi`, `scikit-learn`).

Then, once per shell, source the toolchain env:

```bash
pyenv activate tidlrunner-am62d
source devices/am62d_env.sh
```

## Download Datasets

##### UrbanSound8K (~5.6 GB) — sound classification

To download the UrbanSound8K dataset, run the script:

```bash
bash examples/audio/scripts/download_urbansound8k.sh
```

Downloads to: `data/datasets/UrbanSound8K/`

Expected structure:

```
data/datasets/UrbanSound8K/
  audio/
    fold1/ … fold10/
  metadata/
    UrbanSound8K.csv
```

##### VoiceBank-DEMAND-16k (~2 GB) — speech enhancement

To download the VoiceBank-DEMAND-16k dataset, run the script:

```bash
python3 examples/audio/scripts/download_voicebank_demand.py
```

Downloads to: `data/datasets/VoiceBank-DEMAND-16k/`

Expected structure:

```
data/datasets/VoiceBank-DEMAND-16k/
  train/
    clean/   # p226_001.wav, …
    noisy/
  test/
    clean/
    noisy/
```

## Download Models

```bash
bash examples/audio/scripts/download_audio_models.sh
```

Downloads VGGish11, YAMNet, GTCRN, GCRN from TI model zoo.

> Models also auto-download at compile time via `.link` files — the script is optional.

Target layout:

```
data/configs/samples/models/audio/
  audio_classification/
    urbansound8k/
      vggish11.onnx
      yamnet.onnx
  speech_enhancement/
    voicebank_demand_16k/
      gtcrn_dns3.onnx
      gcrn_g16_fixed_4sec.onnx   # base gcrn_fixed_4sec.onnx variant also present
```

## Config File Locations

Each model has a single config file used for all pipelines (compile, infer, evaluate). AM62D uses the `*_tvmrt_config.yaml` variants (GTCRN is ARM-only, so it has no `tvmrt` variant):

```
data/configs/samples/models/audio/
  audio_classification/
    urbansound8k/
      vggish11_tvmrt_config.yaml
      yamnet_tvmrt_config.yaml
  speech_enhancement/
    voicebank_demand_16k/
      gtcrn_dns3_config.yaml
      gcrn_g16_fixed_4sec_tvmrt_config.yaml   # base gcrn_fixed_4sec_tvmrt_config.yaml also present
```

## Running Pipelines

All commands are run from the repo root (`edgeai-tidlrunner/`) with the AM62D env sourced (`source devices/am62d_env.sh`; see Prerequisites). `evaluate` takes `--target_device AM62D`.

##### VGGish11 — Sound Classification

```bash
tidlrunner-cli compile  --config_path data/configs/samples/models/audio/audio_classification/urbansound8k/vggish11_tvmrt_config.yaml
tidlrunner-cli infer    --config_path data/configs/samples/models/audio/audio_classification/urbansound8k/vggish11_tvmrt_config.yaml
tidlrunner-cli evaluate --config_path data/configs/samples/models/audio/audio_classification/urbansound8k/vggish11_tvmrt_config.yaml --target_device AM62D
```

##### YAMNet — Sound Classification

```bash
tidlrunner-cli compile  --config_path data/configs/samples/models/audio/audio_classification/urbansound8k/yamnet_tvmrt_config.yaml
tidlrunner-cli infer    --config_path data/configs/samples/models/audio/audio_classification/urbansound8k/yamnet_tvmrt_config.yaml
tidlrunner-cli evaluate --config_path data/configs/samples/models/audio/audio_classification/urbansound8k/yamnet_tvmrt_config.yaml --target_device AM62D
```

##### GTCRN — Speech Enhancement (ARM CPU)

```bash
tidlrunner-cli compile  --config_path data/configs/samples/models/audio/speech_enhancement/voicebank_demand_16k/gtcrn_dns3_config.yaml
tidlrunner-cli infer    --config_path data/configs/samples/models/audio/speech_enhancement/voicebank_demand_16k/gtcrn_dns3_config.yaml
tidlrunner-cli evaluate --config_path data/configs/samples/models/audio/speech_enhancement/voicebank_demand_16k/gtcrn_dns3_config.yaml --target_device AM62D
```

##### GCRN — Speech Enhancement

```bash
tidlrunner-cli compile  --config_path data/configs/samples/models/audio/speech_enhancement/voicebank_demand_16k/gcrn_g16_fixed_4sec_tvmrt_config.yaml
tidlrunner-cli infer    --config_path data/configs/samples/models/audio/speech_enhancement/voicebank_demand_16k/gcrn_g16_fixed_4sec_tvmrt_config.yaml
tidlrunner-cli evaluate --config_path data/configs/samples/models/audio/speech_enhancement/voicebank_demand_16k/gcrn_g16_fixed_4sec_tvmrt_config.yaml --target_device AM62D
```

## Audio-Specific Settings

The following CLI arguments (and YAML config fields) control audio preprocessing:

| Argument | YAML field | Default | Description |
| --- | --- | --- | --- |
| `--audio_model_type` | `preprocess.audio_model_type` | `null` | Model architecture: `vggish11`, `yamnet`, `gtcrn`, `gcrn` |
| `--sample_rate` | `preprocess.sample_rate` | `16000` | Audio sample rate in Hz |
| `--audio_duration` | `preprocess.audio_duration` | `4.0` | Clip duration in seconds (used by VGGish11 and GCRN; YAMNet/GTCRN ignore this) |

The `audio_model_type` controls which preprocessing transform is used:

| `audio_model_type` | Transform | Output shape |
| --- | --- | --- |
| `vggish11` | `VGGishMelSpectrogram` — HTK mel, n_fft=1024, hop=512 | `(1, 1, 64, 126)` |
| `yamnet` | `YAMNetMelSpectrogram` — HTK mel, n_fft=512, hop=160 | `(1, 1, 96, 64)` |
| `gtcrn` | `STFTTransform` — sqrt-Hann, n_fft=512, hop=256, center=False | `(1, 257, T, 2)` |
| `gcrn` | `GCRNSTFTTransform` — Hamming, n_fft=320, hop=160, center=True | `(1, 2, 401, 161)` |

## TIDL Support Notes

- **VGGish11**: FP32 model; TIDL quantizes to INT8 via the TVM runtime at compile time (`tidl_offload: true`).
- **YAMNet**: FP32 model; TIDL quantizes to INT8 via the TVM runtime at compile time (`tidl_offload: true`).
- **GCRN**: Offloads to the AM62D C7x at 16-bit (`tensor_bits: 16`, `tidl_offload: true`) — an AM62D feasibility flow via the TVM runtime.
- **GTCRN**: Dynamic time axis (`T`) and unsupported operators; runs on ARM Cortex-A via ONNX Runtime (`tidl_offload: false`), so there is no `tvmrt` config.
