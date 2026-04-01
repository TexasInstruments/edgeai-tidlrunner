# Audio Models and Datasets

This guide covers how to run audio model pipelines in **edgeai-tidlrunner**: sound classification
and speech enhancement.

## Supported Models

| Model | Task | Dataset | Input Shape | TIDL |
|-------|------|---------|-------------|------|
| VGGish11 | Sound classification | UrbanSound8K | `(1, 1, 64, 126)` | Yes (FP32→INT8) |
| YAMNet | Sound classification | UrbanSound8K | `(1, 1, 96, 64)` | Yes (INT8 QAT) |
| GTCRN | Speech enhancement | VoiceBank-DEMAND-16k | `(1, 257, T, 2)` | No (ARM CPU) |
| GCRN | Speech enhancement | VoiceBank-DEMAND-16k | `(1, 2, 401, 161)` | No (ARM CPU) |

## Expected Accuracy

| Model | Hardware | Dataset (samples) | Metric | Value |
|-------|----------|-------------------|--------|-------|
| VGGish11 | TIDL INT8 | UrbanSound8K fold-10 (837) | top1 / top5 / f1_macro | 77% / 94% / 79% |
| YAMNet | TIDL INT8 | UrbanSound8K fold-10 (837) | top1 / top5 / f1_macro | 54% / 90% / 54% |
| GTCRN | CPU FP32 | VoiceBank-DEMAND-16k test (824) | PESQ / STOI / SI-SDR | 2.508 / 0.915 / 15.7 dB |
| GCRN | CPU FP32 | VoiceBank-DEMAND-16k test (824) | PESQ / STOI / SI-SDR | 2.246 / 0.922 / 17.7 dB |

## Prerequisites

### 1. Activate the virtual environment

```bash
pyenv activate tidlrunner
```

### 2. Install audio dependencies

```bash
cd tidlrunner
pip install -e ".[audio]"
```

This installs: `librosa`, `soundfile`, `scipy`, `pesq`, `pystoi`, `scikit-learn`

## Download Datasets

### UrbanSound8K (~5.6 GB) — sound classification

UrbanSound8K requires registration. Run the script and paste the download URL when prompted:

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

### VoiceBank-DEMAND-16k (~2 GB) — speech enhancement

Install the HuggingFace datasets library first (not included in pyproject.toml):

```bash
pip install datasets
```

Then download:

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

Downloads VGGish11 (~10.4 MB), YAMNet (~14.3 MB), GTCRN (~286 KB) from TI model zoo.

> Models also auto-download at compile time via `.link` files — the script is optional.

Target layout:
```
data/models/audio/
  sound_classification/
    urbansound8k/
      vggish11_20250324-1807.onnx
      yamnet_combined.onnx
  speech_enhancement/
    voicebank_demand_16k/
      gtcrn_dns3.onnx
      gcrn_fixed_4sec.onnx
```

## Config File Locations

All config files live under `data/models/audio/`:

```
data/models/audio/
  sound_classification/
    urbansound8k/
      vggish11_compile.yaml
      vggish11_infer.yaml
      vggish11_accuracy.yaml
      yamnet_compile.yaml
      yamnet_infer.yaml
      yamnet_accuracy.yaml
  speech_enhancement/
    voicebank_demand_16k/
      gtcrn_compile.yaml
      gtcrn_infer.yaml
      gtcrn_accuracy.yaml
      gcrn_compile.yaml
      gcrn_infer.yaml
      gcrn_accuracy.yaml
```

## Running Pipelines

All commands are run from the repo root (`edgeai-tidlrunner/`).

### VGGish11 — Sound Classification

```bash
# Compile (quantize FP32 model for AM62A)
tidlrunner-cli compile --config_path data/models/audio/sound_classification/urbansound8k/vggish11_compile.yaml

# Infer (run on fold-10 test data)
tidlrunner-cli infer --config_path data/models/audio/sound_classification/urbansound8k/vggish11_infer.yaml

# Compile + evaluate accuracy in one step
tidlrunner-cli compile+evaluate --config_path data/models/audio/sound_classification/urbansound8k/vggish11_accuracy.yaml
```

### YAMNet — Sound Classification

```bash
# Compile (INT8 QAT model for AM62A)
tidlrunner-cli compile --config_path data/models/audio/sound_classification/urbansound8k/yamnet_compile.yaml

# Infer
tidlrunner-cli infer --config_path data/models/audio/sound_classification/urbansound8k/yamnet_infer.yaml

# Compile + evaluate accuracy
tidlrunner-cli compile+evaluate --config_path data/models/audio/sound_classification/urbansound8k/yamnet_accuracy.yaml
```

### GTCRN — Speech Enhancement

```bash
# Compile (ARM-only; TIDL offload disabled)
tidlrunner-cli compile --config_path data/models/audio/speech_enhancement/voicebank_demand_16k/gtcrn_compile.yaml

# Infer
tidlrunner-cli infer --config_path data/models/audio/speech_enhancement/voicebank_demand_16k/gtcrn_infer.yaml

# Compile + evaluate accuracy
tidlrunner-cli compile+evaluate --config_path data/models/audio/speech_enhancement/voicebank_demand_16k/gtcrn_accuracy.yaml
```

### GCRN — Speech Enhancement

```bash
# Compile (ARM-only; TIDL offload disabled)
tidlrunner-cli compile --config_path data/models/audio/speech_enhancement/voicebank_demand_16k/gcrn_compile.yaml

# Infer
tidlrunner-cli infer --config_path data/models/audio/speech_enhancement/voicebank_demand_16k/gcrn_infer.yaml

# Compile + evaluate accuracy
tidlrunner-cli compile+evaluate --config_path data/models/audio/speech_enhancement/voicebank_demand_16k/gcrn_accuracy.yaml
```

## Audio-Specific Settings

The following CLI arguments (and YAML config fields) control audio preprocessing:

| Argument | YAML field | Default | Description |
|----------|-----------|---------|-------------|
| `--audio_model_type` | `preprocess.audio_model_type` | `null` | Model architecture: `vggish11`, `yamnet`, `gtcrn`, `gcrn` |
| `--sample_rate` | `preprocess.sample_rate` | `16000` | Audio sample rate in Hz |
| `--audio_duration` | `preprocess.audio_duration` | `4.0` | Clip duration in seconds (used by VGGish11 and GCRN; YAMNet/GTCRN ignore this) |

The `audio_model_type` controls which preprocessing transform is used:

| `audio_model_type` | Transform | Output shape |
|--------------------|-----------|-------------|
| `vggish11` | `VGGishMelSpectrogram` — HTK mel, n_fft=1024, hop=512 | `(1, 1, 64, 126)` |
| `yamnet` | `YAMNetMelSpectrogram` — HTK mel, n_fft=512, hop=160 | `(1, 1, 96, 64)` |
| `gtcrn` | `STFTTransform` — sqrt-Hann, n_fft=512, hop=256, center=False | `(1, 257, T, 2)` |
| `gcrn` | `GCRNSTFTTransform` — Hamming, n_fft=320, hop=160, center=True | `(1, 2, 401, 161)` |

## TIDL Support Notes

- **VGGish11**: FP32 model compiles to TIDL INT8 (`tidl_offload: true`). The PTQ opset-17 variant is not supported by TIDL tools 11.02.04.00.
- **YAMNet**: QAT INT8 model, fully supported by TIDL (`tidl_offload: true`).
- **GTCRN**: Dynamic time axis (`T`) is not supported by TIDL; runs on ARM Cortex-A via ONNX Runtime (`tidl_offload: false`).
- **GCRN**: Not yet supported by TIDL; runs on ARM Cortex-A via ONNX Runtime (`tidl_offload: false`).
