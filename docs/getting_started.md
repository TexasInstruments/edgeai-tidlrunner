# Getting Started

Welcome to **edgeai-tidl-runner**! This guide will help you quickly get up and running with model compilation and inference using TIDL (Texas Instruments Deep Learning) tools.

## What is edgeai-tidl-runner?

edgeai-tidl-runner is a Python package that provides an easy-to-use wrapper around TI's TIDL model compilation and runtime tools. It allows you to:

- Compile AI models for TI edge devices (AM68A, AM69A, etc.)
- Run inference with compiled models
- Evaluate model accuracy
- Analyze model performance

The package offers two main interfaces:
- **edgeai_tidlrunner.runner** (high-level interface) - Recommended for beginners
- **edgeai_tidlrunner.rtwrapper** (advanced interface) - For advanced users who need full control

## Installation

### Setup on PC

For basic setup on PC, run:
```bash
./setup_pc.sh
```

### Setup on PC with GPU acceleration

For faster compilation with CUDA GPU support:
```bash
./setup_pc_gpu.sh
```

### Setup on TI EVM

For setup on TI evaluation modules:
```bash
./setup_evm.sh
```

### Download Example Datasets (Optional)

To run the provided examples:
```bash
./example_download_datasets.sh
```

For detailed installation instructions, see the main [README.md](../README.md).

## Quick Start Usage

### 1. Basic Command Line Usage

The simplest way to get started is using the command line interface with just a model path. This will compile your model using random inputs for calibration:

```bash
tidlrunnercli compile --model_path=./path/to/your/model.onnx
```

This basic command will:
- Compile your model for the default target device (AM68A)
- Use random inputs for quantization calibration
- Save compiled artifacts to `./work_dirs/compile/`

**Example with a sample model:**
```bash
tidlrunnercli compile --model_path=./data/examples/models/mobilenet_v2.onnx
```

### 2. Config File Based Usage

For more control over the compilation process, you can use configuration files. This approach allows you to specify datasets, preprocessing, and all other parameters:

```bash
tidlrunnercli compile --config_path ./path/to/config.yaml
```

**Example with provided configs:**
```bash
# Compile multiple models using the aggregate config
tidlrunnercli compile --config_path ./data/models/configs.yaml

# Run inference on compiled models
tidlrunnercli infer --config_path ./data/models/configs.yaml
```

Config files allow you to specify:
- Input datasets for calibration
- Preprocessing parameters
- Target device settings
- Quantization options
- And much more...

## Available Commands

| Command | Description |
|---------|-------------|
| `compile` | Compile models for TI devices |
| `infer` | Run inference with compiled models |
| `accuracy` | Evaluate model accuracy |
| `analyze` | Analyze layer outputs and performance |
| `optimize` | Apply model optimizations |
| `extract` | Extract layers or submodules |
| `report` | Generate compilation reports |

## Getting Help

### Command Line Help
Get help for any command:
```bash
tidlrunnercli --help
tidlrunnercli compile --help
tidlrunnercli infer --help
```

### Documentation Links

- **[Command Line Interface](./commandline_interface.md)** - Detailed command line usage examples
- **[Config File Interface](./configfile_interface.md)** - How to use YAML configuration files
- **[Command Line Arguments](./command_line_arguments.md)** - Complete list of all available arguments
- **[Pythonic Interface](./pythonic_interface.md)** - Using the Python API directly
- **[Custom Models](./custom_models.md)** - Working with your own models
- **[Custom Datasets](./custom_datasets.md)** - Using your own datasets
- **[Runtime Settings](./runtime_settings.md)** - Understanding advanced settings

## Next Steps

1. **Try the basic command** with one of the example models
2. **Explore config files** in `./data/models/vision/` for more advanced usage
3. **Check the examples** in `./examples/vision/scripts/` 
4. **Review the documentation links** above for detailed information

## Need More Help?

- Check the main [README.md](../README.md) for comprehensive information
- Look at the example scripts in the repository
- Review the detailed documentation linked above

Happy compiling! 🚀
