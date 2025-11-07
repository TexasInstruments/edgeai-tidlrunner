# Getting Started

Welcome to the **edgeai-tidlrunner** repository! This guide will help you quickly get up and running with model compilation and inference using TIDL (Texas Instruments Deep Learning) tools.

## What is edgeai-tidlrunner?

edgeai-tidlrunner is a comprehensive toolkit that provides easy-to-use interfaces for compiling AI models to run on TI edge devices. It supports various operations including model compilation, inference, accuracy evaluation, and performance analysis.

## Usage

There are two primary ways to use this repository:

### 1. Basic Command Line Usage

The simplest way to get started is by providing only the model path. This approach uses random inputs for calibration, making it perfect for quick testing and evaluation.

**Example:**
```bash
tidlrunner-cli compile --model_path data/models/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv.onnx
```

This command will:
- Compile the model for the default target device (AM68A)
- Use random inputs for quantization calibration
- Save the compiled artifacts to the default output directory

For a complete list of available command line arguments, see [command_line_arguments.md](./command_line_arguments.md).

### 2. Config File Based Usage

For more control and reproducible workflows, you can use configuration files. This approach allows you to specify all parameters including datasets, preprocessing options, target devices, and more.

**Example:**
```bash
tidlrunner-cli compile --config_path data/models/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv_config.yaml
```

This approach provides:
- Full control over all compilation parameters
- Ability to specify custom datasets for calibration
- Reproducible configurations
- Support for complex preprocessing pipelines

The configuration file can contain any of the fields documented in [command_line_arguments.md](./command_line_arguments.md). The Config Field column in that document shows exactly which fields can be populated in the YAML configuration file.

## Available Commands

The repository supports several commands for different workflows:

- **compile** - Compile models for TI edge devices
- **infer** - Run inference with compiled models
- **accuracy** - Evaluate model accuracy against ground truth
- **analyze** - Analyze model performance and layer outputs
- **optimize** - Apply model optimization techniques
- **extract** - Extract specific layers or submodules
- **report** - Generate detailed compilation reports

## Documentation

For detailed information about each command and its parameters, refer to:

- **[Command Line Arguments](./command_line_arguments.md)** - Complete reference for all available arguments and configuration fields
- **[Command Line Interface](./commandline_interface.md)** - Detailed usage examples
- **[Config File Interface](./configfile_interface.md)** - How to create and use YAML configuration files
- **[Usage](./usage.md)** - General usage patterns and workflows

## Getting Help

For command-specific help, use:
```bash
tidlrunner-cli --help
tidlrunner-cli <command> --help
```

This will show you all available options for each specific command.
