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

## Basic Documentation

### List of commands supported
| Command          | Description                                                               |
|------------------|---------------------------------------------------------------------------|
| compile          | Compile the given model(s)                                                |
| infer            | Run inference using using already compiled model artifacts                |
| accuracy         | Analyze compiled artifacts, run inference and analyze layerwise deviations|
| optimize         | Optimize - simplifier, layer optimizations, shape inference (included in compile)|
| analyze          | Analyze layer outputs, compare them to onnxruntime and write statistics  |
| report           | Generate overall csv report of infer or accuracy                          |
| extract          | Extract layers or submodules from a model                                 |
| compile+infer    | compile the model and run inference                                       |
| compile+analyze  | Compile the model and analyze the outputs of different layers             |
| compile+accuracy | Compile the model, run inference and compute accuracy                     |


<hr>

### Basic interface - tidlrunner-cli Commandline interface
The commandline interface allows to provide the model and a few arguments directly in the commandline.
[runner Commandline interface](./commandline_interface.md)

The commandline options supported for each command are listed [here](./command_line_arguments.md)

<hr>

### Basic interface - tidlrunner-cli Configfile interface
The configfile interface allows to parse all parameters from a yaml file. 
[runner Commandline config file interface](./configfile_interface.md)

<hr>

### Report generation after model compilation

A consolidated csv report will be generated with the report command.
```
tidlrunner-cli report
```

## Detailed Documentation

For detailed information about each command and its parameters, refer to:

- **[Command Line Arguments](./command_line_arguments.md)** - Complete reference for all available arguments and configuration fields
- **[Command Line Interface](./commandline_interface.md)** - Detailed usage examples
- **[Config File Interface](./configfile_interface.md)** - How to create and use YAML configuration files

## Getting Help

For command-specific help, use:
```bash
tidlrunner-cli --help
```

Detailed help is available for each command - for example:
```
tidlrunner-cli <command> --help
```

This will show all the available options for each specific command.

For example:
```
tidlrunner-cli compile --help
```
```
tidlrunner-cli infer --help
```