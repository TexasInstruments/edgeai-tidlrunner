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

Because this uses random inputs by default, it may not produce good outputs while inference. To be able to generate correct outputs, we have to use actual data by specifying dataloader arguments - eg: data_name, data_path.

More details are here: [commandline_interface.md](./commandline_interface.md)

For complete list of available command line arguments, see [command_line_arguments.md](./command_line_arguments.md).

### 2. Config File Based Usage

For more control and reproducible workflows, you can use configuration files. This approach allows you to specify all parameters including datasets, preprocessing options, target devices, and much more.

**Example:**
```bash
tidlrunner-cli compile --config_path data/models/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv_config.yaml
```

This approach provides:
- Full control over all compilation parameters
- Ability to specify custom datasets for calibration
- Reproducible configurations
- Support for complex preprocessing pipelines

More details are here: [configfile_interface.md](./configfile_interface.md)

The configuration file can contain any of the fields documented in [command_line_arguments.md](./command_line_arguments.md). The Config Field column in that document shows exactly which fields can be populated in the YAML configuration file.


## List of commands supported

| Command          | Description                                                               |
|------------------|---------------------------------------------------------------------------|
| compile          | Compile the given model(s)                                                |
| infer            | Run inference using using already compiled model artifacts                |
| accuracy         | Analyze compiled artifacts, run inference and analyze layerwise deviations|
| compile+infer    | compile the model and run inference                                       |
| compile+analyze  | Compile the model and analyze the outputs of different layers             |
| compile+accuracy | Compile the model, run inference and compute accuracy                     |
| analyze          | Analyze TIDL layer outputs, compare them to onnxruntime outputs and write statistics - can be used to identify layer level issues |
| report           | Generate overall csv report of infer or accuracy                          |
| surgery          | Perform model surgery - simplifier, layer optimizations, shape inference (included in compile)|
| extract          | Extract layers or submodules from a model                                 |


<hr>


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
