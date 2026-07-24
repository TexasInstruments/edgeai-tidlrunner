# Getting Started

Welcome to the **edgeai-tidlrunner** repository! This guide will help you quickly get up and running with model compilation and inference using TIDL (Texas Instruments Deep Learning) tools.

## What is edgeai-tidlrunner?

edgeai-tidlrunner is a comprehensive toolkit that provides easy-to-use interfaces for compiling AI models to run on TI Processors devices with the C7 NPU accelerator. It supports various operations including model compilation, inference, evaluate evaluation, and performance analysis.

## Interface

`tidlrunner-cli` is the primary commandline tool interface that can be used. It is generally assumed that the `tidlrunner-cli` tool will be run from the root of this repository, but it is not an explicit requirement. Outputs of the tool will be placed into [work_dirs](../../work_dirs) by default.

Note: alternatively, you can choose to run this script with python and provide arguments [tidlrunner/edgeai_tidlrunner/main.py](../edgeai_tidlrunner/main.py)


## Usage - compile & evaluate on PC

There are two primary ways to use this tool:

### 1. Config File Based Usage

For more control and reproducible workflows, you can use configuration files. This approach allows you to specify all parameters including datasets, preprocessing options, target devices, and much more.

**Example:**
```bash
tidlrunner-cli compile --config_path data/configs/samples/models/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv_config.yaml --target_device AM62A
```
```bash
tidlrunner-cli evaluate --config_path data/configs/samples/models/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv_config.yaml --target_device AM62A
```

This approach provides:
- Full control over all compilation parameters
- Ability to specify custom datasets for calibration
- Reproducible configurations
- Support for complex preprocessing pipelines

More details are here: [configfile_interface.md](./configfile_interface.md)

The configuration file can contain any of the fields documented in [command_line_arguments.md](./command_line_arguments.md). The Config Field column in that document shows exactly which fields can be populated in the YAML configuration file.


### 2. Direct Command Line Usage

*Note: This direct command line usage is useful for quick compilation - to check whether compilation is working or not. This is not our recommended method of usage for actual compilation and inference as it is easy to miss some required arguments (in that case random inputs may get used for quantization calibration and the outputs may not usable). Because of this, the above configfile based interface is our recommended method of usage.*

The simplest way to get started is by providing only the model path. This approach uses random inputs for calibration, making it perfect for quick testing and evaluation.

**Example:**
```bash
tidlrunner-cli compile --model_path data/configs/samples/models/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv.onnx  --target_device AM62A
```
```bash
tidlrunner-cli infer --model_path data/configs/samples/models/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv.onnx  --target_device AM62A
```

This command will:
- Compile the model for the default target device (AM62A)
- **Uses random inputs for quantization calibration**
- Save the compiled artifacts to the default output directory

Because this uses random inputs by default, it **may not produce correct outputs during inference**. To be able to generate correct outputs, actual data has to be used by specifying dataloader arguments - eg: data_name, data_path.

For complete list of commands and arguments, see [command_line_arguments.md](./command_line_arguments.md) or use:

```bash
tidlrunner-cli --help
```

More details on the commandline interface is here: [commandline_interface.md](./commandline_interface.md)

## Usage - actual infer or evaluate on EVM
To run the compiled model artifacts on EVM, follow these instructions:
[running_on_evm](./running_on_evm.md)


## Compiling models for a specific device
It is important to use the correct target device while compiling the model. By default, this tool assumes AM62A, but that may not be the device/EVM that you have. 

List of devices supported by TIDL are listed in the page [Supported Devices & SDKs](https://github.com/TexasInstruments/edgeai/blob/main/edgeai-mpu/readme_sdk.md).

All those devices are supported by this tool. A specific device can be specified using the option target_device. For example:

```bash
tidlrunner-cli compile --config_path data/configs/samples/models/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv_config.yaml --target_device TDA4VH
```

If you need more details, please refer to [this script that downloads tidl_tools](../../tools/tidl_tools_package/download.py)


## Getting Help

For basic help, use:
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

For other types of help, check the [FAQ](./faq.md) and the other docs within this repo. For bugs or issues, create an Issue on this repository or create a support ticket on TI's [e2e forums](https://e2e.ti.com/support/processors-group/processors/f/processors-forum)
