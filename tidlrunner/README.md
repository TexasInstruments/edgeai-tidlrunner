# Introduction

Welcome to the **edgeai-tidlrunner** repository! This guide will help you quickly get up and running with model compilation and inference using TIDL (Texas Instruments Deep Learning) tools.

## What is edgeai-tidlrunner?

edgeai-tidlrunner is a comprehensive toolkit that provides easy-to-use interfaces for compiling AI models to run on TI Processors devices with the C7 NPU accelerator. It supports various operations including model compilation, inference, evaluate evaluation, and performance analysis.

This package provides a wrapper over the core [edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools) to make  model compilation,  inference, accuracy & performance benchmark easy to use.

The off-the-shelf config files in [data/configs/modelzoo](../data/configs/modelzoo) uses models in [edgeai-modelzoo](https://github.com/TexasInstruments/edgeai-tensorlab/blob/main/edgeai-modelzoo) - but these are just examples. The focus in this repository is a simpler user interface and support for compiling user's models and datasets. 

### Steps in TIDL model compilation and inference
This package provides all the resources to enables all these steps. This will be installed as **edgeai_tidlrunner** Python package.
* TIDL model compilation is done in a PC (Ubuntu Linux typically). 
* The inference can be verified on PC Host emulation (for checking the correctness of output). 
* Finally the artifacts can be mounted on, or copied to an EVM/device and the actual inference can be done on the EVM/device. 

### Interfaces in edgeai_tidlrunner package

* **edgeai_tidlrunner.runner** (high level interface) - runner has additional pipeline functionalities such as data loaders and preprocess required to run the entire pipeline correctly. This is a high level interface that hides most of the details and provides a Pythonic and command line APIs. (Recommended)

* **edgeai_tidlrunner.rtwrapper** (advanced interface) - rtwrapper is a thin wrapper over the core OSRT and TIDL-RT runtimes - the wrapper is provided for ease of use and also to make the usage of various runtimes consistent. This is an advanced wrapper does not impose much restrictions on the usage and the full flexibility and functionality of the underlying runtimes are available to the user. 


# Setup & preparation instructions

## Setup

[Setup instructions](docs/setup.md)

Note: The environment variable **TIDL_TOOLS_VERSION** defined in [setup_runner_pc.sh](../setup_runner_pc.sh) determines the version of tidl_tools downloaded and installed - when doing the setup using [setup_runner_pc.sh](../setup_runner_pc.sh) or [setup_runner_pc_gpu.sh](../setup_runner_pc_gpu.sh). Change the value of this variable (if needed) and run setup to download and install the required version of tidl_tools.

## Preparation - download datasets to run examples in this repository. 
Model compilation can be run using random data - if the intention is just to measure latency / FPS. However, to actually check the correctness of output / accuracy, actual data is required.

To run example models in this repository with actual data, download example datasets:
```
./examples/example_download_datasets.sh
```

# Usage documentation

## User guide

This section has details on getting started, usage instructions (including direct command line usage and config file based usage) and examples. See the [user guide](docs/user_guide.md)

## Quick start with benchmark sripts

Easy to use benchmark script wrappers are in [../scripts](../scripts/) to try out and benchmark example configs in this repository. Use these wrapper scripts for preset-based runs using the [configs for models in the modelzoo](../data/configs/modelzoo/).

The benchmark wrappers under `scripts/run_benchmark_*.sh` are thin entry points that call internal helpers (`scripts/_run_benchmark_pc.sh` and `scripts/_run_benchmark_evm.sh`) with preset arguments.

## Model Inspector

An interactive HTML visualization tool for analyzing ONNX models compiled with TIDL. Model Inspector provides comprehensive insights into model structure, performance, accuracy, and hardware acceleration. A modelinspector.html file is generated when you invoke compile, infer or inspect commands in tidlrunner.

[Model Inspector](./edgeai_tidlrunner/modelinspector/README.md)

<hr>
