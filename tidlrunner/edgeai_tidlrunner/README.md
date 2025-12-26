# edgeai-tidl-runner

Bring Your Edge AI Models For Compilation, Inference, Analysis or Benchmark (BYOM) using TI Deep Learning (TIDL) - for TI MPU Processors for Edge AI. More details are in the [Edge AI developer landing page](https://github.com/TexasInstruments/edgeai).


### Notice
#### Purpose
The basic functionality in this repository is similar to that of [edgeai-benchmark](https://github.com/TexasInstruments/edgeai-tensorlab/tree/main/edgeai-benchmark). edgeai-benchmark is primarily focused on compiling and benchmarking models in [edgeai-modelzoo](https://github.com/TexasInstruments/edgeai-modelzoo), but the focus of the tool in this repository is a simpler user interface and support for compiling user's models and datasets. This makes it easy to compile and benchmark custom models.

#### Current status and limitations
This is a work in progress with the following status and limitations 
* onnx models and tflite models are supported. onnx models will use onnxruntime with TIDL by default and tflite models will use tidlruntime with TIDL by default. 
* Other runtimes are not yet supported, but there is a work in progress to support tvm runtime for better offloading capability to DSP(C7x/MMA)
* Work in progress to support most of the [models in edgeai-modelzoo/models/configs.yaml](https://github.com/TexasInstruments/edgeai-modelzoo/blob/main/models/configs.yaml). As of now task types such as image classification, detection and segmentation are supported out-of-the-box. 
* The main purpose of this tool is to enable compilation and benchmarking of user's models - this requires support for generic user defined datasets - this is also a work in progress.


## Introduction

This package provides a wrapper over the core [edgeai-tidl-tools runtimes and model compilation](https://github.com/TexasInstruments/edgeai-tidl-tools) to make  model compilation, inference and benchmark easy to use.

### Steps in TIDL model compilation and inference
This package provides all the resources to enables all these steps. This will be installed as **edgeai_tidlrunner** Python package.
* TIDL model compilation is done in a PC (Ubuntu Linux typically). 
* The inference can be verified on PC Host emulation (for checking the correctness of output). 
* Finally the artifacts can be mounted on, or copied to an EVM/device and the actual inference can be done on the EVM/device. 

### Interfaces in edgeai_tidlrunner package

* **edgeai_tidlrunner.runner** (high level interface) - runner has additional pipeline functionalities such as data loaders and preprocess required to run the entire pipeline correctly. This is a high level interface that hides most of the details and provides a Pythonic and command line APIs. (Recommended for beginners)

* **edgeai_tidlrunner.rtwrapper** (advanced interface) - rtwrapper is a thin wrapper over the core OSRT and TIDL-RT runtimes - the wrapper is provided for ease of use and also to make the usage of various runtimes consistent. This is an advanced wrapper does not impose much restrictions on the usage and the full flexibility and functionality of the underlying runtimes are available to the user. 


## Setup tidl-tools

[Setup instructions](docs/setup.md)


## Getting Started

[Getting started instructions and examples](docs/getting_started.md)


## Detailed Documentation

[Custom models and datasets](./custom_models_and_datasets.md): You have tried the off-the-shelf examples provided in this repository and is ready to compile own models and datasets - then look as this section on custom datasets & models

[Advanced usage documentation](docs/usage_advanced.md): Advanced usage (for experts only)

<hr>
