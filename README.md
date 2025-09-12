# edgeai-tidl-runner

Bring Your Edge AI Models For Compilation and Inference (BYOM) using TI Deep Learning (TIDL) - for TI MPU Processors for Edge AI. More details are in the [Edge AI developer landing page](https://github.com/TexasInstruments/edgeai).

## Introduction

This package provides a wrapper over the core [tidl-tools model compilation and runtimes](https://github.com/TexasInstruments/edgeai-tidl-tools) to make  model compilation and inference interface easy to use.

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


## Download datasets to run examples in this repository (optional)

Model compilation can be run using random data - if the intention is just to measure latency / FPS. However, to actually check the correctness of output / accuracy, actual data is required.

To run example models in this repository with actual data, download example datasets:
```
./example_download_datasets.sh
```


## Getting Started

[Getting started instructions and examples](docs/getting_started.md)


## Detailed Documentation

[Usage documentation](docs/usage_basic.md): Basic usage 

[Custom models and datasets](./custom_models_and_datasets.md): You have tried the off-the-shelf examples provided in this repository and is ready to compile own models and datasets - then look as this section on custom datasets & models

[Advanced usage documentation](docs/usage_advanced.md): Advanced usage (for experts)

<hr>
