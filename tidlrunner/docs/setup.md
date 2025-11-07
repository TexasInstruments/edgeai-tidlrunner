
## Setup

Model compilation is done on on PC (Ubuntu Linux). Scripts starting with setup_pc are to prepare and install the dependencies on PC. Scripts starting with scripts_evm are for installing dependences on EVM/device.

### To setup on PC, run:

```
./setup_runner_pc.sh
```

This will download the tidl_tools in the [tools](../tools) folder. 


### Setup on PC with gpu based tidl-tools (faster to run, but has more dependencies)

Running with CUDA GPU has dependencies - the details of dependencies are in the file [setup_pc_gpu.sh](../setup_pc_gpu.sh)

Example:
```
./setup_runner_pc_gpu.sh
```

This script installs the CUDA based tidl-tools and nvidia-hpc-sdk. The user has to make sure the system has CUDA gpus appropriate nvidia graphics drivers. 

### Changing the tidl-tools version
Version of tidl-tools can be specified during the setup process.

```
TIDL_TOOLS_VERSION="11.1" ./setup_runner_pc_gpu.sh
```

```
TIDL_TOOLS_VERSION="11.0" ./setup_runner_pc_gpu.sh
```

**Important Note**: The version of tidl-tools that is installed will be used for model compilation. The version of tidl-tools used for compiling a generating model artifacts has to match with the version on the EVM/device. Other the model artifacts will not run on the device.

### Environment variables (for information only)
* tidl-tools require TIDL_TOOLS_PATH and LD_LIBRARY_PATH to be set to appropriate folder.  For example: tools/tidl_tools_package/<target_device>/tidl_tools. 
* This is automatically taken care [restart_with_proper_environment in rtwrapper here](edgeai_tidlrunner/rtwrapper/set_env.py). See how it is used in [main.py](edgeai_tidlrunner/main.py)


### To setup on EVM
Run this on the EVM to setup on the EVM

```
./setup_runner_evm.sh
```


### Download datasets to run examples in this repository (optional)

Model compilation can be run using random data - if the intention is just to measure latency / FPS. However, to actually check the correctness of output / accuracy, actual data is required.

To run example models in this repository with actual data, download example datasets:
```
./examples/example_download_datasets.sh
```
