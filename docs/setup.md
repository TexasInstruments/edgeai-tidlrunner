
## Setup

### To setup on PC, run:

```
./setup_pc.sh
```

This will download the tidl_tools in the [tools](../tools) folder. The runtimes require TIDL_TOOLS_PATH and LD_LIBRARY_PATH to be set to appropriate folder inside this folder. For more details see [set_env.sh](../set_env.sh)

### Setup on PC with gpu based tidl-tools (faster to run, but has more dependencies)

Running with CUDA GPU has dependencies - the details of dependencies are in the file [setup_pc_gpu.sh](../setup_pc_gpu.sh)

Example:
```
./setup_pc_gpu.sh
```

This script installs the CUDA based tidl-tools and nvidia-hpc-sdk. The user has to make sure the system has CUDA gpus appropriate nvidia graphics drivers. 

### To setup on EVM, run:

```
./setup_evm.sh
```

<hr>

### Download datasets to run examples in this repository (optional)
To run example models in this repository, download example datasets:
```
./example_download_datasets.sh
```
