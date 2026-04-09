
## Setup

Model compilation is done on an x86 PC (Ubuntu Linux recommended). Scripts starting with setup_pc are to prepare and install the dependencies on PC. Scripts starting with scripts_evm are for installing dependencies on EVM/device.

### Python environment
We recommend to create a new Python environment for with tidlrunner in the Python environment name. (tidlrunner or my-tidlrunner or similar -- the name "tidlrunner" is recommended in the Python environment name to avoid confusion with tidloptimizer, which has a different set of requirements and needs a different environment) 

We also recommend to use Python 3.10 as of now as the tidl-tools used for model compilation on PC are compatible with that version of Python.

For simplicity, these instructions assume that you are using pyenv Python environment manager on Linux OS with bash shell. Any Python environment manager could be used, like venv or conda, but we have tested these scripts with pyenv Python environment manager. If you would like to use pyenv, please find the instructions here: https://github.com/pyenv/pyenv

Once pyenv is installed and your .bashrc is configured to use it, make sure Python 3.10 is installed.
```
pyenv install 3.10
```

Create a virtual environment and activate it:
```
pyenv virtualenv 3.10 tidlrunner
pyenv activate tidlrunner
```

### To setup on PC, run:

```
./setup_runner_pc.sh
```

This will download the tidl_tools in the [tools](../../tools) folder as part of the tidl_tools_package. The actual device-specific tools will be held in the corresponding [bin directory](../../tools/tidl_tools_package/bin/). 


### Setup on PC with gpu based tidl-tools (faster to run, but has more dependencies)

Running with CUDA GPU has dependencies - the details of dependencies are in the file [setup_runner_pc_gpu.sh](../../setup_runner_pc_gpu.sh)

Example:
```
./setup_runner_pc_gpu.sh
```

This script installs the CUDA based tidl-tools and nvidia-hpc-sdk. It is up to the user ha to make sure the system has a CUDA-compatible GPU with appropriate Nvidia graphics drivers. 

### Changing the tidl-tools version
The version of tidl-tools can be specified in setup_runner_pc.sh - open this file and change the line that specifies TIDL_TOOLS_VERSION on top. Tt may also be specified from commandline

```
TIDL_TOOLS_VERSION="11.2" ./setup_runner_pc.sh
```

OR for gpu based tidl-tools:
```
TIDL_TOOLS_VERSION="11.2" ./setup_runner_pc_gpu.sh
```


**Important Note**: The version of tidl-tools that is installed will be used for model compilation. The version of tidl-tools used for compiling a generating model artifacts has to match with the version on the EVM/device.  Model artifacts compiled for another SDK will not run on the device. Please also note that artifacts are specific to the target device and will not run on a different device. 

### Environment variables (for information only)
* tidl-tools require TIDL_TOOLS_PATH and LD_LIBRARY_PATH to be set to appropriate folder.  For example: tools/tidl_tools_package/bin/<target_device>/tidl_tools. 
* This is automatically taken care of by [`restart_with_proper_environment` in rtwrapper here](../edgeai_tidlrunner/rtwrapper/set_env.py). See how it is used in [main.py](../edgeai_tidlrunner/main.py)


### To setup on EVM
Run this on the EVM to setup on the EVM

```
./setup_runner_evm.sh
```



