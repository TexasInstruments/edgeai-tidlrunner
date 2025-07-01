# edgeai-tidl-runner

This package provides edgeai-tidl-runner which is a wrapper over the core TIDL and OSRT runtimes - provide functionality to easily compile and infer using TIDL runtimes. 

This wrapper makes the tidl model compilation and inference interface easy to use. This will be installed as **edgeai_tidlrunner** Python package.

**edgeai_tidlrunner package has three parts**:
* **edgeai_tidlrunner.rtwrapper** - rtwrapper is a thin wrapper over the core OSRT and TIDL-RT runtimes - the wrapper is provided for ease of use and also to make the usage of various runtimes consistent. This low level wrapper does not impose mush restrictions on the usage and the full flexibility and functionality of the underlying runtimes are available to the user. 
* **edgeai_tidlrunner.runner** - runner has additional pipleline functionalities such as data loaders and preprocess required to run the entire pipeline correctly. This is a high level interface that hides most of the details and provides a Pythonic and command line APIs.
* **edgeai_tidlrunner.tools** - this is where the TIDL binaries are installed. The runtimes require TIDL_TOOLS_PATH and LD_LIBRARY_PATH to be set to appropriate folder inside this folder. For more details see [set_env.sh](./set_env.sh)

<hr>

## Setup

To setup on PC, run:

```
./setup_pc.sh
```

Setup with gpu based tidl_tools (faster to run)

Example:
```
./setup_pc_gpu.sh
```

<hr>

## Usage of runner (basic interface)

runner is a basic interface which hides most of the complexity of the underlying runtimes. It can be used either from Python script or from command line.

### tidlrunner-cli commandline interface

tidl-run is the interface script to run model compilation and inference via commandline:

```
tidlrunner-cli import_model --model_path=./data/examples/models/mobilenet_v2.onnx
```

More options can be specified to configure the run:

```
tidlrunner-cli compile_model --model_path=./data/examples/models/mobilenet_v2.onnx --data_path=./data/datasets/vision/imagenetv2c/val
```

See the commandline example in [example_runner_cli.sh](./example_runner_cli.sh)

The options that can be provided can be obtained using the help option. Examples
```
tidlrunner-cli --help
tidlrunner-cli compile_model --help
```

The options that can be passed can also be seen in this file [settings_default.py](./edgeai_tidlrunner/runner/modules/vision/settings/settings_default.py)
Note: The options can be provided either using the short option that is the key or using the long option that is given against 'dest'

### edgeai_tidlrunner Pythonic interface

edgeai_tidlrunner.runner.run is the Pythonic API of runner

```
kwargs = {
    'session': {
        'model_path': ./data/examples/models/mobilenet_v2.onnx',
    }
    'dataloader': {
        'path': ./data/datasets/vision/imagenetv2c/val',
     }
}

edgeai_tidlrunner.runner.run('compile_model', **kwargs)
```

See the Pythonic example in [example_runner_py.py](./examples/vision/scripts/example_runner_py.py) which is invoked via [example_runner_py.sh](./example_runner_py.sh)

The options that can be passed as **kwargs can be seen in this file [settings_default.py](./edgeai_tidlrunner/runner/modules/vision/settings/settings_default.py)
Note: that this file lists the options in a flat syntax using '.' separator for the fields - it is possible to use the '.' syntax or the nested dict as shown above. 

<hr>

## Usage of runtimes wrappers (advanced interface)

The runtime wrappers [edgeai_tidlrunner/rtwrapper](edgeai_tidlrunner/rtwrapper) provides an advanced low level interface beyond what the runner provides. 
An example of this is in [example_advanced_rtwrapper.py](./examples/vision/scripts/example_advanced_rtwrapper.py)

<hr>
