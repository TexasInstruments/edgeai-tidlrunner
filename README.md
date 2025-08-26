# edgeai-tidl-runner

This package provides edgeai-tidl-runner which is a wrapper over the core TIDL model compilation and runtimes. This wrapper makes the TIDL model compilation and inference interface easy to use. 

This will be installed as **edgeai_tidlrunner** Python package.

edgeai_tidlrunner package has two parts:
* **edgeai_tidlrunner.runner** (high level interface - recommended) - runner has additional pipleline functionalities such as data loaders and preprocess required to run the entire pipeline correctly. This is a high level interface that hides most of the details and provides a Pythonic and command line APIs.
* **edgeai_tidlrunner.rtwrapper** (advanced interface) - rtwrapper is a thin wrapper over the core OSRT and TIDL-RT runtimes - the wrapper is provided for ease of use and also to make the usage of various runtimes consistent. This low level wrapper does not impose mush restrictions on the usage and the full flexibility and functionality of the underlying runtimes are available to the user. 

<hr>
<hr>

## Setup

### To setup on PC, run:

```
./setup_pc.sh
```

This will download the tidl_tools in the [tools](./tools) folder. The runtimes require TIDL_TOOLS_PATH and LD_LIBRARY_PATH to be set to appropriate folder inside this folder. For more details see [set_env.sh](./set_env.sh)

### Setup on PC with gpu based tidl-tools (faster to run, but has more dependencies)

Running with CUDA GPU has dependencies - the details of dependencies are in the file [setup_pc_gpu.sh](./setup_pc_gpu.sh)

Example:
```
./setup_pc_gpu.sh
```

This script installs the CUDA based tidl-tools and nvidia-hpc-sdk. The user ha to make sure the system has CUDA gpus appropriate nvidia graphics drivers. 

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

<hr>
<hr>

## Usage of runner (edgeai_tidlrunner.runner interface)

runner is a basic interface which hides most of the complexity of the underlying runtimes. It can be used either from Python script or from command line.


### See the options supported with help command
All the options supported can be obtained using the help option. Examples
```
tidlrunnercli --help
```

Detailed help is available for each command - for example:
```
tidlrunnercli compile --help
```
```
tidlrunnercli infer --help
```

<hr>

### tidlrunnercli Commandline interface
The commandline interface allows to provide the model and a few arguments dirctly in the commandline.
[runner Commandline interface](./docs/commandline_interface.md)

<hr>

### tidlrunnercli Configfile interface
The configfile interface allows to parse all parameters from a yaml file. 
[runner Commandline config file interface](./docs/configfile_interface.md)

<hr>

### edgeai_tidlrunner.runner Pythonic interface
There is also a Pythonic interface for the runner module, for more flexibility.
[runner Pythonic interface](./docs/pythonic_interface.md)

<hr>

### List of commands supported
| Command          | Internal Pipeline(s)        | Description                                                               |
|------------------|-----------------------------|---------------------------------------------------------------------------|
| compile          | CompileModel                | Compile the given model(s)                                                |
| infer            | InferModel                  | Run inference using using already compiled model artifacts                |
| accuracy         | InferAnalyze                | Analyze complied artifacts, run inference and analyze layerwise deviations|
| optimize         | OptimizeModel               | Optimize - simpifier, layer optimizations, shape inference (included in compile)|
| compile+infer    | CompileModel, InferModel    | compile, infer                                                            |
| compile+analyze  | CompileModel, InferAnalyze  | Compile the model, infer and analyze                                      |
| compile+accuracy | CompileModel, InferAccuracy | Compile the model, infer and compute accuracy                             |
| analyze          | <multiple pipelines>        | Analyze layer oiutputs, compare them to onnxruntime and write statistics  |
| report           | GenReport                   | Generate overall csv report of infer or accuracy                          |
| extract          | ExtractNodes                | Extract layers or submodules from a model                                 |



But as explained above, the easiest way to see list of options supported for a command is to use the help - for example:
```
tidlrunnercli compile --help
```


<hr>

## Using custom datasets & models
You have tried the off-the-shelf examples provided in this repository and is ready to compile own models and datasets - then look as this section on custom datasets & models: 
* [Custom models](./docs/custom_models.md)
* [Custom datasets](./docs/custom_datasets.md)

<hr>
<hr>

## Settings/Options Deep dive

[More details of settings](./docs/runtime_settings.md)

<hr>
<hr>

## Usage of rtwrapper (edgeai_tidlrunner.rtwrapper advanced interface)
Abstractions are sometimes a hindrance to understand what is really happening under the hood or to easily modify and extend. rtwrapper is a thin, low level interface to the core tidl-tools, without much overhead. Use it to understand how the core tidl-tools work or to integrate into your application.

[rtwrapper advanced interface](./docs/rtwrapper_interface.md)

<hr>
<hr>
