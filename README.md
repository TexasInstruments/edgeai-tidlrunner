# edgeai-tidl-runner

This package provides edgeai-tidl-runner which is a wrapper over the core TIDL model compilation and runtimes. This wrapper makes the TIDL model compilation and inference interface easy to use. 

This will be installed as **edgeai_tidlrunner** Python package.

edgeai_tidlrunner package has two parts:
* **edgeai_tidlrunner.runner** (high level interface - recommended) - runner has additional pipleline functionalities such as data loaders and preprocess required to run the entire pipeline correctly. This is a high level interface that hides most of the details and provides a Pythonic and command line APIs.
* **edgeai_tidlrunner.rtwrapper** (advanced interface) - rtwrapper is a thin wrapper over the core OSRT and TIDL-RT runtimes - the wrapper is provided for ease of use and also to make the usage of various runtimes consistent. This low level wrapper does not impose mush restrictions on the usage and the full flexibility and functionality of the underlying runtimes are available to the user. 

<hr>

## Setup

### To setup on PC, run:

```
./setup_pc.sh
```

This will download the tidl_tools in the [tools](./tools) folder. The runtimes require TIDL_TOOLS_PATH and LD_LIBRARY_PATH to be set to appropriate folder inside this folder. For more details see [set_env.sh](./set_env.sh)

### Setup with gpu based tidl_tools (faster to run)

Example:
```
./setup_pc_gpu.sh
```

<hr>

## Usage of runner (edgeai_tidlrunner.runner basic interface)

runner is a basic interface which hides most of the complexity of the underlying runtimes. It can be used either from Python script or from command line.


### See the options supported with help command
All the options supported can be obtained using the help option. Examples
```
tidlrunner-cli --help
tidlrunner-cli compile_model --help
```


### tidlrunner-cli Commandline interface
[runner Commandline interface](./docs/commandline.md)


### tidlrunner-cli Config file interface
[runner Commandline interface](./docs/configfile.md)


### edgeai_tidlrunner.runner Pythonic interface
[runner Pythonic interface](./docs/pythonic.md)


### List of commands supported
| Command          | Internal Pipeline           | Description                                                               |
|------------------|-----------------------------|---------------------------------------------------------------------------|
| compile_model    | CompileModel                | Compile the given model(s)                                                |
| infer_model      | InferModel                  | Run inference using using already compiled model artifacts                |
| compile_infer    | CompileModel, InferModel    | compile_model, infer_model                                                |


[//]: # (| infer_analyze    | InferAnalyze                | Run inference and analysis using already compiled model artifacts         |)

[//]: # (| infer_accuracy   | InferAccuracy               | Run inference and compute accuracy using already compiled model artifacts |)

[//]: # (| compile_analyze  | CompileModel, InferAnalyze  | Run inference using compiled model artifacts                              |)

[//]: # (| compile_analyze  | CompileModel, AnalyzeModel  | compile_model, infer_analyze                                              |)

[//]: # (| compile_accuracy | CompileModel, InferAccuracy | Compile the model, infer and compute accuracy                             |)

[//]: # (| optimize_model   | OptimizeModel               | Optimize - shape inference, layer transformations etc.                    |)


### Example Arguments / options

| Shortcut Style Names (For Commandline) | Explicit Dot Style Names (Internal Names - Can be used if needed)            | YAML Config file for Commandline (and equivalent dict format for Pythonic interface) |
|----------------------------------------|------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
|                                        |                                                                              | session:                                                                             |
| model_path                             | session.model_path                                                           | &nbsp; model_path: mobilenet_v2.onnx                                                 |
|                                        |                                                                              | &nbsp; runtime_settings:                                                             |
| target_device                          | session.runtime_settings.target_device                                       | &nbsp; &nbsp; target_device: AM68A                                                   |
|                                        |                                                                              | &nbsp; &nbsp; runtime_options:                                                       |
| tensor_bits                            | session.runtime_settings.runtime_options.target_device                       | &nbsp; &nbsp; &nbsp; tensor_bits: 8                                                  |
| calibration_frames                     | session.runtime_settings.runtime_options.advanced_options:calibration_frames | &nbsp; &nbsp; &nbsp; advanced_options:calibration_frames: 12                         |
|                                        |                                                                              |                                                                                      |

As can be seen from this example, there is a one-to-one mapping between the internal Dot style names and the dictionary format. The YAML Config file can be used in the Commandline and the Python dict can be used in Pythonic interface.

All the supported options and how they map to internal names can be seen in this file [settings_default.py](./edgeai_tidlrunner/runner/modules/vision/settings/settings_default.py) and this file [settings_base.py](./edgeai_tidlrunner/runner/bases/settings_base.py)

<hr>

## Usage of rtwrapper (edgeai_tidlrunner.rtwrapper advanced interface)
[rtwrapper advanced interface](./docs/rtwrapper.md)

<hr>
