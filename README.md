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

tidlrunner-cli is the interface script to run model compilation and inference via commandline:

import_model is one of the most basic commands - it needs only the model path to be provided. The given model is imported with TIDL using random inputs. It can be used to quickly check whether a model works in TIDL or not. 
```
tidlrunner-cli import_model --model_path=./data/examples/models/mobilenet_v2.onnx
```
The compiled artifacts will be placed under [./runs/runner](./runs/runner) in a folder with the model name.

More options can be specified to configure the run with compile_model.

#### Examples

This is the example for an image classification model:
```
tidlrunner-cli compile_model --model_path=./data/examples/models/mobilenet_v2.onnx --data_name image_classification_dataloader --data_path=./data/datasets/vision/imagenetv2c/val
```

This is the example for an object detection model:
```
tidlrunner-cli compile_model --model_path=./data/models/vision/detection/coco/edgeai-mmdet/ssd_mobilenetv2_lite_512x512_20201214_model.onnx --data_name coco_detection_dataloader --data_path=./data/datasets/vision/coco
```

This is the example for a semantic detection model:
```
tidlrunner-cli compile_model --model_path=./data/models/vision/segmentation/cocoseg21/edgeai-tv/deeplabv3plus_mobilenetv2_edgeailite_512x512_20210405.onnx --data_name coco_segmentation_dataloader --data_path=./data/datasets/vision/coco
```

#### Commandline example
See the commandline example in [example_runner_cli.sh](./example_runner_cli.sh)

#### Running multiple models together 
config files can be provided either as a single config file or as an aggregate config file to operate on multiple models in parallel. They will run in parallel and the log will go into a log file specific to each model (will not be displayed on screen)
```
tidlrunner-cli import_model --config_path ./data/models/configs.yaml
```

#### Detailed help
All the options supported can be btained using the help option. Examples
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
