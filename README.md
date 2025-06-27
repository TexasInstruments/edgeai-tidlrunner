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

Additional arguments such as tools_version and tools_type can be provided to configure the setup.

Example:
```
./setup_pc.sh --tools_version 10.1 --tools_type gpu
```

<hr>

## Usage of runner (basic interface)

runner is a basic interface which hides most of the complexity of the underlying runtimes. It can be used either from Python script or from command line.

### tidl-runner commandline interface

tidl-run is the interface script to run model compilation and inference via commandline:

```
tidl-runner-cli compile --model_path=./data/examples/models/mobilenet_v2.onnx
```

More options can be specified to configure the run:

```
tidl-runner-cli compile --model_path=./data/examples/models/mobilenet_v2.onnx --run_dir=./run/tidl-runner/mobilenet_v2 --input_data.path=./data/examples/coco_ccby/images
```

See the commandline example in [run_runner_cli_example.sh](./run_runner_cli_example.sh)

### edgeai_tidlrunner Pythonic interface

edgeai_tidlrunner.runner.run is the Pythonic API of runner

```
kwargs = {
    'model': './data/examples/models/mobilenet_v2.onnx',
    'run_dir': './run/tidl-runner/mobilenet_v2',
    'input_data.path': './data/examples/datasets/coco_ccby/images',
}

edgeai_tidlrunner.runner.run('compile', **kwargs)
```

Several additional options such as runtime_settings can be provided in via this API. 

See the Pythonic example in [run_runner_py_example.py](./run_runner_py_example.py) which is invoked via [run_runner_py_example.sh](./run_runner_py_example.sh)

<hr>

## Usage of runtimes wrappers (advanced interface)

### Create a runtime_settings dict

The default value for runtime_settings is in the file [edgeai_tidlrunner/runtimes/settings/default_settings.py](./edgeai_tidlrunner/runtimes/settings/default_settings.py). Any overrides can be passed here:

```
# runtime_settings and runtime_options
settings_kwargs = {
    # add any runtime_settings overrides here
    'target_device': 'AM68A',
    'runtime_options': {
        # add any runtime_options overrides here
        'advanced_options:c7x_firmware_version':'10_01_04_00',
        'advanced_options:calibration_frames':12,
        'advanced_options:calibration_iterations':12
    }
}
runtime_settings = edgeai_tidlrunner.runtimes.settings.RuntimeSettings(**settings_kwargs)
```

### Run model compilation
```
images_path = f'./data/examples/datasets/coco_ccby/images'
calib_dataset = glob.glob(f'{images_path}/*.*')

onnxruntime_wrapper = edgeai_tidlrunner.runtimes.core.ONNXRuntimeWrapper(
        runtime_options=runtime_settings['runtime_options'],
        model_path=model_path,
        artifacts_folder=artifacts_folder,
        tidl_tools_path=os.environ['TIDL_TOOLS_PATH'])

for input_index in range(runtime_settings['runtime_options']['advanced_options:calibration_frames']):
    input_data = preprocess_input(calib_dataset[input_index])
    onnxruntime_wrapper.run_import(input_data)
```


### Run model inference
```
images_path = f'./data/examples/datasets/coco_ccby/images'
val_dataset = glob.glob(f'{images_path}/*.*')

onnxruntime_wrapper = edgeai_tidlrunner.runtimes.core.ONNXRuntimeWrapper(
        runtime_options=runtime_settings['runtime_options'],
        model_path=model_path,
        artifacts_folder=artifacts_folder,
        tidl_tools_path=os.environ['TIDL_TOOLS_PATH'],
        tidl_offload=True)

for input_index in range(num_frames):
    input_data = preprocess_input(val_dataset[input_index])
    outputs = onnxruntime_wrapper.run_inference(input_data)
    print(outputs)
```

### Examples

A full example that demonstrates the usage of this runtimes: [tidl_onnxrt_example.py](./examples/tidl_onnxrt_example.py)

<hr>

## Usage of TIDL tools (low level interface)

The API and usage of [edgeai_tidlrunner/tools](./edgeai_tidlrunner/tools) is described in [edgeai-tidl-tools](https://github.com/TexasInstruments/edgeai-tidl-tools)

<hr>
