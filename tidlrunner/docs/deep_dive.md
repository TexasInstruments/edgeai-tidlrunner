# Deep dive

## edgeai_tidlrunner.runner Pythonic interface

edgeai_tidlrunner.runner.run is the Pythonic API of runner. There can be used if more flexibility is required.

##### Example Arguments and mapping to internal options
The parameters used in the commandline or in the configfile - one is a shortcut style name, second is an explicit style name and third is a proper Python dictionary style. Any of these can be used - wherever appropriate. All the styles given from interface are first converted to a common style internally. But typically the shortcut names are used in the commandline and dictionary style names are used in yaml file or in Pythonic interface. 

| Shortcut Style Names (For Commandline) | Explicit Dot Style Names (Internal Names - Can be used if needed)            | YAML Configfile (and equivalent dict format for Pythonic interface) |
|----------------------------------------|------------------------------------------------------------------------------|---------------------------------------------------------------------|
|                                        |                                                                              | session:                                                            |
| model_path                             | session.model_path                                                           | &nbsp; model_path: mobilenet_v2.onnx                                |
|                                        |                                                                              | &nbsp; runtime_settings:                                            |
| target_device                          | session.runtime_settings.target_device                                       | &nbsp; &nbsp; target_device: AM62A                                  |
|                                        |                                                                              | &nbsp; &nbsp; runtime_options:                                      |
| tensor_bits                            | session.runtime_settings.runtime_options.target_device                       | &nbsp; &nbsp; &nbsp; tensor_bits: 8                                 |
| calibration_frames                     | session.runtime_settings.runtime_options.advanced_options:calibration_frames | &nbsp; &nbsp; &nbsp; advanced_options:calibration_frames: 12        |
|                                        |                                                                              |                                                                     |

As can be seen from this example, there is a one-to-one mapping between the shortcut style names, internal dot style names and the dictionary format.

There are many more arguments that are supported. All the supported options and how they map to internal names can be seen in this file [settings_default.py](../edgeai_tidlrunner/runner/common/settings/settings_default.py) and this file [settings_base.py](../edgeai_tidlrunner/runner/common/bases/settings_base.py)


##### How to use the Pythonic interface 

The arguments can be provided as Shortcut Style Names in a Dictionary, Explicit Dot Style Names in a Dictionary Or as proper Python Dictionary. Here we use a proper Python Dictionary for clarity of explanation:


```
kwargs = {
    'session': {
        'model_path': './data/configs/samples/models/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv.onnx',
    },
    'dataloader': {
        'name': 'image_classification_dataloader',
        'path': './data/datasets/imagenetv2c/val',
     },
    'preprocess': {
        'name': 'image_preprocess',
    },
}

edgeai_tidlrunner.run('compile', **kwargs)
```

See the Pythonic example in [examples/vision/scripts/example_runner_py.py](../../examples/vision/scripts/example_runner_py.py) which is invoked via [examples/example_advanced_runner_py.sh](../../examples/example_advanced_runner_py.sh)



## Deep dive - Runtime Session Settings/Options Explained


##### input_mean and input_scale 

The input_mean and input_scale parameters are options that are unique for every model. If these are not provided, default values will be used. Mean subtraction with input_mean and multiplication by input_scale is applied to normalize the input and then it is given to the model. Setting correct values for this is important to get a functionally correct model. It is assumed that inputs are 3-channel images

In [settings_default.py](../edgeai_tidlrunner/runner/modules/vision/settings/settings_default.py) the values are set as follows:
```
'input_mean': {'dest': 'session.input_mean', 'default': (123.675, 116.28, 103.53), 'type': float, 'nargs': '*', 'metavar': 'value'},
'input_scale': {'dest': 'session.input_scale', 'default': (0.017125, 0.017507, 0.017429), 'type': float, 'nargs': '*', 'metavar': 'value'},
```

Indicating that the default values in this package are:
```
input_mean = (123.675, 116.28, 103.53)
input_scale = (0.017125, 0.017507, 0.017429)
```

###### torchvision presets
In torchvision and many other popular training source codes, these values are set in a cryptic way that is difficult to understand. 
For example, see the settings in [torchvision](https://github.com/pytorch/vision) classification training scripts [preset](https://github.com/pytorch/vision/blob/f52c4f1afd7d/references/classification/presets.py#L25)
```
mean=(0.485, 0.456, 0.406),
std=(0.229, 0.224, 0.225),
```

It requires careful analyzing of the torchvision code to understand how these values are used. What's happening under the hood is that the image data is first divided by 255 (max value of uint8) and then passed on to [this normalize function](https://github.com/pytorch/vision/blob/f52c4f1afd7d/torchvision/transforms/_functional_tensor.py#L905) that subtracts mean and divides by std. It can be explained as follows. Let x be the image data being read. Then the equation becomes.
```
x_normalized = (x/255 - mean)/std                          (eqn. 1)
```

These particular values are common because they represent the mean and standard deviation of the ImageNet1k dataset.

###### TIDL example
The input_mean, input_scale definition in this package is used for normalization in a straight forward way:
```
x_normalized = (x - input_mean) * input_scale              (eqn. 2)
```

Comparing eqn. 1 and eqn. 2, we can derive the equivalent values. Taking the 255 outside the bracket and merging with std and then taking reciprocal, we get:
```
input_mean = (0.485, 0.456, 0.406) * 255 = (123.675, 116.28, 103.53)
input_scale = 1/{ (0.229, 0.224, 0.225) * 255 } = (0.017125, 0.017507, 0.017429)
```

###### Note about input_mean and input_scale
Setting the input_mean and input_scale needs careful consideration. If these values are incorrectly set, model compilation and inference may work, but the inference accuracy may not be good. Understanding what is really happening in the model training code and mapping them correctly to input_mean and input_scale is important to get a functionally correct model. 

<hr>
<hr>

##### runtime_settings and runtime_options

Whichever interface (runner cli, runner configfile, runner py or rtwrapper) is being used, there are some common parameters that control the core runtimes. These are called runtime_settings and runtime_options

`runtime_settings`: runtime_settings consists of `runtime_options` that go directly into the underlying inference runtime and also some additional arguments. The runtime_settings is basically the keyword arguments dict that can be passed to [session interface](..//edgeai_tidlrunner/runner/common/blocks/sessions) or the [rtwrapper interface](../edgeai_tidlrunner/rtwrapper/core/). The runtime_options is part of runtime_settings. It also has additional parameters that are needed in the abstractions in runner. Default runtime_settings are in [edgeai_tidlrunner/runner/modules/vision/settings/settings_default.py](../edgeai_tidlrunner/runner/common/settings/runtime_settings.py)

`runtime_options`: runtime_options control the behavior of rtwrapper, which is a wrapper over core runtimes (see the section on rtwrapper below) - default values are specified in [edgeai_tidlrunner/rtwrapper/options/options_default.py](../edgeai_tidlrunner/rtwrapper/options/options_default.py)

#### Example
These settings and options can be passed to the underlying runner interface in one of the several ways - for example in a config file or in the Pythonic interface. Here is an example of the Pythonic form:
```
    runtime_settings = {
        # add any runtime_settings overrides here
        'target_device': args.target_device,
        'input_mean': (123.675, 116.28, 103.53),
        'input_scale': (0.017125, 0.017507, 0.017429),
        'runtime_options': {
            # add any runtime_options overrides here
            'tidl_tools_path': os.environ['TIDL_TOOLS_PATH'],
            'artifacts_folder': artifacts_folder,
        }
    }
```

And here is an example usage through rtwrapper interface:
```
from edgeai_tidlrunner import rtwrapper
session = rtwrapper.core.ONNXRuntimeWrapper(model_path='model.onnx', **runtime_settings)
```


## edgeai_tidlrunner.rtwrapper Pythonic wrapper interface for the core runtimes
Abstractions are sometimes a hindrance to understand what is really happening under the hood or to easily modify and extend. rtwrapper is a thin, low level wapper to the core tidl-tools. Use this to understand how the core tidl-tools work or to integrate into the runner.

##### rtwrapper interface
The runtime wrappers [edgeai_tidlrunner/rtwrapper](../edgeai_tidlrunner/rtwrapper) provides an advanced low level interface beyond what the runner provides. 

#### Example
An example of this is in [examples/example_advanced_rtwrapper.py](../../examples/vision/scripts/example_advanced_rtwrapper.py)

#### Options
The default settings used for compilation are contained in [rtwrapper/options/options_defaults.py](../edgeai_tidlrunner/rtwrapper/options/options_default.py)

