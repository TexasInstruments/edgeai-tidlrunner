## Runtime Session Settings/Options Explained

<hr>
<hr>

### input_mean and input_scale 

input_mean and input_scale are options that are unique for every model. If there are not provided, default values will be used. input_mean subtraction and input_scale is applied to normalize the input and then it is given to the model. Setting correct values for this is important to get a functionally correct model.

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

#### torchvision presets
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


#### TIDL example
The input_mean, input_scale definition in this package is used for normalization in a straight forward way:
```
x_normalized = (x - input_mean) * input_scale              (eqn. 2)
```

Comparing eqn. 1 and eqn. 2, we can derive the equivalent values. Taking the 255 outside the bracket and merging with std and then taking reciprocal, we get:
```
input_mean = (0.485, 0.456, 0.406) * 255 = (123.675, 116.28, 103.53)
input_scale = 1/{ (0.229, 0.224, 0.225) * 255 } = (0.017125, 0.017507, 0.017429)
```

#### Conclusion about input_mean and input_scale
Setting the input_mean and input_scale needs careful consideration. If these values are incorrectly set, model compilation and inference may work, but the inference accuracy may not be good. Understanding what is really happening in the model training code and mapping them correctly to input_mean and input_scale is important to get a functionally correct model. 

<hr>
<hr>

### runtime_settings and runtime_options

Whichever interface (runner cli, runner configfile, runner py or rtwrapper) is being used, there are some common parameters that control the core runtimes. These are called runtime_settings and runtime_options

runtime_settings: runtime_settings consists of runtime_options that go directly into the underlying inference runtime and also some additional arguments. runtime_settings is basically the key words arguments dict that can be passed to [session interface](../edgeai_tidlrunner/runner/modules/vision/blocks/sessions/) or the [rtwrapper interface](../edgeai_tidlrunner/rtwrapper/core/). runtime_options is part of runtime_settings. It also has additional parameters that are needed in the abstractions in runner. Default runtime_settings are in [edgeai_tidlrunner/runner/modules/vision/settings/settings_default.py](../edgeai_tidlrunner/runner/modules/vision/settings/runtime_settings.py)

runtime_options: runtime_options control the behaviour of core runtimes - default values are specified in [edgeai_tidlrunner/rtwrapper/options/options_default.py](../edgeai_tidlrunner/rtwrapper/options/options_default.py)

Example:<br>
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
session = rtwrapper.core.ONNXRuntimeWrapper(model_path='model.onnx', **runtime_settings)
```

<hr>
<hr>
