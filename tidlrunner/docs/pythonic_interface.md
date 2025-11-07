
### edgeai_tidlrunner.runner.run is the Pythonic API of runner

### Example Arguments and mapping to internal options
The parameters used in the commandline or in the configfile - one is a shortcut style name, second is an explicit style name and third is a proper Python dictionary style. Any of these can be used - wherever appropriate. All the styles given from interface are first converted to a common style internally. But typically the shortcut names are used in the commandline and dictionary style names are used in yaml file or in Pythonic interface. 

| Shortcut Style Names (For Commandline) | Explicit Dot Style Names (Internal Names - Can be used if needed)            | YAML Configfile (and equivalent dict format for Pythonic interface) |
|----------------------------------------|------------------------------------------------------------------------------|---------------------------------------------------------------------|
|                                        |                                                                              | session:                                                            |
| model_path                             | session.model_path                                                           | &nbsp; model_path: mobilenet_v2.onnx                                |
|                                        |                                                                              | &nbsp; runtime_settings:                                            |
| target_device                          | session.runtime_settings.target_device                                       | &nbsp; &nbsp; target_device: AM68A                                  |
|                                        |                                                                              | &nbsp; &nbsp; runtime_options:                                      |
| tensor_bits                            | session.runtime_settings.runtime_options.target_device                       | &nbsp; &nbsp; &nbsp; tensor_bits: 8                                 |
| calibration_frames                     | session.runtime_settings.runtime_options.advanced_options:calibration_frames | &nbsp; &nbsp; &nbsp; advanced_options:calibration_frames: 12        |
|                                        |                                                                              |                                                                     |

As can be seen from this example, there is a one-to-one mapping between the shortcut style names, internal dot style names and the dictionary format.

There are many more arguments that are supported. All the supported options and how they map to internal names can be seen in this file [settings_default.py](../edgeai_tidlrunner/runner/modules/vision/settings/settings_default.py) and this file [settings_base.py](../edgeai_tidlrunner/runner/bases/settings_base.py)


### How to use the Pythonic interface 

The arguments can be provided as Shortcut Style Names in a Dictionary, Explicit Dot Style Names in a Dictionary Or as proper Python Dictionary. Here we use a proper Python Dictionary for clarity of explanation:


```
kwargs = {
    'session': {
        'model_path': ./data/examples/models/mobilenet_v2.onnx',
    }
    'dataloader': {
        'name': 'image_classification_dataloader'
        'path': ./data/datasets/vision/imagenetv2c/val',
     }
}

edgeai_tidlrunner.runner.run('compile', **kwargs)
```

See the Pythonic example in [examples/vision/scripts/example_runner_py.py](../examples/vision/scripts/example_runner_py.py) which is invoked via [examples/example_advanced_runner_py.sh](../examples/example_advanced_runner_py.sh)
