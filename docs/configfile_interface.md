### tidlrunnercli configfile interface

tidlrunnercli can also accept a config file as input. The syntax is:

```
tidlrunnercli <command> --config_path <configfile> [overrideoptions...]
```

The configfile can be an aggregate config file listing multiple config files as in [this exammple](../data/models/configs.yaml) or it can be individual config files provided [under this directory here](../data/models/vision/)


#### Example - running using config file
* Config files can be provided either as a single config file or as an aggregate config file to operate on multiple models in parallel. 
* If aggregate configfile is provided, the models will run in parallel and the log will go into a log file specific to each model (will not be displayed on screen)
```
tidlrunnercli compile --config_path ./data/models/configs.yaml
```

To run inference:
```
tidlrunnercli infer --config_path ./data/models/configs.yaml
```

#### Example - running all the models in the edgeai-modelzoo
* It is possible to compile the models in [edgeai-modelzoo](https://github.com/TexasInstruments/edgeai-modelzoo) using a simple command.
* Clone edgeai-modelzoo in the parent folder of this repository.
* Then run using the config_path argument.
* Important Note: edgeai-modelzoo has a large number of models - but for now we have enabled support for only imagenet and coco models. Support for more datasets can be added in [pipelines/compile_/compile_base.py](../edgeai-tidlrunner/edgeai_tidlrunner/runner/modules/vision/pipelines/compile_/compile_base.py) in _upgrade_kwargs() method.

```
tidlrunnercli compile --config_path ../edgeai-modelzoo/models/configs.yaml
```

To run inference:
```
tidlrunnercli infer --config_path ../edgeai-modelzoo/models/configs.yaml
```



