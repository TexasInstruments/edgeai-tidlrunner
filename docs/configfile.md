

#### Example - running using config file

config files can be provided either as a single config file or as an aggregate config file to operate on multiple models in parallel. They will run in parallel and the log will go into a log file specific to each model (will not be displayed on screen)
```
tidlrunner-cli compile_model --config_path ./data/models/configs.yaml
```

#### Example - running all the models in the edgeai-modelzoo
If you have edgeai-tensorlab cloned, it is possible to compile all the models in edgeai-tensorlab/edgeai-modelzoo using the following command.
```
tidlrunner-cli compile_model --config_path ../edgeai-tensorlab/edgeai-modelzoo/models/configs.yaml
```





