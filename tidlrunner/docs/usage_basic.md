

## Usage of runner (edgeai_tidlrunner.runner interface)

runner is a basic interface which hides most of the complexity of the underlying runtimes. It can be used either from Python script or from command line.


### See the options supported with help command
All the options supported can be obtained using the help option. Examples
```
tidlrunner-cli --help
```

Detailed help is available for each command - for example:
```
tidlrunner-cli compile --help
```
```
tidlrunner-cli infer --help
```

<hr>

### List of commands supported
| Command          | Description                                                               |
|------------------|---------------------------------------------------------------------------|
| compile          | Compile the given model(s)                                                |
| infer            | Run inference using using already compiled model artifacts                |
| accuracy         | Analyze compiled artifacts, run inference and analyze layerwise deviations|
| optimize         | Optimize - simplifier, layer optimizations, shape inference (included in compile)|
| analyze          | Analyze layer outputs, compare them to onnxruntime and write statistics  |
| report           | Generate overall csv report of infer or accuracy                          |
| extract          | Extract layers or submodules from a model                                 |
| compile+infer    | compile the model and run inference                                       |
| compile+analyze  | Compile the model and analyze the outputs of different layers             |
| compile+accuracy | Compile the model, run inference and compute accuracy                     |


But as explained above, the easiest way to see list of options supported for a command is to use the help - for example:
```
tidlrunner-cli compile --help
```

<hr>

### Basic interface - tidlrunner-cli Commandline interface
The commandline interface allows to provide the model and a few arguments directly in the commandline.
[runner Commandline interface](./commandline_interface.md)

The commandline options supported for each command are listed [here](./command_line_arguments.md)

<hr>

### Basic interface - tidlrunner-cli Configfile interface
The configfile interface allows to parse all parameters from a yaml file. 
[runner Commandline config file interface](./configfile_interface.md)

<hr>
<hr>

### Report generation after model compilation

A consolidated csv report will be generated with the report command.
```
tidlrunner-cli report
```
