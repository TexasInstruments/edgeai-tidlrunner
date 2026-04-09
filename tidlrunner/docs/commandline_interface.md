
### tidlrunner-cli commandline interface

tidlrunner-cli is the interface script to run model compilation and inference via commandline. The syntax is:

```
tidlrunner-cli <command> --target_device <SOC> [options...]
```

The commandline options supported for each command are listed [here](./command_line_arguments.md)


#### Example - compile model with random inputs
Compile is one of the most basic and necessary commands - it needs only the model path to be provided. The given model will be compiled with TIDL using random inputs for fixed-point calibration (i.e. quantization). It can be used to quickly check whether a model works in TIDL or not. 
```
tidlrunner-cli compile --model_path=./data/models/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv.onnx --target_device AM62A
```
The compiled artifacts will be placed under [../work_dirs/](../work_dirs/) in a folder with the model name.

#### Example - compile_model with actual input data
There are several options can be specified to configure the run when running with compile_model.

This is the example for an image classification model:
```
tidlrunner-cli compile --model_path=./data/models/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv.onnx   --target_device AM62A --data_name image_classification_dataloader --data_path=./data/datasets/vision/imagenetv2c/val --preprocess_name image_preprocess 

```

This is the example for an object detection model:
```
tidlrunner-cli compile --model_path=./data/models/vision/detection/coco/edgeai-mmdet/ssd_mobilenetv2_lite_512x512_20201214_model.onnx --target_device AM62A --data_name coco_detection_dataloader --data_path=./data/datasets/vision/coco --preprocess_name image_preprocess --meta_arch_type 3 --meta_arch_file_path=./data/models/vision/detection/coco/edgeai-mmdet/ssd_mobilenetv2_lite_512x512_20201214_model.prototxt
```
* Note the additional arguments for 'meta_arch'. These are an important argument for accelerating SSD and object detection heads. See the relevant [edgeai-tidl-tools document](https://github.com/TexasInstruments/edgeai-tidl-tools/blob/master/docs/od_meta_arch.md) for more information. 

This is the example for a semantic segmentation model:
```
tidlrunner-cli compile --model_path=./data/models/vision/segmentation/cocoseg21/edgeai-tv/deeplabv3plus_mobilenetv2_edgeailite_512x512_20210405.onnx --target_device AM62A --data_name coco_segmentation_dataloader --data_path=./data/datasets/vision/coco --preprocess_name image_preprocess 
```

#### Example - a commandline example
See the commandline examples in [examples/example_runner_cli_pc.sh](../../examples/example_runner_cli_pc.sh) and [examples/example_runner_cli_evm.sh](../../examples/example_runner_cli_evm.sh). 
* To run this tool on the EVM, it must be setup and the models and artifacts must be present. See [running_on_evm.md](./running_on_evm.md) documentation.