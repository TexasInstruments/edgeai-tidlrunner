
### tidlrunnercli commandline interface

tidlrunnercli is the interface script to run model compilation and inference via commandline. The syntax is:

```
tidlrunnercli <command> [options...]
```

The commandline options supported for each command are listed [here](./command_line_arguments.md)


#### Example - compile_model with random inputs
compile_model is one of the most basic commands - it needs only the model path to be provided. The given model is compiled with TIDL using random inputs. It can be used to quickly check whether a model works in TIDL or not. 
```
tidlrunnercli compile --model_path=./data/examples/models/mobilenet_v2.onnx
```
The compiled artifacts will be placed under [../work_dirs/](../work_dirs/) in a folder with the model name.

#### Example - compile_model with actual input data
There are several options can be specified to configure the run when running with compile_model.

This is the example for an image classification model:
```
tidlrunnercli compile --model_path=./data/examples/models/mobilenet_v2.onnx --data_name image_classification_dataloader --data_path=./data/datasets/vision/imagenetv2c/val --preprocess_name image_preprocess 
```

This is the example for an object detection model:
```
tidlrunnercli compile --model_path=./data/models/vision/detection/coco/edgeai-mmdet/ssd_mobilenetv2_lite_512x512_20201214_model.onnx --data_name coco_detection_dataloader --data_path=./data/datasets/vision/coco --preprocess_name image_preprocess
```

This is the example for a semantic segmentation model:
```
tidlrunnercli compile --model_path=./data/models/vision/segmentation/cocoseg21/edgeai-tv/deeplabv3plus_mobilenetv2_edgeailite_512x512_20210405.onnx --data_name coco_segmentation_dataloader --data_path=./data/datasets/vision/coco --preprocess_name image_preprocess 
```

#### Example - a commandline example
See the commandline examples in [example_runner_cli_pc.sh](../example_runner_cli_pc.sh) and [example_runner_cli_evm.sh](../example_runner_cli_evm.sh)
