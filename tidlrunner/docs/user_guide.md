# Getting Started

## Interface

`tidlrunner-cli` is the primary commandline tool interface that can be used. It is generally assumed that the `tidlrunner-cli` tool will be run from the root of this repository (but it is not an explicit requirement). Outputs of the tool will be placed into [work_dirs](../../work_dirs) by default.

Note: Alternatively, you can choose to run this script with Python and provide arguments: [tidlrunner/edgeai_tidlrunner/main.py](../edgeai_tidlrunner/main.py). This is the interface to be used for debugging using VSCode or any other Python IDE.


There are two primary ways to use this tool: Config file based usage and commandline based usage.

## 1. Configfile based usage

For more control and reproducible workflows, you can use configuration files. This approach allows you to specify all parameters including datasets, preprocessing options, target devices, and much more.

The config file based approach provides:
- Full control over all compilation parameters
- Ability to specify custom datasets for calibration
- Reproducible configurations
- Support for complex preprocessing pipelines

### tidlrunner-cli configfile interface

tidlrunner-cli can also accept a config file as input. The syntax is:

```
tidlrunner-cli <command> --config_path <configfile>  --target_device <SOC> [override-options...]
```

The configfile can be an aggregate config file listing multiple config files as in [this example](../../data/configs/samples/models/configs.yaml) or it can be individual config files provided [under this directory here](../../data/configs/samples/models/vision/)


### Example - running using a single config file
* Config file can be provided with the required arguments, instead of providing them on the commandline. The fields that can be used in the config file is described in the section on commandline arguments.
```
tidlrunner-cli compile --config_path ./data/configs/samples/models/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv_config.yaml --target_device AM62A
```

To run inference:
```
tidlrunner-cli infer --config_path ./data/configs/samples/models/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv_config.yaml --target_device AM62A
```

### Example - running using an aggregate config file
* Using an aggregate config file that lists other config files under the configs field. 
* When aggregate configfile is provided, multiple models will run in parallel and the log will go into a log file specific to each model (will not be displayed on screen)
```
tidlrunner-cli compile --config_path ./data/configs/samples/models/configs.yaml --target_device AM62A
```

To run inference:
```
tidlrunner-cli infer --config_path ./data/configs/samples/models/configs.yaml --target_device AM62A
```

### Example - running all the models in the edgeai-modelzoo
* It is possible to compile the models in [edgeai-modelzoo](https://github.com/TexasInstruments/edgeai-modelzoo) using a simple command.
* Clone edgeai-modelzoo in the parent folder of this repository.
* Then run using the config_path argument.
* Important Note: edgeai-modelzoo has a large number of models - but for now we have enabled support for only imagenet and coco models. Support for more datasets can be added in [pipelines/compile_/compile_base.py](../edgeai_tidlrunner/runner/common/pipelines/common_/compile_base.py) in _upgrade_kwargs() method.

```
tidlrunner-cli compile --config_path ../edgeai-modelzoo/models/configs.yaml --target_device AM62A
```

To run inference:
```
tidlrunner-cli infer --config_path ../edgeai-modelzoo/models/configs.yaml --target_device AM62A
```


## 2. Commandline Usage

### tidlrunner-cli commandline interface

tidlrunner-cli is the interface script to run model compilation and inference via commandline. The syntax is:

```
tidlrunner-cli <command> --target_device <SOC> [options...]
```

#### Example - compile model with random inputs
Compile is one of the most basic and necessary commands - it needs only the model path to be provided. The given model will be compiled with TIDL using random inputs for fixed-point calibration (i.e. quantization). It can be used to quickly check whether a model works in TIDL or not. 

The simplest way to get started is by providing only the model path. This approach uses random inputs for calibration, making it perfect for quick testing and evaluation.

```
tidlrunner-cli compile --model_path=./data/configs/samples/models/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv.onnx --target_device AM62A
```

The compiled artifacts will be placed under [../work_dirs/](../work_dirs/) in a folder with the model name.

*Note: This direct command line usage is useful for quick compilation - to check whether compilation is working or not. This is not our recommended method of usage for actual compilation and inference as it is easy to miss some required arguments (in that case random inputs may get used for quantization calibration and the outputs may not usable). Because of this, the above configfile based interface is our recommended method of usage.*

#### Example - compile_model with actual input data
There are several options can be specified to configure the run when running with compile_model.

This is the example for an image classification model:
```
tidlrunner-cli compile --model_path=./data/configs/samples/models/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv.onnx   --target_device AM62A --data_name image_classification_dataloader --data_path=./data/datasets/imagenetv2c/val --preprocess_name image_preprocess 

```

This is the example for an object detection model:
```
tidlrunner-cli compile --model_path=./data/configs/samples/models/vision/detection/coco/edgeai-mmdet/ssd_mobilenetv2_lite_512x512_20201214_model.onnx --target_device AM62A --data_name coco_detection_dataloader --data_path=./data/datasets/coco --preprocess_name image_preprocess --meta_arch_type 3 --meta_arch_file_path=./data/configs/samples/models/vision/detection/coco/edgeai-mmdet/ssd_mobilenetv2_lite_512x512_20201214_model.prototxt
```
* Note the additional arguments for 'meta_arch'. These are an important argument for accelerating SSD and object detection heads. See the relevant [edgeai-tidl-tools document](https://github.com/TexasInstruments/edgeai-tidl-tools/blob/master/docs/od_meta_arch.md) for more information. 

This is the example for a semantic segmentation model:
```
tidlrunner-cli compile --model_path=./data/configs/samples/models/vision/segmentation/cocoseg21/edgeai-tv/deeplabv3plus_mobilenetv2_edgeailite_512x512_20210405.onnx --target_device AM62A --data_name coco_segmentation_dataloader --data_path=./data/datasets/coco --preprocess_name image_preprocess 
```
* Note: Model simplification with onnxsim may fail on this model for the latest onnxsim installation, but this call to onnxsim can be disabled with the --simplify-model argument as shown.

### Additional examples
See the commandline examples in [examples/example_runner_pc.sh](../../examples/example_runner_pc.sh) and [examples/example_runner_evm.sh](../../examples/example_runner_evm.sh). 


# Quick start with benchmark sripts

Easy to use benchmark script wrappers are in [scripts](../../scripts/) to try out and benchmark example configs in this repository. Use these wrapper scripts for preset-based runs using the [configs for models in the modelzoo](../../data/configs/modelzoo/).

The benchmark wrappers under `scripts/run_benchmark_*.sh` are thin entry points that call internal helpers (`scripts/_run_benchmark_pc.sh` and `scripts/_run_benchmark_evm.sh`) with preset arguments.


# Getting Help

There are many more options that can be used to configure the runs and the next section. Use the help commands below to navigate understand the options supported with each commands.

For basic help, use:
```bash
tidlrunner-cli --help
```

Detailed help is available for each command - for example:
```
tidlrunner-cli <command> --help
```

This will show all the available options for each specific command.

For example:
```
tidlrunner-cli compile --help
```
```
tidlrunner-cli infer --help
```

# Device specific details & running on EVM

## Usage - actual infer or evaluate on EVM
To run the compiled model artifacts on EVM, follow these instructions: [running_on_evm](./running_on_evm.md)


## Compiling models for a specific device
It is important to use the correct target device while compiling the model. By default, this tool assumes AM62A, but that may not be the device/EVM that you have. 

List of devices supported by TIDL are listed in the page [Supported Devices & SDKs](https://github.com/TexasInstruments/edgeai/blob/main/edgeai-mpu/readme_sdk.md).

All those devices are supported by this tool. A specific device can be specified using the option target_device. For example:

```bash
tidlrunner-cli compile --config_path data/configs/samples/models/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv_config.yaml --target_device TDA4VH
```

If you need more details, please refer to [this script that downloads tidl_tools](../../tools/tidl_tools_package/download.py)


# Commandline arguments

## Commands

Supported commands and the options that can be used with them are listed below. These options have a short form that is easy to use in the commandline and an equivalent long form that can be used in the config file. to understand how these short options maps to the structured options used in config files, see the default settings [](../edgeai_tidlrunner/runner/common/settings/settings_default.py) and the [example config files](../../data/configs/)


| Section | Name | Task | Required Args | Optional Args | Description |
|---|---|---|---|---|---|
| command | analyze |  |  | command, capture_log, parallel_processes, parallel_devices, target_machine, target_device, log_file, pipeline_type, verbose, model_path, config_path, work_path, run_label, run_dir, model_surgery, simplify_model, shape_inference, input_optimization, input_mean, input_scale, task_type, task_name, num_frames, input_dataset, display_step, upgrade_config, model_selection, model_shortlist, preset_selection, config_template, incremental, clear_run_dir, save_tensors, instance_timeout, overall_timeout, model_id, artifacts_folder, runtime_name, session_type_dict, dataset_type_dict, data_name, data_path, tidl_offload, graph_optimization_level, tensor_bits, debug_level, deny_list_layer_type, deny_list_layer_name, deny_list_layer_name_search, accuracy_level, enable_tfr_optimization, quantization_scale_type, calibration_frames, calibration_iterations, prequantized_model, quant_params_file_path, max_num_subgraph_nodes, add_data_convert_ops, output_feature_16bit_names_list, output_feature_16bit_names_search, meta_arch_type, meta_arch_file_path, detection_threshold, detection_top_k, nms_threshold, keep_top_k, runtime_options_dict, preprocess_name, resize, crop, data_layout, reverse_channels, resize_with_pad, sample_rate, audio_duration, audio_model_type, postprocess_enable, postprocess_name, display_benchmark, save_output, save_output_frames, show_output, analyze_level, act_data | Inspect model outputs and activation data |
| command | compile |  |  | command, capture_log, parallel_processes, parallel_devices, target_machine, target_device, log_file, pipeline_type, verbose, model_path, config_path, work_path, run_label, run_dir, model_surgery, simplify_model, shape_inference, input_optimization, input_mean, input_scale, task_type, task_name, num_frames, input_dataset, display_step, upgrade_config, model_selection, model_shortlist, preset_selection, config_template, incremental, clear_run_dir, save_tensors, instance_timeout, overall_timeout, model_id, artifacts_folder, runtime_name, session_type_dict, dataset_type_dict, data_name, data_path, tidl_offload, graph_optimization_level, tensor_bits, debug_level, deny_list_layer_type, deny_list_layer_name, deny_list_layer_name_search, accuracy_level, enable_tfr_optimization, quantization_scale_type, calibration_frames, calibration_iterations, prequantized_model, quant_params_file_path, max_num_subgraph_nodes, add_data_convert_ops, output_feature_16bit_names_list, output_feature_16bit_names_search, meta_arch_type, meta_arch_file_path, detection_threshold, detection_top_k, nms_threshold, keep_top_k, runtime_options_dict, preprocess_name, resize, crop, data_layout, reverse_channels, resize_with_pad, sample_rate, audio_duration, audio_model_type, postprocess_enable, postprocess_name | Compile models and generate target-specific artifacts |
| command | evaluate |  |  | command, capture_log, parallel_processes, parallel_devices, target_machine, target_device, log_file, pipeline_type, verbose, model_path, config_path, work_path, run_label, run_dir, model_surgery, simplify_model, shape_inference, input_optimization, input_mean, input_scale, task_type, task_name, num_frames, input_dataset, display_step, upgrade_config, model_selection, model_shortlist, preset_selection, config_template, incremental, clear_run_dir, save_tensors, instance_timeout, overall_timeout, model_id, artifacts_folder, runtime_name, session_type_dict, dataset_type_dict, data_name, data_path, tidl_offload, graph_optimization_level, tensor_bits, debug_level, deny_list_layer_type, deny_list_layer_name, deny_list_layer_name_search, accuracy_level, enable_tfr_optimization, quantization_scale_type, calibration_frames, calibration_iterations, prequantized_model, quant_params_file_path, max_num_subgraph_nodes, add_data_convert_ops, output_feature_16bit_names_list, output_feature_16bit_names_search, meta_arch_type, meta_arch_file_path, detection_threshold, detection_top_k, nms_threshold, keep_top_k, runtime_options_dict, preprocess_name, resize, crop, data_layout, reverse_channels, resize_with_pad, sample_rate, audio_duration, audio_model_type, postprocess_enable, postprocess_name, label_path, postprocess_resize_with_pad, postprocess_normalized_detections, postprocess_formatter, postprocess_shuffle_indices, postprocess_squeeze_axis, postprocess_reshape_list, postprocess_ignore_index, postprocess_logits_bbox_to_bbox_ls, postprocess_keypoint, save_output, save_output_frames, show_output | Evaluate model accuracy against ground truth labels |
| command | extract |  |  | command, capture_log, parallel_processes, parallel_devices, target_machine, target_device, log_file, pipeline_type, verbose, model_path, config_path, work_path, run_label, run_dir, extract_mode, submodule_name, max_depth, start_names, end_names | Extract model submodules, operators, or layer ranges |
| command | infer |  |  | command, capture_log, parallel_processes, parallel_devices, target_machine, target_device, log_file, pipeline_type, verbose, model_path, config_path, work_path, run_label, run_dir, model_surgery, simplify_model, shape_inference, input_optimization, input_mean, input_scale, task_type, task_name, num_frames, input_dataset, display_step, upgrade_config, model_selection, model_shortlist, preset_selection, config_template, incremental, clear_run_dir, save_tensors, instance_timeout, overall_timeout, model_id, artifacts_folder, runtime_name, session_type_dict, dataset_type_dict, data_name, data_path, tidl_offload, graph_optimization_level, tensor_bits, debug_level, deny_list_layer_type, deny_list_layer_name, deny_list_layer_name_search, accuracy_level, enable_tfr_optimization, quantization_scale_type, calibration_frames, calibration_iterations, prequantized_model, quant_params_file_path, max_num_subgraph_nodes, add_data_convert_ops, output_feature_16bit_names_list, output_feature_16bit_names_search, meta_arch_type, meta_arch_file_path, detection_threshold, detection_top_k, nms_threshold, keep_top_k, runtime_options_dict, preprocess_name, resize, crop, data_layout, reverse_channels, resize_with_pad, sample_rate, audio_duration, audio_model_type, postprocess_enable, postprocess_name, display_benchmark, save_output, save_output_frames, show_output | Run inference using compiled models |
| command | inspect |  |  | command, capture_log, parallel_processes, parallel_devices, target_machine, target_device, log_file, pipeline_type, verbose, model_path, config_path, work_path, run_label, run_dir, model_surgery, simplify_model, shape_inference, input_optimization, input_mean, input_scale, task_type, task_name, num_frames, input_dataset, display_step, upgrade_config, model_selection, model_shortlist, preset_selection, config_template, incremental, clear_run_dir, save_tensors, instance_timeout, overall_timeout, model_id, artifacts_folder, runtime_name, session_type_dict, dataset_type_dict, data_name, data_path, tidl_offload, graph_optimization_level, tensor_bits, debug_level, deny_list_layer_type, deny_list_layer_name, deny_list_layer_name_search, accuracy_level, enable_tfr_optimization, quantization_scale_type, calibration_frames, calibration_iterations, prequantized_model, quant_params_file_path, max_num_subgraph_nodes, add_data_convert_ops, output_feature_16bit_names_list, output_feature_16bit_names_search, meta_arch_type, meta_arch_file_path, detection_threshold, detection_top_k, nms_threshold, keep_top_k, runtime_options_dict, preprocess_name, resize, crop, data_layout, reverse_channels, resize_with_pad, sample_rate, audio_duration, audio_model_type, postprocess_enable, postprocess_name, display_benchmark, save_output, save_output_frames, show_output, analyze_level, act_data | Analyze model runtime and layer-level statistics |
| command | package |  |  | command, capture_log, parallel_processes, parallel_devices, target_machine, target_device, log_file, pipeline_type, verbose, tensor_bits, work_path, run_label, package_path, param_template | Package artifacts for deployment |
| command | report |  |  | command, capture_log, parallel_processes, parallel_devices, target_machine, target_device, log_file, pipeline_type, verbose, report_mode, report_path, run_label, report_perfsim | Generate compile and performance reports |
| command | surgery |  |  | command, capture_log, parallel_processes, parallel_devices, target_machine, target_device, log_file, pipeline_type, verbose, model_path, config_path, work_path, run_label, run_dir, model_surgery, simplify_model, shape_inference, input_optimization, input_mean, input_scale | Run model surgery optimizations on the input model |

## Dataloaders

Name and path can be used with data_name and data_path commanline arguments. These arguments can also be used in the `dataloader` section in  config file - for example name and path entries as in the [sample config files](../../data/configs/samples/models)

These dataloaders are defined in the [dataloaders module](../../tidlrunner/edgeai_tidlrunner/runner/common/blocks/dataloaders/)

| Section | Name | Task | Required Args | Optional Args | Description |
|---|---|---|---|---|---|
| dataloader | ade20k32_segmentation_dataloader | segmentation | path | label_path, split, num_classes |  |
| dataloader | ade20k_segmentation_dataloader | segmentation | path | label_path, split |  |
| dataloader | audio_classification_dataloader | audio_classification | path |  |  |
| dataloader | camera_capture_dataloader | video-capture |  | source, num_frames |  |
| dataloader | cityscapes_segmentation_dataloader | segmentation | path | label_path |  |
| dataloader | coco_detection_dataloader | detection | path | label_path |  |
| dataloader | coco_keypoint_detection_dataloader | keypoint_detection | path | label_path |  |
| dataloader | coco_segmentation_dataloader | segmentation | path | label_path |  |
| dataloader | image_classification_dataloader | classification | path | label_path |  |
| dataloader | image_files_dataloader | classification | path | label_path |  |
| dataloader | image_list_dataloader | classification | path |  |  |
| dataloader | image_pix2pix_dataloader | image-to-image | path | label_path |  |
| dataloader | image_segmentation_dataloader | segmentation | path | label_path |  |
| dataloader | imagenet_classification_dataloader | classification |  |  |  |
| dataloader | imagenet_dataloader | classification | path | label_path |  |
| dataloader | imagenetv2c_classification_dataloader | classification |  |  |  |
| dataloader | imagenetv2c_dataloader | classification | path | label_path, variant |  |
| dataloader | modelmaker_classification_dataloader | classification | path, label_path |  |  |
| dataloader | modelmaker_detection_dataloader | detection | path | label_path |  |
| dataloader | modelmaker_segmentation_dataloader | segmentation | path, label_path |  |  |
| dataloader | nuscenes_frame_dataloader | 3d-detection | path | version, load_type |  |
| dataloader | nuscenes_mv_image_dataloader | 3d-detection | path | version, load_type |  |
| dataloader | nyudepthv2_dataloader | depth-estimation | path | label_path, split |  |
| dataloader | pandaset_frame_dataloader | 3d-detection | path | version, load_type |  |
| dataloader | pandaset_mv_image_dataloader | 3d-detection | path | version, load_type |  |
| dataloader | random_dataloader |  |  |  |  |
| dataloader | robokit_segmentation_dataloader | segmentation | path | label_path |  |
| dataloader | robokit_visloc_dataloader | visual-localization | path | label_path |  |
| dataloader | speech_enhancement_dataloader | speech-enhancement | path |  |  |
| dataloader | video_capture_dataloader | video-capture |  | source, num_frames |  |
| dataloader | video_file_dataloader | video-capture | video_path | num_frames |  |
| dataloader | voc_segmentation_dataloader | segmentation | path | label_path |  |
| dataloader | widerface_detection_dataloader | detection | path | label_path, split |  |
| dataloader | ycbv_object_6d_pose_dataloader | 6d-pose | path | label_path |  |


# Audio Models and Datasets
To run audio inference pipelines — sound classification (VGGish11, YAMNet on UrbanSound8K) and speech enhancement (GTCRN, GCRN on VoiceBank-DEMAND-16k) — see the [Audio Models and Datasets](audio_models_and_datasets.md) guide. 


# Advanced Documentation (for experts)

[Generic datasets](docs/generic_datasets.md): More details on some of the generic dataset formats supported.

[Deep dive](docs/deep_dive.md) documentation for internal details.


# Help & Support
## FAQ
For other types of help, check the [FAQ](./faq.md) and the other docs within this repo. 

## Reporting Issues
For bugs or issues, create an Issue on this repository or create a support ticket on TI's [e2e forums](https://e2e.ti.com/support/processors-group/processors/f/processors-forum)
