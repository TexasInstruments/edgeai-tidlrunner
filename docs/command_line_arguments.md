# Command Line Arguments

This document explains the command line arguments available for different commands in the edgeai-tidlrunner. The Config Field column shows the corresponding configuration file field, expressed as a flat name separated by '.'. This has equivalence to the dict style used in the config file:
- `session.model_path` corresponds to `session['model_path']` in the configuration dictionary
- `session.runtime_options.tensor_bits` corresponds to `session['runtime_options']['tensor_bits']`
- `session.runtime_options.advanced_options:calibration_frames` corresponds to `session['runtime_options']['advanced_options:calibration_frames']`

## compile

| Argument | Type | Default | Description | Config Field |
|----------|------|---------|-------------|--------------|
| `--pipeline_type` | str | compile | type of pipeline to run | common.pipeline_type |
| `--model_path` | str | None | input model | session.model_path |
| `--config_path` | str | None | path to configuration file | common.config_path |
| `--output_path` | str | ./work_dirs/{pipeline_type}/{target_device}/{tensor_bits}/{model_id}_{runtime_name}_{model_path}_{model_ext} | output model path | session.run_dir |
| `--simplify_model` | bool | True | enable model simplification optimizations | common.optimize.simplify_model |
| `--optimize_model` | bool | True | enable model optimization | common.optimize.optimize_model |
| `--shape_inference` | bool | True | enable shape inference during optimization | common.optimize.shape_inference |
| `--task_type` | str | None | type of AI task (classification, detection, segmentation etc.) | common.task_type |
| `--num_frames` | int | 10 | number of frames to process | common.num_frames |
| `--display_step` | str | 100 | interval for displaying progress information | common.display_step |
| `--model_id` | str | None | unique id of a model - optional | session.model_id |
| `--artifacts_folder` | str | None | folder to store compilation artifacts | session.artifacts_folder |
| `--packaged_path` | str | ./work_dirs/{pipeline_type}_package/{target_device}/{tensor_bits}/{model_id}_{runtime_name}_{model_path}_{model_ext} | packaged model path | session.packaged_path |
| `--runtime_name` | str | None | name of the runtime session | session.name |
| `--input_mean` | float* | (123.675, 116.28, 103.53) | mean values for input normalization (RGB channels) | session.input_mean |
| `--input_scale` | float* | (0.017125, 0.017507, 0.017429) | scale values for input normalization (RGB channels) | session.input_scale |
| `--data_name` | str | None | name of the input dataset | dataloader.name |
| `--data_path` | str | None | path to the input data directory | dataloader.path |
| `--target_device` | str | AM68A | target device for inference (AM68A, AM69A, etc.) | session.target_device |
| `--target_machine` | str | pc | target machine for running the inference (pc, evm) | session.target_machine |
| `--tidl_offload` | bool | True | enable TIDL acceleration for inference | session.tidl_offload |
| `--graph_optimization_level` | int | 0 | ONNX Runtime graph optimization level | session.onnxruntime:graph_optimization_level |
| `--tensor_bits` | int | 8 | quantization bit-width for tensors (8 or 16) | session.runtime_options.tensor_bits |
| `--accuracy_level` | int | 1 | accuracy level for TIDL offload (0, 1, 2) | session.runtime_options.accuracy_level |
| `--debug_level` | int | 0 | debug level for compile and infer | session.runtime_options.debug_level |
| `--deny_list_layer_type` | str | "" | comma separated layer types to exclude from TIDL offload | session.runtime_options.deny_list:layer_type |
| `--deny_list_layer_name` | str | "" | comma separated layer names to exclude from TIDL offload | session.runtime_options.deny_list:layer_name |
| `--quantization_scale_type` | int | None | type of quantization scale to use | session.runtime_options.advanced_options:quantization_scale_type |
| `--calibration_frames` | int | 12 | number of frames for quantization calibration | session.runtime_options.advanced_options:calibration_frames |
| `--calibration_iterations` | int | 12 | number of calibration iterations | session.runtime_options.advanced_options:calibration_iterations |
| `--quant_params_file_path` | str | - | path to quantization parameters file | session.runtime_options.advanced_options:quant_params_proto_path |
| `--max_num_subgraph_nodes` | int | 1536 | maximum number of nodes in a subgraph | session.runtime_options.advanced_options:max_num_subgraph_nodes |
| `--output_feature_16bit_names_list` | str | - | list of output layers to keep in 16-bit precision | session.runtime_options.advanced_options:output_feature_16bit_names_list |
| `--meta_arch_type` | int | - | meta architecture type for object detection | session.runtime_options.object_detection:meta_arch_type |
| `--meta_arch_file_path` | str | - | path to meta architecture file | session.runtime_options.object_detection:meta_layers_names_list |
| `--detection_threshold` | float | 0.3 | confidence threshold for object detection | session.runtime_options.object_detection:confidence_threshold |
| `--detection_top_k` | int | 200 | number of top detections to keep before NMS | session.runtime_options.object_detection:top_k |
| `--nms_threshold` | float | 0.45 | NMS threshold for object detection | session.runtime_options.object_detection:nms_threshold |
| `--keep_top_k` | int | 200 | number of top detections to keep after NMS | session.runtime_options.object_detection:keep_top_k |
| `--preprocess_name` | str | None | name of the preprocessing pipeline | preprocess.name |
| `--resize` | int* | None | resize dimensions for input images (height width) | preprocess.resize |
| `--crop` | int* | None | crop dimensions for input images (height width) | preprocess.crop |
| `--data_layout` | str | None | data layout format (NCHW, NHWC) | preprocess.data_layout |
| `--reverse_channels` | bool | False | reverse color channel order (RGB to BGR) | preprocess.reverse_channels |
| `--resize_with_pad` | bool | False | resize image with padding to maintain aspect ratio | preprocess.resize_with_pad |
| `--postprocess_enable` | bool | False | enable postprocessing after inference | common.postprocess_enable |
| `--postprocess_name` | str | None | name of the postprocessing pipeline | postprocess.name |

## infer

Inherits all arguments from the `compile` command.

## accuracy

| Argument | Type | Default | Description | Config Field |
|----------|------|---------|-------------|--------------|
| `--label_path` | str | None | path to ground truth labels for accuracy evaluation | dataloader.label_path |
| `--num_frames` | int | 1000 | number of frames to process for accuracy evaluation | common.num_frames |
| `--postprocess_enable` | bool | True | enable postprocessing after inference | common.postprocess_enable |
| `--postprocess_resize_with_pad` | bool | False | resize output with padding to maintain aspect ratio | postprocess.resize_with_pad |
| `--postprocess_normalized_detections` | bool | False | whether detections are normalized coordinates | postprocess.normalized_detections |
| `--postprocess_formatter` | str | None | format for postprocessing output | postprocess.formatter |
| `--postprocess_shuffle_indices` | int* | None | indices for shuffling postprocess output | postprocess.shuffle_indices |
| `--postprocess_squeeze_axis` | int | None | axis to squeeze from output tensor | postprocess.squeeze_axis |
| `--postprocess_reshape_list` | list | None | list of reshape operations for output tensors | postprocess.reshape_list |
| `--postprocess_ignore_index` | str | None | index to ignore during accuracy calculation | postprocess.ignore_index |
| `--postprocess_logits_bbox_to_bbox_ls` | bool | False | convert logits bounding box format to bounding box list | postprocess.logits_bbox_to_bbox_ls |
| `--postprocess_keypoint` | bool | False | enable keypoint postprocessing | postprocess.keypoint |
| `--postprocess_save_output` | bool | False | save postprocessed output to files | postprocess.save_output |
| `--postprocess_save_output_frames` | int | 1 | number of output frames to save | postprocess.save_output_frames |

Inherits all other arguments from the `compile` command.

## analyze

| Argument | Type | Default | Description | Config Field |
|----------|------|---------|-------------|--------------|
| `--pipeline_type` | str | analyze | type of pipeline to run | common.pipeline_type |

Inherits all other arguments from the `infer` command.

## optimize

| Argument | Type | Default | Description | Config Field |
|----------|------|---------|-------------|--------------|
| `--model_path` | str | None | input model | session.model_path |
| `--config_path` | str | None | path to configuration file | common.config_path |
| `--output_path` | str | ./work_dirs/{pipeline_type}/{target_device}/{tensor_bits}/{model_id}_{runtime_name}_{model_path}_{model_ext} | output model path | session.run_dir |
| `--pipeline_type` | str | optimize | type of pipeline to run | common.pipeline_type |
| `--simplify_model` | bool | True | enable model simplification optimizations | common.optimize.simplify_model |
| `--optimize_model` | bool | True | enable model optimization | common.optimize.optimize_model |
| `--shape_inference` | bool | True | enable shape inference during optimization | common.optimize.shape_inference |

## extract

| Argument | Type | Default | Description | Config Field |
|----------|------|---------|-------------|--------------|
| `--model_path` | str | None | input model | session.model_path |
| `--config_path` | str | None | path to configuration file | common.config_path |
| `--output_path` | str | ./work_dirs/{pipeline_type}/{target_device}/{tensor_bits}/{model_id}_{runtime_name}_{model_path}_{model_ext} | output model path | session.run_dir |
| `--pipeline_type` | str | extract | type of pipeline to run | common.pipeline_type |
| `--extract_mode` | str | operators | extraction mode (submodules, submodule, start2end, operators) | common.extract.mode |
| `--submodule_name` | str | None | name of specific submodule to extract | common.extract.submodule_name |
| `--max_depth` | int | 3 | maximum depth for submodule extraction | common.extract.max_depth |
| `--start_names` | str | None | starting layer names for start2end extraction | common.extract.start_names |
| `--end_names` | str | None | ending layer names for start2end extraction | common.extract.end_names |

## report

| Argument | Type | Default | Description | Config Field |
|----------|------|---------|-------------|--------------|
| `--pipeline_type` | str | compile | type of pipeline to run | common.pipeline_type |
| `--report_mode` | str | detailed | report generation mode (summary or detailed) | common.report.mode |
| `--report_path` | str | ./work_dirs/compile | path where reports will be generated | common.report.path |
