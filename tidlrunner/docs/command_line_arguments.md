Note: the "Config Field" column uses the argparse `dest` value expressed as a flat name separated by '.' and ':' as used in `SETTINGS_DEFAULT`. This corresponds to the configuration dictionary using bracket notation. For example:

- `session.model_path` corresponds to `session['model_path']` in the configuration dictionary.
- `session.runtime_options.tensor_bits` corresponds to `session['runtime_options']['tensor_bits']`.
- `session.runtime_options.advanced_options:calibration_frames` corresponds to `session['runtime_options']['advanced_options:calibration_frames']` in the flat dest notation.

## compile

| Option | Type | Default | Help | Config Field |
|---|---:|---|---|---|
| model_path | str | None | input model | session.model_path |
| config_path | str | None | path to configuration file | common.config_path |
| work_path | str | ./work_dirs/{pipeline_type}/{target_device}/{tensor_bits}bits | work path | common.work_path |
| run_dir | str | {work_path}/{model_id}_{runtime_name}_{model_path}_{model_ext} | run_dir | session.run_dir |
| pipeline_type | str | compile | type of pipeline to run | common.pipeline_type |
| task_type | str | None | type of AI task (classification, detection, segmentation etc.) | common.task_type |
| num_frames | int | 10 | number of frames to process | common.num_frames |
| display_step | str | 0.1 | interval for displaying progress information | common.display_step |
| upgrade_config | str | True | upgrade edgeai-benchmark config to work with tidlrunner | common.upgrade_config |
| session_type_dict | str | None | mapping of model extensions to session names | common.session_type_dict |
| model_selection | str | None | select a subset of models to run - path of the model is compared using this model_selection regex to select a particular model or not | common.model_selection |
| model_shortlist | str | None | select a subset of models to run - models configs with model_shortlist value <= this specified value will be used | common.model_shortlist |
| preset_selection | utils.str_or_none | None | select a preset for speed accuracy trade-off: None, SPEED, ACCURACY, BALANCED | common.preset_selection |
| config_template | str | data/templates/configs/param_template_config.yaml | param template path | common.config_template |
| incremental | utils.str_to_bool | False | param template path | common.incremental |
| model_id | str | None | unique id of a model - optional | session.model_id |
| artifacts_folder | str | None | folder to store compilation artifacts | session.artifacts_folder |
| runtime_name | str | None | name of the runtime session | session.name |
| data_name | str | None | name of the input dataset | dataloader.name |
| data_path | str | None | path to the input data directory | dataloader.path |
| target_device | str | TargetDeviceType.TARGET_DEVICE_AM68A | target device for inference (AM68A, AM69A, etc.) | session.target_device |
| tidl_offload | utils.str_to_bool | True | enable TIDL acceleration for inference | session.tidl_offload |
| graph_optimization_level | int | GraphOptimizationLevel.ORT_DISABLE_ALL | ONNX Runtime graph optimization level | session.onnxruntime:graph_optimization_level |
| tensor_bits | int | 8 | quantization bit-width for tensors (8 or 16) | session.runtime_options.tensor_bits |
| accuracy_level | int | 1 | accuracy level for TIDL offload (0, 1, 2) | session.runtime_options.accuracy_level |
| debug_level | int | 0 | debug level for compile and infer | session.runtime_options.debug_level |
| deny_list_layer_type | str | '' | comma separated layer types to exclude from TIDL offload | session.runtime_options.deny_list:layer_type |
| deny_list_layer_name | str | '' | comma separated layer names to exclude from TIDL offload | session.runtime_options.deny_list:layer_name |
| deny_list_start_end_dict | utils.aststr_to_object | '' | a dict contaning start and end nodes - it will be used to generate deny_list:layer_name. example: {"/decoder/Concat_3":None, "/aux/Relu_5":None} | session.deny_list_start_end_dict |
| output_16bit_names_start_end_dict | utils.aststr_to_object | '' | a dict contaning start and end nodes - it will be used to generate advanced_options:output_feature_16bit_names_list. example: {"/decoder/Concat_3":None, "/aux/Relu_5":None} | session.output_16bit_names_start_end_dict |
| quantization_scale_type | int | None | type of quantization scale to use | session.runtime_options.advanced_options:quantization_scale_type |
| calibration_frames | int | 12 | number of frames for quantization calibration | session.runtime_options.advanced_options:calibration_frames |
| calibration_iterations | int | 12 | number of calibration iterations | session.runtime_options.advanced_options:calibration_iterations |
| prequantized_model | utils.int_or_none | argparse.SUPPRESS | whether prequantized model | session.runtime_options.advanced_options:prequantized_model |
| quant_params_file_path | utils.str_or_none_or_bool | argparse.SUPPRESS | path to quantization parameters file | session.runtime_options.advanced_options:quant_params_proto_path |
| max_num_subgraph_nodes | int | 3000 | maximum number of nodes in a subgraph | session.runtime_options.advanced_options:max_num_subgraph_nodes |
| output_feature_16bit_names_list | str | argparse.SUPPRESS | list of output layers to keep in 16-bit precision | session.runtime_options.advanced_options:output_feature_16bit_names_list |
| add_data_convert_ops | int | DataConvertOps.DATA_CONVERT_OPS_INPUT_OUTPUT | data convert in DSP (0: disable, 1: input, 2: output, 3: input and output) - otherwise it will happen in ARM | session.runtime_options.advanced_options:add_data_convert_ops |
| meta_arch_type | int | argparse.SUPPRESS | meta architecture type for object detection | session.runtime_options.object_detection:meta_arch_type |
| meta_arch_file_path | str | argparse.SUPPRESS | path to meta architecture file | session.runtime_options.object_detection:meta_layers_names_list |
| detection_threshold | float | 0.3 | confidence threshold for object detection | session.runtime_options.object_detection:confidence_threshold |
| detection_top_k | int | 200 | number of top detections to keep before NMS | session.runtime_options.object_detection:top_k |
| nms_threshold | float | 0.45 | NMS threshold for object detection | session.runtime_options.object_detection:nms_threshold |
| keep_top_k | int | 200 | number of top detections to keep after NMS | session.runtime_options.object_detection:keep_top_k |
| preprocess_name | str | None | name of the preprocessing pipeline | preprocess.name |
| resize | int | None | resize dimensions for input images (height width) | preprocess.resize |
| crop | int | None | crop dimensions for input images (height width) | preprocess.crop |
| data_layout | str | None | data layout format (NCHW, NHWC) | preprocess.data_layout |
| reverse_channels | utils.str_to_bool | False | reverse color channel order (RGB to BGR) | preprocess.reverse_channels |
| resize_with_pad | utils.str_to_bool | False | resize image with padding to maintain aspect ratio | preprocess.resize_with_pad |
| postprocess_enable | utils.str_to_bool | False | enable postprocessing after inference | common.postprocess_enable |
| postprocess_name | str | None | name of the postprocessing pipeline | postprocess.name |

## infer

(Options are the same as `compile`)

## accuracy

| Option | Type | Default | Help | Config Field |
|---|---:|---|---|---|
| label_path | str | None | path to ground truth labels for accuracy evaluation | dataloader.label_path |
| num_frames | int | 1000 | number of frames to process for accuracy evaluation | common.num_frames |
| postprocess_enable | utils.str_to_bool | True | enable postprocessing after inference | common.postprocess_enable |
| postprocess_resize_with_pad | utils.str_to_bool | False | resize output with padding to maintain aspect ratio | postprocess.resize_with_pad |
| postprocess_normalized_detections | utils.str_to_bool | False | whether detections are normalized coordinates | postprocess.normalized_detections |
| postprocess_formatter | str | None | format for postprocessing output | postprocess.formatter |
| postprocess_shuffle_indices | int | None | indices for shuffling postprocess output | postprocess.shuffle_indices |
| postprocess_squeeze_axis | utils.str_to_int | None | axis to squeeze from output tensor | postprocess.squeeze_axis |
| postprocess_reshape_list | utils.str_to_list_of_tuples | None | list of reshape operations for output tensors | postprocess.reshape_list |
| postprocess_ignore_index | str | None | index to ignore during accuracy calculation | postprocess.ignore_index |
| postprocess_logits_bbox_to_bbox_ls | utils.str_to_bool | False | convert logits bounding box format to bounding box list | postprocess.logits_bbox_to_bbox_ls |
| postprocess_keypoint | utils.str_to_bool | False | enable keypoint postprocessing | postprocess.keypoint |
| postprocess_save_output | bool | True | save postprocessed output to files | postprocess.save_output |
| postprocess_save_output_frames | int | 10 | number of output frames to save | postprocess.save_output_frames |

## analyze

(Options are the same as `infer` plus the following)

| Option | Type | Default | Help | Config Field |
|---|---:|---|---|---|
| pipeline_type | str | analyze | type of pipeline to run | common.pipeline_type |
| analyze_level | int | 2 | analyze_level - 0: basic, 1: whole model stats, 2: whole model and per layer stats | common.analyze_level |
| num_frames | int | 1 | number of frames to process for accuracy evaluation | common.num_frames |

## surgery

| Option | Type | Default | Help | Config Field |
|---|---:|---|---|---|
| model_path | str | None | input model | session.model_path |
| config_path | str | None | path to configuration file | common.config_path |
| work_path | str | ./work_dirs/{pipeline_type}/{target_device}/{tensor_bits}bits | work path | common.work_path |
| run_dir | str | {work_path}/{model_id}_{runtime_name}_{model_path}_{model_ext} | run_dir | session.run_dir |
| pipeline_type | str | optimize | type of pipeline to run | common.pipeline_type |
| model_surgery | utils.str_to_bool_or_none_or_dict | True | enable model surgery optimizations | common.surgery.model_surgery |
| simplify_model | utils.str_to_bool | pre | enable model simplification optimizations | common.surgery.simplify_mode |
| shape_inference | utils.str_or_none_or_bool | all | enable shape inference during surgery optimization | common.surgery.shape_inference_mode |
| input_optimization | utils.str_to_bool | False | merge in input_mean and input_scale into the model if possible, so that model input can be in uint8 and not float32 | session.input_optimization |
| input_mean | float | (123.675, 116.28, 103.53) | mean values for input normalization (RGB channels) | session.input_mean |
| input_scale | float | (0.017125, 0.017507, 0.017429) | scale values for input normalization (RGB channels) | session.input_scale |

## extract

| Option | Type | Default | Help | Config Field |
|---|---:|---|---|---|
| model_path | str | None | input model | session.model_path |
| config_path | str | None | path to configuration file | common.config_path |
| work_path | str | ./work_dirs/{pipeline_type}/{target_device}/{tensor_bits}bits | work path | common.work_path |
| run_dir | str | {work_path}/{model_id}_{runtime_name}_{model_path}_{model_ext} | run_dir | session.run_dir |
| pipeline_type | str | extract | type of pipeline to run | common.pipeline_type |
| extract_mode | str | operators | extraction mode (submodules, submodule, start2end, operators) | common.extract.mode |
| submodule_name | str | None | name of specific submodule to extract | common.extract.submodule_name |
| max_depth | int | 3 | maximum depth for submodule extraction | common.extract.max_depth |
| start_names | str | None | starting layer names for start2end extraction | common.extract.start_names |
| end_names | str | None | ending layer names for start2end extraction | common.extract.end_names |

## report

| Option | Type | Default | Help | Config Field |
|---|---:|---|---|---|
| pipeline_type | str | compile | type of pipeline to run | common.pipeline_type |
| report_mode | str | detailed | report generation mode (summary or detailed) | common.report.mode |
| report_path | str | ./work_dirs/compile | path where reports will be generated | common.report.path |

## package

| Option | Type | Default | Help | Config Field |
|---|---:|---|---|---|
| pipeline_type | str | package | type of pipeline to run | common.pipeline_type |
| target_device | str | TargetDeviceType.TARGET_DEVICE_AM68A | target device for inference (AM68A, AM69A, etc.) | session.target_device |
| tensor_bits | int | 8 | quantization bit-width for tensors (8 or 16) | session.runtime_options.tensor_bits |
| work_path | str | ./work_dirs/compile/{target_device}/{tensor_bits}bits | work path | common.work_path |
| package_path | str | ./work_dirs/{pipeline_type}/{target_device}/{tensor_bits}bits | packaged path | common.package_path |
| param_template | str | data/templates/configs/param_template_package.yaml | param template path | common.param_template |
