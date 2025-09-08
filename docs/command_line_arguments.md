# Command Line Arguments

## compile

| Argument | Default | Type | Help |
|----------|---------|------|------|
| --model_path | None | str | input model |
| --config_path | None | str | path to configuration file |
| --output_path | ./work_dirs/{pipeline_type}/{target_device}/{tensor_bits}/{model_id}_{runtime_name}_{model_path}_{model_ext} | str | output model path |
| --pipeline_type | compile | str | type of pipeline to run |
| --simplify_model | True | bool | enable model simplification optimizations |
| --optimize_model | True | bool | enable model optimization |
| --shape_inference | True | bool | enable shape inference during optimization |
| --task_type | None | str | type of AI task (classification, detection, segmentation etc.) |
| --num_frames | 10 | int | number of frames to process |
| --display_step | 100 | str | interval for displaying progress information |
| --model_id | None | str | unique id of a model - optional |
| --artifacts_folder | None | str | folder to store compilation artifacts |
| --packaged_path | ./work_dirs/{pipeline_type}_package/{target_device}/{tensor_bits}/{model_id}_{runtime_name}_{model_path}_{model_ext} | str | packaged model path |
| --runtime_name | None | str | name of the runtime session |
| --input_mean | (123.675, 116.28, 103.53) | float | mean values for input normalization (RGB channels) |
| --input_scale | (0.017125, 0.017507, 0.017429) | float | scale values for input normalization (RGB channels) |
| --data_name | None | str | name of the input dataset |
| --data_path | None | str | path to the input data directory |
| --target_device | AM68A | str | target device for inference (AM68A, AM69A, etc.) |
| --target_machine | PC_EMULATION | str | target machine for running the inference (pc, evm) |
| --tidl_offload | True | bool | enable TIDL acceleration for inference |
| --graph_optimization_level | ORT_DISABLE_ALL | int | ONNX Runtime graph optimization level |
| --tensor_bits | 8 | int | quantization bit-width for tensors (8 or 16) |
| --quantization_scale_type | None | int | type of quantization scale to use |
| --calibration_frames | 12 | int | number of frames for quantization calibration |
| --calibration_iterations | 12 | int | number of calibration iterations |
| --quant_params_file_path | - | str | path to quantization parameters file |
| --max_num_subgraph_nodes | 1536 | int | maximum number of nodes in a subgraph |
| --output_feature_16bit_names_list | - | str | list of output layers to keep in 16-bit precision |
| --meta_arch_type | - | int | meta architecture type for object detection |
| --meta_arch_file_path | - | str | path to meta architecture file |
| --detection_threshold | 0.3 | float | confidence threshold for object detection |
| --detection_top_k | 200 | int | number of top detections to keep before NMS |
| --nms_threshold | 0.45 | float | NMS threshold for object detection |
| --keep_top_k | 200 | int | number of top detections to keep after NMS |
| --preprocess_name | None | str | name of the preprocessing pipeline |
| --resize | None | int | resize dimensions for input images (height width) |
| --crop | None | int | crop dimensions for input images (height width) |
| --data_layout | None | str | data layout format (NCHW, NHWC) |
| --reverse_channels | False | bool | reverse color channel order (RGB to BGR) |
| --resize_with_pad | False | bool | resize image with padding to maintain aspect ratio |
| --postprocess_enable | False | bool | enable postprocessing after inference |
| --postprocess_name | None | str | name of the postprocessing pipeline |

## infer

| Argument | Default | Type | Help |
|----------|---------|------|------|
| --model_path | None | str | input model |
| --config_path | None | str | path to configuration file |
| --output_path | ./work_dirs/{pipeline_type}/{target_device}/{tensor_bits}/{model_id}_{runtime_name}_{model_path}_{model_ext} | str | output model path |
| --pipeline_type | compile | str | type of pipeline to run |
| --simplify_model | True | bool | enable model simplification optimizations |
| --optimize_model | True | bool | enable model optimization |
| --shape_inference | True | bool | enable shape inference during optimization |
| --task_type | None | str | type of AI task (classification, detection, segmentation etc.) |
| --num_frames | 10 | int | number of frames to process |
| --display_step | 100 | str | interval for displaying progress information |
| --model_id | None | str | unique id of a model - optional |
| --artifacts_folder | None | str | folder to store compilation artifacts |
| --packaged_path | ./work_dirs/{pipeline_type}_package/{target_device}/{tensor_bits}/{model_id}_{runtime_name}_{model_path}_{model_ext} | str | packaged model path |
| --runtime_name | None | str | name of the runtime session |
| --input_mean | (123.675, 116.28, 103.53) | float | mean values for input normalization (RGB channels) |
| --input_scale | (0.017125, 0.017507, 0.017429) | float | scale values for input normalization (RGB channels) |
| --data_name | None | str | name of the input dataset |
| --data_path | None | str | path to the input data directory |
| --target_device | AM68A | str | target device for inference (AM68A, AM69A, etc.) |
| --target_machine | PC_EMULATION | str | target machine for running the inference (pc, evm) |
| --tidl_offload | True | bool | enable TIDL acceleration for inference |
| --graph_optimization_level | ORT_DISABLE_ALL | int | ONNX Runtime graph optimization level |
| --tensor_bits | 8 | int | quantization bit-width for tensors (8 or 16) |
| --quantization_scale_type | None | int | type of quantization scale to use |
| --calibration_frames | 12 | int | number of frames for quantization calibration |
| --calibration_iterations | 12 | int | number of calibration iterations |
| --quant_params_file_path | - | str | path to quantization parameters file |
| --max_num_subgraph_nodes | 1536 | int | maximum number of nodes in a subgraph |
| --output_feature_16bit_names_list | - | str | list of output layers to keep in 16-bit precision |
| --meta_arch_type | - | int | meta architecture type for object detection |
| --meta_arch_file_path | - | str | path to meta architecture file |
| --detection_threshold | 0.3 | float | confidence threshold for object detection |
| --detection_top_k | 200 | int | number of top detections to keep before NMS |
| --nms_threshold | 0.45 | float | NMS threshold for object detection |
| --keep_top_k | 200 | int | number of top detections to keep after NMS |
| --preprocess_name | None | str | name of the preprocessing pipeline |
| --resize | None | int | resize dimensions for input images (height width) |
| --crop | None | int | crop dimensions for input images (height width) |
| --data_layout | None | str | data layout format (NCHW, NHWC) |
| --reverse_channels | False | bool | reverse color channel order (RGB to BGR) |
| --resize_with_pad | False | bool | resize image with padding to maintain aspect ratio |
| --postprocess_enable | False | bool | enable postprocessing after inference |
| --postprocess_name | None | str | name of the postprocessing pipeline |

## accuracy

| Argument | Default | Type | Help |
|----------|---------|------|------|
| --model_path | None | str | input model |
| --config_path | None | str | path to configuration file |
| --output_path | ./work_dirs/{pipeline_type}/{target_device}/{tensor_bits}/{model_id}_{runtime_name}_{model_path}_{model_ext} | str | output model path |
| --pipeline_type | compile | str | type of pipeline to run |
| --simplify_model | True | bool | enable model simplification optimizations |
| --optimize_model | True | bool | enable model optimization |
| --shape_inference | True | bool | enable shape inference during optimization |
| --task_type | None | str | type of AI task (classification, detection, segmentation etc.) |
| --num_frames | 1000 | int | number of frames to process for accuracy evaluation |
| --display_step | 100 | str | interval for displaying progress information |
| --model_id | None | str | unique id of a model - optional |
| --artifacts_folder | None | str | folder to store compilation artifacts |
| --packaged_path | ./work_dirs/{pipeline_type}_package/{target_device}/{tensor_bits}/{model_id}_{runtime_name}_{model_path}_{model_ext} | str | packaged model path |
| --runtime_name | None | str | name of the runtime session |
| --input_mean | (123.675, 116.28, 103.53) | float | mean values for input normalization (RGB channels) |
| --input_scale | (0.017125, 0.017507, 0.017429) | float | scale values for input normalization (RGB channels) |
| --data_name | None | str | name of the input dataset |
| --data_path | None | str | path to the input data directory |
| --target_device | AM68A | str | target device for inference (AM68A, AM69A, etc.) |
| --target_machine | PC_EMULATION | str | target machine for running the inference (pc, evm) |
| --tidl_offload | True | bool | enable TIDL acceleration for inference |
| --graph_optimization_level | ORT_DISABLE_ALL | int | ONNX Runtime graph optimization level |
| --tensor_bits | 8 | int | quantization bit-width for tensors (8 or 16) |
| --quantization_scale_type | None | int | type of quantization scale to use |
| --calibration_frames | 12 | int | number of frames for quantization calibration |
| --calibration_iterations | 12 | int | number of calibration iterations |
| --quant_params_file_path | - | str | path to quantization parameters file |
| --max_num_subgraph_nodes | 1536 | int | maximum number of nodes in a subgraph |
| --output_feature_16bit_names_list | - | str | list of output layers to keep in 16-bit precision |
| --meta_arch_type | - | int | meta architecture type for object detection |
| --meta_arch_file_path | - | str | path to meta architecture file |
| --detection_threshold | 0.3 | float | confidence threshold for object detection |
| --detection_top_k | 200 | int | number of top detections to keep before NMS |
| --nms_threshold | 0.45 | float | NMS threshold for object detection |
| --keep_top_k | 200 | int | number of top detections to keep after NMS |
| --preprocess_name | None | str | name of the preprocessing pipeline |
| --resize | None | int | resize dimensions for input images (height width) |
| --crop | None | int | crop dimensions for input images (height width) |
| --data_layout | None | str | data layout format (NCHW, NHWC) |
| --reverse_channels | False | bool | reverse color channel order (RGB to BGR) |
| --resize_with_pad | False | bool | resize image with padding to maintain aspect ratio |
| --postprocess_enable | True | bool | enable postprocessing after inference |
| --postprocess_name | None | str | name of the postprocessing pipeline |
| --label_path | None | str | path to ground truth labels for accuracy evaluation |
| --postprocess_resize_with_pad | False | bool | resize output with padding to maintain aspect ratio |
| --postprocess_normalized_detections | False | bool | whether detections are normalized coordinates |
| --postprocess_formatter | None | str | format for postprocessing output |
| --postprocess_shuffle_indices | None | int | indices for shuffling postprocess output |
| --postprocess_squeeze_axis | None | str | axis to squeeze from output tensor |
| --postprocess_reshape_list | None | str | list of reshape operations for output tensors |
| --postprocess_ignore_index | None | str | index to ignore during accuracy calculation |
| --postprocess_logits_bbox_to_bbox_ls | False | bool | convert logits bounding box format to bounding box list |
| --postprocess_keypoint | False | bool | enable keypoint postprocessing |
| --postprocess_save_output | False | bool | save postprocessed output to files |
| --postprocess_save_output_frames | 1 | int | number of output frames to save |

## analyze

| Argument | Default | Type | Help |
|----------|---------|------|------|
| --model_path | None | str | input model |
| --config_path | None | str | path to configuration file |
| --output_path | ./work_dirs/{pipeline_type}/{target_device}/{tensor_bits}/{model_id}_{runtime_name}_{model_path}_{model_ext} | str | output model path |
| --pipeline_type | analyze | str | type of pipeline to run |
| --simplify_model | True | bool | enable model simplification optimizations |
| --optimize_model | True | bool | enable model optimization |
| --shape_inference | True | bool | enable shape inference during optimization |
| --task_type | None | str | type of AI task (classification, detection, segmentation etc.) |
| --num_frames | 10 | int | number of frames to process |
| --display_step | 100 | str | interval for displaying progress information |
| --model_id | None | str | unique id of a model - optional |
| --artifacts_folder | None | str | folder to store compilation artifacts |
| --packaged_path | ./work_dirs/{pipeline_type}_package/{target_device}/{tensor_bits}/{model_id}_{runtime_name}_{model_path}_{model_ext} | str | packaged model path |
| --runtime_name | None | str | name of the runtime session |
| --input_mean | (123.675, 116.28, 103.53) | float | mean values for input normalization (RGB channels) |
| --input_scale | (0.017125, 0.017507, 0.017429) | float | scale values for input normalization (RGB channels) |
| --data_name | None | str | name of the input dataset |
| --data_path | None | str | path to the input data directory |
| --target_device | AM68A | str | target device for inference (AM68A, AM69A, etc.) |
| --target_machine | PC_EMULATION | str | target machine for running the inference (pc, evm) |
| --tidl_offload | True | bool | enable TIDL acceleration for inference |
| --graph_optimization_level | ORT_DISABLE_ALL | int | ONNX Runtime graph optimization level |
| --tensor_bits | 8 | int | quantization bit-width for tensors (8 or 16) |
| --quantization_scale_type | None | int | type of quantization scale to use |
| --calibration_frames | 12 | int | number of frames for quantization calibration |
| --calibration_iterations | 12 | int | number of calibration iterations |
| --quant_params_file_path | - | str | path to quantization parameters file |
| --max_num_subgraph_nodes | 1536 | int | maximum number of nodes in a subgraph |
| --output_feature_16bit_names_list | - | str | list of output layers to keep in 16-bit precision |
| --meta_arch_type | - | int | meta architecture type for object detection |
| --meta_arch_file_path | - | str | path to meta architecture file |
| --detection_threshold | 0.3 | float | confidence threshold for object detection |
| --detection_top_k | 200 | int | number of top detections to keep before NMS |
| --nms_threshold | 0.45 | float | NMS threshold for object detection |
| --keep_top_k | 200 | int | number of top detections to keep after NMS |
| --preprocess_name | None | str | name of the preprocessing pipeline |
| --resize | None | int | resize dimensions for input images (height width) |
| --crop | None | int | crop dimensions for input images (height width) |
| --data_layout | None | str | data layout format (NCHW, NHWC) |
| --reverse_channels | False | bool | reverse color channel order (RGB to BGR) |
| --resize_with_pad | False | bool | resize image with padding to maintain aspect ratio |
| --postprocess_enable | False | bool | enable postprocessing after inference |
| --postprocess_name | None | str | name of the postprocessing pipeline |

## optimize

| Argument | Default | Type | Help |
|----------|---------|------|------|
| --model_path | None | str | input model |
| --config_path | None | str | path to configuration file |
| --output_path | ./work_dirs/{pipeline_type}/{target_device}/{tensor_bits}/{model_id}_{runtime_name}_{model_path}_{model_ext} | str | output model path |
| --pipeline_type | optimize | str | type of pipeline to run |
| --simplify_model | True | bool | enable model simplification optimizations |
| --optimize_model | True | str_to_bool_or_none_or_dict | enable model optimization |
| --shape_inference | True | bool | enable shape inference during optimization |

## extract

| Argument | Default | Type | Help |
|----------|---------|------|------|
| --model_path | None | str | input model |
| --config_path | None | str | path to configuration file |
| --output_path | ./work_dirs/{pipeline_type}/{target_device}/{tensor_bits}/{model_id}_{runtime_name}_{model_path}_{model_ext} | str | output model path |
| --pipeline_type | extract | str | type of pipeline to run |
| --extract_mode | operators | str | extraction mode (submodules, submodule, start2end, operators) |
| --submodule_name | None | str | name of specific submodule to extract |
| --max_depth | 3 | int | maximum depth for submodule extraction |
| --start_names | None | str | starting layer names for start2end extraction |
| --end_names | None | str | ending layer names for start2end extraction |

## report

| Argument | Default | Type | Help |
|----------|---------|------|------|
| --pipeline_type | compile | str | type of pipeline to run |
| --report_mode | detailed | str | report generation mode (summary or detailed) |
| --report_path | ./work_dirs/compile | str | path where reports will be generated |
