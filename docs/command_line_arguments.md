# Command Line Arguments

## compile

| Argument | Type | Default | Help |
|----------|------|---------|------|
| --target_module | str | vision | specify the target module to be used. default: vision eg. --target_module vision |
| --parallel_processes | int | 8 | - |
| --log_file | str | run.log | - |
| --capture_log | str | False | - |
| --pipeline_type | str | compile | type of pipeline to run |
| --model_path | str | None | input model |
| --output_path | str | ./work_dirs/{pipeline_type}/{target_device}/{tensor_bits}/{model_id}_{runtime_name}_{model_path}_{model_ext} | output model path |
| --config_path | str | None | path to configuration file |
| --simplify_model | bool | True | enable model simplification optimizations |
| --optimize_model | bool | True | enable model optimization |
| --shape_inference | bool | True | enable shape inference during optimization |
| --task_type | str | None | type of AI task (classification, detection, segmentation etc.) |
| --num_frames | int | 10 | number of frames to process |
| --display_step | str | 100 | interval for displaying progress information |
| --model_id | str | None | unique id of a model - optional |
| --artifacts_folder | str | None | folder to store compilation artifacts |
| --packaged_path | str | ./work_dirs/{pipeline_type}_package/{target_device}/{tensor_bits}/{model_id}_{runtime_name}_{model_path}_{model_ext} | packaged model path |
| --runtime_name | str | None | name of the runtime session |
| --input_mean | float | (123.675, 116.28, 103.53) | mean values for input normalization (RGB channels) |
| --input_scale | float | (0.017125, 0.017507, 0.017429) | scale values for input normalization (RGB channels) |
| --data_name | str | None | name of the input dataset |
| --data_path | str | None | path to the input data directory |
| --target_device | str | TARGET_DEVICE_AM68A | target device for inference (AM68A, AM69A, etc.) |
| --tidl_offload | bool | True | enable TIDL acceleration for inference |
| --graph_optimization_level | int | ORT_DISABLE_ALL | ONNX Runtime graph optimization level |
| --tensor_bits | int | 8 | quantization bit-width for tensors (8 or 16) |
| --quantization_scale_type | int | None | type of quantization scale to use |
| --calibration_frames | int | 12 | number of frames for quantization calibration |
| --calibration_iterations | int | 12 | number of calibration iterations |
| --quant_params_file_path | str/bool/None | SUPPRESS | path to quantization parameters file |
| --max_num_subgraph_nodes | int | 1536 | maximum number of nodes in a subgraph |
| --output_feature_16bit_names_list | str | SUPPRESS | list of output layers to keep in 16-bit precision |
| --meta_arch_type | int | SUPPRESS | meta architecture type for object detection |
| --meta_arch_file_path | str | SUPPRESS | path to meta architecture file |
| --detection_threshold | float | 0.3 | confidence threshold for object detection |
| --detection_top_k | int | 200 | number of top detections to keep before NMS |
| --nms_threshold | float | 0.45 | NMS threshold for object detection |
| --keep_top_k | int | 200 | number of top detections to keep after NMS |
| --preprocess_name | str | None | name of the preprocessing pipeline |
| --resize | int | None | resize dimensions for input images (height width) |
| --crop | int | None | crop dimensions for input images (height width) |
| --data_layout | str | None | data layout format (NCHW, NHWC) |
| --reverse_channels | bool | False | reverse color channel order (RGB to BGR) |
| --resize_with_pad | bool | False | resize image with padding to maintain aspect ratio |
| --postprocess_name | str | None | name of the postprocessing pipeline |

## infer

| Argument | Type | Default | Help |
|----------|------|---------|------|
| --target_module | str | vision | specify the target module to be used. default: vision eg. --target_module vision |
| --parallel_processes | int | 8 | - |
| --log_file | str | run.log | - |
| --capture_log | str | False | - |
| --pipeline_type | str | compile | type of pipeline to run |
| --model_path | str | None | input model |
| --output_path | str | ./work_dirs/{pipeline_type}/{target_device}/{tensor_bits}/{model_id}_{runtime_name}_{model_path}_{model_ext} | output model path |
| --config_path | str | None | path to configuration file |
| --simplify_model | bool | True | enable model simplification optimizations |
| --optimize_model | bool | True | enable model optimization |
| --shape_inference | bool | True | enable shape inference during optimization |
| --task_type | str | None | type of AI task (classification, detection, segmentation etc.) |
| --num_frames | int | 10 | number of frames to process |
| --display_step | str | 100 | interval for displaying progress information |
| --model_id | str | None | unique id of a model - optional |
| --artifacts_folder | str | None | folder to store compilation artifacts |
| --packaged_path | str | ./work_dirs/{pipeline_type}_package/{target_device}/{tensor_bits}/{model_id}_{runtime_name}_{model_path}_{model_ext} | packaged model path |
| --runtime_name | str | None | name of the runtime session |
| --input_mean | float | (123.675, 116.28, 103.53) | mean values for input normalization (RGB channels) |
| --input_scale | float | (0.017125, 0.017507, 0.017429) | scale values for input normalization (RGB channels) |
| --data_name | str | None | name of the input dataset |
| --data_path | str | None | path to the input data directory |
| --target_device | str | TARGET_DEVICE_AM68A | target device for inference (AM68A, AM69A, etc.) |
| --tidl_offload | bool | True | enable TIDL acceleration for inference |
| --graph_optimization_level | int | ORT_DISABLE_ALL | ONNX Runtime graph optimization level |
| --tensor_bits | int | 8 | quantization bit-width for tensors (8 or 16) |
| --quantization_scale_type | int | None | type of quantization scale to use |
| --calibration_frames | int | 12 | number of frames for quantization calibration |
| --calibration_iterations | int | 12 | number of calibration iterations |
| --quant_params_file_path | str/bool/None | SUPPRESS | path to quantization parameters file |
| --max_num_subgraph_nodes | int | 1536 | maximum number of nodes in a subgraph |
| --output_feature_16bit_names_list | str | SUPPRESS | list of output layers to keep in 16-bit precision |
| --meta_arch_type | int | SUPPRESS | meta architecture type for object detection |
| --meta_arch_file_path | str | SUPPRESS | path to meta architecture file |
| --detection_threshold | float | 0.3 | confidence threshold for object detection |
| --detection_top_k | int | 200 | number of top detections to keep before NMS |
| --nms_threshold | float | 0.45 | NMS threshold for object detection |
| --keep_top_k | int | 200 | number of top detections to keep after NMS |
| --preprocess_name | str | None | name of the preprocessing pipeline |
| --resize | int | None | resize dimensions for input images (height width) |
| --crop | int | None | crop dimensions for input images (height width) |
| --data_layout | str | None | data layout format (NCHW, NHWC) |
| --reverse_channels | bool | False | reverse color channel order (RGB to BGR) |
| --resize_with_pad | bool | False | resize image with padding to maintain aspect ratio |
| --postprocess_name | str | None | name of the postprocessing pipeline |

## accuracy

| Argument | Type | Default | Help |
|----------|------|---------|------|
| --target_module | str | vision | specify the target module to be used. default: vision eg. --target_module vision |
| --parallel_processes | int | 8 | - |
| --log_file | str | run.log | - |
| --capture_log | str | False | - |
| --pipeline_type | str | compile | type of pipeline to run |
| --model_path | str | None | input model |
| --output_path | str | ./work_dirs/{pipeline_type}/{target_device}/{tensor_bits}/{model_id}_{runtime_name}_{model_path}_{model_ext} | output model path |
| --config_path | str | None | path to configuration file |
| --simplify_model | bool | True | enable model simplification optimizations |
| --optimize_model | bool | True | enable model optimization |
| --shape_inference | bool | True | enable shape inference during optimization |
| --task_type | str | None | type of AI task (classification, detection, segmentation etc.) |
| --num_frames | int | 1000 | number of frames to process for accuracy evaluation |
| --display_step | str | 100 | interval for displaying progress information |
| --model_id | str | None | unique id of a model - optional |
| --artifacts_folder | str | None | folder to store compilation artifacts |
| --packaged_path | str | ./work_dirs/{pipeline_type}_package/{target_device}/{tensor_bits}/{model_id}_{runtime_name}_{model_path}_{model_ext} | packaged model path |
| --runtime_name | str | None | name of the runtime session |
| --input_mean | float | (123.675, 116.28, 103.53) | mean values for input normalization (RGB channels) |
| --input_scale | float | (0.017125, 0.017507, 0.017429) | scale values for input normalization (RGB channels) |
| --data_name | str | None | name of the input dataset |
| --data_path | str | None | path to the input data directory |
| --target_device | str | TARGET_DEVICE_AM68A | target device for inference (AM68A, AM69A, etc.) |
| --tidl_offload | bool | True | enable TIDL acceleration for inference |
| --graph_optimization_level | int | ORT_DISABLE_ALL | ONNX Runtime graph optimization level |
| --tensor_bits | int | 8 | quantization bit-width for tensors (8 or 16) |
| --quantization_scale_type | int | None | type of quantization scale to use |
| --calibration_frames | int | 12 | number of frames for quantization calibration |
| --calibration_iterations | int | 12 | number of calibration iterations |
| --quant_params_file_path | str/bool/None | SUPPRESS | path to quantization parameters file |
| --max_num_subgraph_nodes | int | 1536 | maximum number of nodes in a subgraph |
| --output_feature_16bit_names_list | str | SUPPRESS | list of output layers to keep in 16-bit precision |
| --meta_arch_type | int | SUPPRESS | meta architecture type for object detection |
| --meta_arch_file_path | str | SUPPRESS | path to meta architecture file |
| --detection_threshold | float | 0.3 | confidence threshold for object detection |
| --detection_top_k | int | 200 | number of top detections to keep before NMS |
| --nms_threshold | float | 0.45 | NMS threshold for object detection |
| --keep_top_k | int | 200 | number of top detections to keep after NMS |
| --preprocess_name | str | None | name of the preprocessing pipeline |
| --resize | int | None | resize dimensions for input images (height width) |
| --crop | int | None | crop dimensions for input images (height width) |
| --data_layout | str | None | data layout format (NCHW, NHWC) |
| --reverse_channels | bool | False | reverse color channel order (RGB to BGR) |
| --resize_with_pad | bool | False | resize image with padding to maintain aspect ratio |
| --postprocess_name | str | None | name of the postprocessing pipeline |
| --label_path | str | None | path to ground truth labels for accuracy evaluation |
| --postprocess_resize_with_pad | bool | False | resize output with padding to maintain aspect ratio |
| --postprocess_normalized_detections | bool | False | whether detections are normalized coordinates |
| --postprocess_formatter | str | None | format for postprocessing output |
| --postprocess_shuffle_indices | int | None | indices for shuffling postprocess output |
| --postprocess_squeeze_axis | int | None | axis to squeeze from output tensor |
| --postprocess_reshape_list | list | None | list of reshape operations for output tensors |
| --postprocess_ignore_index | str | None | index to ignore during accuracy calculation |
| --postprocess_logits_bbox_to_bbox_ls | bool | False | convert logits bounding box format to bounding box list |
| --postprocess_detection_threshold | float | 0.3 | detection confidence threshold for postprocessing |
| --postprocess_detection_top_k | int | 200 | top-k detections to keep in postprocessing |
| --postprocess_detection_keep_top_k | float | 200 | number of detections to keep after NMS in postprocessing |
| --postprocess_keypoint | bool | False | enable keypoint postprocessing |
| --postprocess_save_output | bool | False | save postprocessed output to files |
| --postprocess_save_output_frames | int | 1 | number of output frames to save |

## analyze

| Argument | Type | Default | Help |
|----------|------|---------|------|
| --target_module | str | vision | specify the target module to be used. default: vision eg. --target_module vision |
| --parallel_processes | int | 8 | - |
| --log_file | str | run.log | - |
| --capture_log | str | False | - |
| --pipeline_type | str | analyze | type of pipeline to run |
| --model_path | str | None | input model |
| --output_path | str | ./work_dirs/{pipeline_type}/{target_device}/{tensor_bits}/{model_id}_{runtime_name}_{model_path}_{model_ext} | output model path |
| --config_path | str | None | path to configuration file |
| --simplify_model | bool | True | enable model simplification optimizations |
| --optimize_model | bool | True | enable model optimization |
| --shape_inference | bool | True | enable shape inference during optimization |
| --task_type | str | None | type of AI task (classification, detection, segmentation etc.) |
| --num_frames | int | 10 | number of frames to process |
| --display_step | str | 100 | interval for displaying progress information |
| --model_id | str | None | unique id of a model - optional |
| --artifacts_folder | str | None | folder to store compilation artifacts |
| --packaged_path | str | ./work_dirs/{pipeline_type}_package/{target_device}/{tensor_bits}/{model_id}_{runtime_name}_{model_path}_{model_ext} | packaged model path |
| --runtime_name | str | None | name of the runtime session |
| --input_mean | float | (123.675, 116.28, 103.53) | mean values for input normalization (RGB channels) |
| --input_scale | float | (0.017125, 0.017507, 0.017429) | scale values for input normalization (RGB channels) |
| --data_name | str | None | name of the input dataset |
| --data_path | str | None | path to the input data directory |
| --target_device | str | TARGET_DEVICE_AM68A | target device for inference (AM68A, AM69A, etc.) |
| --tidl_offload | bool | True | enable TIDL acceleration for inference |
| --graph_optimization_level | int | ORT_DISABLE_ALL | ONNX Runtime graph optimization level |
| --tensor_bits | int | 8 | quantization bit-width for tensors (8 or 16) |
| --quantization_scale_type | int | None | type of quantization scale to use |
| --calibration_frames | int | 12 | number of frames for quantization calibration |
| --calibration_iterations | int | 12 | number of calibration iterations |
| --quant_params_file_path | str/bool/None | SUPPRESS | path to quantization parameters file |
| --max_num_subgraph_nodes | int | 1536 | maximum number of nodes in a subgraph |
| --output_feature_16bit_names_list | str | SUPPRESS | list of output layers to keep in 16-bit precision |
| --meta_arch_type | int | SUPPRESS | meta architecture type for object detection |
| --meta_arch_file_path | str | SUPPRESS | path to meta architecture file |
| --detection_threshold | float | 0.3 | confidence threshold for object detection |
| --detection_top_k | int | 200 | number of top detections to keep before NMS |
| --nms_threshold | float | 0.45 | NMS threshold for object detection |
| --keep_top_k | int | 200 | number of top detections to keep after NMS |
| --preprocess_name | str | None | name of the preprocessing pipeline |
| --resize | int | None | resize dimensions for input images (height width) |
| --crop | int | None | crop dimensions for input images (height width) |
| --data_layout | str | None | data layout format (NCHW, NHWC) |
| --reverse_channels | bool | False | reverse color channel order (RGB to BGR) |
| --resize_with_pad | bool | False | resize image with padding to maintain aspect ratio |
| --postprocess_name | str | None | name of the postprocessing pipeline |

## optimize

| Argument | Type | Default | Help |
|----------|------|---------|------|
| --target_module | str | vision | specify the target module to be used. default: vision eg. --target_module vision |
| --parallel_processes | int | 8 | - |
| --log_file | str | run.log | - |
| --capture_log | str | False | - |
| --pipeline_type | str | optimize | type of pipeline to run |
| --model_path | str | None | input model |
| --output_path | str | ./work_dirs/{pipeline_type}/{target_device}/{tensor_bits}/{model_id}_{runtime_name}_{model_path}_{model_ext} | output model path |
| --config_path | str | None | - |
| --simplify_model | bool | True | enable model simplification optimizations |
| --optimize_model | bool | True | enable model optimization |
| --shape_inference | bool | True | enable shape inference during optimization |

## extract

| Argument | Type | Default | Help |
|----------|------|---------|------|
| --target_module | str | vision | specify the target module to be used. default: vision eg. --target_module vision |
| --parallel_processes | int | 8 | - |
| --log_file | str | run.log | - |
| --capture_log | str | False | - |
| --pipeline_type | str | extract | type of pipeline to run |
| --model_path | str | None | input model |
| --output_path | str | ./work_dirs/{pipeline_type}/{target_device}/{tensor_bits}/{model_id}_{runtime_name}_{model_path}_{model_ext} | output model path |
| --config_path | str | None | - |
| --extract_mode | str | operators | extraction mode (submodules, submodule, start2end, operators) |
| --submodule_name | str | None | name of specific submodule to extract |
| --max_depth | int | 3 | maximum depth for submodule extraction |
| --start_names | str | None | starting layer names for start2end extraction |
| --end_names | str | None | ending layer names for start2end extraction |

## report

| Argument | Type | Default | Help |
|----------|------|---------|------|
| --target_module | str | vision | specify the target module to be used. default: vision eg. --target_module vision |
| --parallel_processes | int | 8 | - |
| --log_file | str | run.log | - |
| --capture_log | str | False | - |
| --pipeline_type | str | compile | type of pipeline to run |
| --report_mode | str | detailed | report generation mode (summary or detailed) |
| --report_path | str | ./work_dirs/compile | path where reports will be generated |
