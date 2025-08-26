# Command Line Arguments Reference

## compile

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_path` | str | None | Input model path |
| `--output_path` | str | `./work_dirs/{pipeline_type}/{target_device}/{tensor_bits}/{model_id}_{runtime_name}_{model_path}_{model_ext}` | Output model path |
| `--target_module` | str | vision | Target module to be used (choices: vision) |
| `--parallel_processes` | int | 8 | Number of parallel processes |
| `--log_file` | str | run.log | Log file name |
| `--capture_log` | str | False | Log capture mode |
| `--simplify_model` | bool | True | Enable model simplification optimizations |
| `--optimize_model` | bool | True | Enable model optimization |
| `--shape_inference` | bool | True | Enable shape inference during optimization |
| `--task_type` | str | None | Type of AI task (classification, detection, segmentation etc.) |
| `--num_frames` | int | 10 | Number of frames to process |
| `--config_path` | str | None | Path to configuration file |
| `--display_step` | str | 100 | Interval for displaying progress information |
| `--model_id` | str | None | Unique id of a model - optional |
| `--artifacts_folder` | str | None | Folder to store compilation artifacts |
| `--packaged_path` | str | `./work_dirs/{pipeline_type}_package/{target_device}/{tensor_bits}/{model_id}_{runtime_name}_{model_path}_{model_ext}` | Packaged model path |
| `--runtime_name` | str | None | Name of the runtime session |
| `--input_mean` | float[] | (123.675, 116.28, 103.53) | Mean values for input normalization (RGB channels) |
| `--input_scale` | float[] | (0.017125, 0.017507, 0.017429) | Scale values for input normalization (RGB channels) |
| `--data_name` | str | None | Name of the input dataset |
| `--data_path` | str | None | Path to the input data directory |
| `--target_device` | str | AM68A | Target device for inference (AM68A, AM69A, etc.) |
| `--tidl_offload` | bool | True | Enable TIDL acceleration for inference |
| `--graph_optimization_level` | int | ORT_DISABLE_ALL | ONNX Runtime graph optimization level |
| `--tensor_bits` | int | 8 | Quantization bit-width for tensors (8 or 16) |
| `--quantization_scale_type` | int | None | Type of quantization scale to use |
| `--calibration_frames` | int | 12 | Number of frames for quantization calibration |
| `--calibration_iterations` | int | 12 | Number of calibration iterations |
| `--quant_params_file_path` | str/bool | SUPPRESS | Path to quantization parameters file |
| `--max_num_subgraph_nodes` | int | 1536 | Maximum number of nodes in a subgraph |
| `--output_feature_16bit_names_list` | str | SUPPRESS | List of output layers to keep in 16-bit precision |
| `--meta_arch_type` | int | SUPPRESS | Meta architecture type for object detection |
| `--meta_arch_file_path` | str | SUPPRESS | Path to meta architecture file |
| `--detection_threshold` | float | 0.3 | Confidence threshold for object detection |
| `--detection_top_k` | int | 200 | Number of top detections to keep before NMS |
| `--nms_threshold` | float | 0.45 | NMS threshold for object detection |
| `--keep_top_k` | int | 200 | Number of top detections to keep after NMS |
| `--preprocess_name` | str | None | Name of the preprocessing pipeline |
| `--resize` | int[] | None | Resize dimensions for input images (height width) |
| `--crop` | int[] | None | Crop dimensions for input images (height width) |
| `--data_layout` | str | None | Data layout format (NCHW, NHWC) |
| `--reverse_channels` | bool | False | Reverse color channel order (RGB to BGR) |
| `--resize_with_pad` | bool | False | Resize image with padding to maintain aspect ratio |
| `--postprocess_name` | str | None | Name of the postprocessing pipeline |

## infer

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_path` | str | None | Input model path |
| `--output_path` | str | `./work_dirs/{pipeline_type}/{target_device}/{tensor_bits}/{model_id}_{runtime_name}_{model_path}_{model_ext}` | Output model path |
| `--target_module` | str | vision | Target module to be used (choices: vision) |
| `--parallel_processes` | int | 8 | Number of parallel processes |
| `--log_file` | str | run.log | Log file name |
| `--capture_log` | str | False | Log capture mode |
| `--simplify_model` | bool | True | Enable model simplification optimizations |
| `--optimize_model` | bool | True | Enable model optimization |
| `--shape_inference` | bool | True | Enable shape inference during optimization |
| `--task_type` | str | None | Type of AI task (classification, detection, segmentation etc.) |
| `--num_frames` | int | 10 | Number of frames to process |
| `--config_path` | str | None | Path to configuration file |
| `--display_step` | str | 100 | Interval for displaying progress information |
| `--model_id` | str | None | Unique id of a model - optional |
| `--artifacts_folder` | str | None | Folder to store compilation artifacts |
| `--packaged_path` | str | `./work_dirs/{pipeline_type}_package/{target_device}/{tensor_bits}/{model_id}_{runtime_name}_{model_path}_{model_ext}` | Packaged model path |
| `--runtime_name` | str | None | Name of the runtime session |
| `--input_mean` | float[] | (123.675, 116.28, 103.53) | Mean values for input normalization (RGB channels) |
| `--input_scale` | float[] | (0.017125, 0.017507, 0.017429) | Scale values for input normalization (RGB channels) |
| `--data_name` | str | None | Name of the input dataset |
| `--data_path` | str | None | Path to the input data directory |
| `--target_device` | str | AM68A | Target device for inference (AM68A, AM69A, etc.) |
| `--tidl_offload` | bool | True | Enable TIDL acceleration for inference |
| `--graph_optimization_level` | int | ORT_DISABLE_ALL | ONNX Runtime graph optimization level |
| `--tensor_bits` | int | 8 | Quantization bit-width for tensors (8 or 16) |
| `--quantization_scale_type` | int | None | Type of quantization scale to use |
| `--calibration_frames` | int | 12 | Number of frames for quantization calibration |
| `--calibration_iterations` | int | 12 | Number of calibration iterations |
| `--quant_params_file_path` | str/bool | SUPPRESS | Path to quantization parameters file |
| `--max_num_subgraph_nodes` | int | 1536 | Maximum number of nodes in a subgraph |
| `--output_feature_16bit_names_list` | str | SUPPRESS | List of output layers to keep in 16-bit precision |
| `--meta_arch_type` | int | SUPPRESS | Meta architecture type for object detection |
| `--meta_arch_file_path` | str | SUPPRESS | Path to meta architecture file |
| `--detection_threshold` | float | 0.3 | Confidence threshold for object detection |
| `--detection_top_k` | int | 200 | Number of top detections to keep before NMS |
| `--nms_threshold` | float | 0.45 | NMS threshold for object detection |
| `--keep_top_k` | int | 200 | Number of top detections to keep after NMS |
| `--preprocess_name` | str | None | Name of the preprocessing pipeline |
| `--resize` | int[] | None | Resize dimensions for input images (height width) |
| `--crop` | int[] | None | Crop dimensions for input images (height width) |
| `--data_layout` | str | None | Data layout format (NCHW, NHWC) |
| `--reverse_channels` | bool | False | Reverse color channel order (RGB to BGR) |
| `--resize_with_pad` | bool | False | Resize image with padding to maintain aspect ratio |
| `--postprocess_name` | str | None | Name of the postprocessing pipeline |

## accuracy

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_path` | str | None | Input model path |
| `--output_path` | str | `./work_dirs/{pipeline_type}/{target_device}/{tensor_bits}/{model_id}_{runtime_name}_{model_path}_{model_ext}` | Output model path |
| `--target_module` | str | vision | Target module to be used (choices: vision) |
| `--parallel_processes` | int | 8 | Number of parallel processes |
| `--log_file` | str | run.log | Log file name |
| `--capture_log` | str | False | Log capture mode |
| `--simplify_model` | bool | True | Enable model simplification optimizations |
| `--optimize_model` | bool | True | Enable model optimization |
| `--shape_inference` | bool | True | Enable shape inference during optimization |
| `--task_type` | str | None | Type of AI task (classification, detection, segmentation etc.) |
| `--num_frames` | int | 1000 | Number of frames to process for accuracy evaluation |
| `--config_path` | str | None | Path to configuration file |
| `--display_step` | str | 100 | Interval for displaying progress information |
| `--model_id` | str | None | Unique id of a model - optional |
| `--artifacts_folder` | str | None | Folder to store compilation artifacts |
| `--packaged_path` | str | `./work_dirs/{pipeline_type}_package/{target_device}/{tensor_bits}/{model_id}_{runtime_name}_{model_path}_{model_ext}` | Packaged model path |
| `--runtime_name` | str | None | Name of the runtime session |
| `--input_mean` | float[] | (123.675, 116.28, 103.53) | Mean values for input normalization (RGB channels) |
| `--input_scale` | float[] | (0.017125, 0.017507, 0.017429) | Scale values for input normalization (RGB channels) |
| `--data_name` | str | None | Name of the input dataset |
| `--data_path` | str | None | Path to the input data directory |
| `--label_path` | str | None | Path to ground truth labels for accuracy evaluation |
| `--target_device` | str | AM68A | Target device for inference (AM68A, AM69A, etc.) |
| `--tidl_offload` | bool | True | Enable TIDL acceleration for inference |
| `--graph_optimization_level` | int | ORT_DISABLE_ALL | ONNX Runtime graph optimization level |
| `--tensor_bits` | int | 8 | Quantization bit-width for tensors (8 or 16) |
| `--quantization_scale_type` | int | None | Type of quantization scale to use |
| `--calibration_frames` | int | 12 | Number of frames for quantization calibration |
| `--calibration_iterations` | int | 12 | Number of calibration iterations |
| `--quant_params_file_path` | str/bool | SUPPRESS | Path to quantization parameters file |
| `--max_num_subgraph_nodes` | int | 1536 | Maximum number of nodes in a subgraph |
| `--output_feature_16bit_names_list` | str | SUPPRESS | List of output layers to keep in 16-bit precision |
| `--meta_arch_type` | int | SUPPRESS | Meta architecture type for object detection |
| `--meta_arch_file_path` | str | SUPPRESS | Path to meta architecture file |
| `--detection_threshold` | float | 0.3 | Confidence threshold for object detection |
| `--detection_top_k` | int | 200 | Number of top detections to keep before NMS |
| `--nms_threshold` | float | 0.45 | NMS threshold for object detection |
| `--keep_top_k` | int | 200 | Number of top detections to keep after NMS |
| `--preprocess_name` | str | None | Name of the preprocessing pipeline |
| `--resize` | int[] | None | Resize dimensions for input images (height width) |
| `--crop` | int[] | None | Crop dimensions for input images (height width) |
| `--data_layout` | str | None | Data layout format (NCHW, NHWC) |
| `--reverse_channels` | bool | False | Reverse color channel order (RGB to BGR) |
| `--resize_with_pad` | bool | False | Resize image with padding to maintain aspect ratio |
| `--postprocess_name` | str | None | Name of the postprocessing pipeline |
| `--postprocess_resize_with_pad` | bool | False | Resize output with padding to maintain aspect ratio |
| `--postprocess_normalized_detections` | bool | False | Whether detections are normalized coordinates |
| `--postprocess_formatter` | str | None | Format for postprocessing output |
| `--postprocess_shuffle_indices` | int[] | None | Indices for shuffling postprocess output |
| `--postprocess_squeeze_axis` | int | None | Axis to squeeze from output tensor |
| `--postprocess_reshape_list` | str | None | List of reshape operations for output tensors |
| `--postprocess_ignore_index` | str | None | Index to ignore during accuracy calculation |
| `--postprocess_logits_bbox_to_bbox_ls` | bool | False | Convert logits bounding box format to bounding box list |
| `--postprocess_detection_threshold` | float | 0.3 | Detection confidence threshold for postprocessing |
| `--postprocess_detection_top_k` | int | 200 | Top-k detections to keep in postprocessing |
| `--postprocess_detection_keep_top_k` | float | 200 | Number of detections to keep after NMS in postprocessing |
| `--postprocess_keypoint` | bool | False | Enable keypoint postprocessing |
| `--postprocess_save_output` | bool | False | Save postprocessed output to files |
| `--postprocess_save_output_frames` | int | 1 | Number of output frames to save |

## analyze

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_path` | str | None | Input model path |
| `--output_path` | str | `./work_dirs/{pipeline_type}/{target_device}/{tensor_bits}/{model_id}_{runtime_name}_{model_path}_{model_ext}` | Output model path |
| `--target_module` | str | vision | Target module to be used (choices: vision) |
| `--parallel_processes` | int | 8 | Number of parallel processes |
| `--log_file` | str | run.log | Log file name |
| `--capture_log` | str | False | Log capture mode |
| `--simplify_model` | bool | True | Enable model simplification optimizations |
| `--optimize_model` | bool | True | Enable model optimization |
| `--shape_inference` | bool | True | Enable shape inference during optimization |
| `--task_type` | str | None | Type of AI task (classification, detection, segmentation etc.) |
| `--num_frames` | int | 10 | Number of frames to process |
| `--config_path` | str | None | Path to configuration file |
| `--display_step` | str | 100 | Interval for displaying progress information |
| `--model_id` | str | None | Unique id of a model - optional |
| `--artifacts_folder` | str | None | Folder to store compilation artifacts |
| `--packaged_path` | str | `./work_dirs/{pipeline_type}_package/{target_device}/{tensor_bits}/{model_id}_{runtime_name}_{model_path}_{model_ext}` | Packaged model path |
| `--runtime_name` | str | None | Name of the runtime session |
| `--input_mean` | float[] | (123.675, 116.28, 103.53) | Mean values for input normalization (RGB channels) |
| `--input_scale` | float[] | (0.017125, 0.017507, 0.017429) | Scale values for input normalization (RGB channels) |
| `--data_name` | str | None | Name of the input dataset |
| `--data_path` | str | None | Path to the input data directory |
| `--target_device` | str | AM68A | Target device for inference (AM68A, AM69A, etc.) |
| `--tidl_offload` | bool | True | Enable TIDL acceleration for inference |
| `--graph_optimization_level` | int | ORT_DISABLE_ALL | ONNX Runtime graph optimization level |
| `--tensor_bits` | int | 8 | Quantization bit-width for tensors (8 or 16) |
| `--quantization_scale_type` | int | None | Type of quantization scale to use |
| `--calibration_frames` | int | 12 | Number of frames for quantization calibration |
| `--calibration_iterations` | int | 12 | Number of calibration iterations |
| `--quant_params_file_path` | str/bool | SUPPRESS | Path to quantization parameters file |
| `--max_num_subgraph_nodes` | int | 1536 | Maximum number of nodes in a subgraph |
| `--output_feature_16bit_names_list` | str | SUPPRESS | List of output layers to keep in 16-bit precision |
| `--meta_arch_type` | int | SUPPRESS | Meta architecture type for object detection |
| `--meta_arch_file_path` | str | SUPPRESS | Path to meta architecture file |
| `--detection_threshold` | float | 0.3 | Confidence threshold for object detection |
| `--detection_top_k` | int | 200 | Number of top detections to keep before NMS |
| `--nms_threshold` | float | 0.45 | NMS threshold for object detection |
| `--keep_top_k` | int | 200 | Number of top detections to keep after NMS |
| `--preprocess_name` | str | None | Name of the preprocessing pipeline |
| `--resize` | int[] | None | Resize dimensions for input images (height width) |
| `--crop` | int[] | None | Crop dimensions for input images (height width) |
| `--data_layout` | str | None | Data layout format (NCHW, NHWC) |
| `--reverse_channels` | bool | False | Reverse color channel order (RGB to BGR) |
| `--resize_with_pad` | bool | False | Resize image with padding to maintain aspect ratio |
| `--postprocess_name` | str | None | Name of the postprocessing pipeline |

## optimize

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_path` | str | None | Input model path |
| `--output_path` | str | `./work_dirs/{pipeline_type}/{target_device}/{tensor_bits}/{model_id}_{runtime_name}_{model_path}_{model_ext}` | Output model path |
| `--target_module` | str | vision | Target module to be used (choices: vision) |
| `--parallel_processes` | int | 8 | Number of parallel processes |
| `--log_file` | str | run.log | Log file name |
| `--capture_log` | str | False | Log capture mode |
| `--simplify_model` | bool | True | Enable model simplification optimizations |
| `--optimize_model` | bool | True | Enable model optimization |
| `--shape_inference` | bool | True | Enable shape inference during optimization |

## extract

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_path` | str | None | Input model path |
| `--output_path` | str | `./work_dirs/{pipeline_type}/{target_device}/{tensor_bits}/{model_id}_{runtime_name}_{model_path}_{model_ext}` | Output model path |
| `--target_module` | str | vision | Target module to be used (choices: vision) |
| `--parallel_processes` | int | 8 | Number of parallel processes |
| `--log_file` | str | run.log | Log file name |
| `--capture_log` | str | False | Log capture mode |
| `--extract_mode` | str | operators | Extraction mode (submodules, submodule, start2end, operators) |
| `--submodule_name` | str | None | Name of specific submodule to extract |
| `--max_depth` | int | 3 | Maximum depth for submodule extraction |
| `--start_names` | str | None | Starting layer names for start2end extraction |
| `--end_names` | str | None | Ending layer names for start2end extraction |

## report

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_path` | str | None | Input model path |
| `--output_path` | str | `./work_dirs/{pipeline_type}/{target_device}/{tensor_bits}/{model_id}_{runtime_name}_{model_path}_{model_ext}` | Output model path |
| `--target_module` | str | vision | Target module to be used (choices: vision) |
| `--parallel_processes` | int | 8 | Number of parallel processes |
| `--log_file` | str | run.log | Log file name |
| `--capture_log` | str | False | Log capture mode |
| `--report_mode` | str | detailed | Report generation mode (summary or detailed) |
| `--report_path` | str | ./work_dirs/compile | Path where reports will be generated |

