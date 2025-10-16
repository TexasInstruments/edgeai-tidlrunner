# Copyright (c) 2018-2025, Texas Instruments
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import enum
import copy
import os
import sys
import warnings
import argparse

from edgeai_tidlrunner.rtwrapper.options import options_default
from . import constants
from .constants import presets
from ...common import utils
from ...common.bases import settings_base


RUNTIME_SETTINGS_DEFAULT = {
    # whether to run the inference on TIDL or on ARM without TIDL
    # True: with TIDL
    # False: No TIDL
    'tidl_offload': True,

    'target_device': presets.TargetDeviceType.TARGET_DEVICE_AM68A,
    'target_machine': presets.TargetMachineType.TARGET_MACHINE_PC_EMULATION,
    'target_device_preset': True,

    'runtime_options': options_default.RUNTIME_OPTIONS_DEFAULT
}


##########################################################################
SETTINGS_DEFAULT = {}
COPY_SETTINGS_DEFAULT = {}

SETTINGS_DEFAULT['basic'] = settings_base.SETTING_PIPELINE_RUNNER_ARGS_DICT | {
    'pipeline_type': {'dest': 'common.pipeline_type', 'default': None, 'type': str, 'metavar': 'value', 'help': 'type of pipeline to run'},
    'verbose':       {'dest': 'common.verbose', 'default': 0, 'type': int, 'metavar': 'value', 'help': 'verbosity level'},
}

COPY_SETTINGS_DEFAULT['basic'] = {}

##########################################################################
SETTINGS_DEFAULT['optimize'] = SETTINGS_DEFAULT['basic'] | {
    'model_path':                       {'dest': 'session.model_path', 'default': None, 'type': str, 'group':'model', 'metavar': 'value', 'help': 'input model'},
    'config_path':                      {'dest': 'common.config_path', 'default': None, 'type': str, 'group':'model', 'metavar': 'value', 'help': 'path to configuration file'},    
    'work_path':                        {'dest': 'common.work_path', 'default':'./work_dirs/{pipeline_type}/{target_device}/{tensor_bits}bits', 'type':str, 'metavar':'value', 'help':'work path'},   
    'run_dir':                      {'dest': 'session.run_dir', 'default':'{work_path}/{model_id}_{runtime_name}_{model_path}_{model_ext}', 'type':str, 'metavar':'value', 'help':'run_dir'},
    'pipeline_type':                    {'dest': 'common.pipeline_type', 'default': 'optimize', 'type': str, 'metavar': 'value', 'help': 'type of pipeline to run'},    
    'optimize_model':                   {'dest': 'common.optimize.optimize_model', 'default': True, 'type': utils.str_to_bool_or_none_or_dict, 'metavar': 'value', 'help': 'enable model optimization'},
    'simplify_model':                   {'dest': 'common.optimize.simplify_mode', 'default': 'pre', 'type': utils.str_to_bool, 'metavar': 'value', 'help': 'enable model simplification optimizations'},
    'shape_inference':                  {'dest': 'common.optimize.shape_inference_mode', 'default': 'all', 'type': utils.str_or_none_or_bool, 'metavar': 'value', 'help': 'enable shape inference during optimization'},
    'input_optimization':               {'dest': 'session.input_optimization', 'default': False, 'type': utils.str_to_bool, 'metavar': 'value', 'help': 'merge in input_mean and input_scale into the model if possible, so that model input can be in uint8 and not float32'},
    'input_mean':                       {'dest': 'session.input_mean', 'default': (123.675, 116.28, 103.53), 'type': float, 'nargs': '*', 'metavar': 'value', 'help': 'mean values for input normalization (RGB channels)'},
    'input_scale':                      {'dest': 'session.input_scale', 'default': (0.017125, 0.017507, 0.017429), 'type': float, 'nargs': '*', 'metavar': 'value', 'help': 'scale values for input normalization (RGB channels)'},
}

COPY_SETTINGS_DEFAULT['optimize'] = COPY_SETTINGS_DEFAULT['basic'] | {
}

##########################################################################
# compile can be followed by infer, analyze or accuracy
# compile is used to indicate a more sophisticated import - populate real data_path for that.
##########################################################################
SETTINGS_DEFAULT['compile'] = SETTINGS_DEFAULT['basic'] | SETTINGS_DEFAULT['optimize'] | {
    'model_path':               {'dest': 'session.model_path', 'default': None, 'type': str, 'group':'model', 'metavar': 'value', 'help': 'input model'},
    'config_path':              {'dest': 'common.config_path', 'default': None, 'type': str, 'group':'model', 'metavar': 'value', 'help': 'path to configuration file'}, 
    'work_path':                {'dest': 'common.work_path', 'default':'./work_dirs/{pipeline_type}/{target_device}/{tensor_bits}bits', 'type':str, 'metavar':'value', 'help':'work path'},
    'run_dir':              {'dest': 'session.run_dir', 'default':'{work_path}/{model_id}_{runtime_name}_{model_path}_{model_ext}', 'type':str, 'metavar':'value', 'help':'run_dir'},
    'pipeline_type':            {'dest': 'common.pipeline_type', 'default': 'compile', 'type': str, 'metavar': 'value', 'help': 'type of pipeline to run'},
    # common options
    'task_type':                {'dest': 'common.task_type', 'default': None, 'type': str, 'metavar': 'value', 'help': 'type of AI task (classification, detection, segmentation etc.)'},
    'num_frames':               {'dest': 'common.num_frames', 'default': 10, 'type': int, 'metavar': 'value', 'help': 'number of frames to process'},
    'display_step':             {'dest': 'common.display_step', 'default': 100, 'type': str, 'metavar': 'value', 'help': 'interval for displaying progress information'},
    'upgrade_config':           {'dest': 'common.upgrade_config', 'default': True, 'type': str, 'metavar': 'value', 'help': 'upgrade edgeai-benchmark config to work with tidlrunner'},
    'session_type_dict':        {'dest': 'common.session_type_dict', 'default': None, 'type': str, 'metavar': 'value', 'help': 'mapping of model extensions to session names'},
    'model_selection':          {'dest': 'common.model_selection', 'default': None, 'type': str, 'metavar': 'value', 'help': 'select a subset of models to run - path of the model is compared using this model_selection regex to select a particular model or not'},
    'model_shortlist':          {'dest': 'common.model_shortlist', 'default': None, 'type': str, 'metavar': 'value', 'help': 'select a subset of models to run - models configs with model_shortlist value <= this specified value will be used'},
    'preset_selection':         {'dest': 'common.preset_selection', 'default': None, 'type': utils.str_or_none, 'metavar': 'value', 'help': 'select a preset for speed accuracy trade-off: None, SPEED, ACCURACY, BALANCED'},
    'config_template':          {'dest': 'common.config_template', 'default':'data/templates/configs/param_template_config.yaml', 'type':str, 'metavar':'value', 'help':'param template path'},
    'incremental':              {'dest': 'common.incremental', 'default':True, 'type':utils.str_to_bool, 'metavar':'value', 'help':'param template path'},
    # compile/infer session
    ## model
    'model_id':                 {'dest': 'session.model_id', 'default': None, 'type': str, 'metavar': 'value', 'help': 'unique id of a model - optional'},
    'artifacts_folder':         {'dest': 'session.artifacts_folder', 'default': None, 'type': str, 'metavar': 'value', 'help': 'folder to store compilation artifacts'},
    ## runtime
    'runtime_name':             {'dest': 'session.name', 'default': None, 'type': str, 'metavar': 'value', 'help': 'name of the runtime session'},
    # input_data
    'data_name':                {'dest': 'dataloader.name', 'default': None, 'type': str, 'metavar': 'value', 'help': 'name of the input dataset'},
    'data_path':                {'dest': 'dataloader.path', 'default': None, 'type': str, 'metavar': 'path', 'help': 'path to the input data directory'},
    # runtime_settings
    'target_device':            {'dest': 'session.target_device', 'default': presets.TargetDeviceType.TARGET_DEVICE_AM68A, 'type': str, 'metavar': 'value', 'help': 'target device for inference (AM68A, AM69A, etc.)'},
    'tidl_offload':             {'dest': 'session.tidl_offload', 'default': True, 'type': utils.str_to_bool, 'metavar': 'value', 'help': 'enable TIDL acceleration for inference'},
    'graph_optimization_level': {'dest': 'session.onnxruntime:graph_optimization_level', 'default': presets.GraphOptimizationLevel.ORT_DISABLE_ALL, 'type': int, 'metavar': 'value', 'help': 'ONNX Runtime graph optimization level'},
    # runtime_settings.runtime_options
    'tensor_bits':              {'dest': 'session.runtime_options.tensor_bits', 'default': 8, 'type': int, 'metavar': 'value', 'help': 'quantization bit-width for tensors (8 or 16)'},
    'accuracy_level':           {'dest': 'session.runtime_options.accuracy_level', 'default': 1, 'type': int, 'metavar': 'value', 'help': 'accuracy level for TIDL offload (0, 1, 2)'},
    'debug_level':              {'dest': 'session.runtime_options.debug_level', 'default': 0, 'type': int, 'metavar': 'value', 'help': 'debug level for compile and infer'},
    'deny_list_layer_type':     {'dest': 'session.runtime_options.deny_list:layer_type', 'default': '', 'type': str, 'metavar': 'value', 'help': 'comma separated layer types to exclude from TIDL offload'},
    'deny_list_layer_name':     {'dest': 'session.runtime_options.deny_list:layer_name', 'default': '', 'type': str, 'metavar': 'value', 'help': 'comma separated layer names to exclude from TIDL offload'},
    'quantization_scale_type':  {'dest': 'session.runtime_options.advanced_options:quantization_scale_type', 'default': None, 'type': int, 'metavar': 'value', 'help': 'type of quantization scale to use'},
    'calibration_frames':       {'dest': 'session.runtime_options.advanced_options:calibration_frames', 'default': 12, 'type': int, 'metavar': 'value', 'help': 'number of frames for quantization calibration'},
    'calibration_iterations':   {'dest': 'session.runtime_options.advanced_options:calibration_iterations', 'default': 12, 'type': int, 'metavar': 'value', 'help': 'number of calibration iterations'},
    'prequantized_model':   {'dest': 'session.runtime_options.advanced_options:prequantized_model', 'default': argparse.SUPPRESS, 'type': utils.int_or_none, 'metavar': 'value', 'help': 'whether prequantized model'},
    'quant_params_file_path':   {'dest': 'session.runtime_options.advanced_options:quant_params_proto_path', 'default': argparse.SUPPRESS, 'type': utils.str_or_none_or_bool, 'metavar': 'value', 'help': 'path to quantization parameters file'},
    'max_num_subgraph_nodes':   {'dest': 'session.runtime_options.advanced_options:max_num_subgraph_nodes', 'default': 1536, 'type': int, 'metavar': 'value', 'help': 'maximum number of nodes in a subgraph'},    
    'output_feature_16bit_names_list':   {'dest': 'session.runtime_options.advanced_options:output_feature_16bit_names_list', 'default': argparse.SUPPRESS, 'type': str, 'metavar': 'value', 'help': 'list of output layers to keep in 16-bit precision'},
    'add_data_convert_ops':    {'dest': 'session.runtime_options.advanced_options:add_data_convert_ops', 'default': presets.DataConvertOps.DATA_CONVERT_OPS_INPUT_OUTPUT, 'type': int, 'metavar': 'value', 'help': 'data convert in DSP (0: disable, 1: input, 2: output, 3: input and output) - otherwise it will happen in ARM'},        
    # runtime_settings.runtime_options.object_detection
    'meta_arch_type':           {'dest': 'session.runtime_options.object_detection:meta_arch_type', 'default': argparse.SUPPRESS, 'type': int, 'metavar': 'value', 'help': 'meta architecture type for object detection'},
    'meta_arch_file_path':      {'dest': 'session.runtime_options.object_detection:meta_layers_names_list', 'default': argparse.SUPPRESS, 'type': str, 'metavar': 'value', 'help': 'path to meta architecture file'},
    'detection_threshold':      {'dest': 'session.runtime_options.object_detection:confidence_threshold', 'default': 0.3, 'type': float, 'metavar': 'value', 'help': 'confidence threshold for object detection'},
    'detection_top_k':          {'dest': 'session.runtime_options.object_detection:top_k', 'default': 200, 'type': int, 'metavar': 'value', 'help': 'number of top detections to keep before NMS'},
    'nms_threshold':            {'dest': 'session.runtime_options.object_detection:nms_threshold', 'default': 0.45, 'type': float, 'metavar': 'value', 'help': 'NMS threshold for object detection'},    
    'keep_top_k':               {'dest': 'session.runtime_options.object_detection:keep_top_k', 'default': 200, 'type': int, 'metavar': 'value', 'help': 'number of top detections to keep after NMS'},        
    # preprocess
    'preprocess_name':          {'dest':'preprocess.name', 'default':None, 'type':str, 'metavar':'value', 'group':'preprocess_name', 'help': 'name of the preprocessing pipeline'},
    'resize':                   {'dest':'preprocess.resize', 'default':None, 'type':int, 'nargs':'*', 'metavar':'value', 'help': 'resize dimensions for input images (height width)'},
    'crop':                     {'dest':'preprocess.crop', 'default':None, 'type':int, 'nargs':'*', 'metavar':'value', 'help': 'crop dimensions for input images (height width)'},
    'data_layout':              {'dest':'preprocess.data_layout', 'default':None, 'type':str, 'metavar':'value', 'help': 'data layout format (NCHW, NHWC)'},
    'reverse_channels':         {'dest':'preprocess.reverse_channels', 'default':False, 'type':utils.str_to_bool, 'metavar':'value', 'help': 'reverse color channel order (RGB to BGR)'},
    'resize_with_pad':          {'dest':'preprocess.resize_with_pad', 'default':False, 'type':utils.str_to_bool, 'metavar':'value', 'help': 'resize image with padding to maintain aspect ratio'},
    # postprocess
    'postprocess_enable':       {'dest': 'common.postprocess_enable', 'default': False, 'type': utils.str_to_bool, 'metavar': 'value', 'help': 'enable postprocessing after inference'},
    'postprocess_name':         {'dest': 'postprocess.name', 'default': None, 'type': str, 'metavar': 'value', 'help': 'name of the postprocessing pipeline'},
}

COPY_SETTINGS_DEFAULT['compile'] = COPY_SETTINGS_DEFAULT['basic'] | COPY_SETTINGS_DEFAULT['optimize'] | {
    'session.data_layout': 'preprocess.data_layout', 
    'postprocess.data_layout': 'preprocess.data_layout'     
}

##########################################################################
SETTINGS_DEFAULT['infer'] = SETTINGS_DEFAULT['compile'] | {
}

COPY_SETTINGS_DEFAULT['infer'] = COPY_SETTINGS_DEFAULT['compile'] | {
}


##########################################################################
# accuracy requires label_path as well
SETTINGS_DEFAULT['accuracy'] = SETTINGS_DEFAULT['compile'] | {
    'label_path':                         {'dest': 'dataloader.label_path', 'default':None, 'type':str, 'metavar':'path', 'help': 'path to ground truth labels for accuracy evaluation'},
    # increase number of frames for infer_accuracy
    'num_frames': {'dest': 'common.num_frames', 'default': 1000, 'type': int, 'metavar': 'value', 'help': 'number of frames to process for accuracy evaluation'},
    # postprocess
    'postprocess_enable':                 {'dest':'common.postprocess_enable', 'default': True, 'type': utils.str_to_bool, 'metavar': 'value', 'help': 'enable postprocessing after inference'},    
    'postprocess_resize_with_pad':        {'dest':'postprocess.resize_with_pad', 'default':False, 'type':utils.str_to_bool, 'metavar':'value', 'help': 'resize output with padding to maintain aspect ratio'},
    'postprocess_normalized_detections':  {'dest':'postprocess.normalized_detections', 'default':False, 'type':utils.str_to_bool, 'metavar':'value', 'help': 'whether detections are normalized coordinates'},
    'postprocess_formatter':              {'dest':'postprocess.formatter', 'default':None, 'type':str, 'metavar':'value', 'help': 'format for postprocessing output'},
    'postprocess_shuffle_indices':        {'dest':'postprocess.shuffle_indices', 'default':None, 'type':int, 'metavar':'value', 'nargs':'*', 'help': 'indices for shuffling postprocess output'},
    'postprocess_squeeze_axis':           {'dest':'postprocess.squeeze_axis', 'default':None, 'type':utils.str_to_int, 'metavar':'value', 'help': 'axis to squeeze from output tensor'},
    'postprocess_reshape_list':           {'dest':'postprocess.reshape_list', 'default':None, 'type':utils.str_to_list_of_tuples, 'metavar':'value', 'help': 'list of reshape operations for output tensors'},
    'postprocess_ignore_index':           {'dest':'postprocess.ignore_index', 'default':None, 'type':str, 'metavar':'value', 'help': 'index to ignore during accuracy calculation'},
    'postprocess_logits_bbox_to_bbox_ls': {'dest':'postprocess.logits_bbox_to_bbox_ls', 'default':False, 'type':utils.str_to_bool, 'metavar':'value', 'help': 'convert logits bounding box format to bounding box list'},
    #'postprocess_detection_threshold':    {'dest':'postprocess.detection_threshold', 'default':None, 'type':utils.float_or_none, 'metavar':'value', 'help': 'detection confidence threshold for postprocessing'},
    #'postprocess_detection_top_k':        {'dest':'postprocess.detection_top_k', 'default':None, 'type':utils.int_or_none, 'metavar':'value', 'help': 'top-k detections to keep in postprocessing'},
    #'postprocess_detection_keep_top_k':   {'dest':'postprocess.detection_keep_top_k', 'default':None, 'type':utils.float_or_none, 'metavar':'value', 'help': 'number of detections to keep after NMS in postprocessing'},
    'postprocess_keypoint':               {'dest':'postprocess.keypoint', 'default':False, 'type':utils.str_to_bool, 'metavar':'value', 'help': 'enable keypoint postprocessing'},
    'postprocess_save_output':            {'dest':'postprocess.save_output', 'default':False, 'type':bool, 'metavar':'value', 'help': 'save postprocessed output to files'},
    'postprocess_save_output_frames':     {'dest':'postprocess.save_output_frames', 'default':1, 'type':int, 'metavar':'value', 'help': 'number of output frames to save'},
}

COPY_SETTINGS_DEFAULT['accuracy'] = COPY_SETTINGS_DEFAULT['compile'] | {
    'postprocess.detection_threshold': 'session.runtime_options.object_detection:confidence_threshold',
    # 'postprocess.detection_keep_top_k': 'session.runtime_options.object_detection:keep_top_k',
    # 'postprocess.detection_top_k': 'session.runtime_options.object_detection:top_k'       
}

##########################################################################
SETTINGS_DEFAULT['analyze'] = SETTINGS_DEFAULT['infer'] | {
    'pipeline_type':                      {'dest': 'common.pipeline_type', 'default': 'analyze', 'type': str, 'metavar': 'value', 'help': 'type of pipeline to run'},        
    'analyze_level':                      {'dest': 'common.analyze_level', 'default': 2, 'type': int, 'metavar': 'value', 'help': 'analyze_level - 0: basic, 1: whole model stats, 2: whole model and per layer stats'},        
}

COPY_SETTINGS_DEFAULT['analyze'] = COPY_SETTINGS_DEFAULT['infer'] | {
}


##########################################################################
SETTINGS_DEFAULT['extract'] = SETTINGS_DEFAULT['basic'] | {
    'model_path':             {'dest': 'session.model_path', 'default': None, 'type': str, 'group':'model', 'metavar': 'value', 'help': 'input model'},
    'config_path':            {'dest': 'common.config_path', 'default': None, 'type': str, 'group':'model', 'metavar': 'value', 'help': 'path to configuration file'},
    'work_path':              {'dest': 'common.work_path', 'default':'./work_dirs/{pipeline_type}/{target_device}/{tensor_bits}bits', 'type':str, 'metavar':'value', 'help':'work path'},
    'run_dir':            {'dest': 'session.run_dir', 'default':'{work_path}/{model_id}_{runtime_name}_{model_path}_{model_ext}', 'type':str, 'metavar':'value', 'help':'run_dir'},
    'pipeline_type':          {'dest': 'common.pipeline_type', 'default': 'extract', 'type': str, 'metavar': 'value', 'help': 'type of pipeline to run'}, 
    'extract_mode':           {'dest': 'common.extract.mode', 'default': 'operators', 'type': str, 'metavar': 'value', 'choices': ['submodules', 'submodule', 'start2end', 'operators'], 'help': 'extraction mode (submodules, submodule, start2end, operators)'},
    'submodule_name':         {'dest': 'common.extract.submodule_name', 'default': None, 'type': str, 'metavar': 'value', 'help': 'name of specific submodule to extract'},
    'max_depth':              {'dest': 'common.extract.max_depth', 'default': 3, 'type': int, 'metavar': 'value', 'help': 'maximum depth for submodule extraction'},
    'start_names':            {'dest': 'common.extract.start_names', 'default': None, 'type': str, 'metavar': 'value', 'help': 'starting layer names for start2end extraction'},
    'end_names':              {'dest': 'common.extract.end_names', 'default': None, 'type': str, 'metavar': 'value', 'help': 'ending layer names for start2end extraction'},
}

COPY_SETTINGS_DEFAULT['extract'] = COPY_SETTINGS_DEFAULT['basic'] | {
}


##########################################################################
SETTINGS_DEFAULT['report'] = SETTINGS_DEFAULT['basic'] | {
    'pipeline_type':          {'dest': 'common.pipeline_type', 'default': 'compile', 'type': str, 'metavar': 'value', 'help': 'type of pipeline to run'}, 
    'report_mode':            {'dest': 'common.report.mode', 'default': 'detailed', 'type': str, 'metavar': 'value', 'choices': ['summary', 'detailed'], 'help': 'report generation mode (summary or detailed)'},
    'report_path':            {'dest': 'common.report.path', 'default': './work_dirs/compile', 'type': str, 'metavar': 'value', 'help': 'path where reports will be generated'},    
}

COPY_SETTINGS_DEFAULT['report'] = COPY_SETTINGS_DEFAULT['basic'] | {
}


##########################################################################
SETTINGS_DEFAULT['package'] = SETTINGS_DEFAULT['basic'] | {
    'pipeline_type':        {'dest': 'common.pipeline_type', 'default': 'package', 'type': str, 'metavar': 'value', 'help': 'type of pipeline to run'}, 
    'target_device':        {'dest': 'session.target_device', 'default': presets.TargetDeviceType.TARGET_DEVICE_AM68A, 'type': str, 'metavar': 'value', 'help': 'target device for inference (AM68A, AM69A, etc.)'},
    'tensor_bits':          {'dest': 'session.runtime_options.tensor_bits', 'default': 8, 'type': int, 'metavar': 'value', 'help': 'quantization bit-width for tensors (8 or 16)'},
    'work_path':            {'dest': 'common.work_path', 'default':'./work_dirs/compile/{target_device}/{tensor_bits}bits', 'type':str, 'metavar':'value', 'help':'work path'},
    'package_path':         {'dest': 'common.package_path', 'default':'./work_dirs/{pipeline_type}/{target_device}/{tensor_bits}bits', 'type':str, 'metavar':'value', 'help':'packaged path'},
    'param_template':       {'dest': 'common.param_template', 'default':'data/templates/configs/param_template_package.yaml', 'type':str, 'metavar':'value', 'help':'param template path'},
}

COPY_SETTINGS_DEFAULT['package'] = COPY_SETTINGS_DEFAULT['basic'] | {
}


##########################################################################
SETTINGS_DEFAULT['convert'] = SETTINGS_DEFAULT['basic'] | {
    'model_path':             {'dest': 'session.model_path', 'default': None, 'type': str, 'group':'model', 'metavar': 'value', 'help': 'input model'},
    'config_path':            {'dest': 'common.config_path', 'default': None, 'type': str, 'group':'model', 'metavar': 'value', 'help': 'path to configuration file'},    
    'work_path':              {'dest': 'common.work_path', 'default':'./work_dirs/{pipeline_type}/{target_device}/{tensor_bits}bits', 'type':str, 'metavar':'value', 'help':'work path'},   
    'run_dir':            {'dest': 'session.run_dir', 'default':'{work_path}/{model_id}_{runtime_name}_{model_path}_{model_ext}', 'type':str, 'metavar':'value', 'help':'run_dir'},
    'pipeline_type':          {'dest': 'common.pipeline_type', 'default': 'convert', 'type': str, 'metavar': 'value', 'help': 'type of pipeline to run'},    
    'output_model_path':      {'dest': 'common.output_model_path', 'default': None, 'type': str, 'metavar': 'value', 'help': 'output model'},
    'model_selection':          {'dest': 'common.model_selection', 'default': None, 'type': str, 'metavar': 'value', 'help': 'select a subset of models to run - path of the model is compared using this model_selection regex to select a particular model or not'},
    'model_shortlist':          {'dest': 'common.model_shortlist', 'default': None, 'type': str, 'metavar': 'value', 'help': 'select a subset of models to run - models configs with model_shortlist value <= this specified value will be used'},
}

COPY_SETTINGS_DEFAULT['convert'] = COPY_SETTINGS_DEFAULT['basic'] | {
}


##########################################################################
SETTINGS_DEFAULT['distill'] = SETTINGS_DEFAULT['compile'] | {
    'pipeline_type':            {'dest': 'common.pipeline_type', 'default': 'distill', 'type': str, 'metavar': 'value', 'help': 'type of pipeline to run'},    
    'teacher_model_path':       {'dest': 'common.teacher_model_path', 'default': None, 'type': str, 'metavar': 'value', 'help': 'teacher model'},
    'output_model_path':        {'dest': 'common.output_model_path', 'default': None, 'type': str, 'metavar': 'value', 'help': 'output model'},
    'calibration_frames':       {'dest': 'session.runtime_options.advanced_options:calibration_frames', 'default': 100, 'type': int, 'metavar': 'value', 'help': 'number of frames for quantization calibration'},
    'calibration_iterations':   {'dest': 'session.runtime_options.advanced_options:calibration_iterations', 'default': 5, 'type': int, 'metavar': 'value', 'help': 'number of calibration iterations'},
    'calibration_batch_size':   {'dest': 'session.runtime_options.advanced_options:calibration_batch_size', 'default': 16, 'type': int, 'metavar': 'value', 'help': 'number of calibration batch size'},
    'num_frames':               {'dest': 'common.num_frames', 'default': 1000, 'type': int, 'metavar': 'value', 'help': 'number of frames to process'},
}

COPY_SETTINGS_DEFAULT['distill'] = COPY_SETTINGS_DEFAULT['compile'] | {
}


##########################################################################
SETTINGS_DEFAULT['quantize'] = SETTINGS_DEFAULT['distill'] | {
    'pipeline_type':          {'dest': 'common.pipeline_type', 'default': 'quantize', 'type': str, 'metavar': 'value', 'help': 'type of pipeline to run'}, 
}

COPY_SETTINGS_DEFAULT['quantize'] = COPY_SETTINGS_DEFAULT['distill'] | {
}

