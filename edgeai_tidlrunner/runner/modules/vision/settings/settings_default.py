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

from .....rtwrapper.options import options_default
from . import constants
from .constants import presets
from .... import utils
from ....bases import settings_base


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
}

COPY_SETTINGS_DEFAULT['basic'] = {}

##########################################################################
# compile can be followed by infer, analyze or accuracy
# compile is used to indicate a more sophisticated import - populate real data_path for that.
##########################################################################
SETTINGS_DEFAULT['compile'] = SETTINGS_DEFAULT['basic'] | {
    'pipeline_type':            {'dest': 'common.pipeline_type', 'default': 'compile', 'type': str, 'metavar': 'value', 'help': 'type of pipeline to run'},
    # optimizations
    'simplify_model':          {'dest': 'common.optimize.simplify_model', 'default': True, 'type': utils.str_to_bool, 'metavar': 'value'},
    'optimize_model':          {'dest': 'common.optimize.optimize_model', 'default': True, 'type': utils.str_to_bool, 'metavar': 'value'},
    'shape_inference':          {'dest': 'common.optimize.shape_inference', 'default': True, 'type': utils.str_to_bool, 'metavar': 'value'},
    # common options
    'task_type':                {'dest': 'common.task_type', 'default': None, 'type': str, 'metavar': 'value'},
    'num_frames':               {'dest': 'common.num_frames', 'default': 10, 'type': int, 'metavar': 'value'},
    'config_path':              {'dest': 'common.config_path', 'default': None, 'type': str, 'metavar': 'value'},
    'display_step':             {'dest': 'common.display_step', 'default': 100, 'type': str, 'metavar': 'value'},
    # compile/infer session
    ## model
    'model_id':                 {'dest': 'session.model_id', 'default': None, 'type': str, 'metavar': 'value', 'help': 'unique id of a model - optional'},
    'artifacts_folder':         {'dest': 'session.artifacts_folder', 'default': None, 'type': str, 'metavar': 'value'},
    'packaged_path':            {'dest': 'session.packaged_path', 'default':'./work_dirs/{pipeline_type}_package/{target_device}/{tensor_bits}/{model_id}_{runtime_name}_{model_path}_{model_ext}', 'type':str, 'metavar':'value', 'help':'packaged model path'},
    ## runtime
    'runtime_name':             {'dest': 'session.name', 'default': None, 'type': str, 'metavar': 'value'},
    'input_mean':               {'dest': 'session.input_mean', 'default': (123.675, 116.28, 103.53), 'type': float, 'nargs': '*', 'metavar': 'value'},
    'input_scale':              {'dest': 'session.input_scale', 'default': (0.017125, 0.017507, 0.017429), 'type': float, 'nargs': '*', 'metavar': 'value'},
    # input_data
    'data_name':                {'dest': 'dataloader.name', 'default': None, 'type': str, 'metavar': 'value'},
    'data_path':                {'dest': 'dataloader.path', 'default': None, 'type': str, 'metavar': 'path'},
    # runtime_settings
    'target_device':            {'dest': 'session.runtime_settings.target_device', 'default': presets.TargetDeviceType.TARGET_DEVICE_AM68A, 'type': str, 'metavar': 'value'},
    'tidl_offload':             {'dest': 'session.runtime_settings.tidl_offload', 'default': True, 'type': utils.str_to_bool, 'metavar': 'value'},
    'graph_optimization_level': {'dest': 'session.runtime_settings.onnxruntime:graph_optimization_level', 'default': presets.GraphOptimizationLevel.ORT_DISABLE_ALL, 'type': int, 'metavar': 'value'},
    # runtime_settings.runtime_options
    'tensor_bits':              {'dest': 'session.runtime_settings.runtime_options.tensor_bits', 'default': 8, 'type': int, 'metavar': 'value'},
    'quantization_scale_type':  {'dest': 'session.runtime_settings.runtime_options.advanced_options:quantization_scale_type', 'default': None, 'type': int, 'metavar': 'value'},
    'calibration_frames':       {'dest': 'session.runtime_settings.runtime_options.advanced_options:calibration_frames', 'default': 12, 'type': int, 'metavar': 'value'},
    'calibration_iterations':   {'dest': 'session.runtime_settings.runtime_options.advanced_options:calibration_iterations', 'default': 12, 'type': int, 'metavar': 'value'},
    'quant_params_file_path':   {'dest': 'session.runtime_settings.runtime_options.advanced_options:quant_params_proto_path', 'default': argparse.SUPPRESS, 'type': utils.str_or_none_or_bool, 'metavar': 'value'},
    'max_num_subgraph_nodes':   {'dest': 'session.runtime_settings.runtime_options.advanced_options:max_num_subgraph_nodes', 'default': 1536, 'type': int, 'metavar': 'value'},    
    'output_feature_16bit_names_list':   {'dest': 'session.runtime_settings.runtime_options.advanced_options:output_feature_16bit_names_list', 'default': argparse.SUPPRESS, 'type': str, 'metavar': 'value'},        
    # runtime_settings.runtime_options.object_detection
    'meta_arch_type':           {'dest': 'session.runtime_settings.runtime_options.object_detection:meta_arch_type', 'default': argparse.SUPPRESS, 'type': int, 'metavar': 'value'},
    'meta_arch_file_path':      {'dest': 'session.runtime_settings.runtime_options.object_detection:meta_layers_names_list', 'default': argparse.SUPPRESS, 'type': str, 'metavar': 'value'},
    'detection_threshold':      {'dest': 'session.runtime_settings.runtime_options.object_detection:confidence_threshold', 'default': 0.3, 'type': float, 'metavar': 'value'},
    'detection_top_k':          {'dest': 'session.runtime_settings.runtime_options.object_detection:top_k', 'default': 200, 'type': int, 'metavar': 'value'},
    'nms_threshold':            {'dest': 'session.runtime_settings.runtime_options.object_detection:nms_threshold', 'default': 0.45, 'type': float, 'metavar': 'value'},    
    'keep_top_k':               {'dest': 'session.runtime_settings.runtime_options.object_detection:keep_top_k', 'default': 200, 'type': int, 'metavar': 'value'},        
    # preprocess
    'preprocess_name':         {'dest':'preprocess.name', 'default':None, 'type':str, 'metavar':'value', 'group':'preprocess_name'},
    'resize':                  {'dest':'preprocess.resize', 'default':None, 'type':int, 'nargs':'*', 'metavar':'value'},
    'crop':                    {'dest':'preprocess.crop', 'default':None, 'type':int, 'nargs':'*', 'metavar':'value'},
    'data_layout':             {'dest':'preprocess.data_layout', 'default':None, 'type':str, 'metavar':'value'},
    'reverse_channels':        {'dest':'preprocess.reverse_channels', 'default':False, 'type':utils.str_to_bool, 'metavar':'value'},
    'resize_with_pad':         {'dest':'preprocess.resize_with_pad', 'default':False, 'type':utils.str_to_bool, 'metavar':'value'},
    # postprocess
    'postprocess_name':        {'dest': 'postprocess.name', 'default': None, 'type': str, 'metavar': 'value'},
}

COPY_SETTINGS_DEFAULT['compile'] = COPY_SETTINGS_DEFAULT['basic'] | {
    'session.data_layout': 'preprocess.data_layout'
}

##########################################################################
SETTINGS_DEFAULT['infer'] = SETTINGS_DEFAULT['compile'] | {
}

COPY_SETTINGS_DEFAULT['infer'] = COPY_SETTINGS_DEFAULT['compile'] | {
}


##########################################################################
# accuracy requires label_path as well
SETTINGS_DEFAULT['accuracy'] = SETTINGS_DEFAULT['compile'] | {
    'label_path':                         {'dest': 'dataloader.label_path', 'default':None, 'type':str, 'metavar':'path'},

    # increase number of frames for infer_accuracy
    'num_frames': {'dest': 'common.num_frames', 'default': 1000, 'type': int, 'metavar': 'value'},

    # postprocess
    'postprocess_resize_with_pad':        {'dest':'postprocess.resize_with_pad', 'default':False, 'type':utils.str_to_bool, 'metavar':'value'},
    'postprocess_normalized_detections':  {'dest':'postprocess.normalized_detections', 'default':False, 'type':utils.str_to_bool, 'metavar':'value'},
    'postprocess_formatter':              {'dest':'postprocess.formatter', 'default':None, 'type':str, 'metavar':'value'},
    'postprocess_shuffle_indices':        {'dest':'postprocess.shuffle_indices', 'default':None, 'type':int, 'metavar':'value', 'nargs':'*'},
    'postprocess_squeeze_axis':           {'dest':'postprocess.squeeze_axis', 'default':None, 'type':utils.str_to_int, 'metavar':'value'},
    'postprocess_reshape_list':           {'dest':'postprocess.reshape_list', 'default':None, 'type':utils.str_to_list_of_tuples, 'metavar':'value'},
    'postprocess_ignore_index':           {'dest':'postprocess.ignore_index', 'default':None, 'type':str, 'metavar':'value'},
    'postprocess_logits_bbox_to_bbox_ls': {'dest':'postprocess.logits_bbox_to_bbox_ls', 'default':False, 'type':utils.str_to_bool, 'metavar':'value'},
    'postprocess_detection_threshold':    {'dest':'postprocess.detection_threshold', 'default':0.3, 'type':utils.float_or_none, 'metavar':'value'},
    'postprocess_detection_top_k':        {'dest':'postprocess.detection_top_k', 'default':200, 'type':utils.int_or_none, 'metavar':'value'},
    'postprocess_detection_keep_top_k':   {'dest':'postprocess.detection_keep_top_k', 'default':200, 'type':utils.float_or_none, 'metavar':'value'},
    'postprocess_keypoint':               {'dest':'postprocess.keypoint', 'default':False, 'type':utils.str_to_bool, 'metavar':'value'},
    'postprocess_save_output':            {'dest':'postprocess.save_output', 'default':False, 'type':bool, 'metavar':'value'},
    'postprocess_save_output_frames':     {'dest':'postprocess.save_output_frames', 'default':1, 'type':int, 'metavar':'value'},
}

COPY_SETTINGS_DEFAULT['accuracy'] = COPY_SETTINGS_DEFAULT['compile'] | {
}

##########################################################################
SETTINGS_DEFAULT['analyze'] = SETTINGS_DEFAULT['infer'] | {
    'pipeline_type':                    {'dest': 'common.pipeline_type', 'default': 'analyze', 'type': str, 'metavar': 'value', 'help': 'type of pipeline to run'},        
}

COPY_SETTINGS_DEFAULT['analyze'] = COPY_SETTINGS_DEFAULT['infer'] | {
}

##########################################################################
SETTINGS_DEFAULT['optimize'] = SETTINGS_DEFAULT['basic'] | {
    'pipeline_type':                    {'dest': 'common.pipeline_type', 'default': 'optimize', 'type': str, 'metavar': 'value', 'help': 'type of pipeline to run'},    
    'simplify_model':                   {'dest': 'common.optimize.simplify_model', 'default': True, 'type': utils.str_to_bool, 'metavar': 'value'},
    'optimize_model':                   {'dest': 'common.optimize.optimize_model', 'default': True, 'type': utils.str_to_bool, 'metavar': 'value'},
    'shape_inference':                  {'dest': 'common.optimize.shape_inference', 'default': True, 'type': utils.str_to_bool, 'metavar': 'value'},
}

COPY_SETTINGS_DEFAULT['optimize'] = COPY_SETTINGS_DEFAULT['basic'] | {
}


##########################################################################
SETTINGS_DEFAULT['extract'] = SETTINGS_DEFAULT['basic'] | {
    'pipeline_type':          {'dest': 'common.pipeline_type', 'default': 'extract', 'type': str, 'metavar': 'value', 'help': 'type of pipeline to run'}, 
    'extract_mode':           {'dest': 'common.extract.mode', 'default': 'operators', 'type': str, 'metavar': 'value', 'choices': ['submodules', 'submodule', 'start2end', 'operators']},
    'submodule_name':         {'dest': 'common.extract.submodule_name', 'default': None, 'type': str, 'metavar': 'value'},
    'max_depth':              {'dest': 'common.extract.max_depth', 'default': 3, 'type': int, 'metavar': 'value'},
    'start_names':            {'dest': 'common.extract.start_names', 'default': None, 'type': str, 'metavar': 'value'},
    'end_names':              {'dest': 'common.extract.end_names', 'default': None, 'type': str, 'metavar': 'value'},
}

COPY_SETTINGS_DEFAULT['extract'] = COPY_SETTINGS_DEFAULT['basic'] | {
}


##########################################################################
SETTINGS_DEFAULT['report'] = SETTINGS_DEFAULT['basic'] | {
    'pipeline_type':          {'dest': 'common.pipeline_type', 'default': 'compile', 'type': str, 'metavar': 'value', 'help': 'type of pipeline to run'}, 
    'report_mode':            {'dest': 'common.report.mode', 'default': 'detailed', 'type': str, 'metavar': 'value', 'choices': ['summary', 'detailed']},
    'report_path':            {'dest': 'common.report.path', 'default': './work_dirs/compile', 'type': str, 'metavar': 'value'},    
}

COPY_SETTINGS_DEFAULT['report'] = COPY_SETTINGS_DEFAULT['basic'] | {
}