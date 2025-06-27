# Copyright (c) 2018-2021, Texas Instruments
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


##########################################################################
##########################################################################
SETTINGS_DEFAULT['basic'] = settings_base.SETTINGS_TARGET_MODULE_ARGS_DICT | {
    # model
    'output_path':             {'dest':'session.run_dir', 'default':'./runs/runner/{model_name}', 'type':str, 'metavar':'value', 'help':'output model path'},
    'model_path':              {'dest':'session.model_path', 'default':None, 'type':str, 'metavar':'value', 'help':'input model'},
}


COPY_SETTINGS_DEFAULT['basic'] = {
}

##########################################################################
# import can be followed by infer
# compile and import are related - main difference is import is simplistic.
# we let the model run with random data here.
##########################################################################
SETTINGS_DEFAULT['import_model'] = SETTINGS_DEFAULT['basic'] | {
    'optimize': {'dest': 'common.optimize.model_optimizer', 'default': False, 'type': utils.str_to_bool, 'metavar': 'value'},
    'shape_inference': {'dest': 'common.optimize.shape_inference', 'default': True, 'type': utils.str_to_bool, 'metavar': 'value'},

    # common options
    'task_type':               {'dest':'common.task_type', 'default':constants.TaskType.TASK_TYPE_CLASSIFICATION, 'type':str, 'metavar':'value'},
    'num_frames':              {'dest':'common.num_frames', 'default':1, 'type':int, 'metavar':'value'},
    # compile/infer session
    ## --model
    'artifacts_folder':        {'dest':'session.artifacts_folder', 'default':None, 'type':str, 'metavar':'value'},
    ## --runtime
    'runtime_name':            {'dest':'session.name', 'default':None, 'type':str, 'metavar':'value'},
    'input_mean':              {'dest':'session.input_mean', 'default':(123.675,116.28,103.53), 'type':float, 'nargs':'*', 'metavar':'value'},
    'input_scale':             {'dest':'session.input_scale', 'default':(0.017125,0.017507,0.017429), 'type':float, 'nargs':'*', 'metavar':'value'},
    # input_data
    'data_name':               {'dest':'dataloader.name', 'default':'random_dataloader', 'type':str, 'metavar':'value'},
    # preprocess
    'preprocess_name':         {'dest':'preprocess.name', 'default':'no_preprocess', 'type':str, 'metavar':'value', 'group':'preprocess_name'},
    # postprocess
    'postprocess_name':        {'dest':'postprocess.name', 'default':'no_postprocess', 'type':str, 'metavar':'value'},
    # runtime_settings
    'target_device':           {'dest':'session.runtime_settings.target_device', 'default':presets.TargetDeviceType.TARGET_DEVICE_AM68A, 'type':str, 'metavar':'value'},
    'tidl_offload':            {'dest':'session.runtime_settings.tidl_offload', 'default':True, 'type':utils.str_to_bool, 'metavar':'value'},
    # runtime_options
    'tensor_bits':             {'dest':'session.runtime_settings.runtime_options.tensor_bits', 'default':8, 'type':int, 'metavar':'value'},
    'quantization_scale_type': {'dest':'session.runtime_settings.runtime_options.advanced_options:quantization_scale_type', 'default':None, 'type':int, 'metavar':'value'},
    'calibration_frames':      {'dest':'session.runtime_settings.runtime_options.advanced_options:calibration_frames', 'default':12, 'type':int, 'metavar':'value'},
    'calibration_iterations':  {'dest':'session.runtime_settings.runtime_options.advanced_options:calibration_iterations', 'default':12, 'type':int, 'metavar':'value'},
}


COPY_SETTINGS_DEFAULT['import_model'] = {
    'session.data_layout': 'preprocess.data_layout'
}


##########################################################################
SETTINGS_DEFAULT['infer_model'] = SETTINGS_DEFAULT['import_model'] | {
}

COPY_SETTINGS_DEFAULT['infer_model'] = COPY_SETTINGS_DEFAULT['import_model'] | {
}


##########################################################################
# compile can be followed by infer, analyze or accuracy
# compile is used to indicate a more sophisticated import - populate real data_path for that.
##########################################################################
SETTINGS_DEFAULT['compile_model'] = SETTINGS_DEFAULT['import_model'] | {
   'data_name':                {'dest': 'dataloader.name', 'default': 'image_files_dataloader', 'type': str, 'metavar': 'value'},
   'data_path':                {'dest':'dataloader.path', 'default':'./data/datasets/vision/imagenetv2c/val', 'type':str, 'metavar':'path'},

    # preprocess
    'preprocess_name':         {'dest':'preprocess.name', 'default':'image_preprocess', 'type':str, 'metavar':'value', 'group':'preprocess_name'},
    'preprocess_func':         {'dest':'preprocess.func', 'default':None, 'type':utils.str_to_literal, 'metavar':'value', 'group':'preprocess_name'},
    'resize':                  {'dest':'preprocess.resize', 'default':None, 'type':int, 'nargs':'*', 'metavar':'value'},
    'crop':                    {'dest':'preprocess.crop', 'default':None, 'type':int, 'nargs':'*', 'metavar':'value'},
    'data_layout':             {'dest':'preprocess.data_layout', 'default':None, 'type':str, 'metavar':'value'},
    'reverse_channels':        {'dest':'preprocess.reverse_channels', 'default':False, 'type':utils.str_to_bool, 'metavar':'value'},
    'resize_with_pad':         {'dest':'preprocess.resize_with_pad', 'default':False, 'type':utils.str_to_bool, 'metavar':'value'},
}

COPY_SETTINGS_DEFAULT['compile_model'] = COPY_SETTINGS_DEFAULT['import_model'] | {
}


##########################################################################
SETTINGS_DEFAULT['infer_analyze'] = SETTINGS_DEFAULT['infer_model'] | {
}

COPY_SETTINGS_DEFAULT['infer_analyze'] = COPY_SETTINGS_DEFAULT['infer_model'] | {
}


##########################################################################
# accuracy requires label_path as well
SETTINGS_DEFAULT['infer_accuracy'] = SETTINGS_DEFAULT['compile_model'] | {
    'data_name': {'dest': 'dataloader.name', 'default': 'image_classification_dataloader', 'type': str, 'metavar': 'value'},
    'data_path': {'dest': 'dataloader.path', 'default': './examples/vision/datasets/imagenetv2c/val', 'type': str, 'metavar': 'path'},
    'label_path':{'dest': 'dataloader.label_path', 'default':'./examples/datasets/imagenetv2c/val.txt', 'type':str, 'metavar':'path'},

    # postprocess
    'postprocess_name':                   {'dest':'postprocess.name', 'default':'no_postprocess', 'type':str, 'metavar':'value', 'group':'postprocess_name'},
    'postprocess_func':                   {'dest':'postprocess.func', 'default':None, 'type':utils.str_to_literal, 'metavar':'value', 'group':'postprocess_name'},
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

COPY_SETTINGS_DEFAULT['infer_accuracy'] = COPY_SETTINGS_DEFAULT['compile_model'] | {
}
