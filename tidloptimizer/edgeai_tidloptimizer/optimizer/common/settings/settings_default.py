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
from edgeai_tidlrunner.runner.common.settings.settings_default import SETTINGS_DEFAULT, COPY_SETTINGS_DEFAULT
from edgeai_tidlrunner.runner.common.settings import constants
from edgeai_tidlrunner.runner.common import utils


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
    'calibration_iterations':   {'dest': 'session.runtime_options.advanced_options:calibration_iterations', 'default': 10, 'type': int, 'metavar': 'value', 'help': 'number of calibration iterations'},
    'calibration_batch_size':   {'dest': 'session.runtime_options.advanced_options:calibration_batch_size', 'default': 1, 'type': int, 'metavar': 'value', 'help': 'number of calibration batch size'},
    'num_frames':               {'dest': 'common.num_frames', 'default': 1000, 'type': int, 'metavar': 'value', 'help': 'number of frames to process'},
}

COPY_SETTINGS_DEFAULT['distill'] = COPY_SETTINGS_DEFAULT['compile'] | {
}


##########################################################################
SETTINGS_DEFAULT['qdistill'] = SETTINGS_DEFAULT['distill'] | {
    'pipeline_type':          {'dest': 'common.pipeline_type', 'default': 'qdistill', 'type': str, 'metavar': 'value', 'help': 'type of pipeline to run'}, 
}

COPY_SETTINGS_DEFAULT['qdistill'] = COPY_SETTINGS_DEFAULT['distill'] | {
}


##########################################################################
SETTINGS_DEFAULT['quantize'] = SETTINGS_DEFAULT['distill'] | {
    'pipeline_type':          {'dest': 'common.pipeline_type', 'default': 'quantize', 'type': str, 'metavar': 'value', 'help': 'type of pipeline to run'}, 
}

COPY_SETTINGS_DEFAULT['quantize'] = COPY_SETTINGS_DEFAULT['distill'] | {
}

