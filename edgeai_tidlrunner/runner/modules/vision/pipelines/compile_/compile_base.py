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


import os
import sys
import shutil
import copy
import ast

import yaml

from ...... import rtwrapper
from ......rtwrapper.core import presets
from ..... import bases
from ... import blocks
from ..... import utils
from ...settings.settings_default import SETTINGS_DEFAULT, COPY_SETTINGS_DEFAULT
from ..common_.common_base import CommonPipelineBase


class CompileModelBase(CommonPipelineBase):
    ARGS_DICT=SETTINGS_DEFAULT['compile']
    COPY_ARGS=COPY_SETTINGS_DEFAULT['compile']
    
    def __init__(self, with_postprocess=True, **kwargs):
        self.with_postprocess = with_postprocess
        super().__init__(**kwargs)
        
        self.dataloader = None
        self.preprocess = None
        self.session = None
        self.postprocess = None
        self.run_data = None

        dataloader_kwargs = self.settings[self.dataloader_prefix]
        preprocess_kwargs = self.settings[self.preprocess_prefix]
        session_kwargs = self.settings[self.session_prefix]
        postprocess_kwargs = self.settings[self.postprocess_prefix]
        runtime_settings = session_kwargs['runtime_settings']
        runtime_options = runtime_settings['runtime_options']

        ###################################################################################
        if not dataloader_kwargs['name']:
            print(f'WARNING: dataloader name is was not provided - will use random_dataloader'
                  f'\n  and the resultant compiled artifacts may not be accurate.'
                  f'\n  please specify a dataloader using the argument data_name or dataloader.name'
                  f'\n  in addition data_path or dataloader.path may need to be provided.')
            dataloader_kwargs['name'] = 'random_dataloader'
        #
        if not preprocess_kwargs['name']:
            preprocess_kwargs['name'] = 'no_preprocess'
        #
        if not postprocess_kwargs['name']:
            postprocess_kwargs['name'] = 'no_postprocess'
        #

        ###################################################################################
        model_ext = os.path.splitext(self.model_path)[1] if self.model_path else None
        if preprocess_kwargs.get('data_layout', None) is None:
            data_layout_mapping = {
                '.onnx': presets.DataLayoutType.NCHW,
                '.tflite': presets.DataLayoutType.NHWC,
            }
            data_layout = data_layout_mapping.get(model_ext, None)
            preprocess_kwargs['data_layout'] = data_layout
            session_kwargs['data_layout'] = data_layout
        #
        if session_kwargs.get('name', None) is None:
            session_name_mapping = {
                '.onnx': presets.RuntimeType.RUNTIME_TYPE_ONNXRT,
                '.tflite': presets.RuntimeType.RUNTIME_TYPE_TFLITERT,
            }
            session_name = session_name_mapping.get(model_ext, None)
            session_kwargs['name'] = session_name
        #

        ###################################################################################
        if runtime_settings['tidl_offload']:
            assert os.environ.get('TIDL_TOOLS_PATH', None) is not None, f"WARNING: TIDL_TOOLS_PATH is missing in the environment"
            runtime_options['tidl_tools_path'] = os.environ['TIDL_TOOLS_PATH']

        if not runtime_options.get('artifacts_folder', None):
            runtime_options['artifacts_folder'] = self.artifacts_folder

        self.object_detection_meta_layers_names_list_source = session_kwargs['runtime_settings']['runtime_options'].get('object_detection:meta_layers_names_list', None)
        if self.object_detection_meta_layers_names_list_source:
            if not (self.object_detection_meta_layers_names_list_source.startswith('/') or self.object_detection_meta_layers_names_list_source.startswith('.')):
                object_detection_meta_layers_names_path = os.path.join(self.model_folder, self.object_detection_meta_layers_names_list_source)
            else:
                object_detection_meta_layers_names_path = self.object_detection_meta_layers_names_list_source
            #
            session_kwargs['runtime_settings']['runtime_options']['object_detection:meta_layers_names_list'] = object_detection_meta_layers_names_path
        #

        packaged_path = self.settings[self.session_prefix]['packaged_path']
        self.packaged_path = self.build_run_dir(packaged_path)

    def _upgrade_kwargs(self, **kwargs):
        kwargs_in = copy.deepcopy(kwargs)
        kwargs_out = dict()

        for k, v in kwargs_in.items():
            if k.startswith('session.target_device'):
                # these fields are from edgeai-benchmark - no need to use it here
                kwargs_out.pop(k, None)
            elif k.startswith('session.runtime_options'):
                # kwargs_out.pop(k, None)
                # new_key = k.replace('session.runtime_options', 'session.runtime_settings.runtime_options')
                # kwargs_out[new_key] = v
                pass # do not take runtime_options in the old format (directly under session) from the configfile
            elif k == 'session.session_name':
                kwargs_out['session.name'] = v           
            elif k == 'dataloader.name':
                if kwargs_in[k] is not None:
                    kwargs_out[k] = kwargs_in[k]
                #
            elif k == 'dataloader.path':
                if kwargs_in[k] is not None:
                    kwargs_out[k] = kwargs_in[k]
                #
            elif k == 'preprocess.name':
                if kwargs_in[k] is not None:
                    kwargs_out[k] = kwargs_in[k]
                #
            elif k == 'postprocess.name':
                if kwargs_in[k] is not None:
                    kwargs_out[k] = kwargs_in[k]
                #
            elif k == 'dataset_category':
                kwargs_out.pop(k, None)
            elif k == 'calibration_dataset':
                kwargs_out.pop(k, None)
            elif k == 'task_type':
                kwargs_out.pop(k, None)
                kwargs_out['common.task_type'] = v
            elif k == 'input_dataset':
                kwargs_out.pop(k, None)
                if v == 'imagenet':
                    if kwargs_in['dataloader.name'] is None:
                        kwargs_out['dataloader.name'] = 'image_classification_dataloader'
                    #
                    if kwargs_in['dataloader.path'] is None:
                        kwargs_out['dataloader.path'] = './data/datasets/vision/imagenetv2c/val'
                    #
                    if kwargs_in.get('preprocess.name',None) is None:
                        kwargs_out['preprocess.name'] = 'image_preprocess'
                    #
                elif v == 'coco':
                    if kwargs_in['dataloader.name'] is None:
                        kwargs_out['dataloader.name'] = 'coco_detection_dataloader'
                    #
                    if kwargs_in['dataloader.path'] is None:
                        kwargs_out['dataloader.path'] = './data/datasets/vision/coco'
                    #
                    if kwargs_in.get('preprocess.name',None) is None:
                        kwargs_out['preprocess.name'] = 'image_preprocess'
                    #
                    if kwargs_in.get('postprocess.name',None) is None and self.with_postprocess:
                        kwargs_out['postprocess.name'] = 'object_detection_postprocess'
                    #
                elif v == 'cocoseg21':
                    if kwargs_in['dataloader.name'] is None:
                        kwargs_out['dataloader.name'] = 'coco_segmentation_dataloader'
                    #
                    if kwargs_in['dataloader.path'] is None:
                        kwargs_out['dataloader.path'] = './data/datasets/vision/coco'
                    #
                    if kwargs_in.get('preprocess.name',None) is None:
                        kwargs_out['preprocess.name'] = 'image_preprocess'
                    #
                #
            else:
                kwargs_out[k] = v
            #
        #
        return kwargs_out
