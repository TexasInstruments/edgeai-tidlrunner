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

from ...blocks.dataloaders import coco_detection_dataloader
from ...... import rtwrapper
from ......rtwrapper.core import presets
from ......rtwrapper.options import attr_dict
from ..... import bases
from ... import blocks
from ..... import utils
from ...settings.settings_default import SETTINGS_DEFAULT, COPY_SETTINGS_DEFAULT


class CompileModelPipelineBase(bases.PipelineBase):
    ARGS_DICT=SETTINGS_DEFAULT['import_model']
    COPY_ARGS=COPY_SETTINGS_DEFAULT['import_model']
    
    def __init__(self, **kwargs):
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
            dataloader_kwargs['name'] = 'random_dataloader'
        #
        if not preprocess_kwargs['name']:
            preprocess_kwargs['name'] = 'no_preprocess'
        #
        if not postprocess_kwargs['name']:
            postprocess_kwargs['name'] = 'no_postprocess'
        #

        ###################################################################################
        if 'session' in self.settings and self.settings[self.session_prefix].get('model_path', None):
            self.model_source = self.settings[self.session_prefix]['model_path']

            run_dir = self.settings[self.session_prefix]['run_dir']
            model_basename = os.path.basename(self.model_source)
            model_basename_wo_ext = os.path.splitext(model_basename)[0]
            model_id = self.settings[self.session_prefix]['model_id'] or ''
            model_id_underscore = model_id + '_' if model_id else ''

            tensor_bits = self.kwargs.get('session.runtime_settings.runtime_options.tensor_bits', '')
            tensor_bits_str = f'{str(tensor_bits)}bits' if tensor_bits else ''
            tensor_bits_slash = f'{str(tensor_bits)}bits' + os.sep if tensor_bits else ''

            target_device = self.kwargs.get('session.runtime_settings.target_device', 'NONE')
            target_device_str = target_device if target_device else ''
            target_device_slash = target_device + os.sep if target_device else ''

            self.run_dir = run_dir.replace('{model_name}', model_basename_wo_ext) \
                .replace('{model_id}_', model_id_underscore).replace('{model_id}', model_id) \
                .replace('{tensor_bits}/', tensor_bits_slash).replace('{tensor_bits}', tensor_bits_str) \
                .replace('{target_device}/', target_device_slash).replace('{target_device}', target_device_str)

            self.model_folder = os.path.join(self.run_dir, 'model')
            self.model_path = os.path.join(self.model_folder, model_basename)
            self.settings[self.session_prefix]['model_path'] = self.model_path
            self.artifacts_folder = self.settings[self.session_prefix].get('artifactrs_folder', os.path.join(self.run_dir, 'artifacts'))
            self.settings[self.session_prefix]['artifacts_folder'] = self.artifacts_folder
        else:
            self.run_dir = None
            self.model_folder = None
            self.model_path = None
            self.artifacts_folder = None
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
        if not os.environ.get('TIDL_TOOLS_PATH', None):
            target_device = runtime_settings['target_device']
            tools_base_path = os.path.abspath(os.path.normpath(os.path.join(os.path.dirname(rtwrapper.__file__), '..', 'tools')))
            tools_bin_path = os.path.join(tools_base_path, target_device, 'tidl_tools')
            os.environ['TIDL_TOOLS_PATH'] = tools_bin_path
            os.environ['LD_LIBRARY_PATH'] = f":{tools_bin_path}:{os.environ.get('LD_LIBRARY_PATH','')}"

        if not runtime_options.get('tidl_tools_path', None):
            runtime_options['tidl_tools_path'] = os.environ['TIDL_TOOLS_PATH']

        assert runtime_options['tidl_tools_path'] == os.environ['TIDL_TOOLS_PATH'], \
            f"path mismatch: {runtime_options['tidl_tools_path']}, {os.environ['TIDL_TOOLS_PATH']}"

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

    def download_file(self, model_source, model_folder, source_dir=None):
        is_web_link = model_source.startswith('http')
        is_simple_path = os.sep not in os.path.normpath(model_source)
        if is_web_link:
            utils.download_file(model_source, model_folder)
        else:
            model_source = os.path.join(source_dir, model_source) if source_dir and is_simple_path else model_source
            if not os.path.exists(model_source) and os.path.exists(model_source + '.link'):
                model_download_folder = os.path.dirname(model_source)
                utils.download_file(model_source, model_download_folder)
            #
            shutil.copy2(model_source, model_folder)
        #

    def _upgrade_kwargs(self, **kwargs):
        kwargs_out = {}
        for k, v in kwargs.items():
            if k.startswith('session.runtime_options'):
                new_key = k.replace('session.runtime_options', 'session.runtime_settings.runtime_options')
                kwargs_out[new_key] = v
            elif k == 'input_dataset':
                if v == 'imagenet':
                    kwargs_out['dataloader.name'] = 'image_classification_dataloader'
                    kwargs_out['dataloader.path'] = './data/datasets/vision/imagenetv2c/val'
                elif v == 'coco':
                    kwargs_out['dataloader.name'] = 'coco_detection_dataloader'
                    kwargs_out['dataloader.path'] = './data/datasets/vision/coco'
                elif v == 'cocoseg21':
                    kwargs_out['dataloader.name'] = 'coco_segmentation_dataloader'
                    kwargs_out['dataloader.path'] = './data/datasets/vision/coco'
                #
            elif k == 'session.target_device':
                # kwargs_out['session.runtime_settings.target_device'] = v
                pass
            else:
                kwargs_out[k] = v
            #
        #
        return kwargs_out

    def info(self):
        print(f'INFO: Model compile base - {__file__}')

    def run(self):
        return None

    def _write_params(self, filename):
        param_file = os.path.join(self.run_dir, filename)
        with open(param_file, 'w') as fp:
            yaml.dump(self.settings, fp)
        #