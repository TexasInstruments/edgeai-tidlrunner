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

from edgeai_tidlrunner.rtwrapper.options import runtime_options

from ...... import rtwrapper
from ......rtwrapper.core import presets
from ..... import bases
from ... import blocks
from ..... import utils
from ...settings.settings_default import SETTINGS_DEFAULT, COPY_SETTINGS_DEFAULT
from .common_base import CommonPipelineBase
from ...settings import constants


class CompileModelBase(CommonPipelineBase):
    ARGS_DICT=SETTINGS_DEFAULT['compile']
    COPY_ARGS=COPY_SETTINGS_DEFAULT['compile']
    
    def __init__(self, with_postprocess=True, **kwargs):
        self.with_postprocess = with_postprocess
        super().__init__(**kwargs)
        if 'session' in self.settings and self.settings[self.session_prefix].get('model_path', None):
            common_kwargs = self.settings['common']                 
            config_path = os.path.dirname(common_kwargs['config_path']) if common_kwargs['config_path'] else None            
            model_source = self.settings[self.session_prefix]['model_path']       
            self.model_source = self._get_file_path(model_source, source_dir=config_path)

            run_dir = self.settings[self.session_prefix]['run_dir']
            self.run_dir = self._build_run_dir(run_dir)

            model_basename = os.path.basename(self.model_source)
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

    def _build_run_dir(self, run_dir):
        pipeline_type = self.kwargs.get('common.pipeline_type', 'compile')
        task_type = self.kwargs.get('common.task_type', None) or None
    
        model_basename = os.path.basename(self.model_source)
        model_id = self.settings[self.session_prefix].get('model_id', '')
        if not model_id:
            unique_id = utils.generate_unique_id(model_basename, num_characters=8)            
            if task_type in constants.TaskTypeShortNames:
                task_type_short_name = constants.TaskTypeShortNames[task_type]
                model_id = task_type_short_name + '-' + unique_id                
            else:
                pipeline_type = self.kwargs.get('common.pipeline_type', 'x') or 'x'
                model_id = pipeline_type + '-' + unique_id
            #
        #
        model_id_underscore = model_id + '_'

        tensor_bits = self.kwargs.get('session.runtime_options.tensor_bits', '') or ''
        tensor_bits_str = f'{str(tensor_bits)}bits' if tensor_bits else ''
        tensor_bits_slash = f'{str(tensor_bits)}bits' + os.sep if tensor_bits else ''

        target_device = self.kwargs.get('session.target_device', 'NONE') or 'NONE'
        target_device_str = target_device if target_device else ''
        target_device_slash = target_device + os.sep if target_device else ''

        model_basename_wo_ext, model_ext = os.path.splitext(model_basename)
        model_ext = model_ext[1:] if len(model_ext)>0 else model_ext

        runtime_name = self.kwargs.get('session.name', '') or ''

        run_dir = run_dir.replace('{pipeline_type}', pipeline_type)
        run_dir = run_dir.replace('{model_id}_', model_id_underscore)
        run_dir = run_dir.replace('{model_id}', model_id)
        run_dir = run_dir.replace('{runtime_name}', runtime_name)        
        run_dir = run_dir.replace('{model_ext}', model_ext)        
        run_dir = run_dir.replace('{tensor_bits}/', tensor_bits_slash)
        run_dir = run_dir.replace('{tensor_bits}', tensor_bits_str)
        run_dir = run_dir.replace('{target_device}/', target_device_slash)
        run_dir = run_dir.replace('{target_device}', target_device_str)
        run_dir = run_dir.replace('{model_name}', model_basename_wo_ext)
        run_dir = self._replace_model_path(run_dir, '{model_path}', run_dir_tree_depth=3)
        return run_dir
    
    def _replace_model_path(self, run_dir, model_path_str, run_dir_tree_depth):
        run_dir_tree_depth = 3
        model_path_tree = os.path.abspath(os.path.splitext(self.model_source)[0]).split(os.sep)
        if len(model_path_tree) > run_dir_tree_depth:
            model_path_tree = model_path_tree[-run_dir_tree_depth:]
        #
        run_dir = run_dir.replace(model_path_str, '_'.join(model_path_tree))    
        return run_dir    

    def _prepare(self):
        super()._prepare()
        
        self.dataloader = None
        self.preprocess = None
        self.session = None
        self.postprocess = None
        self.run_data = None

        if 'session' in self.settings and self.settings[self.session_prefix].get('model_path', None):
            dataloader_kwargs = self.settings[self.dataloader_prefix]
            preprocess_kwargs = self.settings[self.preprocess_prefix]
            session_kwargs = self.settings[self.session_prefix]
            postprocess_kwargs = self.settings[self.postprocess_prefix]
            runtime_options = session_kwargs['runtime_options']

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
            if session_kwargs['tidl_offload']:
                assert os.environ.get('TIDL_TOOLS_PATH', None) is not None, f"WARNING: TIDL_TOOLS_PATH is missing in the environment"
                runtime_options['tidl_tools_path'] = os.environ['TIDL_TOOLS_PATH']

            if not runtime_options.get('artifacts_folder', None):
                runtime_options['artifacts_folder'] = self.artifacts_folder

            self.object_detection_meta_layers_names_list_source = session_kwargs['runtime_options'].get('object_detection:meta_layers_names_list', None)
            if self.object_detection_meta_layers_names_list_source:
                if not (self.object_detection_meta_layers_names_list_source.startswith('/') or self.object_detection_meta_layers_names_list_source.startswith('.')):
                    object_detection_meta_layers_names_path = os.path.join(self.model_folder, self.object_detection_meta_layers_names_list_source)
                else:
                    object_detection_meta_layers_names_path = self.object_detection_meta_layers_names_list_source
                #
                session_kwargs['runtime_options']['object_detection:meta_layers_names_list'] = object_detection_meta_layers_names_path
            #

            packaged_path = self.settings[self.session_prefix]['packaged_path']
            self.packaged_path = self._build_run_dir(packaged_path)
        #

    @classmethod
    def _upgrade_kwargs(cls, **kwargs):
        kwargs_in = copy.deepcopy(kwargs)
        kwargs_out = dict()

        ###################################################################################
        for k, v in kwargs_in.items():
            if k in ('session.target_device',):
                # options that are not allowed to be None
                if v is not None:
                    kwargs_out[k] = v
                #
            elif k.startswith('runtime_options.'):
                # options that are not allowed to be None
                if v is not None:
                    kwargs_out[k] = v
                #                
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
                    if kwargs_in.get('dataloader.name', None) is None:
                        kwargs_out['dataloader.name'] = 'image_classification_dataloader'
                    #
                    if kwargs_in.get('dataloader.path', None) is None:
                        kwargs_out['dataloader.path'] = './data/datasets/vision/imagenetv2c/val'
                    #
                    if kwargs_in.get('preprocess.name',None) is None:
                        kwargs_out['preprocess.name'] = 'image_preprocess'
                    #
                elif v == 'coco':
                    if kwargs_in.get('dataloader.name', None) is None:
                        kwargs_out['dataloader.name'] = 'coco_detection_dataloader'
                    #
                    if kwargs_in.get('dataloader.path', None) is None:
                        kwargs_out['dataloader.path'] = './data/datasets/vision/coco'
                    #
                    if kwargs_in.get('preprocess.name',None) is None:
                        kwargs_out['preprocess.name'] = 'image_preprocess'
                    #
                    if kwargs_in.get('postprocess.name',None) is None:
                        kwargs_out['postprocess.name'] = 'object_detection_postprocess'
                    #
                elif v == 'cocoseg21':
                    if kwargs_in.get('dataloader.name', None) is None:
                        kwargs_out['dataloader.name'] = 'coco_segmentation_dataloader'
                    #
                    if kwargs_in.get('dataloader.path', None) is None:
                        kwargs_out['dataloader.path'] = './data/datasets/vision/coco'
                    #
                    if kwargs_in.get('preprocess.name',None) is None:
                        kwargs_out['preprocess.name'] = 'image_preprocess'
                    #
                #
            elif k.startswith('model_info'):
                kwargs_out[k] = v                
            else:
                kwargs_out[k] = v
            #
        #

        ###################################################################################
        model_path = kwargs_out.get('session.model_path', None)
        if model_path:
            model_ext = os.path.splitext(model_path)[1]
            if kwargs_out.get('preprocess.data_layout', None) is None:
                data_layout_mapping = {
                    '.onnx': presets.DataLayoutType.NCHW,
                    '.tflite': presets.DataLayoutType.NHWC,
                }
                data_layout = data_layout_mapping.get(model_ext, None)
                kwargs_out['preprocess.data_layout'] = data_layout
                kwargs_out['session.data_layout'] = data_layout
            #
            if kwargs_out.get('session.name', None) is None:
                session_name_mapping = {
                    '.onnx': presets.RuntimeType.RUNTIME_TYPE_ONNXRT,
                    '.tflite': presets.RuntimeType.RUNTIME_TYPE_TFLITERT,
                }
                session_name = session_name_mapping.get(model_ext, None)
                kwargs_out['session.name'] = session_name
            #
        #
        return kwargs_out
