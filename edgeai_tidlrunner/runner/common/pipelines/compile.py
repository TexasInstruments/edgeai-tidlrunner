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

from edgeai_tidlrunner.rtwrapper.core import presets

from ...common import bases
from ...common import utils
from .. import blocks
from ..settings.settings_default import SETTINGS_DEFAULT, COPY_SETTINGS_DEFAULT
from .common_.compile_base import CompileModelBase
from . import surgery


class CompileModel(CompileModelBase):
    ARGS_DICT=SETTINGS_DEFAULT['compile']
    COPY_ARGS=COPY_SETTINGS_DEFAULT['compile']
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _prepare(self):
        super()._prepare()
        self.param_yaml = os.path.join(self.run_dir,'param.yaml')
        
        common_kwargs = self.settings[self.common_prefix]
        dataloader_kwargs = self.settings[self.dataloader_prefix]
        session_kwargs = self.settings[self.session_prefix]
        preprocess_kwargs = self.settings[self.preprocess_prefix]
        postprocess_kwargs = self.settings[self.postprocess_prefix]
        runtime_options = session_kwargs['runtime_options']

        if common_kwargs['incremental']:
            if os.path.exists(self.param_yaml):
                return
            #
        #

        if os.path.exists(self.run_dir):
            print(f'INFO: clearing run_dir folder before compile: {self.run_dir}')
            shutil.rmtree(self.run_dir, ignore_errors=True)
        #

        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.artifacts_folder, exist_ok=True)
        os.makedirs(self.model_folder, exist_ok=True)

        # write config to a file
        self._write_params(self.settings, os.path.join(self.run_dir,'config.yaml'), param_template=common_kwargs.get('config_template', None))

        config_path = os.path.dirname(common_kwargs['config_path']) if common_kwargs['config_path'] else None
        self.download_file(self.model_source, model_folder=self.model_folder, source_dir=config_path)

        if self.object_detection_meta_layers_names_list_source:
            self.download_file(self.object_detection_meta_layers_names_list_source, model_folder=self.model_folder,
                               source_dir=config_path)
        #

        # input_data
        if self.pipeline_config and 'dataloader' in self.pipeline_config:
            self.dataloader = self.pipeline_config['dataloader']
        elif self.pipeline_config and 'calibration_dataset' in self.pipeline_config:
            self.dataloader = self.pipeline_config['calibration_dataset']
        elif self.pipeline_config and 'input_dataset' in self.pipeline_config:
            self.dataloader = self.pipeline_config['input_dataset']
        elif callable(dataloader_kwargs['name']):
            dataloader_method = dataloader_kwargs['name']
            self.dataloader = dataloader_method()
        elif hasattr(blocks.dataloaders, dataloader_kwargs['name']):
            dataloader_method = getattr(blocks.dataloaders, dataloader_kwargs['name'])
            self.dataloader = dataloader_method(self.settings, shuffle=True, **dataloader_kwargs)
            if hasattr(self.dataloader, 'set_size_details'):
                input_details, output_details = self.session.get_input_output_details()
                self.dataloader.set_size_details(input_details)
            #
        else:
            raise RuntimeError(f'ERROR: invalid dataloader args: {dataloader_kwargs}')
        #

        # preprocess
        if self.pipeline_config and 'preprocess' in self.pipeline_config:
            self.preprocess = self.pipeline_config['preprocess']
        elif callable(preprocess_kwargs['name']):
            preprocess_method = preprocess_kwargs['name']
            self.preprocess = preprocess_method()
        elif hasattr(blocks.preprocess, preprocess_kwargs['name']):
            preprocess_method = getattr(blocks.preprocess, preprocess_kwargs['name'])
            if not (preprocess_kwargs.get('resize', None) and preprocess_kwargs.get('crop', None)):
                # input shape was not provided - use the model input size
                input_details, output_details = self.session.get_input_output_details()
                if preprocess_kwargs.get('data_layout') == presets.DataLayoutType.NCHW:
                    preprocess_kwargs['resize'] = copy.deepcopy(tuple(input_details[0]['shape'][-2:]))
                    preprocess_kwargs['crop'] = copy.deepcopy(tuple(input_details[0]['shape'][-2:]))
                elif preprocess_kwargs.get('data_layout') == presets.DataLayoutType.NHWC:
                    preprocess_kwargs['resize'] = copy.deepcopy(tuple(input_details[0]['shape'][-3:-1]))
                    preprocess_kwargs['crop'] = copy.deepcopy(tuple(input_details[0]['shape'][-3:-1]))
                #
            #
            self.preprocess = preprocess_method(self.settings, **preprocess_kwargs)
        else:
            raise RuntimeError(f'ERROR: invalid preprocess args: {preprocess_kwargs}')
        #

        # postprocess
        if self.pipeline_config and 'postprocess' in self.pipeline_config:
            self.postprocess = self.pipeline_config['postprocess']
        elif self.kwargs['common.postprocess_enable']:
            if callable(postprocess_kwargs['name']):
                postprocess_method = postprocess_kwargs['name']
                self.postprocess = postprocess_method()
            elif hasattr(blocks.postprocess, postprocess_kwargs['name']):
                postprocess_method = getattr(blocks.postprocess, postprocess_kwargs['name'])
                self.postprocess = postprocess_method(self.settings, **postprocess_kwargs)
            else:
                raise RuntimeError(f'ERROR: invalid postprocess args: {postprocess_kwargs}')
            #
        #
        self._prepare_model()
        
    def info(self):
        print(f'INFO: Model import - {__file__}')

    def _prepare_model(self):
        print(f'INFO: running model optimize {self.model_path}')
        common_kwargs = self.settings[self.common_prefix]
        optimize_kwargs = common_kwargs['optimize']
        surgery.ModelSurgery._run_func(self.settings, self.model_path, self.model_path, **optimize_kwargs)

    def _run(self):
        print(f'INFO: starting model import')
        common_kwargs = self.settings[self.common_prefix]
        dataloader_kwargs = self.settings[self.dataloader_prefix]
        session_kwargs = self.settings[self.session_prefix]
        preprocess_kwargs = self.settings[self.preprocess_prefix]
        postprocess_kwargs = self.settings[self.postprocess_prefix]
        runtime_options = session_kwargs['runtime_options']

        # session
        session_name = session_kwargs['name']
        session_type = blocks.sessions.SESSION_TYPES_MAPPING[session_name]
        self.session = session_type(self.settings, **session_kwargs)
        self.session.start_import()

        if common_kwargs['incremental']:
            if os.path.exists(self.param_yaml):
                print(f'INFO: incremental {common_kwargs["incremental"]} param.yaml exists: {self.param_yaml}')
                print(f'INFO: skipping compile')
                return
            #
        #

        super()._run()

        # infer model
        run_data = []
        print(f'INFO: running model import {self.model_path}')
        calibration_frames = runtime_options['advanced_options:calibration_frames']
        for input_index in range(min(len(self.dataloader), calibration_frames)):
            print(f'INFO: import frame: {input_index}')
            run_dict = self._run_frame(input_index)
            run_data.append(run_dict)
        #
        print(f'INFO: model import done. output is in: {self.run_dir}')
        self.run_data = run_data

        # TODO - cleanup the parameters and write param.yaml
        self._write_params(self.settings, self.param_yaml)
        return run_data
    
    def _run_frame(self, input_index):
        info_dict = self.get_info_dict(input_index)
        input_data, info_dict = self.dataloader(input_index, info_dict)
        input_data, info_dict = self.preprocess(input_data, info_dict=info_dict) if self.preprocess else (input_data, info_dict)
        output_dict = self.session.run_import(input_data)
        if self.postprocess:
            outputs = list(output_dict.values())
            outputs, info_dict = self.postprocess(outputs, info_dict=info_dict) 
            run_data = {'input':input_data, 'output':outputs, 'info_dict':info_dict}
        else:
            run_data = {'input':input_data, 'output':output_dict, 'info_dict':info_dict}
        #
        return run_data
    
    def get_info_dict(self, input_index):
        info_dict = {'dataset_info': getattr(self.dataloader, 'dataset_info', None),
                     'label_offset_pred': self.pipeline_config.get('metric',{}).get('label_offset_pred',None) if self.pipeline_config else None,
                     'sample_idx': input_index,
                     'task_name': self.pipeline_config.get('task_name',{}) if self.pipeline_config else None}
        return info_dict