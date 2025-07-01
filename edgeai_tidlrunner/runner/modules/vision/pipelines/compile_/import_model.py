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

from ...... import rtwrapper
from ......rtwrapper.core import presets
from ......rtwrapper.options import attr_dict
from ..... import bases
from ... import blocks
from ..... import utils
from ...settings.settings_default import SETTINGS_DEFAULT, COPY_SETTINGS_DEFAULT
from .compile_base import CompileModelPipelineBase
from ..optimize_ import optimize_model


class ImportModelPipeline(CompileModelPipelineBase):
    ARGS_DICT=SETTINGS_DEFAULT['import_model']
    COPY_ARGS=COPY_SETTINGS_DEFAULT['import_model']
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def info(self):
        print(f'INFO: Model import - {__file__}')

    def run(self):
        print(f'INFO: starting model import')
        super().run()

        common_kwargs = self.settings[self.common_prefix]
        dataloader_kwargs = self.settings[self.dataloader_prefix]
        session_kwargs = self.settings[self.session_prefix]
        preprocess_kwargs = self.settings[self.preprocess_prefix]
        postprocess_kwargs = self.settings[self.postprocess_prefix]
        runtime_settings = session_kwargs['runtime_settings']
        runtime_options = runtime_settings['runtime_options']

        if os.path.exists(self.run_dir):
            print(f'INFO: clearing run_dir folder before compile: {self.run_dir}')
            shutil.rmtree(self.run_dir, ignore_errors=True)
        #

        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.artifacts_folder, exist_ok=True)
        os.makedirs(self.model_folder, exist_ok=True)

        config_path = os.path.dirname(common_kwargs['config_path']) if common_kwargs['config_path'] else None
        self.download_file(self.model_source, model_folder=self.model_folder, source_dir=config_path)

        print(f'INFO: running model optimize {self.model_path}')
        optimize_kwargs = common_kwargs.get('optimize', {})
        optimize_model.OptimizeModelPipeline._run(self.model_path, self.model_path, **optimize_kwargs)

        # session
        session_name = session_kwargs['name']
        session_type = blocks.sessions.SESSION_TYPES_MAPPING[session_name]
        self.session = session_type(**session_kwargs)
        self.session.start_import()
        
        # input_data
        dataloader_name = dataloader_kwargs['name']
        dataloader_func = getattr(blocks.dataloaders, dataloader_name)
        self.dataloader = dataloader_func(**dataloader_kwargs)
        if hasattr(self.dataloader, 'set_size_details'):
            input_details, output_details = self.session.get_input_output_details()
            self.dataloader.set_size_details(input_details)
            
        # preprocess
        if preprocess_kwargs['name']:
            preprocess_kwargs.pop('func', None)
            preprocess_name = preprocess_kwargs['name']
            preprocess_method = getattr(blocks.preprocess, preprocess_name)
            if not (preprocess_kwargs.get('resize',None) and preprocess_kwargs.get('crop',None)):
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
        elif isinstance(preprocess_kwargs['func'], str):
            preprocess_method = ast.literal_eval(preprocess_kwargs['func'])
        else:
            raise RuntimeError('ERROR: invalid preprocess args')
        #
        self.preprocess = preprocess_method(self.settings, **preprocess_kwargs)

        # postprocess
        if postprocess_kwargs['name']:
            postprocess_kwargs.pop('func', None)
            postprocess_name = postprocess_kwargs['name']
            postprocess_method = getattr(blocks.postprocess, postprocess_name)
            self.postprocess = postprocess_method(self.settings, **postprocess_kwargs)
        elif postprocess_kwargs['func']:
            postprocess_method = ast.literal_eval(postprocess_kwargs['func'])
        else:
            raise RuntimeError('ERROR: invalid postprocess args')
        
        # infer model
        run_data = []
        outputs = []
        print(f'INFO: running model import {self.model_path}')
        for input_index in range(min(len(self.dataloader), runtime_options['advanced_options:calibration_frames'])):
            print(f'INFO: import frame: {input_index}')
            input_data, info_dict = self.preprocess(self.dataloader[input_index], info_dict={})
            outputs = self.session.run_import(input_data)
            output, info_dict = self.postprocess(outputs, info_dict=info_dict)
            outputs.append(output)
            run_data.append({'input':input_data, 'output':outputs, 'info_dict':info_dict})

        print(f'INFO: model import done. output is in: {self.run_dir}')
        self.run_data = run_data
        return outputs