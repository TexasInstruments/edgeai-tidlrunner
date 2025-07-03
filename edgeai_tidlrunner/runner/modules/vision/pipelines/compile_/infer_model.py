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

from ......rtwrapper.core import presets
from ..... import bases
from ... import blocks
from ..... import utils
from ...settings.settings_default import SETTINGS_DEFAULT, COPY_SETTINGS_DEFAULT
from .compile_base import CompileModelBase


class InferModel(CompileModelBase):
    ARGS_DICT=SETTINGS_DEFAULT['infer_model']
    COPY_ARGS=COPY_SETTINGS_DEFAULT['infer_model']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def info(self):
        print(f'INFO: Model inference - {__file__}')

    def _run(self):
        print(f'INFO: starting model infer')
        super()._run()
        self._write_params('config.yaml')

        common_kwargs = self.settings[self.common_prefix]
        dataloader_kwargs = self.settings[self.dataloader_prefix]
        session_kwargs = self.settings[self.session_prefix]
        preprocess_kwargs = self.settings[self.preprocess_prefix]
        postprocess_kwargs = self.settings[self.postprocess_prefix]
        runtime_settings = session_kwargs['runtime_settings']
        runtime_options = runtime_settings['runtime_options']


        if not os.path.exists(self.run_dir) and not os.path.exists(self.model_folder) and not os.path.exists(self.artifacts_folder):
            raise RuntimeWarning(f'self.run_dir does not exist {self.run_dir} - compile the model before inference')

        # shutil.copy2(self.model_source, self.model_path)


        # session
        session_name = session_kwargs['name']
        session_type = blocks.sessions.SESSION_TYPES_MAPPING[session_name]
        self.session = session_type(**session_kwargs)
        self.session.start_inference()
        
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
        elif postprocess_kwargs.get('func', None):
            self.postprocess = ast.literal_eval(postprocess_kwargs['func'])
        else:
            raise RuntimeError('ERROR: invalid postprocess args')
        
        # infer model
        run_data = []
        num_frames = min(len(self.dataloader), common_kwargs['num_frames'])
        for input_index in range(num_frames):
            print(f'INFO: inference frame: {input_index}')
            input_data, info_dict = self.preprocess(self.dataloader[input_index], info_dict={})
            outputs = self.session.run_inference(input_data)
            outputs, info_dict = self.postprocess(outputs, info_dict=info_dict)
            run_data.append({'input':input_data, 'output':outputs, 'info_dict':info_dict})      

        print(f'INFO: model infer done. output is in: {self.run_dir}')
        self.run_data = run_data
        self._write_params('param.yaml')
        return outputs

