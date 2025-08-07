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
import tqdm

from ......rtwrapper.core import presets
from ..... import bases
from ... import blocks
from ..... import utils
from ...settings.settings_default import SETTINGS_DEFAULT, COPY_SETTINGS_DEFAULT
from .compile_base import CompileModelBase


class InferModel(CompileModelBase):
    ARGS_DICT=SETTINGS_DEFAULT['infer']
    COPY_ARGS=COPY_SETTINGS_DEFAULT['infer']

    def __init__(self, with_postprocess=False, **kwargs):
        super().__init__(with_postprocess=with_postprocess,**kwargs)

        common_kwargs = self.settings[self.common_prefix]
        dataloader_kwargs = self.settings[self.dataloader_prefix]
        session_kwargs = self.settings[self.session_prefix]
        preprocess_kwargs = self.settings[self.preprocess_prefix]
        postprocess_kwargs = self.settings[self.postprocess_prefix]
        runtime_settings = session_kwargs['runtime_settings']
        runtime_options = runtime_settings['runtime_options']

        if not os.path.exists(self.run_dir) and not os.path.exists(self.model_folder) and not os.path.exists(
                self.artifacts_folder):
            raise RuntimeWarning(f'self.run_dir does not exist {self.run_dir} - compile the model before inference')

        # shutil.copy2(self.model_source, self.model_path)

        # session
        session_name = session_kwargs['name']
        session_type = blocks.sessions.SESSION_TYPES_MAPPING[session_name]
        self.session = session_type(**session_kwargs)
        self.session.start_inference()

        # input_data
        if callable(dataloader_kwargs['name']):
            dataloader_method = dataloader_kwargs['name']
            self.dataloader = dataloader_method()
        elif hasattr(blocks.dataloaders, dataloader_kwargs['name']):
            dataloader_method = getattr(blocks.dataloaders, dataloader_kwargs['name'])
            self.dataloader = dataloader_method(**dataloader_kwargs)
            if hasattr(self.dataloader, 'set_size_details'):
                input_details, output_details = self.session.get_input_output_details()
                self.dataloader.set_size_details(input_details)
            #
        else:
            raise RuntimeError(f'ERROR: invalid dataloader args: {dataloader_kwargs}')
        #

        # preprocess
        if callable(preprocess_kwargs['name']):
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
        if callable(postprocess_kwargs['name']):
            postprocess_method = postprocess_kwargs['name']
            self.postprocess = postprocess_method()
        elif hasattr(blocks.postprocess, postprocess_kwargs['name']):
            postprocess_method = getattr(blocks.postprocess, postprocess_kwargs['name'])
            self.postprocess = postprocess_method(self.settings, **postprocess_kwargs)
        else:
            raise RuntimeError(f'ERROR: invalid postprocess args: {postprocess_kwargs}')
        #

    def info(self):
        print(f'INFO: Model inference - {__file__}')

    def _run(self):
        print(f'INFO: starting model infer')
        self.settings['result'] = dict()

        common_kwargs = self.settings[self.common_prefix]
        dataloader_kwargs = self.settings[self.dataloader_prefix]
        session_kwargs = self.settings[self.session_prefix]
        preprocess_kwargs = self.settings[self.preprocess_prefix]
        postprocess_kwargs = self.settings[self.postprocess_prefix]
        runtime_settings = session_kwargs['runtime_settings']
        runtime_options = runtime_settings['runtime_options']

        super()._run()

        # infer model
        run_data = []
        num_frames = min(len(self.dataloader), common_kwargs['num_frames'])
        print(f'INFO: inference starting for {num_frames} frames')
        input_index = 0
        display_step = common_kwargs.display_step
        tqdm_obj = tqdm.tqdm(range(num_frames))
        for input_index in range(num_frames):
            input_data, info_dict = self.preprocess(self.dataloader[input_index], info_dict={}) if self.preprocess else (self.dataloader[input_index], {})
            output_dict = self.session.run_inference(input_data)
            outputs = list(output_dict.values())
            outputs, info_dict = self.postprocess(outputs, info_dict=info_dict) if self.postprocess else (outputs, info_dict)
            output_dict = {output_key:output for output, output_key in zip(outputs, output_dict.keys())}
            run_data.append({'input':input_data, 'output':output_dict, 'info_dict':info_dict})
            if input_index % display_step == 0:
                tqdm_obj.update(input_index - tqdm_obj.n)
            #
        #
        tqdm_obj.update(input_index + 1 - tqdm_obj.n)

        print(f'INFO: model infer done. output is in: {self.run_dir}')
        self.run_data = run_data
        # TODO: populate the result entry
        self._write_params('result.yaml')
        return run_data

