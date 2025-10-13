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

from ....common import utils
from ....common import bases
from ...settings.settings_default import SETTINGS_DEFAULT, COPY_SETTINGS_DEFAULT
from ..compile_ import compile
from ..convert_ import convert


class DistillModel(compile.CompileModel):
    ARGS_DICT=SETTINGS_DEFAULT['distill']
    COPY_ARGS=COPY_SETTINGS_DEFAULT['distill']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _prepare(self):
        super()._prepare()

    def info():
        print(f'INFO: Model distill - {__file__}')

    def _prepare(self):
        print(f'INFO: preparing model distill with parameters: {self.kwargs}')
        super()._prepare()

    def _run(self):
        import torch
        print(f'INFO: starting model quantize with parameters: {self.kwargs}')
        print(f'INFO: running model quantize {self.model_path}')
        common_kwargs = self.settings[self.common_prefix]
        session_kwargs = self.settings[self.session_prefix]
        runtime_options = session_kwargs['runtime_options']
        distill_kwargs = common_kwargs.get('distill', {})

        teacher_model_path = common_kwargs.get('teacher_model_path', None) or self.model_path
        student_model_path = common_kwargs.get('output_model_path', None)
        example_inputs = common_kwargs.get('example_inputs', None)

        teacher_model = teacher_model_path
        if isinstance(teacher_model_path, str):
            teacher_model = convert.ConvertModel._run_func(teacher_model_path)
            if not example_inputs:
                example_inputs = convert.ConvertModel._get_example_inputs(teacher_model_path, to_torch=True)
            #
        #

        student_model = student_model_path
        if isinstance(student_model_path, str):
            student_model = convert.ConvertModel._run_func(student_model_path)
            if not example_inputs:
                example_inputs = convert.ConvertModel._get_example_inputs(student_model_path, to_torch=True)
            #
        #

        # distill loop here
        calibration_iterations = runtime_options['advanced_options:calibration_iterations']
        calibration_frames = runtime_options['advanced_options:calibration_frames']
        calibration_iterations = min(calibration_iterations, len(self.dataloader)) if calibration_iterations else len(self.dataloader)
        calibration_frames = min(calibration_frames, len(self.dataloader)) if calibration_frames else len(self.dataloader)
        for calib_index in range(calibration_iterations):
            print(f'INFO: running model quantize iteration: {calib_index}')
            for input_index in range(calibration_frames):
                print(f'INFO: input frame for quantize: {input_index}')
            #
        #

        if isinstance(student_model_path, str):
            if not example_inputs:
                input_data, info_dict = self.dataloader(0)
                input_data, info_dict = self.preprocess(input_data, info_dict=info_dict) if self.preprocess else (input_data, info_dict)
                example_inputs = tuple([torch.from_numpy(input_tensor) for input_tensor in input_data]) if isinstance(input_data, (list, tuple)) else (torch.from_numpy(input_data),)
            #
            convert.ConvertModel._run_func(student_model, student_model_path, example_inputs)
        #
        return student_model
