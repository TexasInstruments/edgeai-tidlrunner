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

from ...settings.settings_default import SETTINGS_DEFAULT, COPY_SETTINGS_DEFAULT
from ..... import utils
from ..... import bases
from ..convert_ import convert
from . import distill


class QuantizeModel(distill.DistillModel):
    ARGS_DICT=SETTINGS_DEFAULT['quantize']
    COPY_ARGS=COPY_SETTINGS_DEFAULT['quantize']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def info():
        print(f'INFO: Model quantize - {__file__}')

    def _prepare(self):
        print(f'INFO: preparing model quantize with parameters: {self.kwargs}')
        super()._prepare()

        common_kwargs = self.settings[self.common_prefix]

        self.teacher_folder = os.path.join(self.run_dir, 'teacher')
        self.student_folder = os.path.join(self.run_dir, 'student')
        os.makedirs(self.teacher_folder, exist_ok=True)
        os.makedirs(self.student_folder, exist_ok=True)

        self.teacher_model_path = os.path.join(self.teacher_folder, os.path.basename(self.model_path))
        shutil.move(self.model_path, self.teacher_model_path)

        teacher_model = convert.ConvertModel._run_func(self.teacher_model_path)
        example_inputs = convert.ConvertModel._get_example_inputs(self.teacher_model_path, to_torch=True)
        common_kwargs['teacher_model_path'] = teacher_model
        common_kwargs['example_inputs'] = example_inputs

        self.student_model_path = os.path.join(self.student_folder, os.path.basename(self.model_path))
        
        #####################################################
        # TODO: replace this with QDQ model
        student_model = copy.deepcopy(teacher_model)
        common_kwargs['output_model_path'] = student_model
        #####################################################

    def _run(self):
        print(f'INFO: starting model quantize with parameters: {self.kwargs}')
        common_kwargs = self.settings[self.common_prefix]
        super()._run()
        convert.ConvertModel._run_func(common_kwargs['output_model_path'], self.student_model_path, common_kwargs['example_inputs'])
        shutil.copyfile(self.student_model_path, self.model_path)