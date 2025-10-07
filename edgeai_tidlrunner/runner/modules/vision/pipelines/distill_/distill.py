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
from ..common_ import common_base
from ..convert_ import convert


class DistillModel(common_base.CommonPipelineBase):
    ARGS_DICT=SETTINGS_DEFAULT['distill']
    COPY_ARGS=COPY_SETTINGS_DEFAULT['distill']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _prepare(self):
        super()._prepare()

    def info():
        print(f'INFO: Model distill - {__file__}')

    def _run(self):
        print(f'INFO: starting model distill with parameters: {self.kwargs}')

        common_kwargs = self.settings[self.common_prefix]
        distill_kwargs = common_kwargs.get('distill', {})

        if os.path.exists(self.run_dir):
            print(f'INFO: clearing run_dir folder before compile: {self.run_dir}')
            shutil.rmtree(self.run_dir, ignore_errors=True)
        #

        torch_model_name = os.path.splitext(os.path.basename(self.model_path))[0] + '.pt'
        teacher_folder = os.path.join(self.run_dir, 'distill', 'teacher')
        student_folder = os.path.join(self.run_dir, 'distill', 'student')

        os.makedirs(self.run_dir, exist_ok=True)
        # os.makedirs(self.artifacts_folder, exist_ok=True)
        os.makedirs(self.model_folder, exist_ok=True)
        os.makedirs(teacher_folder, exist_ok=True)
        os.makedirs(student_folder, exist_ok=True)

        teacher_path = os.path.join(teacher_folder, torch_model_name)
        student_path = os.path.join(student_folder, torch_model_name)

        config_path = os.path.dirname(common_kwargs['config_path']) if common_kwargs['config_path'] else None
        self.download_file(self.model_source, model_folder=self.model_folder, source_dir=config_path)

        convert_kwargs = common_kwargs.get('convert', {})
        convert.ConvertModel._run_func(self.settings, self.model_path, teacher_path, **convert_kwargs)

        self._run_func(self.settings, teacher_path, student_path, **distill_kwargs)

    @classmethod
    def _run_func(self, settings, teacher_path, student_path, **kwargs):
        try:
            shutil.copy2(teacher_path, student_path)
        except shutil.SameFileError:
            pass
        #