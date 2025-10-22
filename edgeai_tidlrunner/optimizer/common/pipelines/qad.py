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

from edgeai_tidlrunner.runner.common import utils
from edgeai_tidlrunner.runner.common import bases
from edgeai_tidlrunner.runner.common.settings.settings_default import SETTINGS_DEFAULT, COPY_SETTINGS_DEFAULT

from . import convert
from . import distill


class QAD(distill.DistillModel):
    ARGS_DICT=SETTINGS_DEFAULT['qad']
    COPY_ARGS=COPY_SETTINGS_DEFAULT['qad']

    def __init__(self, **kwargs):
        super().__init__(parametrization_types=('clip_const',), **kwargs)

    def info():
        print(f'INFO: Model QAD - {__file__}')

    def _prepare(self):
        import torch
        import torchao

        # from edgeai_torchmodelopt.xmodelopt import quantization
        print(f'INFO: preparing model for QAD with parameters: {self.kwargs}')
        super()._prepare()

        common_kwargs = self.settings[self.common_prefix]
        
        self.teacher_folder = os.path.join(self.run_dir, 'teacher')
        self.student_folder = os.path.join(self.run_dir, 'student')
        os.makedirs(self.teacher_folder, exist_ok=True)

        self.teacher_model_path = os.path.join(self.teacher_folder, os.path.basename(self.model_path))
        shutil.move(self.model_path, self.teacher_model_path)

        teacher_model = convert.ConvertModel._get_torch_model(self.teacher_model_path)
        # it is important to freeze the teacher model's bn
        teacher_model.eval()
        
        student_model = self._prepare_model(teacher_model, self.example_inputs)
        # student_model.train() #eval()
        
        common_kwargs['teacher_model_path'] = teacher_model
        common_kwargs['example_inputs'] = self.example_inputs
        common_kwargs['output_model_path'] = student_model

    def _prepare_model(self, teacher_model, example_inputs, device=None):
        import torch
        import torchao

        student_model = copy.deepcopy(teacher_model)

        if device:
            student_model.to(device)
        #
        return student_model

    def _run(self):
        import torch
        import torchao
        from ..utils import parametrize_wrapper
        from ..utils import hooks_wrapper

        common_kwargs = self.settings[self.common_prefix]
        session_kwargs = self.settings[self.session_prefix]
        runtime_options = session_kwargs['runtime_options']
        calibration_iterations = runtime_options['advanced_options:calibration_iterations']

        print(f'INFO: starting model QAD with parameters: {self.kwargs}')

        parametrize_wrapper.register_parametrizations(self.distill_model.student_model, parametrization_types=('clip_const',))

        self.activations_dict = {}
        hook_handles = hooks_wrapper.register_model_activation_store_hook(self.distill_model.student_model, self.activations_dict)

        super()._run()

        parametrize_wrapper.remove_parametrizations(self.distill_model.student_model)

        hook_handles = list(hook_handles.values()) if isinstance(hook_handles, dict) else hook_handles
        hook_handles = [hook_handles] if not isinstance(hook_handles, list) else hook_handles
        for hook_handle in hook_handles:
            hook_handle.remove()

        student_model = common_kwargs['output_model_path']

        os.makedirs(self.student_folder, exist_ok=True)
        self.student_model_path = os.path.join(self.student_folder, os.path.basename(self.model_path))
        convert.ConvertModel._run_func(student_model, self.student_model_path, common_kwargs['example_inputs'])

        shutil.copyfile(self.student_model_path, self.model_path)

    