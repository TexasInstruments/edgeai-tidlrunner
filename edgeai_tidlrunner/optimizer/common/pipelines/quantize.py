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


class QuantizeModel(distill.DistillModel):
    ARGS_DICT=SETTINGS_DEFAULT['quantize']
    COPY_ARGS=COPY_SETTINGS_DEFAULT['quantize']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def info():
        print(f'INFO: Model quantize - {__file__}')

    def _prepare(self):
        import torch
        import torchao

        # from edgeai_torchmodelopt.xmodelopt import quantization
        print(f'INFO: preparing model quantize with parameters: {self.kwargs}')
        super()._prepare()

        common_kwargs = self.settings[self.common_prefix]
        
        self.teacher_folder = os.path.join(self.run_dir, 'teacher')
        self.student_folder = os.path.join(self.run_dir, 'student')
        os.makedirs(self.teacher_folder, exist_ok=True)

        self.teacher_model_path = os.path.join(self.teacher_folder, os.path.basename(self.model_path))
        shutil.move(self.model_path, self.teacher_model_path)

        example_inputs = convert.ConvertModel._get_example_inputs(self.teacher_model_path, to_torch=True)
        teacher_model = convert.ConvertModel._get_torch_model(self.teacher_model_path)
        # it is important to freeze the teacher model's bn
        teacher_model.eval()
        
        student_model = self._prepare_quantize(teacher_model, example_inputs)
        # student_model.train() #eval()
        
        common_kwargs['teacher_model_path'] = teacher_model
        common_kwargs['example_inputs'] = example_inputs
        common_kwargs['output_model_path'] = student_model

    def _prepare_quantize(self, teacher_model, example_inputs, device=None):
        import torch
        import torchao
        from ..utils.distill_module import DistillWrapperModule
        from torchao.quantization.pt2e.quantize_pt2e import (prepare_qat_pt2e, convert_pt2e)
        from torchao.quantization.pt2e import allow_exported_model_train_eval

        # create student model
        # from edgeai_torchmodelopt.xmodelopt.quantization.v3 import QATPT2EModule 
        # student_model = quantization.v3.QATPT2EModule(teacher_model, example_inputs, total_epochs=calibration_iterations)

        teacher_model_final = teacher_model #torch.export.export(teacher_model, example_inputs).module()

        student_model_initial = copy.deepcopy(teacher_model_final)

        student_model = torch.export.export(student_model_initial, example_inputs).module()
        allow_exported_model_train_eval(student_model)

        # backend developer will write their own Quantizer and expose methods to allow
        # from torchao.quantization.pt2e.quantizer.arm_inductor_quantizer import (ArmInductorQuantizer, get_default_arm_inductor_quantization_config)
        # quantizer = ArmInductorQuantizer().set_global(get_default_arm_inductor_quantization_config(is_qat=True))

        # from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (X86InductorQuantizer, get_default_x86_inductor_quantization_config)
        # quantizer = X86InductorQuantizer().set_global(get_default_x86_inductor_quantization_config(is_qat=True))

        # from torchao.quantization.pt2e.quantizer.xpu_inductor_quantizer import (XPUInductorQuantizer, get_default_xpu_inductor_quantization_config)
        # quantizer = XPUInductorQuantizer().set_global(get_default_xpu_inductor_quantization_config(is_qat=True))

        # install executorch: `pip install executorch`
        from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (get_symmetric_quantization_config, XNNPACKQuantizer)
        # # quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_qat=True, per_channel=True))
        quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config(is_qat=True))

        student_model = prepare_qat_pt2e(student_model, quantizer)
        allow_exported_model_train_eval(student_model)

        if device:
            student_model.to(device)
        #
        return student_model

    def _run(self):
        import torch
        import torchao

        common_kwargs = self.settings[self.common_prefix]
        session_kwargs = self.settings[self.session_prefix]
        runtime_options = session_kwargs['runtime_options']
        calibration_iterations = runtime_options['advanced_options:calibration_iterations']

        print(f'INFO: starting model quantize with parameters: {self.kwargs}')
        super()._run()

        from torchao.quantization.pt2e.quantize_pt2e import (prepare_qat_pt2e, convert_pt2e,)
        student_model = common_kwargs['output_model_path']
        student_model = convert_pt2e(student_model)

        os.makedirs(self.student_folder, exist_ok=True)
        self.student_model_path = os.path.join(self.student_folder, os.path.basename(self.model_path))
        convert.ConvertModel._run_func(student_model, self.student_model_path, common_kwargs['example_inputs'])

        shutil.copyfile(self.student_model_path, self.model_path)