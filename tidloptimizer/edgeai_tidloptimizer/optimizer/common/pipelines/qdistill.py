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
from ..settings.settings_default import SETTINGS_DEFAULT, COPY_SETTINGS_DEFAULT

from . import convert
from . import distill


class QuantAwareDistillation(distill.DistillModel):
    ARGS_DICT=SETTINGS_DEFAULT['qdistill']
    COPY_ARGS=COPY_SETTINGS_DEFAULT['qdistill']

    def __init__(self, **kwargs):
        super().__init__(**kwargs) #parametrization_types=('clip_const',),

        from edgeai_torchmodelopt.xmodelopt.quantization.v3 import QATPT2EModule, QConfigType
        self.qconfig_type = QConfigType.CLIP_RANGE
        self.quantizer_type = 'basic' #'advanced'
        self.with_convert = True #False

    def info():
        print(f'INFO: Model qdistill - {__file__}')

    def _prepare(self):
        super()._prepare()
        
        common_kwargs = self.settings[self.common_prefix]
        
        self.teacher_folder = os.path.join(self.run_dir, 'teacher')
        self.student_folder = os.path.join(self.run_dir, 'student')
        os.makedirs(self.teacher_folder, exist_ok=True)

        teacher_model_path = os.path.join(self.teacher_folder, os.path.basename(self.model_path))
        shutil.move(self.model_path, teacher_model_path)

        os.makedirs(self.student_folder, exist_ok=True)
        student_model_path = os.path.join(self.student_folder, os.path.basename(self.model_path))

    def _run(self):
        common_kwargs = self.settings[self.common_prefix]
        session_kwargs = self.settings[self.session_prefix]
        runtime_options = session_kwargs['runtime_options']
        calibration_iterations = runtime_options['advanced_options:calibration_iterations']

        teacher_model_path = os.path.join(self.teacher_folder, os.path.basename(self.model_path))
        student_model_path = os.path.join(self.student_folder, os.path.basename(self.model_path))

        # get pytorch model
        teacher_model = convert.ConvertModel._get_torch_model(teacher_model_path, example_inputs=self.example_inputs)
        # it is important to freeze the teacher model's BN and Dropouts
        teacher_model.eval()

        # export teacher model - optional
        os.makedirs(self.teacher_folder, exist_ok=True)
        teacher_model_path = os.path.join(self.teacher_folder, os.path.basename(self.model_path))
        teacher_model_path_pt2 = os.path.splitext(teacher_model_path)[0] + ".pt2"
        convert.ConvertModel._run_func(teacher_model, teacher_model_path_pt2, self.example_inputs)
        
        #################################################################################
        # prepare the student model
        import torch

        # create student model
        student_model = torch.export.export(teacher_model, self.example_inputs).module()

        # from torch.ao.quantization.pt2e import allow_exported_model_train_eval
        # allow_exported_model_train_eval(student_model)

        # create student model
        from edgeai_torchmodelopt.xmodelopt.quantization.v3 import QATPT2EModule, QConfigType
        from edgeai_torchmodelopt.xmodelopt.quantization.v3.fake_quantize_types import ADAPTIVE_ACTIVATION_FAKE_QUANT_TYPES
        student_model = QATPT2EModule(teacher_model, example_inputs=self.example_inputs, 
                                      qconfig_type=self.qconfig_type, quantizer_type=self.quantizer_type,
                                      total_epochs=calibration_iterations)

        #################################################################################
        # run the distillation
        common_kwargs['teacher_model_path'] = teacher_model
        common_kwargs['example_inputs'] = self.example_inputs
        common_kwargs['output_model_path'] = student_model

        super()._run()

        student_model = common_kwargs['output_model_path']
        onnx_ir_version = common_kwargs['onnx_ir_version']

        if self.with_convert:
            student_model = self._convert_model(student_model)
        #
        
        # save student model
        # export to pt2 - optional
        student_model_path_pt2 = os.path.splitext(student_model_path)[0] + ".pt2"
        convert.ConvertModel._run_func(student_model, student_model_path_pt2, self.example_inputs)
        # export to onnx
        convert.ConvertModel._run_func(student_model, student_model_path, self.example_inputs, onnx_ir_version=onnx_ir_version)
        # copy to model_path
        shutil.copyfile(student_model_path, self.model_path)

    def _convert_model(self, student_model):
        import torch
        from torch.ao.quantization.quantize_pt2e import convert_pt2e
        from edgeai_torchmodelopt.xmodelopt.quantization.v3 import QATPT2EModule, QConfigType
        if self.qconfig_type == QConfigType.CLIP_RANGE:
            self._convert_layers(student_model)
            student_model.module.graph.lint()
            student_model.module.recompile()
        else:
            student_model = student_model.convert(make_copy=False) \
                if isinstance(student_model, QATPT2EModule) else convert_pt2e(student_model)
        #
        return student_model

    def _convert_layers(self, module):
        import torch
        from edgeai_torchmodelopt.xmodelopt.quantization.v3.fake_quantize_types import ADAPTIVE_ACTIVATION_FAKE_QUANT_TYPES, ADAPTIVE_WEIGHT_FAKE_QUANT_TYPES
        for n, m in list(module.named_children()):
            if isinstance(m, ADAPTIVE_ACTIVATION_FAKE_QUANT_TYPES):
                # print(f'WARNING: Found FakeQuantize in the model at {n}, replace it with Clip operator before convert!')
                min_val, max_val = m.activation_post_process.min_val.item(), m.activation_post_process.max_val.item()
                if max_val > min_val:
                    clip_layer = torch.nn.Hardtanh(min_val=min_val, max_val=max_val)
                    setattr(module, n, clip_layer)
                else:
                    setattr(module, n, torch.nn.Identity())
                #
            #
            if isinstance(m, ADAPTIVE_WEIGHT_FAKE_QUANT_TYPES):
                setattr(module, n, torch.nn.Identity())
            #
            self._convert_layers(m)
        #

