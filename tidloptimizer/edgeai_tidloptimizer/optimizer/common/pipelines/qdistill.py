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
        from edgeai_torchmodelopt.xmodelopt.quantization.v3 import QATPT2EModule, QConfigType, QuantizerTypes
        self.qconfig_type = QConfigType.WF_AFCLIP #WF_AFCLIP #DEFAULT 
        self.quantizer_type = QuantizerTypes.TIDLRT_ADVANCED
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
        distill_kwargs = common_kwargs.get('distill', {})
        torch_device = common_kwargs['torch_device']

        teacher_model_path = os.path.join(self.teacher_folder, os.path.basename(self.model_path))
        student_model_path = os.path.join(self.student_folder, os.path.basename(self.model_path))

        # get pytorch model
        teacher_model = convert.ConvertModel._get_torch_model(teacher_model_path, example_inputs=self.example_inputs)
        teacher_model.to(torch_device)
        # it is important to freeze the teacher model's BN and Dropouts
        teacher_model.eval()

        # export teacher model - optional
        os.makedirs(self.teacher_folder, exist_ok=True)
        teacher_model_path = os.path.join(self.teacher_folder, os.path.basename(self.model_path))
        teacher_model_path_pt2 = os.path.splitext(teacher_model_path)[0] + ".pt2"
        convert.ConvertModel._run_func(teacher_model, teacher_model_path_pt2, self.example_inputs_on_device)
        
        #################################################################################
        # prepare the student model
        import torch

        # create student model - no need to do this - QATPT2EModule will do it
        # student_model = torch.export.export(teacher_model, self.example_inputs_on_device).module()
        # from torch.ao.quantization.pt2e import allow_exported_model_train_eval
        # allow_exported_model_train_eval(student_model)

        # create student model
        from edgeai_torchmodelopt.xmodelopt.quantization.v3 import QATPT2EModule, QConfigType
        from edgeai_torchmodelopt.xmodelopt.quantization.v3.fake_quantize_types import ADAPTIVE_ACTIVATION_FAKE_QUANT_TYPES
        # full: ['linear', 'linear_relu', 'conv', 'conv_relu', 'conv_transpose_relu', 'conv_bn', 'conv_bn_relu', 'conv_transpose_bn', 'conv_transpose_bn_relu', 'gru_io_only', 'adaptive_avg_pool2d', 'add_relu', 'add', 'mul_relu', 'mul', 'cat']
        # minimal: ['linear', 'linear_relu', 'conv', 'conv_relu', 'conv_bn', 'conv_bn_relu', 'conv_transpose_relu', 'conv_transpose_bn', 'conv_transpose_bn_relu']
        # added by us in advanced quantizer: 'matmul'
        annotation_patterns = ['linear', 'linear_relu', 'conv', 'conv_relu', 'conv_bn', 'conv_bn_relu', 'conv_transpose_relu', 'conv_transpose_bn', 'conv_transpose_bn_relu', 'add_relu', 'add',  'cat', 'matmul'] #'mul_relu', 'mul'
        student_model = QATPT2EModule(teacher_model, example_inputs=self.example_inputs_on_device, 
                                      qconfig_type=self.qconfig_type, quantizer_type=self.quantizer_type,
                                      total_epochs=calibration_iterations, annotation_patterns=annotation_patterns)

        #################################################################################
        # run the distillation
        common_kwargs['teacher_model_path'] = teacher_model
        common_kwargs['example_inputs'] = self.example_inputs_on_device
        common_kwargs['output_model_path'] = student_model

        super()._run()

        student_model = common_kwargs['output_model_path']
        onnx_ir_version = common_kwargs['onnx_ir_version']

        if self.with_convert:
            student_model = student_model.convert()
        #
        
        # save student model
        # export to pt2 - optional
        student_model_path_pt2 = os.path.splitext(student_model_path)[0] + ".pt2"
        convert.ConvertModel._run_func(student_model, student_model_path_pt2, self.example_inputs_on_device)
        # export to onnx
        convert.ConvertModel._run_func(student_model, student_model_path, self.example_inputs_on_device, onnx_ir_version=onnx_ir_version)
        # copy to model folder
        pipeline_type = common_kwargs['pipeline_type']
        final_model_path_wo_ext, final_model_ext = os.path.splitext(self.model_path)
        final_model_path = final_model_path_wo_ext + f'_{pipeline_type}' + final_model_ext
        shutil.copyfile(student_model_path, final_model_path)
        # create symlink to model_path
        cur_dir = os.getcwd()
        os.chdir(os.path.dirname(self.model_path))
        os.symlink(os.path.basename(final_model_path), os.path.basename(self.model_path))
        os.chdir(cur_dir)
