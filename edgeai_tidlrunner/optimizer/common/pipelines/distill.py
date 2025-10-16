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
import random
from edgeai_tidlrunner import runner

from edgeai_tidlrunner.runner.common import utils
from edgeai_tidlrunner.runner.common import bases
from edgeai_tidlrunner.runner.common.settings.settings_default import SETTINGS_DEFAULT, COPY_SETTINGS_DEFAULT
from edgeai_tidlrunner.runner.common.blocks import sessions
from edgeai_tidlrunner.runner.common.pipelines import compile

from . import convert


def get_distill_wrapper_module():
    # define inside a function, to avoid torch and torchao depdency at start
    import torch
    import torchao
    class _DistillWrapperModule(torch.nn.Module):
        def __init__(self, student_model, teacher_model, **kwargs):
            super().__init__()
            self.student_model = student_model
            self.teacher_model = teacher_model
            
            self.criterion = torch.nn.SmoothL1Loss() #torch.nn.MSELoss() #torch.nn.KLDivLoss()
            self.epochs = kwargs.get('epochs', 10)
            self.lr = kwargs.get('lr', 0.0001)
            self.lr_min = kwargs.get('lr_min', self.lr/100)
            self.momentum = kwargs.get('momentum', 0.9)
            self.weight_decay = kwargs.get('weight_decay', 1e-4)
            self.temperature = kwargs.get('temperature', 1)

            self.optimizer = torch.optim.SGD(student_model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=self.lr_min)
        
        def forward(self, *inputs):
            with torch.no_grad():
                teacher_outputs = self.teacher_model(*inputs)
            #
            student_outputs = self.student_model(*inputs)
            return student_outputs, teacher_outputs
        
        def eval(self):
            # super().eval()
            self.teacher_model.eval()
            torchao.quantization.pt2e.move_exported_model_to_eval(self.teacher_model)
            self.student_model.eval()
            torchao.quantization.pt2e.move_exported_model_to_eval(self.student_model)
            return self
        
        def train(self):
            # super().train()
            self.teacher_model.eval()
            torchao.quantization.pt2e.move_exported_model_to_eval(self.teacher_model)
            self.student_model.train()
            torchao.quantization.pt2e.move_exported_model_to_train(self.student_model)
            return self
        
        def step_iter(self, outputs, targets):
            if isinstance(outputs, (list, tuple)) and isinstance(targets, (list, tuple)):
                assert len(outputs) == len(targets), f'number of outputs {len(outputs)} and targets {len(targets)} should be same'
                loss = sum([self.criterion(o, t) for o,t in zip(outputs, targets)])
            else:
                loss = self.criterion(outputs, targets)
            #
            lr = round(self.scheduler.get_last_lr()[0], 6)
            loss_value = round(loss.item(), 6)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return {'lr':lr, 'loss':loss_value}
        
        def step_epoch(self):
            self.scheduler.step()

    return _DistillWrapperModule


class DistillModel(compile.CompileModel):
    ARGS_DICT=SETTINGS_DEFAULT['distill']
    COPY_ARGS=COPY_SETTINGS_DEFAULT['distill']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calibration_data_cache = {}

    def _prepare(self):
        super()._prepare()

    def info():
        print(f'INFO: Model distill - {__file__}')

    def _prepare(self):
        print(f'INFO: preparing model distill with parameters: {self.kwargs}')
        common_kwargs = self.settings[self.common_prefix]
        session_kwargs = self.settings[self.session_prefix]
        super()._prepare()
        self.input_normalizer = sessions.create_input_normalizer(**session_kwargs)

    def _run(self):
        import torch
        import torchao
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
            teacher_model = convert.ConvertModel._get_torch_model(teacher_model_path)
            if not example_inputs:
                example_inputs = convert.ConvertModel._get_example_inputs(teacher_model_path, to_torch=True)
            #
        #

        student_model = student_model_path
        if isinstance(student_model_path, str):
            student_model = convert.ConvertModel._get_torch_model(student_model_path)
            if not example_inputs:
                example_inputs = convert.ConvertModel._get_example_inputs(student_model_path, to_torch=True)
            #
        #

        if not example_inputs:
            input_data, info_dict = self._get_input_from_dataloader(0)
        #

        calibration_iterations = runtime_options['advanced_options:calibration_iterations']
        calibration_frames = runtime_options['advanced_options:calibration_frames']
        calibration_batch_size = runtime_options['advanced_options:calibration_batch_size']
        calibration_iterations = min(calibration_iterations, len(self.dataloader)) if calibration_iterations else len(self.dataloader)
        calibration_frames = min(calibration_frames, len(self.dataloader)) if calibration_frames else len(self.dataloader)

        DistillWrapperModule = get_distill_wrapper_module()
        self.distill_model = DistillWrapperModule(student_model, teacher_model, epochs=calibration_iterations, **distill_kwargs)
        self.distill_model.train()

        # distill loop here
        for calib_index in range(calibration_iterations):
            print(f'INFO: running model quantize iteration: {calib_index}')
            for input_index in range(calibration_frames):
                print(f'INFO: input batch for quantize: {input_index}')
                input_data, info_dict = self._get_input_from_dataloader(
                    input_index, calibration_frames, calibration_batch_size, random_shuffle=True, use_cache=True)
                distill_outputs = self.distill_model(*input_data)
                distil_metrics = self.distill_model.step_iter(*distill_outputs)
                print(f'INFO: input batch for quantize: {input_index}, {distil_metrics}')
            #
            self.distill_model.step_epoch()
        #
        if isinstance(student_model_path, str):
            convert.ConvertModel._run_func(student_model, student_model_path, example_inputs)
        #

        self.distill_model.eval()
        return student_model
    
    def _get_input_from_dataloader(self, index, calibration_frames=None, batch_size=1, random_shuffle=False, use_cache=False):
        import torch
        dataset_size = min(calibration_frames, len(self.dataloader)) if calibration_frames else len(self.dataloader)
        input_list = []
        for batch_index in range(batch_size):
            if random_shuffle:
                index = random.randint(0, dataset_size-1)
            else:
                index = (index + batch_index) % dataset_size
            #
            if use_cache and index in self.calibration_data_cache:
                input_data, info_dict = self.calibration_data_cache[index]
            else:
                input_data, info_dict = self.dataloader(index)
                input_data, info_dict = self.preprocess(input_data, info_dict=info_dict) if self.preprocess else (input_data, info_dict)
                input_data, info_dict = self.input_normalizer(input_data, info_dict)
                input_data = copy.deepcopy(input_data)
                # make copy to remove negative indexes in tensors that torch.from_numpy does not like
                input_data = tuple([torch.from_numpy(input_tensor) for input_tensor in input_data]) if isinstance(input_data, (list, tuple)) else (torch.from_numpy(input_data),)
                self.calibration_data_cache[index] = (input_data, info_dict)
            #
            input_list.append(input_data)
        #
        input_batch = tuple([torch.cat([t[idx] for t in input_list], dim=0) for idx in range(len(input_list[0]))]) if batch_size > 1 else input_list[0]
        return input_batch, info_dict