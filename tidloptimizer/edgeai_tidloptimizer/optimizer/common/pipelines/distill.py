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
import tqdm
from edgeai_tidlrunner import runner

from edgeai_tidlrunner.runner.common import utils
from edgeai_tidlrunner.runner.common import bases
from edgeai_tidlrunner.runner.common.blocks import sessions
from edgeai_tidlrunner.runner.common.pipelines import compile
from ..settings.settings_default import SETTINGS_DEFAULT, COPY_SETTINGS_DEFAULT

from . import convert


class DistillModel(compile.CompileModel):
    ARGS_DICT=SETTINGS_DEFAULT['distill']
    COPY_ARGS=COPY_SETTINGS_DEFAULT['distill']

    def __init__(self, parametrization_types=None, **kwargs):
        super().__init__(**kwargs)
        self.calibration_data_cache = {}
        self.parametrization_types = parametrization_types

    def info():
        print(f'INFO: Model distill - {__file__}')

    def _prepare_model(self):
        # no model surgery required - override the method in compile
        return self.model_path

    def _prepare(self):
        super()._prepare()

        # make deterministic random for distill
        random.seed(1)

        print(f'INFO: preparing model distill with parameters: {self.kwargs}')
        common_kwargs = self.settings[self.common_prefix]
        session_kwargs = self.settings[self.session_prefix]
        runtime_options = session_kwargs['runtime_options']
        distill_kwargs = common_kwargs.get('distill', {})
        torch_device = common_kwargs['torch_device']
        
        self.input_normalizer = common_kwargs.get('input_normalizer', None)
        if not self.input_normalizer:
            self.input_normalizer = sessions.create_input_normalizer(**(session_kwargs | dict(input_optimization=False)))
        #
        self.example_inputs = common_kwargs.get('example_inputs', None)
        if not self.example_inputs:
            self.example_inputs, info_dict = self._get_input_from_dataloader(0)
        #
        self.example_inputs_on_device = tuple([tensor.to(torch_device) for tensor in self.example_inputs])

    def _run(self):
        # make deterministic random for distill
        random.seed(1)

        print(f'INFO: starting model quantize with parameters: {self.kwargs}')
        print(f'INFO: running model quantize {self.model_path}')
        common_kwargs = self.settings[self.common_prefix]
        session_kwargs = self.settings[self.session_prefix]
        runtime_options = session_kwargs['runtime_options']
        distill_kwargs = common_kwargs.get('distill', {})
        torch_device = common_kwargs['torch_device']

        calibration_iterations = runtime_options['advanced_options:calibration_iterations']
        calibration_frames = runtime_options['advanced_options:calibration_frames']
        calibration_batch_size = runtime_options['advanced_options:calibration_batch_size']
        calibration_iterations = min(calibration_iterations, len(self.dataloader)) if calibration_iterations else len(self.dataloader)
        calibration_frames = min(calibration_frames, len(self.dataloader)) if calibration_frames else len(self.dataloader)

        ###################################################################3
        # prepare model
        # from ..utils.distiller_wrapper import DistillerBaseModule as DistillerModule
        from ..utils.distiller_wrapper import DistillerModule

        teacher_model_path = common_kwargs.get('teacher_model_path', None)
        student_model_path = common_kwargs.get('output_model_path', None)
        self.example_inputs = common_kwargs.get('example_inputs', None) or self.example_inputs
        onnx_ir_version = common_kwargs['onnx_ir_version']

        teacher_model = teacher_model_path
        if isinstance(teacher_model_path, str):
            teacher_model = convert.ConvertModel._get_torch_model(teacher_model_path, example_inputs=self.example_inputs)
            teacher_model.to(torch_device)
        #

        student_model = student_model_path
        if isinstance(student_model_path, str):
            student_model = convert.ConvertModel._get_torch_model(student_model_path, example_inputs=self.example_inputs)
            student_model.to(torch_device)
        #

        calibration_iterations = runtime_options['advanced_options:calibration_iterations']
        calibration_iterations = min(calibration_iterations, len(self.dataloader)) if calibration_iterations else len(self.dataloader)
        self.distiller_model = DistillerModule(student_model, teacher_model, epochs=calibration_iterations, **distill_kwargs)

        # distill loop here
        tqdm_epoch = tqdm.tqdm(range(calibration_iterations), desc='DistillEpoch', leave=False)
        for calib_index in tqdm_epoch:
            # print(f'INFO: running model quantize iteration: {calib_index}')
            self.distiller_model.train()

            tqdm_batch = tqdm.tqdm(range(calibration_frames), desc='DistillBatch', leave=False)
            for input_index in tqdm_batch:
                # print(f'INFO: input batch for quantize: {input_index}')
                input_data, info_dict = self._get_input_from_dataloader(
                    input_index, calibration_frames, calibration_batch_size, random_shuffle=True, use_cache=True)
                
                input_data = tuple([tensor.to(torch_device) for tensor in input_data])
                
                distill_outputs = self.distiller_model(*input_data)
                distil_metrics = self.distiller_model.step_iter(*distill_outputs)
                
                tqdm_batch.set_postfix(refresh=True, epoch=calib_index, batch=input_index, **distil_metrics)
            #

            self.distiller_model.eval()

            tqdm_epoch.set_postfix(refresh=True, epoch=calib_index, num_batches=calibration_frames, **distil_metrics)
            self.distiller_model.step_epoch()
        #

        self.distiller_model.cleanup()

        if isinstance(student_model_path, str):
            convert.ConvertModel._run_func(self.distiller_model.student_model, student_model_path, self.example_inputs, onnx_ir_version=onnx_ir_version)
            common_kwargs['output_model_path'] = student_model_path
        else:
            common_kwargs['output_model_path'] = self.distiller_model.student_model
        #
        return self.distiller_model.student_model
    
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
                info_dict = self.get_info_dict(index)
                input_data, info_dict = self.dataloader(index, info_dict)
                input_data, info_dict = self.preprocess(input_data, info_dict=info_dict) if self.preprocess else (input_data, info_dict)
                input_data, info_dict = self.input_normalizer(input_data, info_dict) if self.input_normalizer else (input_data, info_dict)
                input_data = copy.deepcopy(input_data)
                # make copy to remove negative indexes in tensors that torch.from_numpy does not like
                input_data = tuple([torch.from_numpy(input_tensor) for input_tensor in input_data]) if isinstance(input_data, (list, tuple)) else (torch.from_numpy(input_data),)
                self.calibration_data_cache[index] = (input_data, info_dict)
            #
            input_list.append(input_data)
        #
        input_batch = tuple([torch.cat([t[idx] for t in input_list], dim=0) for idx in range(len(input_list[0]))]) if batch_size > 1 else input_list[0]
        return input_batch, info_dict
    