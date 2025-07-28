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
import numpy as np
import copy

from ..options import presets
from .basert_wrapper import BaseRuntimeWrapper


class TFLiteRuntimeWrapper(BaseRuntimeWrapper):
    def __init__(self, tidl_offload=True, **kwargs):
        super().__init__(tidl_offload=tidl_offload, **kwargs)
        self._num_run_import = 0

    def start_import(self):
        if self._start_import_done:
            return self.interpreter
        #
        self.is_import = True
        self.kwargs = self._set_default_options(self.kwargs)
        self._calibration_frames = self.kwargs['runtime_options']['advanced_options:calibration_frames']
        self.interpreter = self._create_interpreter(is_import=True)
        self.kwargs['input_details'] = self._get_input_details(self.interpreter, self.kwargs.get('input_details', None))
        self.kwargs['output_details'] = self._get_output_details(self.interpreter, self.kwargs.get('output_details', None))
        self._start_import_done = True
        return self.interpreter

    def run_import(self, input_data, output_keys=None):
        if not self._start_import_done:
            self.start_import()
        #
        input_data = self._format_input_data(input_data)
        output = self._run(input_data, output_keys)

        self._num_run_import += 1
        if self._num_run_import > self._calibration_frames:
            print(f"WARNING: not need to call run_import more than calibration_frames = {self._calibration_frames}")
        #
        return output

    def start_inference(self):
        if self._start_inference_done:
            return self.interpreter
        #
        self.is_import = False
        self.kwargs = self._set_default_options(self.kwargs)
        self._calibration_frames = self.kwargs['runtime_options']['advanced_options:calibration_frames']
        self.interpreter = self._create_interpreter(is_import=False)
        self.kwargs['input_details'] = self._get_input_details(self.interpreter, self.kwargs.get('input_details', None))
        self.kwargs['output_details'] = self._get_output_details(self.interpreter, self.kwargs.get('output_details', None))
        self._start_inference_done = True
        return self.interpreter

    def run_inference(self, input_data, output_keys=None):
        if not self._start_inference_done:
            self.start_inference()
        #
        input_data = self._format_input_data(input_data)
        return self._run(input_data, output_keys)

    def _run(self, input_data, output_keys=None):
        # if model needs additional inputs given in extra_inputs
        if self.kwargs.get('extra_inputs'):
            input_data.update(self.kwargs['extra_inputs'])
        #
        input_details = self.kwargs['input_details']
        output_details = self.kwargs['output_details']
        for (input_detail, c_data_entry) in zip(input_details, input_data):
            self._set_tensor(input_detail, c_data_entry)
        #
        self.interpreter.invoke()
        outputs = [self._get_tensor(output_detail) for output_detail in output_details]
        return outputs

    def _create_interpreter(self, is_import):
        # move the import inside the function, so that tflite_runtime needs to be installed
        # only if someone wants to use it
        import tflite_runtime.interpreter as tflitert_interpreter
        if self.kwargs['tidl_offload']:
            if is_import:
                self.kwargs['runtime_options']['import'] = "yes"
                tidl_delegate = [tflitert_interpreter.load_delegate('tidl_model_import_tflite.so', self.kwargs['runtime_options'])]
            else:
                self.kwargs['runtime_options']['import'] = "no"
                tidl_delegate = [tflitert_interpreter.load_delegate('libtidl_tfl_delegate.so', self.kwargs['runtime_options'])]
            #
            interpreter = tflitert_interpreter.Interpreter(model_path=self.kwargs['model_path'], experimental_delegates=tidl_delegate)
        else:
            interpreter = tflitert_interpreter.Interpreter(model_path=self.kwargs['model_path'])
        #
        interpreter.allocate_tensors()
        return interpreter

    def _format_input_data(self, input_data):
        if isinstance(input_data, dict):
            return input_data

        if not isinstance(input_data, (list,tuple)):
            input_data = (input_data,)

        return input_data

    def _set_tensor(self, model_input, tensor):
        if model_input['type'] == np.int8:
            # scale, zero_point = model_input['quantization']
            # tensor = np.clip(np.round(tensor/scale + zero_point), -128, 127)
            tensor = np.array(tensor, dtype=np.int8)
        elif model_input['type'] == np.uint8:
            # scale, zero_point = model_input['quantization']
            # tensor = np.clip(np.round(tensor/scale + zero_point), 0, 255)
            tensor = np.array(tensor, dtype=np.uint8)
        #
        self.interpreter.set_tensor(model_input['index'], tensor)

    def _get_tensor(self, model_output):
        tensor = self.interpreter.get_tensor(model_output['index'])
        if model_output['type'] == np.int8 or model_output['type']  == np.uint8:
            scale, zero_point = model_output['quantization']
            tensor = np.array(tensor, dtype=np.float32)
            tensor = (tensor - zero_point) / scale
        #
        return tensor

    def _get_input_details(self, interpreter, input_details=None):
        return super()._get_input_details_tflite(interpreter, input_details)

    def _get_output_details(self, interpreter, output_details=None):
        return super()._get_output_details_tflite(interpreter, output_details)

    def _set_default_options(self, kwargs):
        return kwargs

    def set_runtime_option(self, option, value):
        self.kwargs['runtime_options'][option] = value

    def get_runtime_option(self, option, default=None):
        return self.kwargs['runtime_options'].get(option, default)