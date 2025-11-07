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
import copy

from ..options import presets
from .basert_wrapper import BaseRuntimeWrapper


class ONNXRuntimeWrapper(BaseRuntimeWrapper):
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
        super()._pre_inference(input_data, output_keys)
        outputs = self._run(input_data, output_keys)
        super()._post_inference(input_data, output_keys)
        return outputs

    def _run(self, input_data, output_keys=None):
        # if model needs additional inputs given in extra_inputs
        if self.kwargs.get('extra_inputs'):
            input_data.update(self.kwargs['extra_inputs'])
        #
        # output_details is not mandatory, output_keys can be None
        output_keys = output_keys or [d_info['name'] for d_info in self.kwargs['output_details']]
        # run the actual import step
        outputs = self.interpreter.run(output_keys, input_data)
        output_dict = {output_key:output for output_key, output in zip(output_keys, outputs)}
        return output_dict

    def _create_interpreter(self, is_import):
        # move the import inside the function, so that onnxruntime needs to be installed only if someone wants to use it
        import onnxruntime
        # pass options to pybind
        if is_import:
            self.kwargs['runtime_options']['import'] = "yes"
        else:
            self.kwargs['runtime_options']['import'] = "no"
        #
        runtime_options = self.kwargs['runtime_options']

        sess_options = onnxruntime.SessionOptions()

        if self.kwargs['onnxruntime:graph_optimization_level'] is not None:
            # for transformer models, it is necessary to set graph_optimization_level in session options for onnxruntime
            # to onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL so that TIDL can properly handle the model.
            sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel(self.kwargs['onnxruntime:graph_optimization_level'])

        # suppress warnings
        sess_options.log_severity_level = 3

        if self.kwargs['tidl_offload']:
            ep_list = ['TIDLCompilationProvider', 'CPUExecutionProvider'] if is_import else \
                      ['TIDLExecutionProvider', 'CPUExecutionProvider']
            interpreter = onnxruntime.InferenceSession(self.kwargs['model_path'], providers=ep_list,
                            provider_options=[runtime_options, {}], sess_options=sess_options)
        else:
            ep_list = ['CPUExecutionProvider']
            interpreter = onnxruntime.InferenceSession(self.kwargs['model_path'], providers=ep_list,
                            provider_options=[{}], sess_options=sess_options)
        #
        return interpreter

    def _format_input_data(self, input_data):
        if isinstance(input_data, dict):
            return input_data

        if not isinstance(input_data, (list,tuple)):
            input_data = (input_data,)

        input_details = self.kwargs['input_details']
        return {d_info['name']:dat for d_info, dat in zip(input_details,input_data)}

    def _get_input_details(self, interpreter, input_details=None):
        return super()._get_input_details_onnx(interpreter, input_details)

    def _get_output_details(self, interpreter, output_details=None):
        return super()._get_output_details_onnx(interpreter, output_details)

    def _set_default_options(self, kwargs):
        kwargs['onnxruntime:graph_optimization_level'] = self.kwargs.get('onnxruntime:graph_optimization_level', 
                                                               presets.GraphOptimizationLevel.ORT_DISABLE_ALL)
        return kwargs

    def set_runtime_option(self, option, value):
        self.kwargs['runtime_options'][option] = value

    def get_runtime_option(self, option, default=None):
        return self.kwargs['runtime_options'].get(option, default)