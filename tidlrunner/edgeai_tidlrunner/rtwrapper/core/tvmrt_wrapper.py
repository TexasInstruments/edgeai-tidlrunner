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


class TVMRuntimeWrapper(BaseRuntimeWrapper):
    def __init__(self, tidl_offload=True, **kwargs):
        super().__init__(tidl_offload=tidl_offload, **kwargs)
        self._num_run_import = 0
        self._input_list = []
        self.supported_machines = (
            presets.TargetMachineType.TARGET_MACHINE_PC_EMULATION,
            presets.TargetMachineType.TARGET_MACHINE_EVM
        )

        self.platform_mapping_dict = {
            'AM62'   : 'AM62',
            'AM62X'   : 'AM62',
            'AM62A'  : 'AM62A',
            'AM62AX'  : 'AM62A',
            'J722S'  : 'AM67A',
            'AM67A'  : 'AM67A',
            'TDA4AEN' : 'AM67A',
            'J721E'  : 'AM68PA',
            'AM68PA'  : 'AM68PA',
            'TDA4VM'  : 'AM68PA',
            'J721S2' : 'AM68A',
            'AM68A' : 'AM68A',
            'TDA4VL' : 'AM68A',
            'TDA4AL' : 'AM68A',
            'J742S2' : 'AM69A', # to be corrected?
            'TDA4VP' : 'AM69A', # to be corrected?
            'TDA4AP' : 'AM69A', # to be corrected?
            'J784S4' : 'AM69A',
            'AM69A' : 'AM69A',
            'TDA4VH' : 'AM69A',
            'TDA4AH' : 'AM69A',
        }

    def start_import(self):
        if self._start_import_done:
            return self.interpreter
        #
        self.is_import = True
        self.kwargs = self._set_default_options(self.kwargs)
        self._calibration_frames = self.kwargs['runtime_options']['advanced_options:calibration_frames']
       # tvm/dlr requires input shape in prepare_for_import - so moved this ahead
        self.kwargs['input_details'] = self._get_input_details(None, self.kwargs.get('input_details', None))
        self.kwargs['output_details'] = self._get_output_details(None, self.kwargs.get('output_details', None))
        self._input_list = []
        self._start_import_done = True
        return True

    def run_import(self, input_data, output_keys=None):
        if not self._start_import_done:
            self.start_import()
        #
        self._num_run_import += 1

        #_format_input_data was not called yet, as shapes were not available - call it here:
        input_data = self._format_input_data(input_data)
        self._input_list.append(input_data)

        output = None
        if len(self._input_list) == self._calibration_frames:
            self.interpreter = self._create_interpreter_for_import(self._input_list)
        elif len(self._input_list) > self._calibration_frames:
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
        self.kwargs['input_details'] = self._get_input_details(None, self.kwargs.get('input_details', None))
        self.kwargs['output_details'] = self._get_output_details(None, self.kwargs.get('output_details', None))
        # moved the import inside the function, so that dlr needs to be installed only if someone wants to use it
        import tvm
        from tvm.contrib import graph_executor as runtime
        artifacts_folder = self.kwargs['artifacts_folder']
        if not os.path.exists(artifacts_folder):
            return False
        
        loaded_json = open(artifacts_folder + "/deploy_graph.json").read()
        loaded_lib = tvm.runtime.load_module(artifacts_folder + "/deploy_lib.so","so")
        loaded_params = bytearray(open(artifacts_folder + "/deploy_param.params", "rb").read())
        # create a runtime executor module
        sess = runtime.create(loaded_json, loaded_lib, tvm.cpu())
        sess.load_params(loaded_params)
        self.interpreter = sess

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
        # feed input data
        for key, value in input_data.items():
            self.interpreter.set_input(key, value)
        self.interpreter.run()
        outputs = []
        for i in range(self.interpreter.get_num_outputs()):
            outputs.append(self.interpreter.get_output(i).asnumpy())

        output_keys = output_keys or [d_info['name'] for d_info in self.kwargs['output_details']]
        output_dict = {output_key:output for output_key, output in zip(output_keys, outputs)}
        return output_dict

    def _create_interpreter_for_import(self, calib_list):
        # onnx and tvm are required only for model import
        # so import inside the function so that inference can be done without it
        from tvm import relay
        from tvm.relay.backend.contrib import tidl

        target_machine = self.kwargs['target_machine']
        target_device = self.kwargs['target_device']
        platform_name = self.platform_mapping_dict[target_device]

        model_path = self.kwargs['model_path']
        model_path0 = model_path[0] if isinstance(model_path, (list,tuple)) else model_path
        model_type = self.kwargs['model_type'] or os.path.splitext(model_path0)[1][1:]

        input_details = self.kwargs['input_details']
        input_shape = {inp_d['name']:inp_d['shape'] for inp_d in input_details}
        input_keys = list(input_shape.keys())

        artifacts_folder = self.kwargs['artifacts_folder']
        os.makedirs(artifacts_folder, exist_ok=True)

        from tvm.contrib import tidl

        # the artifact files that are generated
        deploy_lib = 'deploy_lib.so'
        deploy_graph = 'deploy_graph.json'
        deploy_params = 'deploy_param.params'

        for target_machine in self.supported_machines:
            if target_machine == presets.TargetMachineType.TARGET_MACHINE_EVM:
                if(os.path.exists(os.path.join(artifacts_folder, f'{deploy_lib}.pc'))):
                    print("INFO: Reusing TIDL artifacts from x86 compilation for target compilation")
                    os.environ["REUSE_TIDL_ARTIFACTS"] = '1'

            print(f"INFO: Compiling for target device -- {target_machine}")
            status = tidl.compile_model(
                                        platform = platform_name.lower(),
                                        compile_for_device = (True if (target_machine == presets.TargetMachineType.TARGET_MACHINE_EVM) else False),
                                        enable_tidl_offload = self.kwargs.get('tidl_offload', True),
                                        delegate_options = self.kwargs['runtime_options'],
                                        calibration_input_list = calib_list,
                                        model_path = model_path0,
                                        input_shape_dict = input_shape
                                        ### Optional arguments: This API does model to Relay conversion internally, however it can be overridden using already converted IR module and params 
                                        # mod = mod,   # Input Relay IR module.
                                        # params = params   # The parameter dict used by Relay.
                                        )
            assert(status)
            os.listdir(artifacts_folder)
            
            # save the deployables
            path_lib_orig = os.path.join(artifacts_folder, f'{deploy_lib}')
            path_graph_orig = os.path.join(artifacts_folder, f'{deploy_graph}')
            path_params_orig = os.path.join(artifacts_folder, f'{deploy_params}')

            path_lib_target_machine = os.path.join(artifacts_folder, f'{deploy_lib}.{target_machine}')
            path_graph_target_machine = os.path.join(artifacts_folder, f'{deploy_graph}.{target_machine}')
            path_params_target_machine = os.path.join(artifacts_folder, f'{deploy_params}.{target_machine}')

            os.rename(path_lib_orig, path_lib_target_machine)
            os.rename(path_graph_orig, path_graph_target_machine)
            os.rename(path_params_orig, path_params_target_machine)

            os.environ.pop("REUSE_TIDL_ARTIFACTS", None) # Clean up this env variable for next run

        # create a symbolic link to the deploy_lib specified in target_machine
        artifacts_folder = self.kwargs['artifacts_folder']

        cwd = os.getcwd()
        os.chdir(artifacts_folder)
        artifact_files = [deploy_lib, deploy_graph, deploy_params]
        for artifact_file in artifact_files:
            os.symlink(f'{artifact_file}.{target_machine}', artifact_file)
        #
        os.chdir(cwd)

    def _format_input_data(self, input_data):
        if isinstance(input_data, dict):
            return input_data

        if not isinstance(input_data, (list,tuple)):
            input_data = (input_data,)

        input_details = self.kwargs['input_details']
        input_shape = {inp_d['name']:inp_d['shape'] for inp_d in input_details}
        input_keys = list(input_shape.keys())
        input_data = {d_name:d for d_name, d in zip(input_keys,input_data)}
        return input_data

    def _set_default_options(self, kwargs):
        # tvm need advanced settings as a dict
        # convert the entries starting with advanced_options: to a dict
        runtime_options = kwargs['runtime_options']
        advanced_options_prefix = 'advanced_options:'
        advanced_options = {k.replace(advanced_options_prefix,''):v for k,v in runtime_options.items() \
                            if k.startswith(advanced_options_prefix)}
        runtime_options = {k:v for k,v in runtime_options.items() if not k.startswith(advanced_options_prefix)}
        runtime_options.update(dict(advanced_options=advanced_options))
        kwargs['runtime_options'] = runtime_options
        return kwargs

    def set_runtime_option(self, option, value):
        advanced_options_prefix = 'advanced_options:'
        if advanced_options_prefix in option:
            option = option.replace(advanced_options_prefix, '')
            self.kwargs['runtime_options']['advanced_options'][option] = value
        else:
            self.kwargs['runtime_options'][option] = value

    def get_runtime_option(self, option, default=None):
        advanced_options_prefix = 'advanced_options:'
        if advanced_options_prefix in option:
            option = option.replace(advanced_options_prefix, '')
            return self.kwargs['runtime_options']['advanced_options'].get(option, default)
        else:
            return self.kwargs['runtime_options'].get(option, default)

    def _get_input_details(self, dlr_interpreter, input_details=None):
        if input_details is None:
            model_path = self.kwargs['model_path']
            model_path0 = model_path[0] if isinstance(model_path, (list,tuple)) else model_path
            model_type = self.kwargs.get('model_type',None) or os.path.splitext(model_path0)[1][1:]
            if model_type == 'onnx':
                import onnxruntime
                sess_options = onnxruntime.SessionOptions()
                ep_list = ['CPUExecutionProvider']
                interpreter = onnxruntime.InferenceSession(model_path0, providers=ep_list,
                                provider_options=[{}], sess_options=sess_options)
                input_details = super()._get_input_details_onnx(interpreter, input_details)
                del interpreter
            elif model_type == 'tflite':
                import tflite_runtime.interpreter as tflitert_interpreter
                runtime_options_temp = copy.deepcopy(self.kwargs['runtime_options'])
                runtime_options_temp['artifacts_folder'] = os.path.join(runtime_options_temp['artifacts_folder'], '_temp_details')
                self.kwargs['runtime_options']['import'] = 'yes'
                os.makedirs(runtime_options_temp['artifacts_folder'], exist_ok=True)
                self._clear_folder(runtime_options_temp['artifacts_folder'])
                interpreter = tflitert_interpreter.Interpreter(model_path0)
                input_details = self._get_input_details_tflite(interpreter, input_details)
                self._clear_folder(runtime_options_temp['artifacts_folder'], remove_base_folder=True)
                del interpreter
                del runtime_options_temp
            else:
                raise RuntimeError('input_details can be obtained for onnx and tiflite models - for others, it must be provided')
            #
        #
        return input_details

    def _get_output_details(self, dlr_interpreter, output_details=None):
        if output_details is None:
            model_path = self.kwargs['model_path']
            model_path0 = model_path[0] if isinstance(model_path, (list,tuple)) else model_path
            model_type = self.kwargs.get('model_type',None) or os.path.splitext(model_path0)[1][1:]
            if model_type == 'onnx':
                import onnxruntime
                sess_options = onnxruntime.SessionOptions()
                ep_list = ['CPUExecutionProvider']
                interpreter = onnxruntime.InferenceSession(model_path0, providers=ep_list,
                                provider_options=[{}], sess_options=sess_options)
                output_details = super()._get_output_details_onnx(interpreter, output_details)
                del interpreter
            elif model_type == 'tflite':
                import tflite_runtime.interpreter as tflitert_interpreter
                runtime_options_temp = copy.deepcopy(self.kwargs['runtime_options'])
                runtime_options_temp['artifacts_folder'] = os.path.join(runtime_options_temp['artifacts_folder'], '_temp_details')
                self.kwargs['runtime_options']['import'] = 'yes'
                os.makedirs(runtime_options_temp['artifacts_folder'], exist_ok=True)
                self._clear_folder(runtime_options_temp['artifacts_folder'])
                interpreter = tflitert_interpreter.Interpreter(model_path0)
                output_details = self._get_output_details_tflite(interpreter, output_details)
                self._clear_folder(runtime_options_temp['artifacts_folder'], remove_base_folder=True)
                del interpreter
                del runtime_options_temp
            else:
                raise RuntimeError('output_details can be obtained for onnx and tiflite models - for others, it must be provided')
            #
        #
        return output_details

    def _load_mxnet_model(self, model_path):
        import mxnet
        assert isinstance(model_path, list) and len(model_path) == 2, 'mxnet model path must be a list of size 2'

        model_json = mxnet.symbol.load(model_path[0])
        save_dict = mxnet.ndarray.load(model_path[1])
        arg_params = {}
        aux_params = {}
        for key, param in save_dict.items():
            tp, name = key.split(':', 1)
            if tp == 'arg':
                arg_params[name] = param
            elif tp == 'aux':
                aux_params[name] = param
            #
        #
        return model_json, arg_params, aux_params
