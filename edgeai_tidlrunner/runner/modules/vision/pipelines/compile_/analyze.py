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
import numpy as np
import onnx
import glob
import xlsxwriter

from ..... import utils
from ..... import bases
from ...settings.settings_default import SETTINGS_DEFAULT, COPY_SETTINGS_DEFAULT
from ..common_ import compile_base
from . import compile
from . import infer


def _write_outputs_to_bin(root, basename, output_dict):
    os.makedirs(root, exist_ok=True)
    for output_name, output_tensor in output_dict.items():
        out = np.array(output_tensor, dtype=np.float32)
        out.tofile(os.path.join(root, basename + output_name.replace("/", "_") + '.bin'))


class CompileAnalyzeNoTIDL(compile.CompileModel):
    ARGS_DICT=SETTINGS_DEFAULT['analyze']
    COPY_ARGS=COPY_SETTINGS_DEFAULT['analyze']

    def __init__(self, **kwargs):
        kargs_copy = copy.deepcopy(kwargs)
        kargs_copy['tidl_offload'] = False
        kargs_copy['session.run_dir'] = os.path.join(kargs_copy['session.run_dir'], 'notidl')
        kargs_copy['common.postprocess_enable'] = False        
        super().__init__(**kargs_copy)

    def modify_model(self):
        super().modify_model()
        if self.kwargs['common.analyze_level'] >= 2:
            model_path = self.model_path
            # Load the original ONNX model and add intermediate outputs
            onnx_model = onnx.load(model_path)
            intermediate_layer_value_info = onnx.helper.ValueInfoProto()
            intermediate_layer_value_info.name = ''
            for i in range(len(onnx_model.graph.node)):
                for j in range(len(onnx_model.graph.node[i].output)):
                    intermediate_layer_value_info.name = onnx_model.graph.node[i].output[j]
                    onnx_model.graph.output.append(intermediate_layer_value_info)
                #
            #
            onnx.save(onnx_model, model_path)

    def _prepare(self):     
        if os.path.exists(self.run_dir):
            base_dir = os.path.dirname(self.run_dir)
            print(f'INFO: clearing run_dir folder before analyze: {base_dir}')
            shutil.rmtree(base_dir, ignore_errors=True)
        #    
        super()._prepare()   
        
    def _run(self):    
        super()._run()


class InferAnalyzeNoTIDL(infer.InferModel):
    ARGS_DICT=SETTINGS_DEFAULT['analyze']
    COPY_ARGS=COPY_SETTINGS_DEFAULT['analyze']

    def __init__(self, **kwargs):
        kargs_copy = copy.deepcopy(kwargs)
        kargs_copy['tidl_offload'] = False
        kargs_copy['session.run_dir'] = os.path.join(kargs_copy['session.run_dir'], 'notidl')
        kargs_copy['common.postprocess_enable'] = False            
        super().__init__(**kargs_copy)

    def _run(self):
        super()._run()
        traces_root = os.path.join(self.run_dir, 'outputs_')
        os.makedirs(traces_root, exist_ok=True)
        for frame_idx in range(len(self.run_data)):
            frame_name = str(frame_idx)
            _write_outputs_to_bin(os.path.join(traces_root, frame_name), '', self.run_data[frame_idx]['output'])
        #
        print('INFO: onnxruntime outputs generated')


class CompileAnalyzeTIDL(compile.CompileModel):
    ARGS_DICT = SETTINGS_DEFAULT['analyze']
    COPY_ARGS = COPY_SETTINGS_DEFAULT['analyze']

    def __init__(self, **kwargs):
        kargs_copy = copy.deepcopy(kwargs)
        kargs_copy['session.run_dir'] = os.path.join(kargs_copy['session.run_dir'], 'tidl')
        kargs_copy['common.postprocess_enable'] = False            
        super().__init__(**kargs_copy)

    def _run(self):
        super()._run()


class InferAnalyzeTIDL(infer.InferModel):
    ARGS_DICT=SETTINGS_DEFAULT['analyze']
    COPY_ARGS=COPY_SETTINGS_DEFAULT['analyze']

    def __init__(self, **kwargs):
        kargs_copy = copy.deepcopy(kwargs)
        kargs_copy['session.run_dir'] = os.path.join(kargs_copy['session.run_dir'], 'tidl')
        kargs_copy['session.runtime_options.debug_level'] = 0 if kargs_copy['common.analyze_level'] == 0 else 4
        kargs_copy['common.postprocess_enable'] = False            
        super().__init__(**kargs_copy)

    def _run(self):
        super()._run()
        traces_root = os.path.join(self.run_dir, 'outputs_')
        os.makedirs(traces_root, exist_ok=True)
        for frame_idx in range(len(self.run_data)):
            frame_name = str(frame_idx)
            _write_outputs_to_bin(os.path.join(traces_root, frame_name), '', self.run_data[frame_idx]['output'])
        #
        print('INFO: TIDL outputs generated')
        print('INFO: TIDL traces generated')

    def _run_frame(self, input_index, *args, **kwargs):
        if self.kwargs['common.analyze_level'] >= 2:
            os.system('rm -f /tmp/tidl_*.bin')
        #
        run_dict = super()._run_frame(input_index, *args, **kwargs)
        if self.kwargs['common.analyze_level'] >= 2:
            traces_root = os.path.join(self.run_dir, 'traces_', str(input_index))
            os.makedirs(traces_root, exist_ok=True)
            os.system(f'mv /tmp/tidl_*.bin {traces_root}')
        #
        return run_dict


class InferAnalyzeFinal(compile_base.CompileModelBase):
    ARGS_DICT=SETTINGS_DEFAULT['analyze']
    COPY_ARGS=COPY_SETTINGS_DEFAULT['analyze']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run(self):
        notidl_run_dir = os.path.join(self.run_dir, 'notidl')
        tidl_run_dir = os.path.join(self.run_dir, 'tidl')
        notidl_outputs_dir = os.path.join(notidl_run_dir, 'outputs_')
        tidl_traces_dir = os.path.join(tidl_run_dir, 'traces_')
        tidl_outputs_dir = os.path.join(tidl_run_dir, 'outputs_')

        notidl_outputs_dirs = [os.path.join(notidl_outputs_dir,f) for f in os.listdir(notidl_outputs_dir)]
        tidl_traces_dirs = [os.path.join(tidl_traces_dir,f) for f in os.listdir(tidl_traces_dir)]
        tidl_outputs_dirs = [os.path.join(tidl_outputs_dir,f) for f in os.listdir(tidl_outputs_dir)]

        notidl_outputs_dirs.sort()
        tidl_traces_dirs.sort()
        tidl_outputs_dirs.sort()

        if len(notidl_outputs_dirs) != len(tidl_outputs_dirs):
            print(f"ERROR: Number of outputs in notidl and tidl folders do not match")
            raise ValueError(f"ERROR: Number of outputs in notidl and tidl folders do not match")
        #
        
        analyze_xlsx_path = os.path.join(self.run_dir, "analyze.xlsx")
        analyze_xlsx =  xlsxwriter.Workbook(analyze_xlsx_path, options=dict(nan_inf_to_errors=True))
        self._analyze_model_outputs(notidl_outputs_dirs, tidl_outputs_dirs, analyze_xlsx)
        if self.kwargs['common.analyze_level'] >= 2:
            self._analyze_model_layers(notidl_outputs_dirs, tidl_traces_dirs, analyze_xlsx)
        #
        analyze_xlsx.close()
        print(f"INFO: Data successfully written to {analyze_xlsx_path}")

    def _analyze_model_outputs(self, notidl_outputs_dirs, tidl_outputs_dirs, analyze_xlsx):
        num_traces = len(tidl_outputs_dirs)
        frame_worksheet = analyze_xlsx.add_worksheet('diff_outputs')
        for frame_idx in range(num_traces):
            golden_output_dir = notidl_outputs_dirs[frame_idx]
            tidl_output_dir = tidl_outputs_dirs[frame_idx]
            golden_output_files = [os.path.join(golden_output_dir, f) for f in os.listdir(golden_output_dir)]
            tidl_output_files = [os.path.join(tidl_output_dir, f) for f in os.listdir(tidl_output_dir)]
            golden_output_files.sort()
            tidl_output_files.sort()
            if len(tidl_output_files) == 0 or len(golden_output_files) == 0:
                print(f"ERROR: Number of traces in notidl and tidl outputs do not match")
                raise ValueError(f"ERROR: Number of traces in notidl and tidl outputs do not match")
            #
            num_outputs = len(tidl_output_files)
            column_start_idx = 0
            for output_idx in range(num_outputs):
                tidl_output_file = tidl_output_files[output_idx]
                golden_output_file = os.path.join(golden_output_dir, os.path.basename(tidl_output_file))
                goldenBuffer = np.fromfile(golden_output_file, dtype=np.float32).flatten()
                tidlBuffer = np.fromfile(tidl_output_file, dtype=np.float32).flatten()
                diff_dict = self._get_tensor_diff(goldenBuffer, tidlBuffer)

                frame_worksheet.write_row(0, column_start_idx+1, ['ONNXLayer', 'TIDLONNXLayer'])
                frame_worksheet.write_row(0, column_start_idx+3, diff_dict.keys())

                frame_worksheet.write(frame_idx + 1, column_start_idx+0, str(frame_idx))
                frame_worksheet.write(frame_idx + 1, column_start_idx+1, os.path.basename(golden_output_file))
                frame_worksheet.write(frame_idx + 1, column_start_idx+2, os.path.basename(tidl_output_file))
                frame_worksheet.write_row(frame_idx + 1, column_start_idx+3, diff_dict.values())
                column_start_idx += (3 + len(diff_dict))
            #
        #

    def _analyze_model_layers(self, notidl_traces_dirs, tidl_traces_dirs, analyze_xlsx):
        if len(notidl_traces_dirs) != len(tidl_traces_dirs):
            print(f"ERROR: Number of traces in notidl and tidl folders do not match")
            raise ValueError(f"ERROR: Number of traces in notidl and tidl folders do not match")
        #
        tidl_run_dir = os.path.join(self.run_dir, 'tidl')
        num_traces = len(tidl_traces_dirs)

        layer_info_dir = os.path.join(tidl_run_dir, 'artifacts', 'tempDir')
        layer_info_files = [os.path.join(layer_info_dir,f) for f in os.listdir(layer_info_dir) if f.endswith('layer_info.txt')]
        for frame_idx in range(num_traces):
            frame_worksheet = analyze_xlsx.add_worksheet('diff_frame_' + str(frame_idx))
            row_idx = 1
            for subgraph_idx, layer_info_path in enumerate(layer_info_files):
                tidl_onnx_trace_mapping = self._get_traces(notidl_traces_dirs[frame_idx], tidl_traces_dirs[frame_idx], layer_info_path)
                layers_diff = self._get_layers_diff(notidl_traces_dirs[frame_idx], tidl_traces_dirs[frame_idx], layer_info_path, tidl_onnx_trace_mapping)
                frame_worksheet.write_row(0, 1, ['ONNXLayer', 'TIDLDataID'])
                frame_worksheet.write_row(0, 3, layers_diff[0].keys())
                tidl_onnx_trace_mapping_keys = list(tidl_onnx_trace_mapping.keys())
                for layer_idx in range(len(layers_diff)):
                    tidl_data_id = tidl_onnx_trace_mapping_keys[layer_idx]
                    tidl_onnx_trace_mapping_entry = tidl_onnx_trace_mapping[tidl_data_id]
                    frame_worksheet.write(row_idx+layer_idx, 0, str(subgraph_idx) + '_' + str(layer_idx))
                    frame_worksheet.write(row_idx+layer_idx, 1, tidl_onnx_trace_mapping_entry[2])
                    frame_worksheet.write(row_idx+layer_idx, 2, tidl_onnx_trace_mapping_entry[3])
                    frame_worksheet.write_row(row_idx+layer_idx, 3, layers_diff[layer_idx].values())
                #
                row_idx += len(layers_diff)
            #
        #

    def _get_traces(self, onnx_trace_folder, tidl_trace_folder, layer_info_path):
        """
        Computes the mapping between TIDL and ONNX traces
        """
        entries = [line.strip().split(" ") for line in open(layer_info_path)]
        onnx_entries = os.listdir(onnx_trace_folder)
        tidl_onnx_trace_mapping = {}
        for entry in entries:
            if entry[0] != entry[1]:
                continue
            tidl_data_id = entry[0]
            onnx_layer_name = entry[-1]
            onnx_layer_id = onnx_layer_name.replace("/", "_")

            _tidl_data_id = tidl_data_id
            while len(_tidl_data_id) < 4:
                _tidl_data_id = "0" + _tidl_data_id

            _tidl_trace_path = os.path.join(
                tidl_trace_folder,
                f"tidl_trace_subgraph_0_{_tidl_data_id}*_float.bin",
            )

            onnx_trace_path = list(
                filter(lambda path: onnx_layer_id in path, onnx_entries)
            )
            tidl_trace_path = glob.glob(_tidl_trace_path)

            if len(onnx_trace_path) > 0 and len(tidl_trace_path) > 0:
                onnx_trace_path = os.path.join(
                    onnx_trace_folder, onnx_trace_path[0]
                )
                tidl_trace_path = tidl_trace_path[0]
                tidl_onnx_trace_mapping[tidl_data_id] = [
                    os.path.basename(onnx_trace_path),
                    os.path.basename(tidl_trace_path),
                    onnx_layer_name,
                    tidl_data_id
                ]
            else:
                print(f"WARNING: Traces Not found for outdataId: {_tidl_data_id}")
            #
        #
        return tidl_onnx_trace_mapping

    def _get_tensor_diff(self, goldenBuffer, tidlBuffer):
        try:
            delta = goldenBuffer - tidlBuffer
        except:
            delta = np.zeros_like(tidlBuffer)
        #
        eps = 1e-6
        scale = np.mean(np.absolute(goldenBuffer))
        scale = np.abs(scale)
        abs_delta = np.absolute(delta)
        max = np.max(abs_delta)
        mean = np.mean(abs_delta)
        mae = mean / scale if scale > eps else (mean+eps) / (scale+eps)
        max_abs = np.mean(np.absolute(delta))
        diff_dict = {'MeanAbsRelE': float(mae), 'MaxAbsE': float(max), 'MeanMaxAbsE': float(max_abs)}
        return diff_dict

    def _get_layers_diff(self, onnx_trace_folder, tidl_trace_folder, layer_info_path, tidl_onnx_trace_mapping):
        """
        Generate error summary for the entire network
        """
        traces_files = list(tidl_onnx_trace_mapping.keys())
        num_traces_files = len(traces_files)
        combined_diff = []
        for idx in range(num_traces_files):
            layer_idx = traces_files[idx]
            tidl = os.path.join(
                tidl_trace_folder,
                tidl_onnx_trace_mapping[layer_idx][1],
            )
            golden = os.path.join(
                onnx_trace_folder,
                tidl_onnx_trace_mapping[layer_idx][0],
            )
            goldenBuffer = np.fromfile(golden, dtype=np.float32).flatten()
            tidlBuffer = np.fromfile(tidl, dtype=np.float32).flatten()

            tidl_onnx_trace_mapping[layer_idx].extend([goldenBuffer, tidlBuffer])
            diff_dict = self._get_tensor_diff(goldenBuffer, tidlBuffer)
            combined_diff.append(diff_dict)
        #
        return combined_diff

