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


class CompileAnalyzeNoTIDL(compile.CompileModel):
    ARGS_DICT=SETTINGS_DEFAULT['analyze']
    COPY_ARGS=COPY_SETTINGS_DEFAULT['analyze']

    def __init__(self, **kwargs):
        kargs_copy = copy.deepcopy(kwargs)
        kargs_copy['tidl_offload'] = False
        kargs_copy['session.run_path'] = os.path.join(kargs_copy['session.run_path'], 'notidl')
        kargs_copy['common.postprocess_enable'] = False        
        super().__init__(**kargs_copy)

    def modify_model(self):
        super().modify_model()
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
        if os.path.exists(self.run_path):
            base_dir = os.path.dirname(self.run_path)
            print(f'INFO: clearing run_path folder before analyze: {base_dir}')
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
        kargs_copy['session.run_path'] = os.path.join(kargs_copy['session.run_path'], 'notidl')
        kargs_copy['common.postprocess_enable'] = False            
        super().__init__(**kargs_copy)

    def _run(self):
        super()._run()
        traces_root = os.path.join(self.run_path, 'traces')
        os.makedirs(traces_root, exist_ok=True)
        for frame_idx in range(len(self.run_data)):
            frame_name = str(frame_idx)
            self.write_outputs_to_bin(os.path.join(traces_root, frame_name), '', self.run_data[frame_idx]['output'])
        #
        print('INFO: onnxruntime traces generated')

    def write_outputs_to_bin(self, root, basename, output_dict):
        os.makedirs(root, exist_ok=True)
        for output_name, output_tensor in output_dict.items():
            out = np.array(output_tensor, dtype=np.float32)
            out.tofile(os.path.join(root, basename + output_name.replace("/", "_") + '.bin'))


class CompileAnalyzeTIDL(compile.CompileModel):
    ARGS_DICT = SETTINGS_DEFAULT['analyze']
    COPY_ARGS = COPY_SETTINGS_DEFAULT['analyze']

    def __init__(self, **kwargs):
        kargs_copy = copy.deepcopy(kwargs)
        kargs_copy['session.run_path'] = os.path.join(kargs_copy['session.run_path'], 'tidl')
        kargs_copy['common.postprocess_enable'] = False            
        super().__init__(**kargs_copy)

    def _run(self):
        super()._run()


class InferAnalyzeTIDL(infer.InferModel):
    ARGS_DICT=SETTINGS_DEFAULT['analyze']
    COPY_ARGS=COPY_SETTINGS_DEFAULT['analyze']

    def __init__(self, **kwargs):
        kargs_copy = copy.deepcopy(kwargs)
        kargs_copy['session.run_path'] = os.path.join(kargs_copy['session.run_path'], 'tidl')
        kargs_copy['session.runtime_options.debug_level'] = 4
        kargs_copy['common.postprocess_enable'] = False            
        super().__init__(**kargs_copy)

    def _run(self):
        super()._run()
        print('INFO: TIDL traces generated')

    def _run_frame(self, input_index, *args, **kwargs):
        os.system('rm -f /tmp/tidl_*.bin')
        run_dict = super()._run_frame(input_index, *args, **kwargs)
        traces_root = os.path.join(self.run_path, 'traces', str(input_index))
        os.makedirs(traces_root, exist_ok=True)
        os.system(f'mv /tmp/tidl_*.bin {traces_root}')
        return run_dict


class InferAnalyzeFinal(compile_base.CompileModelBase):
    ARGS_DICT=SETTINGS_DEFAULT['analyze']
    COPY_ARGS=COPY_SETTINGS_DEFAULT['analyze']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run(self):
        notidl_run_path = os.path.join(self.run_path, 'notidl')
        tidl_run_path = os.path.join(self.run_path, 'tidl')
        notidl_traces = os.path.join(notidl_run_path, 'traces')
        tidl_traces = os.path.join(tidl_run_path, 'traces')
        layer_info_dir = os.path.join(tidl_run_path, 'artifacts', 'tempDir')
        layer_info_files = [f for f in os.listdir(layer_info_dir) if f.endswith('layer_info.txt')]
        layer_info_path = os.path.join(layer_info_dir, layer_info_files[0])

        analyze_xlsx_path = os.path.join(self.run_path, "analyze.xlsx")
        analyze_xlsx =  xlsxwriter.Workbook(analyze_xlsx_path, options=dict(nan_inf_to_errors=True))
        num_traces = len(os.listdir(notidl_traces))
        for frame_idx in range(num_traces):
            frame_worksheet = analyze_xlsx.add_worksheet(str(frame_idx))
            notidl_traces_frame_id_dir = os.path.join(notidl_traces, str(frame_idx))
            tidl_traces_frame_id_dir = os.path.join(tidl_traces, str(frame_idx))
            tidl_onnx_trace_mapping = self.get_traces(notidl_traces_frame_id_dir, tidl_traces_frame_id_dir, layer_info_path)
            layers_diff = self.get_layers_diff(notidl_traces_frame_id_dir, tidl_traces_frame_id_dir, layer_info_path, tidl_onnx_trace_mapping)

            frame_worksheet.write_row(0, 0, ['ONNXLayer', 'TIDLDataID'])
            frame_worksheet.write_row(0, 2, layers_diff[0].keys())
            tidl_onnx_trace_mapping_keys = list(tidl_onnx_trace_mapping.keys())
            for layer_idx in range(len(layers_diff)):
                tidl_data_id = tidl_onnx_trace_mapping_keys[layer_idx]
                tidl_onnx_trace_mapping_entry = tidl_onnx_trace_mapping[tidl_data_id]
                frame_worksheet.write(layer_idx + 1, 0, tidl_onnx_trace_mapping_entry[2])
                frame_worksheet.write(layer_idx + 1, 1, tidl_onnx_trace_mapping_entry[3])
                frame_worksheet.write_row(layer_idx + 1, 2, layers_diff[layer_idx].values())
            #
        #
        analyze_xlsx.close()
        print(f"INFO: Data successfully written to {analyze_xlsx_path}")

    def get_traces(self, onnx_trace_folder, tidl_trace_folder, layer_info_path):
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

    def get_layers_diff(self, onnx_trace_folder, tidl_trace_folder, layer_info_path, tidl_onnx_trace_mapping):
        """
        Generate error summary for the entire network
        """
        dropdown_list = list(tidl_onnx_trace_mapping.keys())
        numLayers = len(dropdown_list)
        mae_dict = {}
        max_dict = {}
        mae_abs_dict = {}
        combined_diff = []
        for idx in range(numLayers):
            layer_idx = dropdown_list[idx]
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

            delta = []
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

            mae_dict[layer_idx] = mae
            max_dict[layer_idx] = max
            mae_abs_dict[layer_idx] = max_abs
            combined_diff.append({'MeanAbsRelE': float(mae), 'MaxAbsE': float(max), 'MeanMaxAbsE': float(max_abs)})
        #
        return combined_diff

