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
import glob
import math
import yaml
import xlsxwriter

from ...common import utils
from ...common import bases
from ..settings.settings_default import SETTINGS_DEFAULT, COPY_SETTINGS_DEFAULT
from .common_ import compile_base
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

    def _prepare_model(self):
        import onnx
        super()._prepare_model()
        if self.kwargs['common.analyze_level'] >= 2:
            # Load the original ONNX model and add intermediate outputs
            model_path = self.model_path
            onnx_model = onnx.load(model_path)

            # using native onnx
            # intermediate_layer_value_info = onnx.helper.ValueInfoProto()
            # intermediate_layer_value_info.name = ''
            # for i in range(len(onnx_model.graph.node)):
            #     for j in range(len(onnx_model.graph.node[i].output)):
            #         intermediate_layer_value_info.name = onnx_model.graph.node[i].output[j]
            #         onnx_model.graph.output.append(intermediate_layer_value_info)
            #     #
            # #

            # using onnx graph surgeon
            import onnx_graphsurgeon as gs
            graph = gs.import_onnx(onnx_model)
            for node in list(graph.nodes):
                if node.op == 'DequantizeLinear' and isinstance(node.inputs[0], gs.Constant):
                    # weights need not be output
                    # there is an issue in QDQ models if this is done.
                    continue
                for out in node.outputs:
                    if out not in graph.outputs:
                        graph.outputs.append(out)
            
            onnx_model = gs.export_onnx(graph)
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


class CompileAnalyzeTIDL32(compile.CompileModel):
    ARGS_DICT = SETTINGS_DEFAULT['analyze']
    COPY_ARGS = COPY_SETTINGS_DEFAULT['analyze']

    def __init__(self, **kwargs):
        kargs_copy = copy.deepcopy(kwargs)

        tensor_bits = kargs_copy.get('session.runtime_options.tensor_bits', '') or 'x'
        tensor_bits_str = f'{str(tensor_bits)}' if tensor_bits else ''
        tensor_bits_slash = f'{str(tensor_bits)}' + os.sep if tensor_bits else ''
        run_dir = kargs_copy['session.run_dir']
        run_dir = run_dir.replace('{tensor_bits}/', tensor_bits_slash)
        run_dir = run_dir.replace('{tensor_bits}', tensor_bits_str)
        run_dir = os.path.join(kargs_copy['session.run_dir'], 'tidl32')

        work_path = kargs_copy['common.work_path']
        work_path = work_path.replace('{tensor_bits}/', tensor_bits_slash)
        work_path = work_path.replace('{tensor_bits}', tensor_bits_str)

        kargs_copy['common.work_path'] = work_path
        kargs_copy['session.run_dir'] = run_dir
        kargs_copy['session.runtime_options.tensor_bits'] = 32
        kargs_copy['common.postprocess_enable'] = False            
        super().__init__(**kargs_copy)

    def _run(self):
        super()._run()


class InferAnalyzeTIDL32(infer.InferModel):
    ARGS_DICT=SETTINGS_DEFAULT['analyze']
    COPY_ARGS=COPY_SETTINGS_DEFAULT['analyze']

    def __init__(self, **kwargs):
        kargs_copy = copy.deepcopy(kwargs)

        tensor_bits = kargs_copy.get('session.runtime_options.tensor_bits', '') or 'x'
        tensor_bits_str = f'{str(tensor_bits)}' if tensor_bits else ''
        tensor_bits_slash = f'{str(tensor_bits)}' + os.sep if tensor_bits else ''
        run_dir = kargs_copy['session.run_dir']
        run_dir = run_dir.replace('{tensor_bits}/', tensor_bits_slash)
        run_dir = run_dir.replace('{tensor_bits}', tensor_bits_str)
        run_dir = os.path.join(kargs_copy['session.run_dir'], 'tidl32')

        work_path = kargs_copy['common.work_path']
        work_path = work_path.replace('{tensor_bits}/', tensor_bits_slash)
        work_path = work_path.replace('{tensor_bits}', tensor_bits_str)

        kargs_copy['common.work_path'] = work_path
        kargs_copy['session.run_dir'] = run_dir
        kargs_copy['session.runtime_options.debug_level'] = 0 if kargs_copy['common.analyze_level'] == 0 else 4
        kargs_copy['session.runtime_options.tensor_bits'] = 32
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
    

class CompileAnalyzeTIDL(compile.CompileModel):
    ARGS_DICT = SETTINGS_DEFAULT['analyze']
    COPY_ARGS = COPY_SETTINGS_DEFAULT['analyze']

    def __init__(self, **kwargs):
        kargs_copy = copy.deepcopy(kwargs)
        # kargs_copy['session.run_dir'] = os.path.join(kargs_copy['session.run_dir'], 'tidl')
        kargs_copy['common.postprocess_enable'] = False            
        super().__init__(**kargs_copy)

    def _run(self):
        super()._run()


class InferAnalyzeTIDL(infer.InferModel):
    ARGS_DICT=SETTINGS_DEFAULT['analyze']
    COPY_ARGS=COPY_SETTINGS_DEFAULT['analyze']

    def __init__(self, **kwargs):
        kargs_copy = copy.deepcopy(kwargs)
        # kargs_copy['session.run_dir'] = os.path.join(kargs_copy['session.run_dir'], 'tidl')
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
        print('INFO: Creating symlink')
        cwd = os.getcwd()
        os.chdir(self.run_dir)
        os.symlink(f"../{os.path.basename(self.run_dir)}", 'tidl')
        os.chdir(cwd)

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
        comparison_patterns = [('notidl','tidl'), ('notidl','tidl32'), ('tidl32','tidl')]
        comparison_run_dir = dict()
        comparison_outputs_dir = dict()
        comparison_traces_dir = dict()
        comparison_outputs_dirs = dict()
        comparison_traces_dirs = dict()
        all_pattern_keys = [item for sublist in comparison_patterns for item in sublist]
        for k in all_pattern_keys:
            comparison_run_dir[k] = os.path.join(self.run_dir, k)
            comparison_outputs_dir[k] = os.path.join(comparison_run_dir[k], 'outputs_')
            traces_name = 'outputs_' if k == 'notidl' else 'traces_'
            comparison_traces_dir[k] = os.path.join(comparison_run_dir[k], traces_name)

            comparison_outputs_dirs[k] = [os.path.join(comparison_outputs_dir[k],f) for f in os.listdir(comparison_outputs_dir[k])]
            comparison_traces_dirs[k] = [os.path.join(comparison_traces_dir[k],f) for f in os.listdir(comparison_traces_dir[k])]
            comparison_outputs_dirs[k].sort()
            comparison_traces_dirs[k].sort()
        
        analyze_xlsx_path = os.path.join(self.run_dir, "analyze.xlsx")
        analyze_xlsx =  xlsxwriter.Workbook(analyze_xlsx_path, options=dict(nan_inf_to_errors=True))

        column_start_idx = 0
        frame_worksheet = analyze_xlsx.add_worksheet('diff_outputs')
        for k, v in comparison_patterns:
            column_start_idx = self._analyze_model_outputs(comparison_outputs_dirs[k], comparison_outputs_dirs[v], frame_worksheet, column_start_idx, k, v)

        if self.kwargs['common.analyze_level'] >= 2:
            for k, v in comparison_patterns:
                num_traces = len(comparison_traces_dirs[v])
                layer_info_dir = os.path.join(comparison_run_dir[v], 'artifacts', 'tempDir')
                layer_info_files = [os.path.join(layer_info_dir,f) for f in os.listdir(layer_info_dir) if f.endswith('layer_info.txt')]
                tidl_onnx_trace_mapping_dict = dict()
                for frame_idx in range(num_traces):
                    tidl_onnx_trace_mapping_dict[frame_idx] = dict()
                    for subgraph_idx, layer_info_path in enumerate(layer_info_files):
                        tidl_onnx_trace_mapping = self._get_traces(k, v, subgraph_idx, comparison_traces_dirs[k][frame_idx], comparison_traces_dirs[v][frame_idx], layer_info_path)
                        tidl_onnx_trace_mapping_dict[frame_idx].update(tidl_onnx_trace_mapping)
                    #
                #
                self._analyze_model_layers(k, v, tidl_onnx_trace_mapping_dict, analyze_xlsx)

                tidl_onnx_trace_mapping_obj = dict()
                for frame_idx in range(num_traces):
                        tidl_onnx_trace_mapping = tidl_onnx_trace_mapping_dict[frame_idx]
                        tidl_onnx_trace_mapping_obj[frame_idx] = {lk:[lventry for lventry in lv if not isinstance(lventry, np.ndarray)] for lk, lv in tidl_onnx_trace_mapping.items()}

                analyze_layer_mapping_file = os.path.join(self.run_dir, f'layer_output_mapping_{k}_{v}.yaml')
                with open(analyze_layer_mapping_file, 'w') as fp:
                    yaml.safe_dump(tidl_onnx_trace_mapping_obj, fp)
                #
            #
        #

        analyze_xlsx.close()
        print(f"INFO: Data successfully written to {analyze_xlsx_path}")

    def _analyze_model_outputs(self, notidl_outputs_dirs, tidl_outputs_dirs, frame_worksheet, column_start_idx=0, refname='', tidlname=''):
        num_traces = len(tidl_outputs_dirs)
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
            
            for output_idx in range(num_outputs):
                tidl_output_file = tidl_output_files[output_idx]
                golden_output_file = os.path.join(golden_output_dir, os.path.basename(tidl_output_file))
                goldenBuffer = np.fromfile(golden_output_file, dtype=np.float32).flatten()
                tidlBuffer = np.fromfile(tidl_output_file, dtype=np.float32).flatten()
                diff_dict = self._get_tensor_diff(goldenBuffer, tidlBuffer)

                frame_worksheet.write_row(0, column_start_idx+0, ['FrameNum', f'RefLayer({refname})', f'TIDLONNXLayer({tidlname})'])
                frame_worksheet.write_row(0, column_start_idx+3, diff_dict.keys())

                frame_worksheet.write(frame_idx + 1, column_start_idx+0, str(frame_idx))
                frame_worksheet.write(frame_idx + 1, column_start_idx+1, os.path.splitext(os.path.basename(golden_output_file))[0])
                frame_worksheet.write(frame_idx + 1, column_start_idx+2, os.path.splitext(os.path.basename(tidl_output_file))[0])
                frame_worksheet.write_row(frame_idx + 1, column_start_idx+3, diff_dict.values())
                column_start_idx += (3 + len(diff_dict))
            #
        #
        return column_start_idx

    def _analyze_model_layers(self, refname, tidlname, tidl_onnx_trace_mapping_dict, analyze_xlsx):
        zero_if_nan = lambda v: 0.0 if (math.isnan(v) or v is None) else v
        num_frames = len(tidl_onnx_trace_mapping_dict)
        layers_diff_dict = dict()
        for frame_idx in range(num_frames):
            tidl_onnx_trace_mapping = tidl_onnx_trace_mapping_dict[frame_idx]
            layers_diff_dict[frame_idx] = layers_diff_dict.get(frame_idx, dict())
            layers_diff = self._get_layers_diff(tidl_onnx_trace_mapping)
            layers_diff_dict[frame_idx] = layers_diff

            # find median
            layers_diff_keys = layers_diff[0].keys()
            layers_diff_transposed = {key: [] for key in layers_diff_keys}
            for layer_idx in range(len(layers_diff)):
                for key, v in layers_diff[layer_idx].items():
                    v = zero_if_nan(v)
                    layers_diff[layer_idx][key] = v
                    layers_diff_transposed[key].append(v)
                #
            #
            layers_diff_median = {key: np.median(np.array(vs)) for key, vs in layers_diff_transposed.items()}
        #

        # add percentage values as well
        for frame_idx in range(num_frames):
            layers_diff = layers_diff_dict[frame_idx]
            for layer_idx, layer_diff in enumerate(layers_diff):
                for key in layers_diff_median.keys():
                    median_val = layers_diff_median[key]
                    eps = 1e-6 if median_val == 0 else 0
                    layer_diff[key + '_Median%'] = abs((layer_diff[key] + eps) * 100 / (median_val + eps))
                #
            # 
        #

        for frame_idx in range(num_frames):
            frame_worksheet = analyze_xlsx.add_worksheet(f'diff_{refname}_{tidlname}_' + str(frame_idx))
            row_idx = 1
            tidl_onnx_trace_mapping = tidl_onnx_trace_mapping_dict[frame_idx]
            layers_diff = layers_diff_dict[frame_idx]
            frame_worksheet.write_row(0, 0, ['Subgraph', 'SerialNum', 'ONNXLayer', 'TIDLDataID'])
            frame_worksheet.write_row(0, 4, layers_diff[0].keys())
            tidl_onnx_trace_mapping_keys = list(tidl_onnx_trace_mapping.keys())
            subgraph_idx = 0 # to be corrected
            for layer_idx in range(len(layers_diff)):
                tidl_data_id = tidl_onnx_trace_mapping_keys[layer_idx]
                tidl_onnx_trace_mapping_entry = tidl_onnx_trace_mapping[tidl_data_id]
                layer_id_mapping_list = [str(subgraph_idx), str(layer_idx), tidl_onnx_trace_mapping_entry[2], tidl_onnx_trace_mapping_entry[3]]
                frame_worksheet.write_row(row_idx+layer_idx, 0, layer_id_mapping_list)
                frame_worksheet.write_row(row_idx+layer_idx, 4, layers_diff[layer_idx].values())
            #
            row_idx += len(layers_diff)
        #

    def _get_traces(self, refname, tidlname, subgraph_idx, onnx_trace_folder, tidl_trace_folder, layer_info_path):
        """
        Computes the mapping between TIDL and ONNX traces
        """
        entries = [line.strip().split(" ") for line in open(layer_info_path)]
        tidl_onnx_trace_mapping = {}
        for entry in entries:
            if entry[0] != entry[1]:
                continue
            tidl_data_id = entry[0]

            _tidl_data_id = tidl_data_id
            while len(_tidl_data_id) < 4:
                _tidl_data_id = "0" + _tidl_data_id
            #

            onnx_layer_name = entry[-1]
            if refname == 'notidl':
                onnx_layer_id = onnx_layer_name.replace("/", "_")
                onnx_entries = os.listdir(onnx_trace_folder)
                onnx_trace_path = list(filter(lambda path: onnx_layer_id in path, onnx_entries))
                onnx_trace_path = os.path.join(onnx_trace_folder, onnx_trace_path[0]) if len(onnx_trace_path)>0 else None
            else:
                _onnx_trace_path = os.path.join(onnx_trace_folder, f"tidl_trace_subgraph_{subgraph_idx}_{_tidl_data_id}*_float.bin")
                onnx_trace_path = glob.glob(_onnx_trace_path)
                onnx_trace_path = onnx_trace_path[0] if len(onnx_trace_path)>0 else None
            #

            _tidl_trace_path = os.path.join(tidl_trace_folder, f"tidl_trace_subgraph_{subgraph_idx}_{_tidl_data_id}*_float.bin")
            tidl_trace_path = glob.glob(_tidl_trace_path)
            tidl_trace_path = tidl_trace_path[0] if len(tidl_trace_path)>0 else None

            if onnx_trace_path and tidl_trace_path:
                tidl_onnx_trace_mapping[tidl_data_id] = [
                    onnx_trace_path,
                    tidl_trace_path,
                    onnx_layer_name,
                    tidl_data_id
                ]
            # else:
            #     print(f"WARNING: Traces Not found for outdataId: {_tidl_data_id}")
            # #
        #
        return tidl_onnx_trace_mapping

    def _get_tensor_diff(self, goldenBuffer, tidlBuffer):
        try:
            delta = goldenBuffer - tidlBuffer
        except:
            delta = np.zeros_like(tidlBuffer)
        #
        eps = 1e-6
        abs_delta = np.abs(delta)
        max_delta = np.max(abs_delta)
        mean_delta = np.mean(abs_delta)
        median_delta = np.median(abs_delta)
        mean_scale = np.mean(np.abs(goldenBuffer))
        mean_abs_rel_diff = mean_delta / mean_scale if mean_scale > eps else (mean_delta+eps) / (mean_scale+eps)
        diff_dict = {'MeanAbsRelDiff': float(mean_abs_rel_diff), 'MeanAbsDiff': float(mean_delta), 'MedianAbsDiff': float(median_delta), 'MaxAbsDiff': float(max_delta)}
        return diff_dict

    def _get_layers_diff(self, tidl_onnx_trace_mapping):
        """
        Generate error summary for the entire network
        """
        traces_files = list(tidl_onnx_trace_mapping.keys())
        num_traces_files = len(traces_files)
        combined_diff = []
        for idx in range(num_traces_files):
            layer_idx = traces_files[idx]
            tidl = tidl_onnx_trace_mapping[layer_idx][1]
            golden = tidl_onnx_trace_mapping[layer_idx][0]
            goldenBuffer = np.fromfile(golden, dtype=np.float32).flatten()
            tidlBuffer = np.fromfile(tidl, dtype=np.float32).flatten()
            tidl_onnx_trace_mapping[layer_idx].extend([goldenBuffer, tidlBuffer])
            diff_dict = self._get_tensor_diff(goldenBuffer, tidlBuffer)
            combined_diff.append(diff_dict)
        #
        return combined_diff
