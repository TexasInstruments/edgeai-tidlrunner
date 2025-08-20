# Copyright (c) 2018-2021, Texas Instruments
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
import csv
import time
import re

from .. import options


class BaseRuntimeWrapper:
    def __init__(self, **kwargs):
        if not isinstance(kwargs['runtime_options'], options.RuntimeOptions):
            kwargs['runtime_options'] = options.RuntimeOptions(**kwargs['runtime_options']).get_runtime_options()
        #
        self._start_import_done = False
        self._start_inference_done = False
        if not hasattr(self, 'kwargs'):
            self.kwargs = kwargs
        else:
            self.kwargs.update(kwargs)

        self.infer_stats_dict = {}
        self._infer_stats_sum = {}

    def get_runtime_options(self):
        return self.kwargs['runtime_options']

    def get_input_output_details(self):
        if self._start_import_done or self._start_inference_done:
            return self.kwargs['input_details'], self.kwargs['output_details']
        else:
            return None, None

    def _get_input_details_onnx(self, interpreter, input_details=None):
        if input_details is None:
            properties = {'name':'name', 'shape':'shape', 'type':'type'}
            input_details = []
            model_input_details = interpreter.get_inputs()
            for inp_d in model_input_details:
                inp_dict = {}
                for p_key, p_val in properties.items():
                    inp_d_val = getattr(inp_d, p_key)
                    if p_key == 'type':
                        inp_d_val = str(inp_d_val)
                    #
                    if p_key == 'shape':
                        inp_d_val = list(inp_d_val)
                    #
                    inp_dict[p_val] = inp_d_val
                #
                input_details.append(inp_dict)
            #
        #
        return input_details

    def _get_output_details_onnx(self, interpreter, output_details=None):
        if output_details is None:
            properties = {'name':'name', 'shape':'shape', 'type':'type'}
            output_details = []
            model_output_details = interpreter.get_outputs()
            for oup_d in model_output_details:
                oup_dict = {}
                for p_key, p_val in properties.items():
                    oup_d_val = getattr(oup_d, p_key)
                    if p_key == 'type':
                        oup_d_val = str(oup_d_val)
                    #
                    if p_key == 'shape':
                        oup_d_val = list(oup_d_val)
                    #
                    oup_dict[p_val] = oup_d_val
                #
                output_details.append(oup_dict)
            #
        #
        return output_details

    def _get_input_details_tflite(self, interpreter, input_details=None):
        if input_details is None:
            properties = {'name':'name', 'shape':'shape', 'dtype':'type', 'index':'index'}
            input_details = []
            model_input_details = interpreter.get_input_details()
            for inp_d in model_input_details:
                inp_dict = {}
                for p_key, p_val in properties.items():
                    inp_d_val = inp_d[p_key]
                    if p_key == 'dtype':
                        inp_d_val = str(inp_d_val)
                    #
                    if p_key == 'shape':
                        inp_d_val = [int(val) for val in inp_d_val]
                    #
                    inp_dict[p_val] = inp_d_val
                #
                input_details.append(inp_dict)
            #
        #
        return input_details

    def _get_output_details_tflite(self, interpreter, output_details=None):
        if output_details is None:
            properties = {'name':'name', 'shape':'shape', 'dtype':'type', 'index':'index'}
            output_details = []
            model_output_details = interpreter.get_output_details()
            for oup_d in model_output_details:
                oup_dict = {}
                for p_key, p_val in properties.items():
                    oup_d_val = oup_d[p_key]
                    if p_key == 'dtype':
                        oup_d_val = str(oup_d_val)
                    #
                    if p_key == 'shape':
                        oup_d_val = [int(val) for val in oup_d_val]
                    #
                    oup_dict[p_val] = oup_d_val
                #
                output_details.append(oup_dict)
            #
        #
        return output_details

    def _pre_inference(self, *args, **kwargs):
        self._infer_stats_sum['invoke_start_time'] = time.time()

    def _post_inference(self, *args, **kwargs):
        delta_time = time.time() - self._infer_stats_sum['invoke_start_time']
        self._infer_stats_sum['infer_time_invoke_ms'] = self._infer_stats_sum.get('infer_time_invoke_ms', 0) + delta_time * options.presets.MILLI_CONST
        self._infer_stats()
        
    def _infer_stats(self):
        stats_dict = self._infer_frame_stats()
        self.infer_stats_dict['num_frames'] = self.infer_stats_dict.get('num_frames', 0) + 1
        # compute and populate final stats so that it can be used in result
        self._infer_stats_sum['num_subgraphs'] = self._infer_stats_sum.get('num_subgraphs', 0) + stats_dict.get('num_subgraphs', 0)
        if self.kwargs['target_machine'] == options.presets.TargetMachineType.TARGET_MACHINE_EVM:
            # self._infer_stats_sum['infer_time_invoke_ms'] =  ?? # This is handled in run_inference
            self._infer_stats_sum['infer_time_core_ms'] = self._infer_stats_sum.get('infer_time_core_ms', 0) + stats_dict.get('core_time', 0) * options.presets.MILLI_CONST
            self._infer_stats_sum['infer_time_subgraph_ms'] = self._infer_stats_sum.get('infer_time_subgraph_ms', 0) + stats_dict.get('subgraph_time', 0) * options.presets.MILLI_CONST
            self._infer_stats_sum['ddr_transfer_mb'] = self._infer_stats_sum.get('ddr_transfer_mb', 0) + (stats_dict.get('read_total', 0) + stats_dict.get('write_total', 0)) / options.presets.MEGA_CONST
        #
        for k, v in self._infer_stats_sum.items():
            self.infer_stats_dict[k] = self._infer_stats_sum[k] / self.infer_stats_dict['num_frames']
        #
        if 'perfsim_time' in stats_dict:
            self.infer_stats_dict.update({'perfsim_time_ms': stats_dict['perfsim_time'] * options.presets.MILLI_CONST})
        #
        if 'perfsim_ddr_transfer' in stats_dict:
            self.infer_stats_dict.update({'perfsim_ddr_transfer_mb': stats_dict['perfsim_ddr_transfer'] / options.presets.MEGA_CONST})
        #
        if 'perfsim_macs' in stats_dict:
            self.infer_stats_dict.update({'perfsim_gmacs': stats_dict['perfsim_macs'] / options.presets.GIGA_CONST})
        #

    def _infer_frame_stats(self):
        stats_dict = dict()
        stats_dict['num_subgraphs'] = None
        stats_dict['num_frames'] = None

        stats_dict['total_time'] = None
        stats_dict['core_time'] = None
        stats_dict['subgraph_time'] = None
        stats_dict['read_total'] = None
        stats_dict['write_total'] = None

        stats_dict['perfsim_macs'] = None
        stats_dict['perfsim_time'] = None
        stats_dict['perfsim_ddr_transfer'] = None

        if hasattr(self.interpreter, 'get_TI_benchmark_data'):
            stats_dict = self._tidl_infer_stats()
            stats_dict['num_frames'] = 1
        #
        try:
            perfsim_stats = self._infer_perfsim_stats()
            stats_dict.update(perfsim_stats)
        except Exception as e:
            print(f'WARNING: perfsim stats could not be obtained: {e}')
        #
        return stats_dict
    

    def _tidl_infer_stats(self):
        benchmark_dict = self.interpreter.get_TI_benchmark_data()
        subgraph_time = copy_time = 0
        cp_in_time = cp_out_time = 0
        subgraphIds = []
        for stat in benchmark_dict.keys():
            if 'proc_start' in stat:
                if self.kwargs['name'] == 'onnxrt':
                    value = stat.split("ts:subgraph_")
                    value = value[1].split("_proc_start")
                    subgraphIds.append(value[0])
                else:
                    subgraphIds.append(int(re.sub("[^0-9]", "", stat)))
                #
            #
        #
        for i in range(len(subgraphIds)):
            subgraph_time += benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_proc_end'] - benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_proc_start']
            cp_in_time += benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_copy_in_end'] - benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_copy_in_start']
            cp_out_time += benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_copy_out_end'] - benchmark_dict['ts:subgraph_'+str(subgraphIds[i])+'_copy_out_start']
        #
        copy_time = cp_in_time + cp_out_time
        copy_time = copy_time if len(subgraphIds) == 1 else 0
        total_time = benchmark_dict['ts:run_end'] - benchmark_dict['ts:run_start']
        write_total = benchmark_dict['ddr:read_end'] - benchmark_dict['ddr:read_start']
        read_total = benchmark_dict['ddr:write_end'] - benchmark_dict['ddr:write_start']
        # change units
        total_time = total_time / options.presets.DSP_FREQ
        copy_time = copy_time / options.presets.DSP_FREQ
        subgraph_time = subgraph_time / options.presets.DSP_FREQ
        write_total = write_total
        read_total = read_total
        # core time excluding the copy overhead
        core_time = total_time - copy_time
        stats = {
            'num_subgraphs': len(subgraphIds),
        }
        if self.kwargs['target_machine'] == options.presets.TargetMachineType.TARGET_MACHINE_EVM:
            stats.update({
                'total_time': total_time,
                'core_time': core_time,
                'subgraph_time': subgraph_time,
                'write_total': write_total,
                'read_total': read_total
            })
        #
        return stats

    def _infer_perfsim_stats(self):
        artifacts_folder = self.kwargs['artifacts_folder']
        subgraph_root = os.path.join(artifacts_folder, 'tempDir') \
            if os.path.isdir(os.path.join(artifacts_folder, 'tempDir')) else artifacts_folder
        perfsim_folders = [os.path.join(subgraph_root, d) for d in os.listdir(subgraph_root)]
        perfsim_folders = [d for d in perfsim_folders if os.path.isdir(d)]
        perfsim_dict = {}
        for perfsim_folder in perfsim_folders:
            subgraph_stats = self._subgraph_perfsim_stats(perfsim_folder)
            for k, v in subgraph_stats.items():
                if k in perfsim_dict:
                    perfsim_dict[k] += v
                else:
                    perfsim_dict[k] = v
                #
            #
        #
        return perfsim_dict
    #

    def _subgraph_perfsim_stats(self, perfsim_folder):
        perfsim_files = os.listdir(perfsim_folder)
        if len(perfsim_files) == 0:
            return None
        #
        subgraph_perfsim_dict = {}
        # get the gmac number from netLog file
        netlog_file = perfsim_folder + '.bin_netLog.txt'
        with open(netlog_file) as netlog_fp:
            netlog_reader = csv.reader(netlog_fp)
            netlog_data = [data for data in netlog_reader]
            perfsim_macs = [row for row in netlog_data if 'total giga macs' in row[0].lower()][0][0]
            perfsim_macs = float(perfsim_macs.split(':')[1])
            # change units - convert gmacs to macs
            perfsim_macs = perfsim_macs * options.presets.GIGA_CONST
            subgraph_perfsim_dict.update({'perfsim_macs': perfsim_macs})
        #
        # get the perfsim cycles
        graph_name = os.path.basename(perfsim_folder)
        perfsim_csv = [p for p in perfsim_files if graph_name in p and os.path.splitext(p)[1] == '.csv' and not p.startswith('.')][0]
        perfsim_csv = os.path.join(perfsim_folder, perfsim_csv)
        with open(perfsim_csv) as perfsim_fp:
            perfsim_reader = csv.reader(perfsim_fp)
            perfsim_data = [data for data in perfsim_reader]

            # perfsim time - read from file
            perfsim_time = [row for row in perfsim_data if 'total network time (us)' in row[0].lower()][0][0]
            perfsim_time = float(perfsim_time.split('=')[1])
            # change units - convert from ultrasec to seconds
            perfsim_time = perfsim_time / options.presets.ULTRA_CONST
            subgraph_perfsim_dict.update({'perfsim_time': perfsim_time})

            # perfsim cycles - read from file
            # perfsim_cycles = [row for row in perfsim_data if 'total network cycles (mega)' in row[0].lower()][0][0]
            # perfsim_cycles = float(perfsim_cycles.split('=')[1])
            # change units - convert from mega cycles to cycles
            # perfsim_cycles = perfsim_cycles * options.presets.MEGA_CONST
            # subgraph_perfsim_dict.update({'perfsim_cycles': perfsim_cycles})

            # perfsim ddr transfer - read from file
            perfsim_ddr_transfer = [row for row in perfsim_data if 'ddr bw (mega bytes) : total' in row[0].lower()][0][0]
            perfsim_ddr_transfer = float(perfsim_ddr_transfer.split('=')[1])
            # change units - convert from megabytes to bytes
            perfsim_ddr_transfer = perfsim_ddr_transfer * options.presets.MEGA_CONST
            subgraph_perfsim_dict.update({'perfsim_ddr_transfer': perfsim_ddr_transfer})
        #
        return subgraph_perfsim_dict
