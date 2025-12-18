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


import sys
import os
import copy
import argparse
import ast
import yaml
import functools
import subprocess

import edgeai_tidlrunner
from edgeai_tidlrunner import rtwrapper, runner, interfaces


COMMAND_PIPELINES = edgeai_tidlrunner.get_command_pipelines()
SPECIAL_PIPELINE_NAMES = ('report',)


class StartRunner(runner.common.bases.PipelineBase):
    ARGS_DICT = runner.common.bases.SETTING_PIPELINE_RUNNER_ARGS_DICT
    COPY_ARGS = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._set_parallel_devices()

    def _set_parallel_devices(self):
        try:
            if self.kwargs['session.target_machine'] == 'pc' and self.kwargs['common.parallel_devices'] is None:
                print(f"INFO: model compilation in PC can use CUDA gpus (if it is available) - setup using setup_pc_gpu.sh")
                num_cuda_gpus = self._get_num_cuda_gpus()
                print(f'INFO: setting parallel_devices to the number of CUDA gpus found: {num_cuda_gpus}')
                sys.argv += [ f'--parallel_devices={num_cuda_gpus}' ]
            #
        except:
            print("\nINFO: could not find CUDA gpus - parallel_devices will not be used.")
        #

    def _get_num_cuda_gpus(self):
        nvidia_smi_command = 'nvidia-smi --list-gpus | wc -l'
        proc = subprocess.Popen([nvidia_smi_command], stdout=subprocess.PIPE, shell=True)
        out_ret, err_ret = proc.communicate()
        num_cuda_gpus = int(out_ret)
        return num_cuda_gpus

    def run(self, command):
        if command not in SPECIAL_PIPELINE_NAMES:
            run_dict = interfaces._create_run_dict(command, argparse=True, **self.kwargs)
        else:
            run_dict = interfaces._create_run_dict(command, argparse=True, model_id=command+'_model', **self.kwargs)
        #
        return interfaces._run(run_dict)

    @classmethod
    def main(cls, **kwargs):
        # add args and continue with normal execution
        sys.argv[0] = os.environ.get('RUNNER_INVOKE_NAME', sys.argv[0])
        for k, v in kwargs.items():
            has_arg = any([f'--{k}' in arg for arg in sys.argv])
            if not has_arg:
                sys.argv.append(f'--{k}={v}')
            #
        #

        if len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1] in ('help', 'h', '--help', '-h')):
            print('============================================================')
            sys.argv = [sys.argv[0]]
            parser = cls.get_arg_parser()
            command_args, rest_args = parser.parse_known_args()
            kwargs = vars(command_args)
            command_choices = runner.get_command_pipelines(**kwargs)
            parser.print_help()
            print('============================================================')
            print('for detailed help, use the following options:')
            for command_choice in command_choices:
                print(f'{sys.argv[0]} {command_choice} --help')
            #
        elif (len(sys.argv) == 2 and sys.argv[1] not in SPECIAL_PIPELINE_NAMES) or \
            (len(sys.argv) > 2 and sys.argv[2] in ('help', 'h', '--help', '-h')):
            if len(sys.argv) == 2:
                if sys.argv[1].startswith('-'):
                    sys.argv[1] = '--help'
                #
            #
            sys.argv = [sys.argv[0], sys.argv[1]]
            parser = cls.get_arg_parser()
            command_args, rest_args = parser.parse_known_args()
            kwargs = vars(command_args)
            command_choices = runner.get_command_pipelines(**kwargs)
            command = sys.argv[1]
            # assert command in command_choices, RuntimeError(
            #     f'ERROR: invalid command: {command} - must be one of {command_choices}')
            sys.argv = [sys.argv[0]] + ['--help']
            main_runner = cls()
            main_runner.run(command)
        else:
            parser = cls.get_arg_parser()
            command_args, rest_args = parser.parse_known_args()
            kwargs = vars(command_args)
            command_choices = runner.get_command_pipelines(**kwargs)
            command = sys.argv[1].lower().replace(' ', '')
            # assert command in command_choices, RuntimeError(
            #     f'ERROR: invalid command: {command} - must be one of {command_choices}')
            sys.argv = [sys.argv[0]] + sys.argv[2:]
            main_runner = cls(**kwargs)
            main_runner.run(command)


def start():
    print(f'INFO: running - {sys.argv}')
    StartRunner.main()


def start_with_proper_environment(**kwargs):
    print(f'INFO: running - {sys.argv}')
    target_machine = kwargs['target_machine']
    is_tidl_tools_path_defined = (os.environ.get('TIDL_TOOLS_PATH', None) is not None and os.environ.get('LD_LIBRARY_PATH', None) is not None)

    if target_machine == rtwrapper.core.presets.TargetMachineType.TARGET_MACHINE_PC_EMULATION and (not is_tidl_tools_path_defined):
        print("INFO: TIDL_TOOLS_PATH or LD_LIBRARY_PATH is not set, restarting with proper environment...")
        command_args, rest_args = StartRunner.get_arg_parser().parse_known_args()  
        command_kwargs = vars(command_args)
        start_kwargs = kwargs.copy()
        cmd_keys_mapping = {
            'session.target_device': 'target_device',
            'session.target_machine': 'target_machine',
        }
        for cmd_key in cmd_keys_mapping:
            if cmd_key in command_kwargs:
                kwarg_key = cmd_keys_mapping[cmd_key]
                start_kwargs[kwarg_key] = command_kwargs[cmd_key]
            #
        #
        rtwrapper.restart_with_proper_environment(**start_kwargs)
    else:
        # TIDL_TOOLS_PATH is not needed in EVM, but just set it to empty to pass through checks for it
        os.environ['TIDL_TOOLS_PATH'] = os.environ.get('TIDL_TOOLS_PATH', '')
        StartRunner.main(**kwargs)


if __name__ == "__main__":
    print(f'INFO: running {__file__} __main__')
    print(f'INFO: OR run tidlrunner-cli which is setup to call main:main() in pyproject.toml')    
    start_with_proper_environment()
