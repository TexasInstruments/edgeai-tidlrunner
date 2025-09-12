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

from edgeai_tidlrunner import rtwrapper, runner


class MainRunner(runner.bases.PipelineBase):
    ARGS_DICT = runner.bases.SETTING_PIPELINE_RUNNER_ARGS_DICT
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

    @classmethod
    def _get_target_module(cls, target_module_name):
        target_module = getattr(runner.modules, target_module_name)
        return target_module

    def run(self, command):
        run_dict = runner._create_run_dict(command, argparse=True, **self.kwargs)
        return runner._run(run_dict)

    @classmethod
    def main(cls):
        sys.argv[0] = os.environ.get('RUNNER_INVOKE_NAME', sys.argv[0])

        if len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1] in ('help', 'h', '--help', '-h')):
            print('============================================================')
            sys.argv = [sys.argv[0]]
            parser = cls.get_arg_parser()
            command_args, rest_args = parser.parse_known_args()
            kwargs = vars(command_args)
            target_module = cls._get_target_module(kwargs['common.target_module'])
            command_choices = target_module.get_command_choices()
            parser.print_help()
            print('============================================================')
            print('for detailed help, use the following options:')
            for command_choice in command_choices:
                print(f'{sys.argv[0]} {command_choice} --help')
            #
        elif (len(sys.argv) == 2 and sys.argv[1] != 'report') or \
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
            target_module = cls._get_target_module(kwargs['common.target_module'])
            command_choices = target_module.get_command_choices()
            command = sys.argv[1]
            assert command in command_choices, RuntimeError(
                f'ERROR: invalid command: {command} - must be one of {command_choices}')
            sys.argv = [sys.argv[0]] + ['--help']
            main_runner = cls()
            main_runner.run(command)
        else:
            parser = cls.get_arg_parser()
            command_args, rest_args = parser.parse_known_args()
            kwargs = vars(command_args)
            target_module = cls._get_target_module(kwargs['common.target_module'])
            command_choices = target_module.get_command_choices()
            command = sys.argv[1].lower().replace(' ', '')
            assert command in command_choices, RuntimeError(
                f'ERROR: invalid command: {command} - must be one of {command_choices}')
            sys.argv = [sys.argv[0]] + sys.argv[2:]
            main_runner = cls(**kwargs)
            main_runner.run(command)


def start():
    print(f'INFO: running - {sys.argv}')
    MainRunner.main()


if __name__ == "__main__":
    print(f'INFO: running {__file__} __main__')
    print(f'INFO: This doesn not setup environment variables. Make sure TIDL_TOOLS_PATH and LD_LIBRARY_PATH are set properly.')
    print(f'INFO: OR run tidlrunnercli which is setup to call main:main_auto() in pyproject.toml')    
    start()
