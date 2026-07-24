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
from edgeai_tidlrunner import rtwrapper, runner
from edgeai_tidlrunner.runner.common.settings.settings_help import export_help_markdown


SPECIAL_PIPELINE_NAMES = ('report',)


class StartRunner(runner.common.bases.PipelineBase):
    ARGS_DICT = runner.common.bases.SETTING_PIPELINE_RUNNER_ARGS_BASE
    COPY_ARGS = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        has_parallel_devides_arg = any(['--parallel_devices' in v for v in sys.argv[1:]])
        if not has_parallel_devides_arg:
            self._set_parallel_devices()
        #

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

    def run(self, command=None, **kwargs):
        full_kwargs = self.kwargs | kwargs
        command = command or full_kwargs.pop('command', None)
        
        if command not in SPECIAL_PIPELINE_NAMES:
            return runner.run(command=command, argparse=True, **full_kwargs)
        else:
            return runner.run(command=command, argparse=True, model_id=command+'_model', **full_kwargs)

    @classmethod
    def main(cls, **kwargs):
        # add args and continue with normal execution
        sys.argv[0] = os.environ.get('RUNNER_INVOKE_NAME', sys.argv[0])
        for k, v in kwargs.items():
            has_arg = any([f'--{k}' in arg for arg in sys.argv])
            if not has_arg:
                sys.argv.append(f'--{k}={v}')

        has_help_arg = any([arg in ('help', 'h', '--help', '-h') for arg in sys.argv])
        if len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1] in ('help', 'h', '--help', '-h')):
            print('============================================================')
            sys.argv = [sys.argv[0], 'help']
            
            parser = cls.get_arg_parser()
            command_choices = list(runner.get_command_pipelines().keys())
            parser.print_help()
            
            command_args, rest_args = parser.parse_known_args()
            kwargs = vars(command_args)

            print('============================================================')
            print('for detailed help, use the following options:')
            for command_choice in command_choices:
                print(f'{sys.argv[0]} {command_choice} --help')
            
            help_markdown = export_help_markdown()
            if help_markdown:
                print('============================================================')
                print('registered command help (markdown):')
                print(help_markdown)

        else:
            parser = cls.get_arg_parser()
            command = sys.argv[1]
            main_runner = cls(**kwargs)
            main_runner.run(command, **kwargs)


def start():
    print(f'INFO: running - {sys.argv}')
    StartRunner.main()


def start_with_proper_environment(**kwargs):
    if len(sys.argv) == 1:
        sys.argv.append('help')
    
    print(f'INFO: running - {sys.argv}')

    target_machine = kwargs['target_machine']
    is_tidl_tools_path_defined = (os.environ.get('TIDL_TOOLS_PATH', None) is not None and os.environ.get('LD_LIBRARY_PATH', None) is not None)
    has_help_arg = any([arg in ('help', 'h', '--help', '-h') for arg in sys.argv])

    if (not has_help_arg) and target_machine == rtwrapper.core.presets.TargetMachineType.TARGET_MACHINE_PC_EMULATION and (not is_tidl_tools_path_defined):
        print("INFO: TIDL_TOOLS_PATH or LD_LIBRARY_PATH is not set, restarting with proper environment...")
        parser = StartRunner.get_arg_parser()
        command_args, rest_args = parser.parse_known_args()
        command_kwargs = vars(command_args)
        if 'session.target_device' not in command_kwargs:
            print('INFO: provide target_device argument - this has to match to with your device. eg: --target_device=AM62A')
            print('INFO: list of supported devices can be found here:\n      https://github.com/TexasInstruments/edgeai/blob/main/edgeai-mpu/readme_sdk.md \n      https://github.com/TexasInstruments/edgeai-tidlrunner/blob/main/tools/tidl_tools_package/download.py#L50')
            exit(0)

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
