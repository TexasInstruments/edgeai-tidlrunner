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

    @classmethod
    def _get_target_module(cls, target_module_name):
        target_module = getattr(runner.modules, target_module_name)
        return target_module

    def _create_run_dict(self, command, **kwargs):
        # which target module to use
        target_module = self._get_target_module(kwargs['common.target_module'])
        pipeline_names = target_module.pipelines.command_module_name_dict[command]
        pipeline_names = pipeline_names if isinstance(pipeline_names, list) else [pipeline_names]

        run_dict = {}
        for pipeline_name in pipeline_names:
            # remove spaces from command
            command_module = getattr(target_module.pipelines, pipeline_name)
            command_args, rest_args = command_module.get_arg_parser().parse_known_args()    
            kwargs_cmd = vars(command_args)
            provided_args = kwargs_cmd.pop('_provided_args')
            # rest_args = [arg for arg in rest_args if 'config_path' not in arg]
            # rest_args = [arg for arg in rest_args if '.yaml' not in arg]
            # if rest_args:
            #     raise RuntimeError(f"WARNING: unrecognized arguments for {command_entry}: {rest_args}")
            # #
                        
            config_path = kwargs_cmd.get('common.config_path', None)
            if config_path:
                with open(config_path) as fp:
                    kwargs_config = yaml.safe_load(fp)
                #
                kwargs_config.pop('command', None)
            else:
                kwargs_config = dict()
            #
            kwargs_config.pop('command', None)
            if 'configs' not in kwargs_config:
                configs = {'config':config_path}
            else:
                configs = kwargs_config.pop('configs')
            #

            for model_key, config_entry_file in configs.items():
                if config_entry_file:
                    if not (config_entry_file.startswith('/') or config_entry_file.startswith('.')):
                        config_base_path = os.path.dirname(config_path)
                        config_entry_file = os.path.join(config_base_path, config_entry_file)
                    #
                    with open(config_entry_file) as fp:
                        kwargs_cfg = yaml.safe_load(fp)
                    #                    
                else:
                    kwargs_cfg = dict()
                #
                kwargs_model = dict()    

                # set defaults+command line args
                kwargs_model.update(kwargs_cmd)  
                # override with cfg                        
                kwargs_model.update(kwargs_cfg)  
                # now override with command line args
                kwargs_provided = {k:v for k,v in kwargs_cmd.items() if k in provided_args}
                kwargs_model.update(kwargs_provided)
                # correct config_path is required to form the full model_path
                kwargs_model.update({'common.config_path': config_entry_file})

                run_dict_enties = run_dict.get(model_key, [])
                run_dict_enties.append((command,pipeline_name,kwargs_model))
                run_dict[model_key] = run_dict_enties
            #
        #
        return run_dict

    def run(self, command):
        run_dict = self._create_run_dict(command, **self.kwargs)
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


#################################################################################
def main_with_proper_environment(target_machine='pc'):
    print(f'INFO: running - {sys.argv}')
    if os.environ.get('TIDL_TOOLS_PATH', None) is None or \
       os.environ.get('LD_LIBRARY_PATH', None) is None:
        print("INFO: TIDL_TOOLS_PATH or LD_LIBRARY_PATH is not set, restarting with proper environment...")
        rtwrapper.restart_with_proper_environment()
    else:
        # Continue with normal execution
        with_target_machine = any(['--target_machine' in arg for arg in sys.argv])
        if not with_target_machine:
            sys.argv.append(f'--target_machine={target_machine}')
        #
        MainRunner.main()


def main_pc():
    main_with_proper_environment(target_machine='pc')


def main_evm():
    main_with_proper_environment(target_machine='evm')


def main():
    print(f"INFO: checking machine architecture...")
    result = subprocess.run(['uname', '-m'], capture_output=True, text=True)
    arch = result.stdout.strip()
    print(f"INFO: machine architecture found: {arch}")   
    target_machine = 'pc' if 'x86' in arch or 'amd64' in arch else 'evm'
    print(f"INFO: setting target_machine to: {target_machine}")
    main_with_proper_environment(target_machine=target_machine)


#################################################################################
if __name__ == "__main__":
    print(f'INFO: running {__file__} __main__')  
    main()
