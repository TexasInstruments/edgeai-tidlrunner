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

    @classmethod
    def _create_run_dict(cls, command, model_key=None, **kwargs):
        config_path = kwargs.get('common.config_path', None)
        # which target module to use
        target_module = cls._get_target_module(kwargs['common.target_module'])
        pipeline_names = target_module.pipelines.command_module_name_dict[command]
        pipeline_names = pipeline_names if isinstance(pipeline_names, list) else [pipeline_names]

        command_list = []
        for pipeline_name in pipeline_names:
            # remove spaces from command
            command_module = getattr(target_module.pipelines, pipeline_name)
            if config_path:
                with open(config_path) as fp:
                    kwargs_cfg = yaml.safe_load(fp)
                #
                kwargs_cfg.pop('command', None)
            else:
                kwargs_cfg = dict()
            #
            command_args, rest_args = command_module.get_arg_parser().parse_known_args()
            kwargs_cmd = vars(command_args)
            # rest_args = [arg for arg in rest_args if 'config_path' not in arg]
            # rest_args = [arg for arg in rest_args if '.yaml' not in arg]
            # if rest_args:
            #     raise RuntimeError(f"WARNING: unrecognized arguments for {command_entry}: {rest_args}")
            # #
            kwargs_cfg.update(kwargs_cmd)
            kwargs_cfg.update(kwargs)
            command_list.append((command,pipeline_name,kwargs_cfg))
        #
        return command_list

    def run(self, command):
        run_dict = {}
        config_path = self.kwargs.pop('common.config_path', None)
        if not config_path:
            kwargs = copy.deepcopy(self.kwargs)
            run_dict_entry = self._create_run_dict(command, model_key=None, **kwargs)
            run_dict.update({'command':run_dict_entry})
        else:
            with open(config_path) as fp:
                kwargs_cfg = yaml.safe_load(fp)
            #
            kwargs_cfg.pop('command', None)
            if 'configs' not in kwargs_cfg:
                configs = {'config':config_path}
            else:
                configs = kwargs_cfg.pop('configs')
            #
            for model_key, config_entry_file in configs.items():
                if not (config_entry_file.startswith('/') or config_entry_file.startswith('.')):
                    config_base_path = os.path.dirname(config_path)
                    config_entry_file = os.path.join(config_base_path, config_entry_file)
                #
                kwargs = copy.deepcopy(self.kwargs)
                kwargs.update({'common.config_path': config_entry_file})
                run_dict_entry = self._create_run_dict(command, model_key=model_key, **kwargs)
                run_dict.update({model_key:run_dict_entry})
            #
        #
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
        elif len(sys.argv) == 2 or (len(sys.argv) > 2 and sys.argv[2] in ('help', 'h', '--help', '-h')):
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


def main():
    print(f'INFO: running - {sys.argv}')
    MainRunner.main()


def main_pc():
    has_targe_machine = any([arg for arg in sys.argv if 'targe_machine' in arg])
    if not has_targe_machine:
        sys.argv.append('--target_machine=pc')
    #
    main()


def main_evm():
    has_targe_machine = any([arg for arg in sys.argv if 'targe_machine' in arg])
    if not has_targe_machine:
        sys.argv.append('--target_machine=evm')
    #
    main()


if __name__ == '__main__':
    main()
