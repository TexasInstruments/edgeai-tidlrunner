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

import edgeai_tidlrunner
from edgeai_tidlrunner import runner


class MainRunner(runner.bases.PipelineBase):
    args_dict = runner.bases.SETTINGS_TARGET_MODULE_ARGS_DICT
    copy_args = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def _get_target_module(cls, target_module_name):
        target_module = getattr(runner.modules, target_module_name)
        return target_module

    def _run_command(self, command, config_path=None, **kwargs):
        # which target module to use
        target_module = self._get_target_module(kwargs['common.target_module'])

        if '[' in command:
            commands = command.replace('[', '').replace(']', '').split(',')
        elif isinstance(command, list):
            commands = command
        else:
            commands = [command]

        commands = [cmd.lower().replace(' ', '') for cmd in commands]

        command_dict = {}
        for commandi in commands:
            # remove spaces from command
            command_module_name = target_module.pipelines.command_module_name_dict[commandi]
            command_module = getattr(target_module.pipelines, command_module_name)
            if not config_path:
                command_args, rest_args = command_module.get_arg_parser().parse_known_args()
                kwargs_cmd = vars(command_args)
                rest_args = [arg for arg in rest_args if 'config_path' not in arg]
                rest_args = [arg for arg in rest_args if '.yaml' not in arg]
                if rest_args:
                    raise RuntimeError(f"WARNING: unrecognized arguments for {commandi}: {rest_args}")
                #
            else:
                kwargs_cmd = kwargs
            #
            command_dict[commandi] = kwargs_cmd

        runner.run(command_dict)

    def run(self, command):
        config_path = self.kwargs.pop('common.config_path', None)
        if not config_path:
            return self._run_command(command, **self.kwargs)
        else:
            config_base_path = os.path.dirname(config_path)
            with open(config_path) as fp:
                kwargs_cfg = yaml.safe_load(fp)
            #
            kwargs_cfg.pop('command', None)
            if 'configs' in kwargs_cfg:
                if len(kwargs_cfg['configs']) > 1:
                    configs = kwargs_cfg.pop('configs')
                    def command_proc(func, *args, **kwargs):
                        print(f'INFO: running - {config_key}')
                        target = functools.partial(func, *args, **kwargs)
                        proc = runner.utils.ProcessWithQueue(name=config_key, target=target)
                        proc.start()
                        return proc
                    #
                    task_entries = {}
                    for config_key, config_entry_file in configs.items():
                        if not (config_entry_file.startswith('/') or config_entry_file.startswith('http')):
                            config_entry_file = os.path.join(config_base_path, config_entry_file)
                        #
                        with open(config_entry_file) as fp:
                            command_kwargs = yaml.safe_load(fp)
                        #
                        command_kwargs.update(self.kwargs)
                        command_kwargs.update({'common.config_path': config_entry_file})
                        command_func = functools.partial(command_proc, self._run_command, command, config_path=config_entry_file, **command_kwargs)
                        task_entry = {'proc_name': config_key, 'proc_func': command_func}
                        task_list = [task_entry]
                        task_entries.update({config_key: task_list})
                    #
                    parallel_processes = self.kwargs['common.parallel_processes'] or 1
                    return runner.utils.ParallelRunner(parallel_processes=parallel_processes).run(task_entries)
                else:
                    configs = kwargs_cfg.pop('configs')
                    config_entry_file = list(configs.values())[0]
                    if not (config_entry_file.startswith('/') or config_entry_file.startswith('http')):
                        config_entry_file = os.path.join(config_base_path, config_entry_file)
                    #
                    with open(config_entry_file) as fp:
                        command_kwargs = yaml.safe_load(fp)
                    #
                    command_kwargs.update(self.kwargs)
                    command_kwargs.update({'common.config_path': config_path})
                    return self._run_command(command, config_path=config_path, **command_kwargs)
                #
            else:
                command_kwargs = kwargs_cfg
                command_kwargs.update(self.kwargs)
                command_kwargs.update({'common.config_path': config_path})
                return self._run_command(command, config_path=config_path, **command_kwargs)
            #
        #

    @classmethod
    def main(cls):
        sys.argv[0] = os.environ.get('RUNNER_INVOKE_NAME', sys.argv[0])

        if len(sys.argv) == 1 or (len(sys.argv) > 1 and sys.argv[1] in ('help', 'h', '--help', '-h')):
            print('============================================================')
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
            command = sys.argv[1]
            assert command in command_choices, RuntimeError(
                f'ERROR: invalid command: {command} - must be one of {command_choices}')
            sys.argv = [sys.argv[0]] + sys.argv[2:]
            main_runner = cls(**kwargs)
            main_runner.run(command)


def main():
    print(f'INFO: running - {sys.argv}')
    MainRunner.main()


if __name__ == '__main__':
    main()
