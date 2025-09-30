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
import copy
import functools
import yaml
import warnings

from .. import rtwrapper
from . import utils
from . import bases
from . import modules


def get_target_module(target_module_name):
    target_module = getattr(modules, target_module_name)
    return target_module


def get_command_pipelines(**kwargs):
    if 'common.target_module' not in kwargs:
        raise RuntimeError('ERROR: common.target_module is not specified')
    #
    target_module = get_target_module(kwargs['common.target_module'])
    command_choices = target_module.get_command_pipelines(**kwargs)
    return command_choices
    

def _run_command(task_index, command_key, pipeline_name, command_kwargs, capture_log):
    command_kwargs = copy.deepcopy(command_kwargs)
    parallel_devices = command_kwargs['common.parallel_devices']
    if parallel_devices is not None and parallel_devices > 0:
        parallel_devices_index = task_index % parallel_devices
        os.environ['CUDA_VISIBLE_DEVICES'] = str(parallel_devices_index)
    #
    command_kwargs['common.capture_log'] = capture_log
    target_module_name = command_kwargs['common.target_module']
    target_module = getattr(modules, target_module_name)
    command_module_name_dict = target_module.pipelines.get_command_pipelines(**command_kwargs)
    assert command_key in command_module_name_dict, f'ERROR: unknown command: {command_key}'
    command_module = getattr(target_module.pipelines, pipeline_name)
    runner_obj = command_module(**command_kwargs)
    runner_obj.prepare()
    runner_obj.run()


def _model_selection(config_entry_file, model_path, model_selection):
    is_selected = True
    if model_path is not None and model_selection is not None:
        import re
        is_selected = re.search(model_selection, model_path) is not None
        if config_entry_file is not None:
            is_selected = is_selected or re.search(model_selection, config_entry_file) is not None
        #
    #
    return is_selected


def _create_run_dict(command, argparse=False, ignore_unknown_args=False, **kwargs):
    # which target module to use
    target_module = get_target_module(kwargs['common.target_module'])
    pipeline_names = target_module.pipelines.get_command_pipelines(**kwargs)[command]
    pipeline_names = pipeline_names if isinstance(pipeline_names, list) else [pipeline_names]

    rest_args_list = []
    run_dict = {}
    for pipeline_name in pipeline_names:
        if argparse:
            command_module = getattr(target_module.pipelines, pipeline_name)
            command_args, rest_args = command_module.get_arg_parser().parse_known_args()    
            rest_args_list.append(rest_args)

            kwargs_cmd = vars(command_args)
            provided_args = kwargs_cmd.pop('_provided_args', set())
                        
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
            model_id = kwargs_config.get('session',{}).get('model_id', None) or kwargs.get('session',{}).get('model_id', None)
            if 'configs' not in kwargs_config:
                configs = {model_id:config_path}
            else:
                configs = kwargs_config.get('configs')
            #
        else:
            configs = {'config': None}
        #

        for model_id, config_entry_file in configs.items():
            if config_entry_file:
                if not (config_entry_file.startswith('/') or config_entry_file.startswith('.')):
                    config_base_path = os.path.dirname(config_path)
                    config_entry_file = os.path.join(config_base_path, config_entry_file)
                #
                with open(config_entry_file) as fp:
                    kwargs_cfg = yaml.safe_load(fp)
                #                    
            else:
                kwargs_cfg = kwargs #dict()
            #

            kwargs_cfg = command_module._flatten_dict(**kwargs_cfg)
            kwargs_cfg = command_module._expand_short_args(**kwargs_cfg)                
            kwargs_cfg = command_module._upgrade_kwargs(**kwargs_cfg)

            kwargs_model = dict()    
            # set defaults+command line args
            kwargs_model.update(kwargs_cmd)  
            # override with cfg                        
            kwargs_model.update(kwargs_cfg)  
            # now override with command line args that were provided
            kwargs_provided = {k:v for k,v in kwargs_cmd.items() if k in provided_args}
            kwargs_model.update(kwargs_provided)
            # correct config_path is required to form the full model_path
            kwargs_model.update({'common.config_path': config_entry_file})

            model_selection = kwargs_model.get('common.model_selection', None)
            model_path = kwargs_model.get('session.model_path', None)
            if _model_selection(config_entry_file, model_path, model_selection):
                # append to command_list for the model
                model_command_list = run_dict.get(model_id, [])
                model_command_list.append((command,pipeline_name,kwargs_model))
                run_dict[model_id] = model_command_list
            else:
                print(f'INFO: skipping entry: {model_path} or config {config_entry_file} does not match model_selection: {model_selection}')
            #
        #
    #

    rest_args = rest_args_list[0]        
    for rest_args_i in rest_args_list[1:]:
        rest_args = [arg for arg in rest_args if arg in rest_args_i]
    #
    # ignore the option --target_machine since it could have been added in main.py
    rest_args = [arg for arg in rest_args if '--target_machine' not in arg]
    if rest_args:
        if ignore_unknown_args:
            warnings.warn(f'WARNING: unknown args found for command: {command} - {rest_args}')
        else:
            raise RuntimeError(f'WARNING: unknown args found for command: {command} - {rest_args}')
        #
    #
    return run_dict


def _run(model_command_dict):
    assert isinstance(model_command_dict, dict) and \
           isinstance(list(model_command_dict.values())[0],list) and \
            isinstance(list(model_command_dict.values())[0][0], tuple), 'expecting a dict of list of tuples'

    parallel_processes = None
    multiple_models = len(model_command_dict) > 1
    multiple_commands = len(list(model_command_dict.values())[0]) > 1

    task_index = 0
    task_entries = {}
    for model_key, model_command_list in model_command_dict.items():
        task_list = []
        for model_command_entry in model_command_list:
            command_key, pipeline_name, command_kwargs = model_command_entry
            # while running multiple configs, it is better to use parallel processing
            parallel_processes = command_kwargs['common.parallel_processes']
            capture_log = bases.settings_base.CaptureLogModes.CAPTURE_LOG_MODE_ON if parallel_processes and multiple_models else command_kwargs['common.capture_log']
            task_func = functools.partial(_run_command, task_index, command_key, pipeline_name, command_kwargs, capture_log)
            model_key = model_key or 'model'
            proc_name = f'{model_key}:{command_key}:{pipeline_name}'
            task_entry = {'proc_name':proc_name, 'proc_func':task_func}
            task_list.append(task_entry)
            task_index = task_index + 1
        #
        task_entries.update({model_key:task_list})
    #

    # if there is more than one model or command or parallel_processes is set, we need to launch in ParallelRunner
    # or else we can directly run it
    if parallel_processes and (multiple_models or multiple_commands):
        # there are multiple commands given to be run back to back - running them on the same process can be problematic
        # so we will run them using multiprocessing - using separate process for each sub-command
        # this is useful for cases like 'compile,accuracy' or 'import,infer'
        def command_proc(proc_name, proc_func):
            print(f'INFO: running - {proc_name}')
            proc = utils.ProcessWithQueue(name=proc_name, target=proc_func)
            proc.start()
            return proc

        for task_list in task_entries.values():
            for task_entry in task_list:
                proc_name = task_entry['proc_name']
                proc_func = task_entry['proc_func']
                task_entry['proc_func'] = functools.partial(command_proc, proc_name, proc_func)
            #
        #
        return utils.ParallelRunner(parallel_processes=parallel_processes).run(task_entries)
    else:
        return utils.SequentialRunner().run(task_entries)


def run(command, **kwargs):
    """
    Run the given command with the provided keyword arguments.
    
    :param command: The command to run, can be a string or a dictionary.
    :param kwargs: Additional keyword arguments to pass to the command.
    :return: The result of the command execution.
    """
    if isinstance(command, str):
        run_dict = _create_run_dict(command, argparse=False, **kwargs)
        return _run(run_dict)
    else:
        raise RuntimeError(f"ERROR: run() got unexpected command {command} with type {type(command)}. Expected str or dict.")
    