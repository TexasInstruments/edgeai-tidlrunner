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
import copy
import yaml
import difflib
import warnings
import functools
import re

import edgeai_tidlrunner
from edgeai_tidlrunner import runner, optimizer
from edgeai_tidlrunner.runner.common import utils
from edgeai_tidlrunner.runner.common import bases


def get_package_names():
    return ['runner', 'optimizer']


def get_command_pipelines(package_name=None, **kwargs):
    command_pipelines = {}
    package_names = get_package_names() if package_name is None else [package_name]
    for package_module_name in package_names:
        package_module = getattr(edgeai_tidlrunner, package_module_name)
        for target_module_name in package_module.get_target_modules():
            target_module = getattr(package_module, target_module_name)
            target_comand_pipelines = target_module.get_command_pipelines(**kwargs)
            target_comand_pipelines = {(package_module_name + '.' + target_module_name + '.' + k):v for k,v in target_comand_pipelines.items()}
            command_pipelines.update(target_comand_pipelines)
        #
    #
    return command_pipelines


def matching_command_name(command_name, package_name=None):
    command_pipelines = get_command_pipelines(package_name=package_name)
    command_names = list(command_pipelines.keys())
    if command_name not in command_pipelines.keys():
        command_name_matches = difflib.get_close_matches(command_name, command_names)
        if not command_name_matches:
            command_name_matches = [cname for cname in command_names if command_name == cname.split('.')[-1]]
        #
        if not command_name_matches or len(command_name_matches) > 1:
            raise RuntimeError(f'ERROR: invalid command_name: {command_name}, available commands are: {list(command_names)}')
        #
        command_name = command_name_matches[0]
    #
    if command_name not in command_names:
        raise RuntimeError(f'ERROR: invalid command_name: {command_name}, available commands are: {list(command_names)}')
    #
    return command_name


def get_target_module(command_name, **kwargs):
    package_name = kwargs.get('common.package_name', None)
    
    command_name = matching_command_name(command_name, package_name=package_name)
    command_name_split = command_name.split('.')
    assert len(command_name_split) == 3, f'ERROR: invalid command_name: {command_name}'

    if package_name:
        assert package_name in get_package_names(), f'ERROR: invalid package_name: {package_name}'
        assert package_name == command_name_split[0], f'ERROR: given package_name: {package_name} does not match: {command_name_split[0]}'
    else:
        package_name = command_name_split[0]
    #
    target_module = getattr(getattr(edgeai_tidlrunner, package_name), command_name_split[1])
    return target_module


def _run_command(task_index, command_name, pipeline_name, command_kwargs, capture_log):
    command_kwargs = copy.deepcopy(command_kwargs)
    parallel_devices = command_kwargs['common.parallel_devices']
    if parallel_devices is not None and parallel_devices > 0:
        parallel_devices_index = task_index % parallel_devices
        os.environ['CUDA_VISIBLE_DEVICES'] = str(parallel_devices_index)
    #
    command_kwargs['common.capture_log'] = capture_log

    target_module = get_target_module(command_name, **command_kwargs)
    command_module_name_dict = target_module.get_command_pipelines(**command_kwargs)
    assert command_name in command_module_name_dict, f'ERROR: unknown command: {command_name}'
    command_module = getattr(target_module.pipelines, pipeline_name)

    runner_obj = command_module(**command_kwargs)
    runner_obj.prepare()
    runner_obj.run()


def _model_selection(model_selection, *args):
    if model_selection is None:
        is_selected = True
    else:
        model_selection = utils.formatted_nargs(model_selection)
        is_selected = False
        for m in model_selection:
            for arg in args:
                if isinstance(arg, str):
                    is_selected = is_selected or (re.search(m, arg) is not None)
                #
            #
        #
    #
    return is_selected


def _model_shortlist(model_shortlist, model_shortlist_for_model):
    if model_shortlist:
        shortlisted_model = model_shortlist_for_model and int(model_shortlist_for_model) <= int(model_shortlist)
    else:
        shortlisted_model = True
    #
    return shortlisted_model


def _get_configs(config_path, **kwargs):
    if isinstance(config_path, str):
        if config_path.endswith('.yaml'):
            with open(config_path) as fp:
                kwargs_config = yaml.safe_load(fp)
            #
            kwargs_config.pop('command', None)
            if 'configs' in kwargs_config:
                configs = kwargs_config.get('configs')
            else:
                model_id = kwargs_config.get('session',{}).get('model_id', None) or kwargs.get('session.model_id', None)
                configs = {model_id:config_path}
            #
        elif os.path.exists(config_path) and os.path.isdir(config_path):
            print(f"INFO: config_path is a configs module from edgeai-benchmark: {config_path}")
            import edgeai_benchmark

            runner_settings = bases.pipeline_base.PipelineBase._parse_to_dict(**kwargs)
            runtime_options = runner_settings.get('session', {}).get('runtime_options', {})
            calibration_frames = runtime_options.get('advanced_options:calibration_frames', None)
            calibration_iterations = runtime_options.get('advanced_options:calibration_iterations', None)
            num_frames = runner_settings.get('common', {}).get('num_frames', None)
            settings_kwargs = {}
            for arg in ['runtime_options', 'calibration_frames', 'calibration_iterations', 'num_frames']:
                arg_value = locals()[arg]
                if arg_value:
                    settings_kwargs[arg] = arg_value
                #
            #
            
            settings_file = edgeai_benchmark.get_settings_file()

            model_shortlist = kwargs.get('common.model_shortlist', None)
            model_shortlist = int(model_shortlist) if model_shortlist is not None else None
            model_selection=kwargs.get('common.model_selection', None)
            model_selection = utils.formatted_nargs(model_selection)
            target_device = kwargs.get('session.target_device', None)

            settings = edgeai_benchmark.config_settings.ConfigSettings(
                settings_file, model_shortlist=model_shortlist, model_selection=model_selection, 
                target_device=target_device, 
                configs_path = os.path.abspath(config_path),
                **settings_kwargs)

            if not os.path.exists(settings.datasets_path):
                benchmark_dependencies_path = '../edgeai-benchmark/dependencies'
                local_dependencies_path = './dependencies'
                if os.path.exists(benchmark_dependencies_path) and not os.path.exists(local_dependencies_path):
                    try:
                        print(f"INFO: creating symlink to: {benchmark_dependencies_path}")
                        print(f"INFO: make sure that datasets required for edgeai-benchmark configs are available in that folder")
                        print(f"INFO: consult the documentation of edgeai-benchmark for more information")
                        os.symlink(benchmark_dependencies_path, local_dependencies_path)
                    except:
                        print(f"INFO: could not create symlink to: {benchmark_dependencies_path}")
                    #
                #
            #
            print(f'settings: {settings}')
            if settings.model_shortlist is not None:
                print('INFO', 'model_shortlist has been set', 'it will cause only a subset of models to run:')
                print('INFO', 'model_shortlist', f'{settings.model_shortlist}')
            #

            work_path = kwargs['common.work_path']
            print(f'\nINFO: work_path: {work_path}')
            pipeline_configs = edgeai_benchmark.interfaces.get_configs(settings, work_path)
            configs = edgeai_benchmark.pipelines.PipelineRunner(settings, pipeline_configs).get_pipeline_configs()
            upgrade_config = {'common.upgrade_config':False}
            configs = {model_id: (upgrade_config | pipeline_config) for model_id, pipeline_config in configs.items()}
        else:
            raise RuntimeError(f'ERROR: invalid config_path: {config_path}')
        #
    elif isinstance(config_path, dict):
        configs = config_path
    else:
        raise RuntimeError(f'ERROR: invalid config_path: {config_path}')
    #
    return configs


def _create_run_dict(command, ignore_unknown_args=False, model_id=None, **kwargs):
    # which target module to use
    target_module = get_target_module(command, **kwargs)
    pipeline_names = target_module.get_command_pipelines(**kwargs)[command]
    pipeline_names = pipeline_names if isinstance(pipeline_names, list) else [pipeline_names]

    selected_models = []
    rest_args_list = []
    run_dict = {}
    for pipeline_name in pipeline_names:
        command_module = getattr(target_module.pipelines, pipeline_name)
        command_args, rest_args = command_module.get_arg_parser().parse_known_args()    
        rest_args_list.append(rest_args)

        kwargs_cmd = vars(command_args)
        provided_args = kwargs_cmd.pop('_provided_args', set())
                    
        config_path = kwargs_cmd.get('common.config_path', None)
        if config_path:
            configs = _get_configs(config_path, **(kwargs_cmd | kwargs))
        else:
            assert model_id is not None, 'ERROR: model_id is required when config_path is not given'
            configs = {model_id:dict()}
        #

        selected_models = []
        for model_id, config_entry in configs.items():
            if isinstance(config_entry, str):
                if not (config_entry.startswith('/') or config_entry.startswith('.')):
                    config_base_path = os.path.dirname(config_path)
                    config_entry = os.path.join(config_base_path, config_entry)
                #
                with open(config_entry) as fp:
                    kwargs_cfg = yaml.safe_load(fp)
                #         
            elif isinstance(config_entry, dict):
                kwargs_cfg = utils.pretty_object(config_entry)
            else:
                kwargs_cfg = dict()
            #
            kwargs_cfg.get('session', {}).pop('run_dir', None)

            # add given args
            kwargs_copy = copy.deepcopy(kwargs)
            kwargs_copy.pop('command', None)
            kwargs_cfg.update(kwargs_copy)

            # process args
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
            if isinstance(config_entry, str):
                config_entry_path = config_entry
                kwargs_model.update({'common.config_path': config_entry})
            else:
                config_entry_path = None
            #
            kwargs_model['common.pipeline_config'] = config_entry

            verbose = kwargs_model.get('common.verbose', 0)
            model_shortlist = kwargs_model.get('common.model_shortlist', None)
            model_selection = kwargs_model.get('common.model_selection', None)
            model_path = kwargs_model.get('session.model_path', None)

            model_shortlist_for_model = kwargs_model.get('model_info.model_shortlist', None)
            shortlisted_model = _model_shortlist(model_shortlist, model_shortlist_for_model)
            selected_model = _model_selection(model_selection, config_entry, model_path, model_id)
            if shortlisted_model and selected_model:
                # append to command_list for the model
                model_command_list = run_dict.get(model_id, [])
                model_command_list.append((command,pipeline_name,kwargs_model))
                run_dict[model_id] = model_command_list
                selected_models.append(model_id)
            elif verbose > 0:
                if config_entry_path:
                    print(f'INFO: skipping entry: {config_entry_path} - does not match model_shortlist: {model_shortlist}, model_selection: {model_selection}')
                else:
                    print(f'INFO: skipping entry: {model_path} - does not match model_shortlist: {model_shortlist}, model_selection: {model_selection}')
                #
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
        run_dict = _create_run_dict(command, **kwargs)
        return _run(run_dict)
    else:
        raise RuntimeError(f"ERROR: run() got unexpected command {command} with type {type(command)}. Expected str or dict.")
    