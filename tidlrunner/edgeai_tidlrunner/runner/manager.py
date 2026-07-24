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
import functools
import copy
import warnings
import yaml
import re

from .common import bases, utils, pipelines
from .common.bases.pipeline_base import PipelineBase


class PipelineManager(PipelineBase):
    ARGS_DICT = {}
    COPY_ARGS = {}

    def __init__(self, command, pipeline_names, **kwargs):
        super().__init__()
        self.command = command
        self.pipeline_names = pipeline_names

    def _run_command(self, task_index, command_name, pipeline_name, command_kwargs, capture_log):
        command_kwargs = copy.deepcopy(command_kwargs)
        command_kwargs['common.capture_log'] = capture_log

        command_module = getattr(pipelines, pipeline_name)

        runner_obj = command_module(**command_kwargs)
        try:
            runner_obj.prepare()
        except Exception as e:
            print(f"ERROR: Failed to prepare runner for command: {command_name} : {e}")
            sys.exit(1)
        
        runner_obj.run()

    def _model_selection(self, model_selection, *args):
        if model_selection is None:
            return True
        
        model_selection = utils.formatted_nargs(model_selection)
        is_selected = False
        for m in model_selection:
            for arg in args:
                if isinstance(arg, str):
                    selected_parts = all([re.search(m_part, arg) is not None for m_part in m.split('+')])
                else:
                    selected_parts = True
                #
                is_selected = is_selected or selected_parts
            #
        #
        return is_selected


    def _model_shortlist(self, model_shortlist, model_shortlist_for_model):
        if model_shortlist:
            shortlisted_model = model_shortlist_for_model is not None and int(model_shortlist_for_model) <= int(model_shortlist)
        else:
            shortlisted_model = True
        #
        return shortlisted_model

    def _get_configs(self, config_path, **kwargs):
        is_aggregate_config_file = False
        if isinstance(config_path, str):
            if config_path.endswith('.yaml'):
                with open(config_path) as fp:
                    kwargs_config = yaml.safe_load(fp)
                #
                kwargs_config.pop('command', None)
                if 'configs' in kwargs_config:
                    configs = kwargs_config.get('configs')
                    is_aggregate_config_file = True
                else:
                    model_id = kwargs_config.get('session',{}).get('model_id', None) or kwargs.get('session.model_id', None)
                    configs = {model_id:config_path}
                #
                if 'session.target_device' in kwargs_config:
                    assert kwargs['target_device'] == kwargs_config['session.target_device'], f"WARNING: config file {config_path} contains session.target_device: {kwargs_config['session.target_device']} - not recommended. To override the default, provide through commandline argument."
                #
                if 'session.target_machine' in kwargs_config:
                    assert kwargs['target_machine'] != kwargs_config['session.target_machine'], f"WARNING: config file {config_path} contains session.target_machine: {kwargs_config['session.target_machine']} - not recommended. To override the default, provide through commandline argument."
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

                print(f'settings: {settings}')
                if settings.model_shortlist is not None:
                    print('INFO', 'model_shortlist has been set', 'it will cause only a subset of models to run:')
                    print('INFO', 'model_shortlist', f'{settings.model_shortlist}')

                work_path = kwargs['common.work_path']
                print(f'\nINFO: work_path: {work_path}')
                pipeline_configs = edgeai_benchmark.interfaces.get_configs(settings, work_path)
                pipeline_configs = edgeai_benchmark.pipelines.PipelineRunner(settings, pipeline_configs).get_pipeline_configs()
                upgrade_config = {'common.upgrade_config': False}
                configs = {}
                for model_id, pipeline_config in pipeline_configs.items():
                    combined_config = upgrade_config | pipeline_config | {'common.pipeline_config': pipeline_config}
                    configs[model_id] = combined_config

            else:
                raise RuntimeError(f'ERROR: invalid config_path: {config_path}')

        elif isinstance(config_path, dict):
            configs = config_path
        else:
            raise RuntimeError(f'ERROR: invalid config_path: {config_path}')

        return configs, is_aggregate_config_file

    def create_run_dict(self, ignore_unknown_args=False, model_id=None, **kwargs):
        is_aggregate_config_file = False

        selected_models = []
        rest_args_list = []
        run_dict = {}
        for pipeline_idx, pipeline_name in enumerate(self.pipeline_names):
            command_module = getattr(pipelines, pipeline_name)
            command_args, rest_args = command_module.get_arg_parser().parse_known_args()    
            rest_args_list.append(rest_args)

            kwargs_with_defaults = vars(command_args)
            provided_arg_names = kwargs_with_defaults.pop('_provided_args', set())
            provided_kwargs = {k:kwargs_with_defaults[k] for k in provided_arg_names}
                        
            config_path = kwargs_with_defaults.get('common.config_path', None)
            model_path = kwargs_with_defaults.get('session.model_path', None)
            pipeline_type = kwargs_with_defaults.get('common.pipeline_type', None)
            if config_path:
                configs, is_aggregate_config_file = self._get_configs(config_path, **kwargs_with_defaults)
                if is_aggregate_config_file:
                    print(f'INFO: aggregate config file given - config_path: {config_path}')

            else:
                if model_id is None:
                    print('WARNING: model_id is not given, generating randomly')
                    model_id = f"{pipeline_type}-" + utils.generate_unique_id(model_path, num_characters=8) if model_path else "x-x"

                configs = {model_id:{'session.model_id':model_id}}


            selected_models = []
            for model_id, config_entry in configs.items():
                pipeline_config = config_entry.pop('common.pipeline_config', None) if isinstance(config_entry, dict) else None

                if isinstance(config_entry, str):
                    if is_aggregate_config_file and not (config_entry.startswith('/') or config_entry.startswith('.')):
                        config_base_path = os.path.dirname(config_path)
                        config_entry = os.path.join(config_base_path, config_entry)

                    with open(config_entry) as fp:
                        kwargs_cfg = yaml.safe_load(fp)
  
                elif isinstance(config_entry, dict):
                    kwargs_cfg = utils.pretty_object(config_entry)
                else:
                    kwargs_cfg = dict()

                kwargs_cfg.get('session', {}).pop('run_dir', None)

                # create preliminary args (without upgrade) - for some basic checks - model_shortlist, model_selection
                kwargs_before_upgrade = copy.deepcopy(kwargs_with_defaults)
                kwargs_before_upgrade.update(kwargs_cfg)
                kwargs_before_upgrade = bases.pipeline_base.PipelineBase.process_args(**kwargs_before_upgrade)
                # now override with command line args that were provided - that has preferance over cfg
                kwargs_before_upgrade.update(provided_kwargs)
                # selected_model, shortlisted_model
                verbose = kwargs_before_upgrade.get('common.verbose', 0)
                model_shortlist = kwargs_before_upgrade.get('common.model_shortlist', None)
                model_selection = kwargs_before_upgrade.get('common.model_selection', None)
                if is_aggregate_config_file:
                    model_path = kwargs_before_upgrade.get('session.model_path', None)
                    model_shortlist_for_model = kwargs_before_upgrade.get('model_info.model_shortlist', None)
                    shortlisted_model = self._model_shortlist(model_shortlist, model_shortlist_for_model)
                    selected_model = self._model_selection(model_selection, config_entry, model_path, model_id)
                else:
                    selected_model = shortlisted_model = True

                if shortlisted_model and selected_model:
                    # now systematically create the final kwargs
                    kwargs_model = dict()
                    # set defaults+command line args - so that we have all the args required
                    kwargs_model.update(kwargs_with_defaults)
                    # upgrade the cfg and override kwargs_model with kwargs_cfg - cfg has preferance over default
                    kwargs_cfg = command_module.process_args(**kwargs_cfg)
                    kwargs_model.update(kwargs_cfg)
                    # now override with command line args that were provided - that has preferance over cfg
                    kwargs_model.update(provided_kwargs)
                    # correct config_path is required to form the full model_path
                    if isinstance(config_entry, str):
                        config_entry_path = config_entry
                        kwargs_model.update({'common.config_path': config_entry})
                    else:
                        config_entry_path = None
                    #
                    kwargs_model['common.pipeline_config'] = pipeline_config

                    # append to command_list for the model
                    model_command_list = run_dict.get(model_id, [])
                    model_command_list.append((self.command,pipeline_name,kwargs_model))
                    run_dict[model_id] = model_command_list
                    selected_models.append(model_id)
                    if pipeline_idx == 0:
                        print(f'INFO: shortlisted/selected - model_id: {kwargs_model.get("session.model_id",None)}, config_path: {kwargs_model.get("common.config_path",None)}, model_path: {kwargs_model.get("session.model_path",None)}')
                elif verbose > 0:
                    if pipeline_idx == 0:
                        if config_entry_path:
                            print(f'INFO: skipping entry: {config_entry_path} - does not match model_shortlist: {model_shortlist}, model_selection: {model_selection}')
                        else:
                            print(f'INFO: skipping entry: {model_path} - does not match model_shortlist: {model_shortlist}, model_selection: {model_selection}')

        rest_args = rest_args_list[0]        
        for rest_args_i in rest_args_list[1:]:
            rest_args = [arg for arg in rest_args if arg in rest_args_i]
        #
        # ignore the option --target_machine since it could have been added in main.py
        rest_args = [arg for arg in rest_args if '--target_machine' not in arg]
        if rest_args:
            if ignore_unknown_args:
                warnings.warn(f'WARNING: unknown args found for command: {self.command} - {rest_args}')
            else:
                print(f'WARNING: unknown args found for command: {self.command} - {rest_args}')
                exit(0)
            #
        #

        if len(run_dict) == 0:
            print(f'ERROR: no models selected for command: {self.command} - model_shortlist: {model_shortlist}, model_selection: {model_selection}')
        #

        return run_dict

    def run(self, model_command_dict):
        assert isinstance(model_command_dict, dict), f'ERROR: {__file__} _run(): expecting a dict of list of tuples'
        if len(model_command_dict) == 0:
            print(f'ERROR: nothing to run - model_command_dict is empty')
            return

        assert isinstance(list(model_command_dict.values())[0],list) and \
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
                parallel_devices = command_kwargs['common.parallel_devices']
                instance_timeout = command_kwargs.get('common.instance_timeout', None)
                overall_timeout = command_kwargs.get('common.overall_timeout', None)

                if command_kwargs['common.capture_log'] == bases.settings_base.CaptureLogModes.CAPTURE_LOG_MODE_ADAPTIVE:
                    # CAPTURE_LOG_MODE_TEE is not working now - need to fix it before using here
                    capture_log = bases.settings_base.CaptureLogModes.CAPTURE_LOG_MODE_ON \
                        if parallel_processes and multiple_models else bases.settings_base.CaptureLogModes.CAPTURE_LOG_MODE_OFF #CAPTURE_LOG_MODE_TEE
                else:
                    capture_log = command_kwargs['common.capture_log']
                #

                proc_env = None
                if parallel_processes and parallel_devices is not None and parallel_devices > 0:
                    parallel_devices_index = task_index % parallel_devices
                    proc_env = dict()
                    proc_env['CUDA_VISIBLE_DEVICES'] = str(parallel_devices_index)
                    proc_env['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
                #
                
                task_func = functools.partial(self._run_command, task_index, command_key, pipeline_name, command_kwargs, capture_log)
                model_key = model_key or 'model'
                proc_name = f'{model_key}:{command_key}:{pipeline_name}'
                proc_info = {'model_path': command_kwargs.get('common.model_path', ''), 'config_path': command_kwargs.get('common.config_path', '')}
                task_entry = {'proc_name':proc_name, 'proc_func':task_func, 'proc_info':proc_info, 'proc_env':proc_env}
                task_list.append(task_entry)
                task_index = task_index + 1
            #
            task_entries.update({model_key:task_list})
        #

        # if there is more than one model or command or parallel_processes is set, we need to launch in ParallelRunner
        # or else we can directly run it
        if (parallel_processes and multiple_models) or (multiple_models or multiple_commands):
            for task_list in task_entries.values():
                for task_entry in task_list:
                    proc_name = task_entry['proc_name']
                    proc_func = task_entry['proc_func']
                    proc_info = task_entry['proc_info']
                    proc_env = task_entry['proc_env'] 
                    # there are multiple commands given to be run back to back - running them on the same process can be problematic
                    # so we will run them using multiprocessing - using separate process for each sub-command
                    # this is useful for cases like 'compile,evaluate' or 'import,infer'
                    task_entry['proc_func'] = functools.partial(utils.ProcessWithQueue.create, proc_name, proc_func, proc_info, proc_env)
                #
            #
            if (parallel_processes and multiple_models):
                runner_obj = utils.ParallelRunner(parallel_processes=parallel_processes, overall_timeout=overall_timeout, instance_timeout=instance_timeout)
            else:
                runner_obj = utils.SequentialRunner(parallel_processes=parallel_processes, with_progressbar=multiple_models)
            #
        else:
            runner_obj = utils.SequentialRunner()
        #
        return runner_obj.run(task_entries)
