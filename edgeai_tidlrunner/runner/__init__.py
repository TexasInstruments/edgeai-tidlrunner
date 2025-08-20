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
import copy
import functools

from . import utils
from . import bases
from . import modules


def _run_command(command_key, pipeline_name, command_kwargs, capture_log):
    command_kwargs = copy.deepcopy(command_kwargs)
    command_kwargs['common.capture_log'] = capture_log
    target_module_name = command_kwargs['common.target_module']
    target_module = getattr(modules, target_module_name)
    command_module_name_dict = target_module.pipelines.command_module_name_dict
    assert command_key in command_module_name_dict, f'ERROR: unknown command: {command_key}'
    command_module = getattr(target_module.pipelines, pipeline_name)
    runner_obj = command_module(**command_kwargs)
    runner_obj.prepare()
    runner_obj.run()


def _run(model_command_dict):
    assert isinstance(model_command_dict, dict) and \
           isinstance(list(model_command_dict.values())[0],list) and \
            isinstance(list(model_command_dict.values())[0][0], tuple), 'expecting a dict of list of tuples'

    parallel_processes = None
    multiple_models = len(model_command_dict) > 1
    multiple_commands = len(list(model_command_dict.values())[0]) > 1

    task_entries = {}
    for model_key, model_command_list in model_command_dict.items():
        task_list = []
        for model_command_entry in model_command_list:
            command_key, pipeline_name, command_kwargs = model_command_entry
            # while running multiple configs, it is better to use parallel processing
            parallel_processes = command_kwargs['common.parallel_processes']
            capture_log = bases.settings_base.CaptureLogModes.CAPTURE_LOG_MODE_ON if parallel_processes and multiple_models else command_kwargs['common.capture_log']
            task_func = functools.partial(_run_command, command_key, pipeline_name, command_kwargs, capture_log)
            model_key = model_key or 'model'
            proc_name = f'{model_key}:{command_key}:{pipeline_name}'
            task_entry = {'proc_name':proc_name, 'proc_func':task_func}
            task_list.append(task_entry)
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
        _run({command:[(command,kwargs)]})
    else:
        raise RuntimeError(f"ERROR: run() got unexpected command {command} with type {type(command)}. Expected str or dict.")
    