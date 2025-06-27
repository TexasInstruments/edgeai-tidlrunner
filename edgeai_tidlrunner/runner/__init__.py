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


import functools

from . import utils
from . import bases
from . import modules


def _run(command_dict): 
    if len(command_dict) == 1:
        # there is only one command to run - running it in the current process
        command_entry = list(command_dict.keys())[0]
        command_kwargs = command_dict[command_entry]
        runner_obj = bases.PipelineRunner(command_entry, **command_kwargs)
        runner_obj.run()
    else:
        # there are multiple commands given to be run back to back - running them on the same process can be problematic
        # so we will run them using multiprocessing - using separate process for each sub-command
        # this is useful for cases like 'compile,accuracy' or 'import,infer'
        def command_proc(command_entry, **kwargs):
            print(f'INFO: running - {command_entry}')
            runner_obj = bases.PipelineRunner(command_entry, **kwargs)
            proc = utils.ProcessWithQueue(name=command_entry, target=runner_obj.run)
            proc.start()
            return proc
        
        task_list = []
        for commandi, commandi_kwargs in command_dict.items():
            command_func = functools.partial(command_proc, commandi, **commandi_kwargs)
            task_entry = {'proc_name':commandi, 'proc_func':command_func}
            task_list.append(task_entry)

        command_key = ','.join(command_dict.keys())
        task_entries = {command_key: task_list}
        return utils.ParallelRunner(parallel_processes=1).run(task_entries)


def run(command, **kwargs):
    """
    Run the given command with the provided keyword arguments.
    
    :param command: The command to run, can be a string or a dictionary.
    :param kwargs: Additional keyword arguments to pass to the command.
    :return: The result of the command execution.
    """
    if isinstance(command, str):
        _run({command:kwargs})
    elif isinstance(command, dict) and len(command) > 0:
        # there is only one command to run - running it in the current process
        command_dict = command
        if len(kwargs) > 0:
            raise RuntimeError("ERROR: when command is a dict, kwargs should be empty. ")
        _run(command_dict)
    else:
        raise RuntimeError(f"ERROR: run() got unexpected command {command} with type {type(command)}. Expected str or dict.")
    