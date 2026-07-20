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
import importlib

from edgeai_tidlrunner.runner import common, manager


def get_package_names():
    return ['edgeai_tidlrunner.runner']


def get_command_pipelines(**kwargs):
    command_pipelines_dict = common.get_command_pipelines(**kwargs)
    return command_pipelines_dict


def get_pipeline(pipeline_name):
    return common.get_pipeline(pipeline_name)


def get_pipeline_manager(command, **kwargs):
    command_pipelines_dict = get_command_pipelines(**kwargs)
    supported_pipeline_names = list(command_pipelines_dict.keys())
    assert command in command_pipelines_dict, f"ERROR: invalid command: {command} - must be one of {supported_pipeline_names}"
    pipeline_names = command_pipelines_dict[command]
    return manager.PipelineManager(command, pipeline_names, **kwargs)


def run(command, **kwargs):
    """
    Run the given command with the provided keyword arguments.
    
    :param command: The command to run, can be a string or a dictionary.
    :param kwargs: Additional keyword arguments to pass to the command.
    :return: The result of the command execution.
    """
    if not isinstance(command, str):
        raise RuntimeError(f"ERROR: run() got unexpected command {command} with type {type(command)}. Expected str or dict.")

    pipeline_manager = get_pipeline_manager(command)
    run_dict = pipeline_manager.create_run_dict(**kwargs)
    return pipeline_manager.run(run_dict)
