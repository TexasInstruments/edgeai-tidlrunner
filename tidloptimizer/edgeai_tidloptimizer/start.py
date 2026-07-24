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
from edgeai_tidlrunner import rtwrapper, runner, start
from edgeai_tidlrunner.runner.common.settings.settings_help import export_help_markdown
from edgeai_tidloptimizer import optimizer

SPECIAL_PIPELINE_NAMES = (,)


class StartOptimizer(start.StartRunner):
    def run(self, command=None, **kwargs):
        full_kwargs = self.kwargs | kwargs
        command = command or full_kwargs.pop('command', None)
        
        if command not in SPECIAL_PIPELINE_NAMES:
            return optimizer.run(command=command, argparse=True, **full_kwargs)
        else:
            return optimizer.run(command=command, argparse=True, model_id=command+'_model', **full_kwargs)


def start():
    print(f'INFO: running - {sys.argv}')
    StartOptimizer.main()


def start_with_proper_environment(START_CLS=StartOptimizer, **kwargs):
    return start.start_with_proper_environment(START_CLS=START_CLS, **kwargs)


if __name__ == "__main__":
    print(f'INFO: running {__file__} __main__')
    print(f'INFO: OR run tidlrunner-cli which is setup to call main:main() in pyproject.toml')    
    start_with_proper_environment()
