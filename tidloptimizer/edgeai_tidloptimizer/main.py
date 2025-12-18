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

from edgeai_tidlrunner.start import StartRunner


def _main(_package_name='edgeai_tidloptimizer.optimizer', **kwargs):
    # TIDL_TOOLS_PATH is not needed in tidloptimizer, but just set it to empty to pass through checks for it
    os.environ['TIDL_TOOLS_PATH'] = os.environ.get('TIDL_TOOLS_PATH', '')
    # start
    StartRunner.main(_package_name=_package_name, **kwargs)



def main(**kwargs):
    _main(_package_name='edgeai_tidloptimizer.optimizer', _package_short_name='tidloptimizer', **kwargs)


if __name__ == "__main__":
    print(f'INFO: running {__file__} __main__')  
    _main(_package_name='edgeai_tidloptimizer.optimizer', _package_short_name='tidloptimizer')

