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

from edgeai_tidlrunner import rtwrapper, runner

from .start import MainRunner


def main_with_proper_environment(target_machine='pc'):
    print(f'INFO: running - {sys.argv}')
    if os.environ.get('TIDL_TOOLS_PATH', None) is None or \
       os.environ.get('LD_LIBRARY_PATH', None) is None:
        print("INFO: TIDL_TOOLS_PATH or LD_LIBRARY_PATH is not set, restarting with proper environment...")
        rtwrapper.restart_with_proper_environment()
    else:
        # Continue with normal execution
        with_target_machine = any(['--target_machine' in arg for arg in sys.argv])
        if not with_target_machine:
            sys.argv.append(f'--target_machine={target_machine}')
        #
        MainRunner.main()


def main_pc():
    main_with_proper_environment(target_machine='pc')


def main_evm():
    main_with_proper_environment(target_machine='evm')


def main():
    print(f"INFO: checking machine architecture...")
    result = subprocess.run(['uname', '-m'], capture_output=True, text=True)
    arch = result.stdout.strip()
    print(f"INFO: machine architecture found: {arch}")   
    target_machine = 'pc' if 'x86' in arch or 'amd64' in arch else 'evm'
    print(f"INFO: setting target_machine to: {target_machine}")
    main_with_proper_environment(target_machine=target_machine)


if __name__ == "__main__":
    print(f'INFO: running {__file__} __main__')  
    main()
