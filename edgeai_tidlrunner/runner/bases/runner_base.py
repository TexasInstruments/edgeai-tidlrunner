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
import wurlitzer

from .. import utils
from . import settings_base
from . import pipeline_base
from .. import modules


class PipelineRunner(pipeline_base.PipelineBase):
    ARGS_DICT = settings_base.SETTING_PIPELINE_RUNNER_ARGS_DICT
    COPY_ARGS = {}


    def __init__(self, command, **kwargs):
        super().__init__(**kwargs)
        target_module_name = self.settings['common']['target_module']
        target_module = getattr(modules, target_module_name)
        command_module_name_dict = target_module.pipelines.command_module_name_dict
        command_module_name = command_module_name_dict[command]
        command_module = getattr(target_module.pipelines, command_module_name)
        self.command_object = command_module(**kwargs)
    
    def run(self):
        capture_log = self.settings['common']['capture_log']
        log_file = self.settings['common']['log_file']
        if capture_log and self.run_dir and log_file:
            if not log_file.startswith('/') and not log_file.startswith('.'):
                log_file =  os.path.join(self.run_dir, log_file)
            #
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            with open(log_file, 'a') as log_fp:
                stdout_tee = utils.TeeLogWriter(sys.stdout, log_fp) if capture_log == settings_base.CaptureLogModes.CAPTURE_LOG_MODE_TEE else log_fp
                stderr_tee = utils.TeeLogWriter(sys.stderr, log_fp) if capture_log == settings_base.CaptureLogModes.CAPTURE_LOG_MODE_TEE else log_fp
                with wurlitzer.pipes(stdout=stdout_tee, stderr=stderr_tee):
                    return self.command_object.run()
                #
            #
        else:
            return self.command_object.run()


