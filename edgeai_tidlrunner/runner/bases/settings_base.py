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


class CaptureLogModes:
    CAPTURE_LOG_MODE_OFF = False  # only to screen
    CAPTURE_LOG_MODE_ON = True    # only to file
    CAPTURE_LOG_MODE_TEE = 'tee'  # to screen and to file


class SettingsBaseDefaults:
    NUM_PARALLEL_PROCESSES = 8
    CAPTURE_LOG_MODE = CaptureLogModes.CAPTURE_LOG_MODE_OFF
    CAPTURE_LOG_FILE = 'run.log'


class TargetModuleType:
    TARGET_MODULE_VISION = 'vision'


SETTINGS_TARGET_MODULE_ARGS_DICT = {
    'target_module':           {'dest':'common.target_module', 'default':TargetModuleType.TARGET_MODULE_VISION, 
                                'type':str, 'metavar':'value', 'choices':[TargetModuleType.TARGET_MODULE_VISION], 'required':False, 
                                'help':'specify the target module to be used. default: vision eg. --target_module vision'},
}


SETTING_PIPELINE_RUNNER_ARGS_DICT = SETTINGS_TARGET_MODULE_ARGS_DICT | {
    # model
    'model_path':               {'dest': 'session.model_path', 'default': None, 'type': str, 'metavar': 'value', 'help': 'input model'},
    'output_path':              {'dest': 'session.run_dir', 'default':'./work_dirs/{pipeline_type}/{target_device}/{tensor_bits}/{model_id}_{model_name}_{model_ext}', 'type':str, 'metavar':'value', 'help':'output model path'},
    'config_path':              {'dest': 'common.config_path', 'default': None, 'type': str, 'metavar': 'value'},
    'parallel_processes':       {'dest': 'common.parallel_processes', 'default': SettingsBaseDefaults.NUM_PARALLEL_PROCESSES, 'type': int, 'metavar': 'value'},
    'log_file':                 {'dest': 'common.log_file', 'default': SettingsBaseDefaults.CAPTURE_LOG_FILE, 'type': str, 'metavar': 'value'},
    'capture_log':              {'dest': 'common.capture_log', 'default': SettingsBaseDefaults.CAPTURE_LOG_MODE, 'type': str, 'metavar': 'value'},
}

