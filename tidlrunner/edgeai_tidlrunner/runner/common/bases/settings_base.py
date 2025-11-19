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


from ....rtwrapper.core import presets

class CaptureLogModes:
    CAPTURE_LOG_MODE_OFF = False  # only to screen
    CAPTURE_LOG_MODE_ON = True    # only to file
    CAPTURE_LOG_MODE_TEE = 'tee'  # to screen and to file


class SettingsBaseDefaults:
    NUM_PARALLEL_PROCESSES = 8
    CAPTURE_LOG_MODE = CaptureLogModes.CAPTURE_LOG_MODE_OFF
    CAPTURE_LOG_FILE = 'run.log'



SETTING_PIPELINE_RUNNER_ARGS_DICT = {
    # model
    'log_file':                 {'dest': 'common.log_file', 'default': SettingsBaseDefaults.CAPTURE_LOG_FILE, 'type': str, 'metavar': 'value'},
    'capture_log':              {'dest': 'common.capture_log', 'default': SettingsBaseDefaults.CAPTURE_LOG_MODE, 'type': str, 'metavar': 'value'},
    'parallel_processes':       {'dest': 'common.parallel_processes', 'default': SettingsBaseDefaults.NUM_PARALLEL_PROCESSES, 'type': int, 'metavar': 'value'},
    'parallel_devices':         {'dest': 'common.parallel_devices', 'default': None, 'type': int, 'metavar': 'value', 'help': 'number of parallel gpu devices to use for compilation (used only if gpu based tidl-tools is installed)'},
    'target_machine':           {'dest': 'session.target_machine', 'default': presets.TargetMachineType.TARGET_MACHINE_PC_EMULATION, 'type': str, 'metavar': 'value', 'help': 'target machine for running the inference (pc, evm)'},
    'package_name':           {'dest': 'common.package_name', 'default': None, 'type': str, 'metavar': 'value', 'help': 'internal argument to select package_name (runner, optimizer etc) - no need to specify explicitly'},
    'target_device':            {'dest': 'session.target_device', 'default': presets.TargetDeviceType.TARGET_DEVICE_AM68A, 'type': str, 'metavar': 'value', 'help': 'target device for inference (AM68A, AM69A, etc.)'},
}

