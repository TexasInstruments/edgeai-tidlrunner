# Copyright (c) 2018-2021, Texas Instruments
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

import enum
import copy
import os
import sys
import warnings

from .....rtwrapper.options import presets, attr_dict
from .....rtwrapper.options.runtime_options import RuntimeOptions
from . import settings_default as runtime_settings_default_module


class RuntimeSettings(attr_dict.AttrDict):
    def __init__(self, **kwargs):
        self.calibration_iterations_factor = None
        self.target_device_preset = True
        self.target_device = presets.TargetDeviceType.TARGET_DEVICE_AM68A
        super().__init__()
        self.update(self._get_runtime_settings(**kwargs))

    def get_runtime_settings(self):
        return self

    def _get_calibration_iterations_factor(self, fast_calibration):
        # model may need higher number of itarations for certain devices (i.e. when per channel quantization is not supported)
        device_needs_more_iterations = (self.calibration_iterations_factor is not None)
        model_needs_more_iterations = (not fast_calibration)
        if device_needs_more_iterations and model_needs_more_iterations:
            return self.calibration_iterations_factor
        else:
            return presets.CalibrationIterationsFactor.CALIBRATION_ITERATIONS_FACTOR_1X

    def _get_runtime_settings(self, model_quant_type=None, settings_type=None, is_qat=False, fast_calibration=True,
            det_options=None, ext_options=None, **kwargs):
        runtime_settings = self._get_runtime_settings_with_default(**kwargs)
        self.update(runtime_settings)

        # target device presets
        if isinstance(self.target_device_preset, dict):
            preset_dict = self.target_device_preset
        elif self.target_device_preset and self.target_device:
            preset_dict = presets.TARGET_DEVICE_SETTINGS_PRESETS[self.target_device]
        else:
            preset_dict = None
        #
        if preset_dict:
            self.update(preset_dict)
        #

        calibration_iterations_factor = self._get_calibration_iterations_factor(fast_calibration)

        runtime_options = RuntimeOptions(
            model_quant_type=model_quant_type, settings_type=settings_type, is_qat=is_qat,
            calibration_iterations_factor=calibration_iterations_factor,
            det_options=det_options, ext_options=ext_options, **runtime_settings['runtime_options'])
        self['runtime_options'] = runtime_options

        return self

    def update_runtime_settings(self, **kwargs):
        runtime_options = kwargs.pop('runtime_options', {})
        self.update(kwargs)
        self.runtime_options.update(runtime_options)
        return self

    @classmethod
    def _get_runtime_settings_with_default(cls, model_quant_type=None, verbose=False, runtime_settings_default=None, **kwargs):
        runtime_options_kwargs = kwargs.pop('runtime_options', {})
        runtime_options = RuntimeOptions(model_quant_type=model_quant_type, verbose=verbose, **runtime_options_kwargs)

        runtime_settings_default = runtime_settings_default or runtime_settings_default_module.RUNTIME_SETTINGS_DEFAULT
        runtime_settings = copy.deepcopy(runtime_settings_default)
        for k, v in kwargs.items():
            if k in runtime_settings:
                runtime_settings[k] = v
            elif (verbose is True or (isinstance(verbose, int) and verbose>=2)):
                warnings.warn(f'\nWARNING: unknown runtime option passed - please check if it is correct: {k}')
            #
        #

        runtime_settings['runtime_options'] = runtime_options
        return runtime_settings
