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

import enum
import copy
import os
import sys
import warnings

from . import attr_dict
from . import presets
from . import options_default


class RuntimeOptions(attr_dict.AttrDict):
    def __init__(self, **kwargs):
        super().__init__()
        self.runtime_options = {}
        self.update(self._get_runtime_options(**kwargs))

    def get_runtime_options(self):
        return self

    def _get_calibration_iterations(self, is_qat, calibration_iterations):
        # note that calibration_iterations has effect only if accuracy_level>0
        # so we can just set it to the max value here.
        # for more information see: get_calibration_accuracy_level()
        # Not overriding for 16b now
        return -1 if is_qat else calibration_iterations

    def _get_calibration_accuracy_level(self, is_qat):
        # For QAT models, simple calibration is sufficient, so we shall use accuracy_level=0
        #use advance calib for 16b too
        return 0 if is_qat else 1

    def _get_runtime_options_base(self, model_quant_type=None, is_qat=False,
            calibration_iterations_factor=None, det_options=None, ext_options=None, **kwargs):
        '''
        example usage for min_options and max_options to set the limit
            settings.runtime_options_onnx_np2(max_options={'advanced_options:calibration_frames':25, 'advanced_options:calibration_iterations':25})
             similarly min_options can be used to set lower limit
             currently only calibration_frames and calibration_iterations are handled in this function.

        model_quant_type: model_quant_type
        det_options: True for detection models, False for other models. Can also be a dictionary.
        '''

        # this is the default runtime_options defined above
        runtime_options = self._get_runtime_options_with_default(
            model_quant_type=model_quant_type, **kwargs)

        calibration_iterations_factor = calibration_iterations_factor or \
            presets.CalibrationIterationsFactor.CALIBRATION_ITERATIONS_FACTOR_1X

        runtime_options['advanced_options:calibration_frames'] = \
                max(int(runtime_options['advanced_options:calibration_frames'] * calibration_iterations_factor), 1)

        runtime_options['advanced_options:calibration_iterations'] = \
                max(int(self._get_calibration_iterations(is_qat, runtime_options['advanced_options:calibration_iterations']) *
                calibration_iterations_factor), 1)

        # this takes care of overrides given as ext_options keyword argument
        if ext_options is not None:
            assert isinstance(ext_options, dict), \
                f'runtime_options provided via kwargs must be dict, got {type(ext_options)}'
            runtime_options.update(ext_options)
        #
        object_detection_meta_arch_type = runtime_options.get('object_detection:meta_arch_type', None)

        # for tflite models, these options are directly processed inside tidl
        # for onnx od models, od post proc options are specified in the prototxt and it is modified with these options
        # use a large top_k, keep_top_k and low confidence_threshold for accuracy measurement
        if det_options:
            # SSD models may need to have a high detection_threshold afor inference since thier runtime is sensitive to this threhold
            is_ssd = det_options == 'SSD' or (det_options is True and object_detection_meta_arch_type in presets.TIDL_DETECTION_META_ARCH_TYPE_SSD_LIST)
            detection_nms_threshold = 0.45
            detection_threshold = (0.3 if is_ssd else 0.05)
            detection_top_k = (200 if is_ssd else 500)
            detection_keep_top_k = 200
            if isinstance(det_options, dict):
                detection_nms_threshold = det_options.get('object_detection:nms_threshold', detection_nms_threshold)
                detection_threshold = det_options.get('object_detection:confidence_threshold', detection_threshold)
                detection_top_k = det_options.get('object_detection:top_k', detection_top_k)
                detection_keep_top_k = det_options.get('object_detection:keep_top_k', detection_keep_top_k)
            #
            runtime_options.update({
                'object_detection:confidence_threshold': detection_threshold
            })
            runtime_options.update({
                'object_detection:top_k': detection_top_k
            })
            runtime_options.update({
                'object_detection:nms_threshold': detection_nms_threshold
            })
            runtime_options.update({
                'object_detection:keep_top_k': detection_keep_top_k
            })
        #
        return runtime_options

    def _get_runtime_options(self, model_quant_type=None, **kwargs):
        if model_quant_type == presets.ModelQuantType.MODEL_QUANT_TYPE_QUANT_CLIP_P2:
            kwargs['is_qat'] = True
            kwargs['advanced_options:quantization_scale_type'] = presets.QUANTScaleType.QUANT_SCALE_TYPE_P2
        elif model_quant_type == presets.ModelQuantType.MODEL_QUANT_TYPE_QUANT_TFLITE:
            kwargs['is_qat'] = True
            kwargs['advanced_options:quantization_scale_type'] = presets.QUANTScaleType.QUANT_SCALE_TYPE_PREQUANT_TFLITE
            kwargs['advanced_options:prequantized_model'] = presets.PreQuantizedModelType.PREQUANTIZED_MODEL_TYPE_NONE
        elif model_quant_type == presets.ModelQuantType.MODEL_QUANT_TYPE_QUANT_QDQ:
            kwargs['is_qat'] = True
            kwargs['advanced_options:quantization_scale_type'] = presets.QUANTScaleType.QUANT_SCALE_TYPE_NP2_PERCHAN
            kwargs['advanced_options:prequantized_model'] = presets.PreQuantizedModelType.PREQUANTIZED_MODEL_TYPE_QDQ
        elif model_quant_type == presets.ModelQuantType.MODEL_QUANT_TYPE_QUANT_QDQ_P2:
            kwargs['is_qat'] = True
            kwargs['advanced_options:quantization_scale_type'] = presets.QUANTScaleType.QUANT_SCALE_TYPE_P2
            kwargs['advanced_options:prequantized_model'] = presets.PreQuantizedModelType.PREQUANTIZED_MODEL_TYPE_QDQ
        #
        return self._get_runtime_options_base(model_quant_type, **kwargs)

    @classmethod
    def _get_runtime_options_with_default(cls, model_quant_type=None, verbose=1, runtime_options_default=None, **kwargs):
        runtime_options_default = runtime_options_default or options_default.RUNTIME_OPTIONS_DEFAULT
        runtime_options = copy.deepcopy(runtime_options_default)

        for k, v in kwargs.items():
            if k in runtime_options:
                runtime_options[k] = v
            elif (verbose is True or (isinstance(verbose, int) and verbose>=2)):
                warnings.warn(f'\nWARNING: unknown runtime option passed - please check if it is correct: {k}')
            #
        #

        # additional options (firmware version)
        c7x_firmware_version = runtime_options.get('advanced_options:c7x_firmware_version', None)
        if (verbose is True or (isinstance(verbose, int) and verbose>=1)) and c7x_firmware_version is not None:
            warnings.warn(f'\nINFO: advanced_options:c7x_firmware_version passed to tidl_tools from this repo for model compilation is: {c7x_firmware_version}'
                          f'\nINFO: for potential firmware update needed in SDK to run this model, see the SDK version compatibiltiy table: '
                          f'\nINFO: https://github.com/TexasInstruments/edgeai-tidl-tools/blob/master/docs/version_compatibility_table.md')

        return runtime_options
