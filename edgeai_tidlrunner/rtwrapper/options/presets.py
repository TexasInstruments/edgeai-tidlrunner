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


from .enumerations import *


# TIDL platform to be used for compilation
# This also depends on the tidl_tools that is installed.
TIDL_PLATFORM = "J7"


# TIDL version that is supported by default - however this is not the only version that is supported.
# This version actually depends on tidl_tools that is being used - so what is populated here is just for guidance.
TIDL_VERSION = (11,1)
TIDL_VERSION_STR = '11.1'


MODEL_TYPES = [ModelType.MODEL_TYPE_ONNX, ModelType.MODEL_TYPE_TFLITE, ModelType.MODEL_TYPE_MXNET]


# runtime_options preferred - may not blindly apply for qat models
TARGET_DEVICE_SETTINGS_PRESETS = {
    TargetDeviceType.TARGET_DEVICE_TDA4VM : {
        # TDA4VM does not support the per-channel asymmetric quantization - p2 can be used as default
        'runtime_options': {'advanced_options:quantization_scale_type': QUANTScaleType.QUANT_SCALE_TYPE_P2},
        # we may need more calibration images and iterations
        'calibration_iterations_factor': CalibrationIterationsFactor.CALIBRATION_ITERATIONS_FACTOR_NX
    },
    TargetDeviceType.TARGET_DEVICE_AM62A : {
        'runtime_options': {'advanced_options:quantization_scale_type': QUANTScaleType.QUANT_SCALE_TYPE_NP2_PERCHAN},
    },
    TargetDeviceType.TARGET_DEVICE_AM67A: {
        'runtime_options': {'advanced_options:quantization_scale_type': QUANTScaleType.QUANT_SCALE_TYPE_NP2_PERCHAN},
    },
    TargetDeviceType.TARGET_DEVICE_AM68A : {
        'runtime_options': {'advanced_options:quantization_scale_type': QUANTScaleType.QUANT_SCALE_TYPE_NP2_PERCHAN},
    },
    TargetDeviceType.TARGET_DEVICE_AM69A : {
        'runtime_options': {'advanced_options:quantization_scale_type': QUANTScaleType.QUANT_SCALE_TYPE_NP2_PERCHAN},
    },
    TargetDeviceType.TARGET_DEVICE_AM62 : {
        'runtime_options': {},
        'tidl_offload': False,
    },
}


# to handle speciall case for runtime_options['object_detection:xx']
TIDL_DETECTION_META_ARCH_TYPE_SSD_LIST = [
    TIDLDetectionMetaArchType.TIDL_DETECTION_META_ARCH_TYPE_SSD_TFLITE,
    TIDLDetectionMetaArchType.TIDL_DETECTION_META_ARCH_TYPE_SSD_ONNX,
]
