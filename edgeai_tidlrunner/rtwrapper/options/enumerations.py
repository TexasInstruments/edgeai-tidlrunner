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


class DebugLevel:
    DEBUG_LEVEL_DISABLE = 0 # 0 - no debug,
    DEBUG_LEVEL_1 = 1 # 1 - Level 1 debug prints
    DEBUG_LEVEL_2 = 2 # 2 - Level 2 debug prints
    DEBUG_LEVEL_3 = 3 # 3 - Level 1 debug prints, fixed point layer traces
    DEBUG_LEVEL_4 = 4 # 4 (experimental) - Level 1 debug prints, Fixed point and floating point traces
    DEBUG_LEVEL_5 = 5 # 5 (experimental) - Level 2 debug prints, Fixed point and floating point traces
    DEBUG_LEVEL_6 = 6 # 6 - Level 3 debug prints


class QUANTScaleType:
    # 0 (non-power of 2, default)
    QUANT_SCALE_TYPE_NP2 = 0

    # 1 (power of 2 quantization - recommended for older devices such as TDA4VM - needed for p2 qat models generated using older qat tool - edgeai_torchmodelopt.quantization.v1)
    QUANT_SCALE_TYPE_P2 = 1

    # 2 UNUSED
    QUANT_SCALE_TYPE_UNUSED = 2

    # these are not supported in TDA4VM, but for other SoCs, these are the recommended modes
    # 3 (prequantized tflite model - non-power of 2, supported in newer devices). for prequantized qdq onnx model, this is not the option. use the flag advanced_options:prequantized_model instead
    QUANT_SCALE_TYPE_PREQUANT_TFLITE = 3

    # per-channel quantization is highy recommended if this feature is supported in hardware
    # 4 per-channel quantization - supported in SoCs other than TDA4VM
    QUANT_SCALE_TYPE_NP2_PERCHAN = 4


class TensorBits:
    TENSOR_BITS_8 = 8
    TENSOR_BITS_16 = 16
    TENSOR_BITS_32 = 32


class AccurcyLevel:
    ACCURACY_LEVEL_BASIC = 0
    ACCURACY_LEVEL_ADVANCED = 1


class DataConvertOps:
    DATA_CONVERT_OPS_DISABLE = 0 #0 - disable,
    DATA_CONVERT_OPS_INPUT = 1 #1 - Input format conversion
    DATA_CONVERT_OPS_OUTPUT = 2 #2 - output format conversion
    DATA_CONVERT_OPS_INPUT_OUTPUT = 3 #3 - Input and output format conversion


class NCPerfsimFlag:
    NC_PERFSIM_DISABLE = 1601
    NC_PERFSIM_ENABLE = 83886080


class PreQuantizedModelType:
    PREQUANTIZED_MODEL_TYPE_NONE = None
    PREQUANTIZED_MODEL_TYPE_CLIP = 0
    PREQUANTIZED_MODEL_TYPE_QDQ = 1


# target devices/socs supported.
class TargetDeviceType:
    TARGET_DEVICE_TDA4VM = 'TDA4VM'
    TARGET_DEVICE_AM62A = 'AM62A'
    TARGET_DEVICE_AM67A = 'AM67A'
    TARGET_DEVICE_AM68A = 'AM68A'
    TARGET_DEVICE_AM69A = 'AM69A'
    TARGET_DEVICE_AM62 = 'AM62'


# compilation can only be run in PC as of now, but inference can be run in both PC and EVM
# whether running in PC/Host Emulation or really running in EVM/device:
class TargetMachineType:
    TARGET_MACHINE_PC_EMULATION = 'pc'
    TARGET_MACHINE_EVM = 'evm'


# data layout constants
class DataLayoutType:
    NCHW = 'NCHW'
    NHWC = 'NHWC'


# supported model types
class ModelType:
    MODEL_TYPE_ONNX = 'onnx'
    MODEL_TYPE_TFLITE = 'tflite'
    MODEL_TYPE_MXNET = 'mxnet'


class RuntimeType:
    RUNTIME_TYPE_ONNXRT = 'onnxrt'
    RUNTIME_TYPE_TFLITERT = 'tflitert'
    RUNTIME_TYPE_TVMDLR = 'tvmdlr'
    RUNTIME_TYPE_TVMRT = 'tvmrt'  # TODO - not supported yet


class CalibrationIterationsFactor:
    CALIBRATION_ITERATIONS_FACTOR_1X = 1.0
    CALIBRATION_ITERATIONS_FACTOR_NX = 2.0


class ModelQuantType:
    MODEL_QUANT_TYPE_DEFAULT = None
    MODEL_QUANT_TYPE_QUANT_CLIP_P2 = "QUANT_CLIP_P2"
    MODEL_QUANT_TYPE_QUANT_TFLITE = "QUANT_TFLITE"
    MODEL_QUANT_TYPE_QUANT_QDQ = "QUANT_QDQ"
    MODEL_QUANT_TYPE_QUANT_QDQ_P2 = "QUANT_QDQ_P2"


class TIDLDetectionMetaArchType:
    TIDL_DETECTION_META_ARCH_TYPE_SSD_TFLITE = 1
    TIDL_DETECTION_META_ARCH_TYPE_SSD_ONNX = 3
    TIDL_DETECTION_META_ARCH_TYPE_YOLOV3 = 4
    TIDL_DETECTION_META_ARCH_TYPE_RETINANET_EFFDET = 5
    TIDL_DETECTION_META_ARCH_TYPE_YOLOX_YOLOV5_YOLOV7 = 6
    TIDL_DETECTION_META_ARCH_TYPE_YOLOV8_RTMDET = 8
    TIDL_DETECTION_META_ARCH_TYPE_3DOD_POINTPILLARS_FASTBEV = 7
    TIDL_DETECTION_META_ARCH_TYPE_3DOD_BEVFORMER = 10


# https://onnxruntime.ai/docs/api/c/group___global.html
class GraphOptimizationLevel:
  ORT_DISABLE_ALL = 0
  ORT_ENABLE_BASIC = 1
  ORT_ENABLE_EXTENDED = 2
  ORT_ENABLE_LAYOUT = 3
  ORT_ENABLE_ALL = 99


# other common constants
MILLI_CONST = 1e3 # multiplication by 1000 is to convert seconds to milliseconds
MEGA_CONST = 1e6  # convert raw data to mega : example bytes to mega bytes (MB)
GIGA_CONST = 1e9
ULTRA_CONST = 1e6

# frequency of the core C7x/MMA processor that accelerates Deep Learning Tasks
# this constant is used to convert cycles to time : time = cycles / DSP_FREQ
# in future, this will need to be device dependent
DSP_FREQ = 1e9

