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
from . import presets


# The following are options for model compilation with TIDL OSRT
# More information is here: https://github.com/TexasInstruments/edgeai-tidl-tools/blob/master/examples/osrt_python/README.md

RUNTIME_OPTIONS_DEFAULT = {
    'platform': presets.TIDL_PLATFORM,

    'version': presets.TIDL_VERSION_STR,

    # to be populated
    'tidl_tools_path': None,

    # to be populated
    'artifacts_folder': None,

    # to be populated in _create_interpreter
    'import': None,

    # This option specifies number of bits for TIDL tensor and weights
    # 8,16 (32 - only for PC inference, not device)
    'tensor_bits': presets.TensorBits.TENSOR_BITS_8,

    # This option specifies level of accuracy desired - specifying higher accuracy_level gives improved accuracy, but may take more time for model compilation
    # 0 - basic calibration,
    # 1 - higher accuracy (advanced bias calibration),
    # 9 - user defined
    'accuracy_level': presets.AccurcyLevel.ACCURACY_LEVEL_ADVANCED,

    # This options enables increasing levels of debug prints and TIDL layer traces
    # 0 - no debug,
    # 1 - Level 1 debug prints
    # 2 - Level 2 debug prints
    # 3 - Level 1 debug prints, fixed point layer traces
    # 4 (experimental) - Level 1 debug prints, Fixed point and floating point traces
    # 5 (experimental) - Level 2 debug prints, Fixed point and floating point traces
    # 6 - Level 3 debug prints
    # Default value: 0
    'debug_level': presets.DebugLevel.DEBUG_LEVEL_DISABLE,

    # This option specifies the feature/mode to be used for inference. This option must be specified during compilation and impacts the artifacts generated
    # 0 (TIDL_inferenceModeDefault)
    # 1 (TIDL_inferenceModeHighThroughput)
    # 2 (TIDL_inferenceModeLowLatency)
    # Default value: 0
    'advanced_options:inference_mode': 0,

    # This option enables performance optimization for high resolution models
    # 0 - disable,
    # 1 enable
    'advanced_options:high_resolution_optimization': 0,

    # fold initial batchnorm into convolution
    # 0 - disable
    # 1 - enable
    'advanced_options:pre_batchnorm_fold': 1,

    # This option specifies type of quantization style to be used for model quantization
    # 0 - non-power-of-2,
    # 1 - power-of-2
    # 3 - TF-Lite pre-quantized model
    # 4 - Asymmetric, Per-channel Quantization (not supported in TDA4VM)
    # Defaults in this code - TDA4VM: 1, Other SOCs: 4
    # 'advanced_options:quantization_scale_type': 4

    # This option specifies number of frames to be used for calibration - min 10 frames recommended
    # Any - min 10 frames recommended
    'advanced_options:calibration_frames': 12,

    # This option specifies number of bias calibration iterations
    # Any - min 10 recommended
    'advanced_options:calibration_iterations': 12,

    # further quantization/calibration options - these take effect only if the accuracy_level in basic options is set to 9
    # if bias_clipping is set to 0 (default), weight scale will be adjusted to avoid bias_clipping
    # if bias_clipping is set to 1, weight scale is computed solely based on weight range.
    # bias_clipping should only affect the mode where the bias is clipped to 16bits (default in TDA4VM).
    'advanced_options:activation_clipping': 1,
    'advanced_options:weight_clipping': 1,
    #'advanced_options:bias_clipping': 1,

    # adjust bias values in iterations to improve accuracy
    # 0 for disable
    # 1 for enable
    'advanced_options:bias_calibration': 1,

    # mixed precision options - this is just a placeholder
    # output/params names need to be specified according to a particular model
    'advanced_options:output_feature_16bit_names_list':'',
    'advanced_options:params_16bit_names_list':'',

    # optimize data conversion options by moving them from arm to c7x
    # 0 - disable,
    # 1 - Input format conversion
    # 2 - output format conversion
    # 3 - Input and output format conversion
    'advanced_options:add_data_convert_ops': presets.DataConvertOps.DATA_CONVERT_OPS_INPUT_OUTPUT,

    # max number of nodes in a subgraph (default is 750)
    #'advanced_options:max_num_subgraph_nodes': 2000,

    # 0 - NC_PERFSIM_DISABLE
    # 1 - NC_PERFSIM_ENABLE
    'ti_internal_nc_flag': presets.NCPerfsimFlag.NC_PERFSIM_ENABLE,

    # In case you are using firmware released as part of processor SDK RTOS, this field can be ignored. If you are using TIDL firmware release with a new patch release of the same "release line" then it is essential to use c7x_firmware_version explicitly
    # None - c7x_firmware_version not used if this is None
    # String - for example: 11_01_00_00
    # Note: None or '' are not supported values
    # Check with edgeai-tidl-tools documentation to understand the latest supported firmware versions.
    # Note: When compiling models custom firmware version, firmware update may be needed in the SDK to run the model. '
    # See the SDK version compatibiltiy table: https://github.com/TexasInstruments/edgeai-tidl-tools/blob/master/docs/version_compatibility_table.md
    #'advanced_options:c7x_firmware_version': ??,

    # object detection settings
    'object_detection:meta_layers_names_list': '',
    'object_detection:meta_arch_type': None,
    'object_detection:nms_threshold': 0.45,
    'object_detection:confidence_threshold': 0.3,
    'object_detection:top_k': 200,
    'object_detection:keep_top_k': 200,
}

