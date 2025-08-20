#!/bin/bash

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


# set target device
export TARGET_DEVICE="AM68A"

# add environment settings as needed
source ./set_env.sh


##################################################################
# Example 1: compile by directly using a model path
# if data_path is not specified, this will use random inputs and it may not be good for accuracy.
# also there are several paameters for which defaults are assumed - it may not be perfect
# to understand the options that can be specified, use: tidlrunner-cli compile --help 
#----------------------------------------------------------------
# Example 1.1 - compile
tidlrunner-cli compile --model_path ./data/models/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv.onnx

##################################################################
# Example 1.2 - infer
# tidlrunner-cli infer --model_path ./data/models/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv.onnx

##################################################################
# Example 1.3 - compile+infer in a single command
# tidlrunner-cli compile+infer --model_path ./data/models/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv.onnx


##################################################################
# Exampe 2: compile+infer using a config file
#----------------------------------------------------------------
# tidlrunner-cli compile+infer --config_path ./data/models/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv_config.yaml


##################################################################
# Example 3: compile+infer using a wrapper configs file that aggregates other config files
#----------------------------------------------------------------
# tidlrunner-cli compile+infer --config_path ./data/models/configs.yaml


##################################################################
# Example 4: compile and evaluate accuracy using aggregate configs file
#----------------------------------------------------------------
# tidlrunner-cli compile+accuracy --config_path ./data/models/configs.yaml


##################################################################
# Example 5: analyze a model using a config file
#----------------------------------------------------------------
# tidlrunner-cli analyze --config_path ./data/models/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv_config.yaml

