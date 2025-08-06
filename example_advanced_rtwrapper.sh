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

##################################################################
# Info: run the cli script
# Environment variables such as TARGET_DEVICE, TARGET_MACHINE can be set in the commandline
# Other arguments can be passed additionally.
# Example 1:
# ./run_test_cli.sh
# Example 2:
# TARGET_DEVICE=AM68A TARGET_MACHINE=pc ./run_test_cli.sh --target_device AM68A

# set target device
export TARGET_DEVICE="AM68A"

# add environemnt settings as needed
source ./set_env.sh

python3 ./examples/vision/scripts/example_advanced_rtwrapper.py "compile" \
  --target_device ${TARGET_DEVICE} \
  --model_path ./data/models/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv.onnx \
  --data_path ./data/datasets/vision/imagenetv2c/val

python3 ./examples/vision/scripts/example_advanced_rtwrapper.py "infer" \
  --target_device ${TARGET_DEVICE} \
  --model_path ./data/models/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv.onnx \
  --data_path ./data/datasets/vision/imagenetv2c/val
