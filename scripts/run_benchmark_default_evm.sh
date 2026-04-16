#!/bin/bash

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


##################################################################
# Note - this is not a complete test script, but a sample to show how to run the runner benchmark for EVM. 
# If a model inference crashes, leaving the EVM in a bad state, you may need to reboot the EVM before running the next model inference - this script currently does not include that feature. 
# You can also run the runner benchmark for each model config separately, instead of using the aggregate configs file, which will allow you to easily identify which model inference is crashing and causing the EVM to be in a bad state. 
# Please refer to the documentation for more details on how to use the runner benchmark and the various options available.

##################################################################
# for convenience, setting TARGET_DEVICE env variable to be used below - this is not needed.
TARGET_DEVICE="AM62A"


##################################################################
# evaluate accuracy using aggregate configs file
#----------------------------------------------------------------
tidlrunner-cli evaluate --config_path ./data/models/configs.yaml --target_device ${TARGET_DEVICE} --parallel_processes 1
