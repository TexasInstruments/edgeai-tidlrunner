#!/usr/bin/env bash

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


######################################################################
pip3 install -e ./[pc] --verbose

# download-tidlrunner-tools is a script that defined in and installed via ./pyproject.toml
# download and install packages - this invokes: python3 edgeai_tidlrunner/download.py
# pip3 install --no-input onnx-graphsurgeon==0.3.26 --extra-index-url https://pypi.ngc.nvidia.com
# pip3 install --no-input osrt_model_tools @ git+https://github.com/TexasInstruments/edgeai-tidl-tools.git@11_00_08_00#subdirectory=osrt-model-tools
echo "Running: download-tidlrunner-tools..."
download-tidlrunner-tools

#######################################################################
pip3 install -e ./tools --verbose

# download-tidl-tools is a script that defined in and installed via tools/pyproject.toml
# download and install tidl-tools - this invokes: python3 tools/tidl_tools_package/download.py
echo "Running: download-tidl-tools..."
download-tidl-tools
