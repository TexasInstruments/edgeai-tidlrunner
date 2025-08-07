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


from .compile_.infer import InferModel
from .compile_.compile import CompileModel
from .compile_.accuracy import InferAccuracy
from .compile_.analyze import *

from .optimize_.optimize import OptimizeModel

# from .optimize_.optimize_gui import OptimizeModelGUI
# from .utils_.split_model import SplitModel


command_module_name_dict_base = {
    'compile':'CompileModel',
    'infer':'InferModel',
    'accuracy':'InferAccuracy',
    #'analyze': ['CompileAnalyzeNoTIDL', 'InferAnalyzeNoTIDL', 'CompileAnalyzeTIDL32', 'InferAnalyzeTIDL32', 'CompileAnalyzeTIDL16', 'InferAnalyzeTIDL16', 'CompileAnalyzeTIDL8', 'InferAnalyzeTIDL8', 'InferAnalyzeFinal'],
    'analyze': ['CompileAnalyzeNoTIDL', 'InferAnalyzeNoTIDL', 'CompileAnalyzeTIDL32', 'InferAnalyzeTIDL32', 'InferAnalyzeFinal'],
    'compile+infer': ['CompileModel', 'InferModel'],
    'compile+accuracy': ['CompileModel', 'InferAccuracy'],
}

command_module_name_dict_ext = {
    'optimize':'OptimizeModel',
    #'optimize_model_gui':'OptimizeModelGUI',
    #'split_model':'SplitModel',
}

command_module_name_dict = command_module_name_dict_base | command_module_name_dict_ext
command_choices = list(command_module_name_dict.keys())
command_choices = list(set(command_choices))


def get_command_choices():
    return command_choices
