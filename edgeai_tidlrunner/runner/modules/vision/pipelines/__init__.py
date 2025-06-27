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


from .compile_.import_model import ImportModelPipeline
from .compile_.infer_model import InferModelPipeline
from .compile_.compile_model import CompileModelPipeline
from .compile_.infer_analyze import InferAnalyzePipeline
from .compile_.infer_accuracy import InferAccuracyPipeline

from .optimize_.optimize_model import OptimizeModelPipeline
from .optimize_.optimize_model_gui import OptimizeModelGUIPipeline

from .utils_.split_model import SplitModelPipeline


command_module_name_dict_base = {
    'import_model':'ImportModelPipeline',
    'infer_model':'InferModelPipeline',

    'compile_model':'CompileModelPipeline',
    'infer_analyze':'InferAnalyzePipeline',
    'infer_accuracy':'InferAccuracyPipeline',
}

command_module_name_dict_ext = {
    'optimize_model':'OptimizeModelPipeline',
    'optimize_model_gui':'OptimizeModelGuiPipeline',
    'split_model':'SplitModelPipeline',
}

command_module_name_dict = command_module_name_dict_base | command_module_name_dict_ext
command_choices = list(command_module_name_dict.keys())
command_choices = list(set(command_choices))
# combined commands
command_choices += ['[import_model,infer_model]', '[compile_model,infer_model]', '[compile_model,infer_analyze]', '[compile_model,infer_accuracy]']


def get_command_choices():
    return command_choices + ['run_config']
