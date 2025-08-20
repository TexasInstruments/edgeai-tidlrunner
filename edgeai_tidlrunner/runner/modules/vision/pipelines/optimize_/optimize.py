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


import os
import sys
import shutil

from ...settings.settings_default import SETTINGS_DEFAULT, COPY_SETTINGS_DEFAULT
from ..... import utils
from ..... import bases
from ..common_ import common_base


class OptimizeModel(common_base.CommonPipelineBase):
    ARGS_DICT=SETTINGS_DEFAULT['optimize']
    COPY_ARGS=COPY_SETTINGS_DEFAULT['optimize']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _prepare(self):
        super()._prepare()
                    
    def info():
        print(f'INFO: Model optimize - {__file__}')

    def _run(self):
        print(f'INFO: starting model optimize with parameters: {self.kwargs}')

        common_kwargs = self.settings[self.common_prefix]
        optimize_kwargs = common_kwargs.get('optimize', {})

        if os.path.exists(self.run_dir):
            print(f'INFO: clearing run_dir folder before compile: {self.run_dir}')
            shutil.rmtree(self.run_dir, ignore_errors=True)
        #

        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(self.artifacts_folder, exist_ok=True)
        os.makedirs(self.model_folder, exist_ok=True)

        config_path = os.path.dirname(common_kwargs['config_path']) if common_kwargs['config_path'] else None
        self.download_file(self.model_source, model_folder=self.model_folder, source_dir=config_path)

        self._run_func(self.model_path, self.model_path, **optimize_kwargs)

    @classmethod
    def _run_func(cls, model_source, model_path, simplify_model=True, shape_inference=True, optimize_model=True, **kwargs):
        if simplify_model:
            import onnxsim
            onnxsim.simplify(model_path)
        #
        if optimize_model:
            from osrt_model_tools.onnx_tools import tidl_onnx_model_optimizer
            tidl_onnx_model_optimizer.optimize(model_source, model_path)
        #
        if shape_inference:
            import onnx
            onnx.shape_inference.infer_shapes_path(model_source, model_path)
        #
