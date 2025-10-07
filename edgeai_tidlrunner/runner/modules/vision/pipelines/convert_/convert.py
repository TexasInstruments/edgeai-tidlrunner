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
import copy

from ...settings.settings_default import SETTINGS_DEFAULT, COPY_SETTINGS_DEFAULT
from ..... import utils
from ..... import bases
from ..common_ import common_base


class ConvertModel(common_base.CommonPipelineBase):
    ARGS_DICT=SETTINGS_DEFAULT['convert']
    COPY_ARGS=COPY_SETTINGS_DEFAULT['convert']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _prepare(self):
        super()._prepare()
        os.makedirs(self.run_dir, exist_ok=True)

    def info():
        print(f'INFO: Model convert - {__file__}')

    def _run(self):
        print(f'INFO: starting model convert with parameters: {self.kwargs}')

        common_kwargs = self.settings[self.common_prefix]
        convert_kwargs = common_kwargs.get('convert', {})

        if os.path.exists(self.run_dir):
            print(f'INFO: clearing run_dir folder before compile: {self.run_dir}')
            shutil.rmtree(self.run_dir, ignore_errors=True)
        #

        output_path = self.model_folder

        os.makedirs(self.run_dir, exist_ok=True)
        # os.makedirs(self.artifacts_folder, exist_ok=True)
        os.makedirs(self.model_folder, exist_ok=True)
        os.makedirs(output_path, exist_ok=True)

        config_path = os.path.dirname(common_kwargs['config_path']) if common_kwargs['config_path'] else None
        self.download_file(self.model_source, model_folder=self.model_folder, source_dir=config_path)

        output_name = os.path.splitext(os.path.basename(self.model_path))[0] + '.pt'
        output_model = os.path.join(output_path, output_name)
        self._run_func(self.settings, self.model_path, output_model, **convert_kwargs)

    @classmethod
    def _run_func(cls, settings, model_source, model_path, **kwargs):
        try:
            shutil.copy2(model_source, model_path)
        except shutil.SameFileError:
            pass
        #
        kwargs = copy.deepcopy(kwargs)

        try:
            import torch
        except Exception as e:
            print(f"ERROR: torch could not be imported: {e}")
            raise
        #

        try:
            import edgeai_onnx2torchmodel
        except Exception as e:
            print(f"WARNING: failed to install edgeai_onnx2torchmodel package, error: {e}")
            install_url = "edgeai_onnx2torchmodel@git+ssh://git@bitbucket.itg.ti.com/edgeai-algo/edgeai-modeloptimization.git@2025_kunal_onnx2torch#subdirectory=onnx2torchmodel", ##"edgeai_onnx2torchmodel@git+https://github.com/TexasInstruments/edgeai-modeloptimization.git@main#subdirectory=onnx2torchmodel"
            print(f"ERROR: trying to install package from url: {install_url}")
            raise
        #

        torch_model = edgeai_onnx2torchmodel.convert(model_source, **kwargs)
        torch.save(torch_model, model_path)
        return torch_model
