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

from ..settings.settings_default import SETTINGS_DEFAULT, COPY_SETTINGS_DEFAULT
from ...common import utils
from ...common import bases
from .common_ import common_base


class ModelSurgery(common_base.CommonPipelineBase):
    ARGS_DICT=SETTINGS_DEFAULT['optimize']
    COPY_ARGS=COPY_SETTINGS_DEFAULT['optimize']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _prepare(self):
        super()._prepare()
        os.makedirs(self.run_dir, exist_ok=True)

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

        output_path = self.model_folder

        os.makedirs(self.run_dir, exist_ok=True)
        # os.makedirs(self.artifacts_folder, exist_ok=True)
        os.makedirs(self.model_folder, exist_ok=True)
        os.makedirs(output_path, exist_ok=True)

        config_path = os.path.dirname(common_kwargs['config_path']) if common_kwargs['config_path'] else None
        self.download_file(self.model_source, model_folder=self.model_folder, source_dir=config_path)

        output_model = os.path.join(output_path, os.path.basename(self.model_path))
        self._run_func(self.settings, self.model_path, output_model, **optimize_kwargs)

    @classmethod
    def _run_func(cls, settings, model_source, model_path, optimize_model=True, **kwargs):
        try:
            shutil.copy2(model_source, model_path)
        except shutil.SameFileError:
            pass
        #

        kwargs = copy.deepcopy(kwargs)

        input_optimization = settings['session'].get('input_optimization', False)
        model_ext = os.path.splitext(model_path)[1].lower()

        if model_ext == '.onnx':
            # input_optimization is set, the input_mean and input_scale are added inside the model
            if input_optimization:
                if not isinstance(kwargs.get('add_input_normalization', None), dict):
                    input_mean = settings['session'].get('input_mean', None)
                    input_scale = settings['session'].get('input_scale', None)
                    if input_mean and input_scale:
                        kwargs['add_input_normalization'] = dict()
                        kwargs['add_input_normalization']['input_mean'] = input_mean
                        kwargs['add_input_normalization']['input_scale'] = input_scale
                    #
                #
            #
            if not optimize_model:
                # optimize_model is false, but shape_inference and input_optimization may still be required
                from osrt_model_tools.onnx_tools import tidl_onnx_model_optimizer
                custom_optimizers = {
                    'shape_inference_mode': kwargs.get('shape_inference_mode', 'pre'), 
                    'simplify_mode': kwargs.get('simplify_mode', None),
                    'add_input_normalization': kwargs.get('add_input_normalization', False)
                }
                tidl_onnx_model_optimizer.optimize(model_path, model_path, custom_optimizers=custom_optimizers)
            else:
                if isinstance(optimize_model, dict):
                    kwargs.update(optimize_model)
                #
                from osrt_model_tools.onnx_tools import tidl_onnx_model_optimizer
                tidl_onnx_model_optimizer.optimize(model_path, model_path, **kwargs)
            #
        elif model_ext == '.tflite':
            if input_optimization:
                input_mean = settings['session'].get('input_mean', None)
                input_scale = settings['session'].get('input_scale', None)
                if input_mean and input_scale:
                    from osrt_model_tools.tflite_tools import tflite_model_opt
                    tflite_model_opt.tidlTfliteModelOptimize(model_path, model_path, scaleList=input_scale, meanList=input_mean)
                #
            #
        else:
            raise RuntimeError(f'ERROR: optimization for supported for model format: {model_ext}')
        #
        return
