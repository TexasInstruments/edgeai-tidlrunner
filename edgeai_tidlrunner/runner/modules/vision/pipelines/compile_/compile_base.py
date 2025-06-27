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
import ast

from ...... import rtwrapper
from ......rtwrapper.core import presets
from ......rtwrapper.options import attr_dict
from ..... import bases
from ... import blocks
from ..... import utils
from ...settings.settings_default import SETTINGS_DEFAULT, COPY_SETTINGS_DEFAULT


class CompileModelPipelineBase(bases.PipelineBase):
    args_dict=SETTINGS_DEFAULT['import_model']
    copy_args=COPY_SETTINGS_DEFAULT['import_model']
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataloader = None
        self.preprocess = None
        self.session = None
        self.postprocess = None
        self.run_data = None

        session_kwargs = self.settings[self.session_prefix]
        runtime_settings = session_kwargs['runtime_settings']
        runtime_options = runtime_settings['runtime_options']

        if not os.environ.get('TIDL_TOOLS_PATH', None):
            target_device = runtime_settings['target_device']
            tools_base_path = os.path.abspath(os.path.normpath(os.path.join(os.path.dirname(rtwrapper.__file__), '..', 'tools')))
            tools_bin_path = os.path.join(tools_base_path, target_device, 'tidl_tools')
            os.environ['TIDL_TOOLS_PATH'] = tools_bin_path
            os.environ['LD_LIBRARY_PATH'] = f":{tools_bin_path}:{os.environ.get('LD_LIBRARY_PATH','')}"

        if not runtime_options.get('tidl_tools_path', None):
            runtime_options['tidl_tools_path'] = os.environ['TIDL_TOOLS_PATH']

        assert runtime_options['tidl_tools_path'] == os.environ['TIDL_TOOLS_PATH'], \
            f"path mismatch: {runtime_options['tidl_tools_path']}, {os.environ['TIDL_TOOLS_PATH']}"

        if not runtime_options.get('artifacts_folder', None):
            runtime_options['artifacts_folder'] = self.artifacts_folder

    def download_file(self, model_source, model_folder, source_dir=None):
        is_web_link = model_source.startswith('http')
        is_simple_path = os.sep not in os.path.normpath(model_source)
        if is_web_link:
            utils.download_file(model_source, model_folder)
        else:
            model_source = os.path.join(source_dir, model_source) if source_dir and is_simple_path else model_source
            if not os.path.exists(model_source) and os.path.exists(model_source + '.link'):
                model_download_folder = os.path.dirname(model_source)
                utils.download_file(model_source, model_download_folder)
            #
            shutil.copy2(model_source, model_folder)
        #

    def _set_default_args(self, **kwargs):
        kwargs_cmd = super()._set_default_args(**kwargs)
        model_path = kwargs_cmd['session.model_path']
        model_ext = os.path.splitext(model_path)[1] if model_path else None
        if kwargs_cmd.get('preprocess.data_layout',None) is None:
            data_layout_mapping = {
                '.onnx': presets.DataLayoutType.NCHW,
                '.tflite': presets.DataLayoutType.NHWC,
            }
            data_layout = data_layout_mapping.get(model_ext, None)
            kwargs_cmd['preprocess.data_layout'] = data_layout
        #
        if kwargs_cmd.get('session.name',None) is None:
            session_name_mapping = {
                '.onnx': presets.RuntimeType.RUNTIME_TYPE_ONNXRT,
                '.tflite': presets.RuntimeType.RUNTIME_TYPE_TFLITERT,
            }
            session_name = session_name_mapping.get(model_ext, None)
            kwargs_cmd['session.name'] = session_name
        #
        return kwargs_cmd

    def _upgrade_kwargs(self, **kwargs):
        if 'session' in kwargs:
            if 'runtime_options' in kwargs['session']:
                runtime_options = kwargs['session'].pop('runtime_options', {})
                runtime_settings = kwargs['session'].get('runtime_settings', {})
                runtime_settings.get('runtime_options', {}).update(runtime_options)
            #
            if 'target_device' in kwargs['session']:
                target_device = kwargs['session'].pop('target_device')
                runtime_settings = kwargs['session'].get('runtime_settings', {})
                runtime_settings['target_device'] = target_device
            #
        #
        return kwargs

    def info(self):
        print(f'INFO: Model compile base - {__file__}')

    def run(self):
        return None