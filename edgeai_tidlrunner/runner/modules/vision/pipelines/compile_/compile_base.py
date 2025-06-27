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
        self.common_prefix = 'common'
        self.dataloader_prefix = 'dataloader'
        self.session_prefix = 'session'
        self.preprocess_prefix = 'preprocess'
        self.postprocess_prefix = 'postprocess'

        self.model_source = self.settings[self.session_prefix]['model_path']
        run_dir = self.settings[self.session_prefix]['run_dir']
        model_basename = os.path.basename(self.model_source)
        model_basename_wo_ext = os.path.splitext(model_basename)[0]
        self.run_dir = run_dir.replace('{model_name}', model_basename_wo_ext)
        self.model_folder = os.path.join(self.run_dir, 'model')
        self.model_path = os.path.join(self.model_folder, model_basename)
        self.settings[self.session_prefix]['model_path'] = self.model_path
        self.artifacts_folder = self.settings[self.session_prefix].get('artifactrs_folder', os.path.join(self.run_dir, 'artifacts'))
        self.settings[self.session_prefix]['artifacts_folder'] = self.artifacts_folder

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

    def info(self):
        print(f'INFO: Model compile base - {__file__}')

    def run(self):
        return None