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
import yaml

from ..... import utils
from ..... import bases
from ...settings.settings_default import SETTINGS_DEFAULT, COPY_SETTINGS_DEFAULT
from ...settings import constants


class CommonPipelineBase(bases.PipelineBase):
    ARGS_DICT = SETTINGS_DEFAULT['basic']
    COPY_ARGS = COPY_SETTINGS_DEFAULT['basic']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
             
    def _prepare(self):
        pass

    def _get_file_path(self, model_source, source_dir=None):
        is_web_link = model_source.startswith('http')
        is_simple_path = os.sep not in os.path.normpath(model_source)
        if is_web_link:
            return model_source
        else:
            model_source = os.path.join(source_dir, model_source) if source_dir and is_simple_path else model_source
            return model_source

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
        print(f'INFO: Common pipeline base - {__file__}')

    def _run(self):
        return None

    def _write_params(self, filename):
        params = utils.pretty_object(self.settings)
        param_file = os.path.join(self.run_dir, filename)
        with open(param_file, 'w') as fp:
            yaml.dump(params, fp)
        #