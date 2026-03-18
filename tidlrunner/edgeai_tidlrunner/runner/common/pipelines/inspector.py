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
import copy
import shutil
import yaml
import json

from ..settings.settings_default import SETTINGS_DEFAULT, COPY_SETTINGS_DEFAULT
from .common_.common_base import CommonPipelineBase
from .common_.compile_base import CompileModelBase

from .... import modelinspector
from ....modelinspector.data_extractor import main as gen_json
from ....modelinspector.html_generator import main as gen_html


class GenerateModelInspectorJSON(CompileModelBase):
    ARGS_DICT=SETTINGS_DEFAULT['analyze']
    COPY_ARGS=COPY_SETTINGS_DEFAULT['analyze']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _prepare(self):
        super()._prepare()
        common_kwargs = self.settings['common']
        if not os.path.exists(self.run_dir):
            print(f'INFO: run_dir does not exist: {self.run_dir}')
        #

    def info(self):
        print(f'INFO: Model Inspector - {__file__}')

    def _run(self):
        print(f'INFO: Model Inspector - JSON generation')
        inspector_base_path = os.path.join(self.run_dir, 'inspector')
        output_json_path = os.path.join(inspector_base_path, 'modelinspector.json')
        os.makedirs(inspector_base_path, exist_ok=True)
        common_kwargs = self.settings['common']
        extract_activations = common_kwargs.get('act_data', True)
        gen_json(self.run_dir, output_json_path, extract_activations)


class GenerateModelInspectorHTML(CompileModelBase):
    ARGS_DICT=SETTINGS_DEFAULT['analyze']
    COPY_ARGS=COPY_SETTINGS_DEFAULT['analyze']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _prepare(self):
        super()._prepare()
        common_kwargs = self.settings['common']
        if not os.path.exists(self.run_dir):
            print(f'INFO: run_dir does not exist: {self.run_dir}')
        #

    def info(self):
        print(f'INFO: Model Inspector - {__file__}')

    def _run(self):
        print(f'INFO: Model Inspector - HTML generation')
        inspector_base_path = os.path.join(self.run_dir, 'inspector')
        output_json_path = os.path.join(inspector_base_path, 'modelinspector.json.gz')
        output_html_path = os.path.join(inspector_base_path, 'modelinspector.html')
        activations_json_path = os.path.join(inspector_base_path, 'modelinspector_activations.json.gz')
        os.makedirs(inspector_base_path, exist_ok=True)
        template_file = os.path.join(os.path.dirname(modelinspector.__file__), 'template.html')

        # Check the act_data flag from settings
        common_kwargs = self.settings['common']
        extract_activations = common_kwargs.get('act_data', True)

        # Pass activations file ONLY if flag is set AND file exists
        if extract_activations and os.path.exists(activations_json_path):
            gen_html(output_json_path, template_file, output_html_path, activations_json_path)
        else:
            gen_html(output_json_path, template_file, output_html_path)

        # Clean up intermediate JSON files — data is now embedded in HTML
        for json_file in [output_json_path, activations_json_path]:
            if os.path.exists(json_file):
                os.remove(json_file)
