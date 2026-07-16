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

        from ....modelinspector.data_extractor import main as gen_json

        inspector_base_path = os.path.join(self.run_dir, 'inspector')
        output_json_path = os.path.join(inspector_base_path, 'modelinspector.json')
        os.makedirs(inspector_base_path, exist_ok=True)

        common_kwargs = self.settings['common']
        extract_activations = common_kwargs.get('act_data', True)
        try:
            gen_json(self.run_dir, output_json_path, extract_activations)
            print(f'INFO: Model Inspector - JSON generation successful')
        except Exception as e:
            print(f'INFO: Model Inspector - JSON generation skipped due to missing compile artifacts: {e}')


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
        from .... import modelinspector
        from ....modelinspector.html_generator import main as gen_html

        inspector_base_path = os.path.join(self.run_dir, 'inspector')
        output_json_path = os.path.join(inspector_base_path, 'modelinspector.json')
        output_html_path = os.path.join(inspector_base_path, 'modelinspector.html')
        template_file = os.path.join(os.path.dirname(modelinspector.__file__), 'template.html')

        if not os.path.exists(output_json_path):
            print(f'INFO: Model Inspector JSON not found, skipping HTML generation')
            return

        # Activation data is embedded in the JSON (no separate activations file needed)
        try:
            gen_html(output_json_path, template_file, output_html_path)
            print(f'INFO: Model Inspector - HTML generation successful')
        except Exception as e:
            print(f'INFO: Model Inspector - HTML generation skipped due to missing compile artifacts: {e}')



class UpdateModelInspectorEVMPerfJSON(CompileModelBase):
    """Update inspector JSON with EVM hardware performance data.

    Only updates JSON — HTML generation is handled by GenerateModelInspectorHTML.
    - On EVM : reads /tmp/tidl_trace_subgraph_<N>_perf.csv (real hardware).
    - On PC  : skipped — compile already populated the JSON.
    """
    ARGS_DICT = SETTINGS_DEFAULT['analyze']
    COPY_ARGS = COPY_SETTINGS_DEFAULT['analyze']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _prepare(self):
        super()._prepare()

    def info(self):
        print(f'INFO: Model Inspector EVM Perf JSON - {__file__}')

    def _run(self):
        try:
            from ....rtwrapper.core import presets
            target_machine = self.settings.get('session', {}).get('target_machine')
            is_evm = (target_machine == presets.TargetMachineType.TARGET_MACHINE_EVM)
        except Exception:
            is_evm = False

        if not is_evm:
            return  # PC: compile already populated JSON; nothing to update here

        inspector_dir = os.path.join(self.run_dir, 'inspector')
        json_path = os.path.join(inspector_dir, 'modelinspector.json')

        if not os.path.exists(json_path):
            print(f'INFO: Model Inspector JSON not found at {json_path}, skipping EVM update')
            return

        # 1 — EVM perf: /tmp/tidl_trace_subgraph_<N>_perf.csv
        print('INFO: Model Inspector - Updating JSON with EVM hardware data')
        from ....modelinspector.data_extractor import update_with_evm_perf
        try:
            update_with_evm_perf(json_path)
        except Exception as e:
            print(f'INFO: Model Inspector - EVM perf update failed: {e}')



class UpdateModelInspectorEVMAccuracyJSON(CompileModelBase):
    """Update inspector JSON with EVM accuracy data from result.yaml.

    Only updates JSON — HTML generation is handled by GenerateModelInspectorHTML.
    Only runs on EVM after evaluate. On PC result.yaml reflects simulation — skipped.
    """
    ARGS_DICT = SETTINGS_DEFAULT['analyze']
    COPY_ARGS = COPY_SETTINGS_DEFAULT['analyze']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _prepare(self):
        super()._prepare()

    def info(self):
        print(f'INFO: Model Inspector EVM Accuracy - {__file__}')

    def _run(self):
        try:
            from ....rtwrapper.core import presets
            target_machine = self.settings.get('session', {}).get('target_machine')
            is_evm = (target_machine == presets.TargetMachineType.TARGET_MACHINE_EVM)
        except Exception:
            is_evm = False

        if not is_evm:
            return  # Ignore result.yaml on PC

        inspector_dir = os.path.join(self.run_dir, 'inspector')
        json_path = os.path.join(inspector_dir, 'modelinspector.json')

        if not os.path.exists(json_path):
            print(f'INFO: Model Inspector JSON not found, skipping accuracy update')
            return

        from ....modelinspector.data_extractor import load_accuracy_from_result_yaml
        try:
            accuracy = load_accuracy_from_result_yaml(self.run_dir)
            if accuracy:
                with open(json_path, 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
                meta = data.setdefault('metadata', {})
                for key, val in accuracy.items():
                    if not key.startswith('_'):
                        meta[key] = val
                meta.pop('evm_accuracy', None)
                with open(json_path, 'w', encoding='utf-8') as fh:
                    json.dump(data, fh, indent=2)
                print('INFO: Model Inspector - Accuracy data written to JSON')
        except Exception as e:
            print(f'INFO: Model Inspector - Accuracy update failed: {e}')
