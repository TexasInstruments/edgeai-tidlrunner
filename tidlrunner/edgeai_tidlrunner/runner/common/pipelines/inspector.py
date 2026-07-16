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


def _json_has_perfsim(json_path):
    """Returns True if JSON already has perfsim (performance) data in any TIDL layer."""
    try:
        with open(json_path) as f:
            d = json.load(f)
        for sg in d.get('runtime', {}).get('subgraphs', {}).values():
            for layer in sg.get('layers', []):
                if layer.get('performance') is not None:
                    return True
    except Exception:
        pass
    return False

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

        if os.path.exists(output_json_path) and _json_has_perfsim(output_json_path):
            print(f'INFO: Model Inspector JSON already has perfsim data, skipping generation')
            return

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


class UpdateModelInspectorActivations(CompileModelBase):
    """Update existing inspector JSON with activation/trace data after an analyze run."""
    ARGS_DICT=SETTINGS_DEFAULT['analyze']
    COPY_ARGS=COPY_SETTINGS_DEFAULT['analyze']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _prepare(self):
        super()._prepare()

    def info(self):
        print(f'INFO: Model Inspector - {__file__}')

    def _run(self):
        print(f'INFO: Model Inspector - Updating JSON with activation data')

        common_kwargs = self.settings['common']
        if not common_kwargs.get('act_data', True):
            print(f'INFO: Model Inspector - act_data=False, skipping activation update')
            return

        inspector_base_path = os.path.join(self.run_dir, 'inspector')
        output_json_path = os.path.join(inspector_base_path, 'modelinspector.json')

        if not os.path.exists(output_json_path):
            print(f'INFO: Model Inspector JSON not found at {output_json_path}, skipping activation update')
            return

        from ....modelinspector.data_extractor import update_with_activations
        try:
            success = update_with_activations(self.run_dir, output_json_path)
            if success:
                print(f'INFO: Model Inspector - Activation update successful')
            else:
                print(f'INFO: Model Inspector - No activation data found, JSON unchanged')
        except Exception as e:
            print(f'INFO: Model Inspector - Activation update skipped: {e}')


class UpdateModelInspectorEVMPerf(CompileModelBase):
    """Update inspector JSON with performance data and regenerate HTML.

    - On EVM : reads /tmp/tidl_trace_subgraph_<N>_perf.csv (real hardware).
               Stamps performance_source = 'evm_hardware' and locks the JSON so
               subsequent PC runs cannot overwrite the real data.
    - On PC  : skipped — compile already wrote artifacts/ CSV data into the JSON
               with performance_source = 'pc_simulation'.
    """
    ARGS_DICT = SETTINGS_DEFAULT['analyze']
    COPY_ARGS = COPY_SETTINGS_DEFAULT['analyze']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _prepare(self):
        super()._prepare()

    def info(self):
        print(f'INFO: Model Inspector EVM Perf - {__file__}')

    def _run(self):
        try:
            from ....rtwrapper.core import presets
            target_machine = self.settings.get('session', {}).get('target_machine')
            is_evm = (target_machine == presets.TargetMachineType.TARGET_MACHINE_EVM)
        except Exception:
            is_evm = False

        # On PC, compile already populated the perf data; nothing to do here.
        if not is_evm:
            return

        inspector_dir = os.path.join(self.run_dir, 'inspector')
        json_path = os.path.join(inspector_dir, 'modelinspector.json')
        html_path = os.path.join(inspector_dir, 'modelinspector.html')

        if not os.path.exists(json_path):
            print(f'INFO: Model Inspector JSON not found at {json_path}, skipping EVM perf update')
            return

        print('INFO: Model Inspector - Updating JSON with EVM hardware performance data')
        from ....modelinspector.data_extractor import update_with_evm_perf
        try:
            updated = update_with_evm_perf(json_path)
            if not updated:
                print('INFO: Model Inspector - No /tmp/ EVM perf CSV found, skipping HTML regeneration')
                return
        except Exception as e:
            print(f'INFO: Model Inspector - EVM perf update failed: {e}')
            return

        try:
            from .... import modelinspector
            from ....modelinspector.html_generator import main as gen_html
            template_file = os.path.join(os.path.dirname(modelinspector.__file__), 'template.html')
            gen_html(json_path, template_file, html_path)
            print('INFO: Model Inspector - HTML regenerated with EVM performance data')
        except Exception as e:
            print(f'INFO: Model Inspector - HTML regeneration failed: {e}')


class UpdateModelInspectorEVMAccuracy(CompileModelBase):
    """Read accuracy/timing from result.yaml and patch the inspector JSON.

    Only runs on EVM after an evaluate pipeline.  On PC the result.yaml exists
    but reflects simulation — we skip it so the HTML never shows misleading
    accuracy numbers from PC emulation.
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
        accuracy = load_accuracy_from_result_yaml(self.run_dir)
        if not accuracy:
            print('INFO: Model Inspector - result.yaml not found or empty, skipping accuracy update')
            return

        print('INFO: Model Inspector - Updating JSON with EVM accuracy data from result.yaml')
        try:
            with open(json_path, 'r', encoding='utf-8') as fh:
                data = json.load(fh)

            # Write accuracy fields flat into metadata (same level as model_name, task_type)
            meta = data.setdefault('metadata', {})
            for key, val in accuracy.items():
                if not key.startswith('_'):  # skip internal keys like _result_path
                    meta[key] = val
            # Clean up any old nested key from previous runs
            meta.pop('evm_accuracy', None)

            with open(json_path, 'w', encoding='utf-8') as fh:
                json.dump(data, fh, indent=2)
            print(f'INFO: Model Inspector - Accuracy data written to JSON')
        except Exception as e:
            print(f'INFO: Model Inspector - Accuracy update failed: {e}')