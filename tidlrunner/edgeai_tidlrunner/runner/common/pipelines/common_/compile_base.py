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
import numpy as np

from edgeai_tidlrunner.rtwrapper.options import runtime_options

from edgeai_tidlrunner import rtwrapper
from edgeai_tidlrunner.rtwrapper.core import presets

from ....common import bases
from ... import blocks
from ....common import utils
from ...settings.settings_default import SETTINGS_DEFAULT, COPY_SETTINGS_DEFAULT
from .common_base import CommonPipelineBase
from ...settings import constants
from . import upgrade_cfg


class CompileModelBase(CommonPipelineBase):
    ARGS_DICT=SETTINGS_DEFAULT['compile']
    COPY_ARGS=COPY_SETTINGS_DEFAULT['compile']
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if 'session' in self.settings and self.settings[self.session_prefix].get('model_path', None):
            self.artifacts_folder = self.settings[self.session_prefix].get('artifactrs_folder', os.path.join(self.run_dir, 'artifacts'))
            self.settings[self.session_prefix]['artifacts_folder'] = self.artifacts_folder
        else:
            self.artifacts_folder = None
        #
        self._run_counter = 0  # Counter for unique NPZ filenames
        
    def _prepare(self):
        super()._prepare()

        self.dataloader = None
        self.preprocess = None
        self.session = None
        self.postprocess = None
        self.run_data = None

        if 'session' in self.settings and self.settings[self.session_prefix].get('model_path', None):
            common_kwargs = self.settings[self.common_prefix]
            dataloader_kwargs = self.settings[self.dataloader_prefix]
            preprocess_kwargs = self.settings[self.preprocess_prefix]
            session_kwargs = self.settings[self.session_prefix]
            postprocess_kwargs = self.settings[self.postprocess_prefix]
            runtime_options = session_kwargs['runtime_options']

            ###################################################################################
            if not self.pipeline_config:
                if not dataloader_kwargs['name']:
                    print(f'WARNING: dataloader name is was not provided - will use random_dataloader'
                        f'\n  and the resultant compiled artifacts may not be accurate.'
                        f'\n  please specify a dataloader using the argument data_name or dataloader.name'
                        f'\n  in addition data_path or dataloader.path may need to be provided.')
                    dataloader_kwargs['name'] = 'random_dataloader'
                #
                if dataloader_kwargs['name'] == 'random_dataloader':
                    print(f'WARNING: preprocess name will be set to no_preprocess since no dataloader was provided.')
                    preprocess_kwargs['name'] = 'no_preprocess'
                elif not preprocess_kwargs.get('name', None):
                    preprocess_kwargs['name'] = 'no_preprocess'
                #
                if not postprocess_kwargs.get('name', None):
                    postprocess_kwargs['name'] = 'no_postprocess'
                #
            #

            ###################################################################################
            if session_kwargs['tidl_offload']:
                assert os.environ.get('TIDL_TOOLS_PATH', None) is not None, f"WARNING: TIDL_TOOLS_PATH is missing in the environment"
                runtime_options['tidl_tools_path'] = os.environ['TIDL_TOOLS_PATH']

            if not runtime_options.get('artifacts_folder', None):
                runtime_options['artifacts_folder'] = self.artifacts_folder

            self.object_detection_meta_layers_names_list_source = session_kwargs['runtime_options'].get('object_detection:meta_layers_names_list', None)
            if self.object_detection_meta_layers_names_list_source:
                if not (self.object_detection_meta_layers_names_list_source.startswith('/') or self.object_detection_meta_layers_names_list_source.startswith('.')):
                    object_detection_meta_layers_names_path = os.path.join(self.model_folder, self.object_detection_meta_layers_names_list_source)
                else:
                    object_detection_meta_layers_names_path = self.object_detection_meta_layers_names_list_source
                #
                session_kwargs['runtime_options']['object_detection:meta_layers_names_list'] = object_detection_meta_layers_names_path
            #
        #

    @classmethod
    def _upgrade_kwargs(cls, **kwargs):
        kwargs_out = upgrade_cfg.upgrade_kwargs(**kwargs)
        return kwargs_out

    def _update_settings_after_init(self):
        if self.session and hasattr(self.session, 'get_runtime_options'):
            session_runtime_options = self.session.get_runtime_options()
            self.settings[self.session_prefix]['runtime_options']['advanced_options:quantization_scale_type'] = \
                session_runtime_options['advanced_options:quantization_scale_type']
        #

    def _write_params(self, settings, filename, param_template=None, cleanup_paths=False):
        # adjustments for backward compatibility with 
        # params.yaml and result.yaml written by edgeai-benchmark
        settings = copy.deepcopy(settings)
        if 'session' in settings:
            settings['session']['session_name'] = settings['session']['name']
            settings['session']['input_details'] = self.session.get_kwargs()['input_details']
            settings['session']['output_details'] = self.session.get_kwargs()['output_details']
            
            if cleanup_paths:
                settings['session']['model_path'] = os.path.join(*os.path.normpath(settings['session']['model_path']).split(os.sep)[-2:])    
                settings['session']['artifacts_folder'] = os.path.normpath(settings['session']['artifacts_folder']).split(os.sep)[-1]  
            #   
        #
        super()._write_params(settings, filename, param_template=param_template)

    def get_info_dict(self, input_index):
        if isinstance(self.pipeline_config, dict):
            label_offset_pred = self.pipeline_config.get('metric',{}).get('label_offset_pred',None)
            task_type = self.pipeline_config.get('task_type', None)
            task_name = self.pipeline_config.get('task_name', None)
        else:
            label_offset_pred = self.kwargs.get('label_offset_pred', None)
            task_type = self.kwargs.get('common.task_type', None)
            task_name = self.kwargs.get('common.task_name', None)
        #
        dataset_info = self.dataloader.peek_param('dataset_info', None)
        label_offset_pred = self.kwargs.get('metric.label_offset_pred', None)
        info_dict = {'dataset_info': dataset_info,
                     'label_offset_pred': label_offset_pred,
                     'sample_idx': input_index,
                     'task_type': task_type,
                     'task_name': task_name,
                     'run_dir': self.run_dir,
                     'label_offset_pred': label_offset_pred,
                     }
        return info_dict
    

    def _save_input_tensors(self, input_data, phase=None):
        """Dump input data to NPZ file for debugging/analysis purposes."""
        # Create output directory if it doesn't exist
        dump_dir = self.kwargs.get('save_input_tensors_dir', os.path.join(self.run_dir, 'save_tensors', 'inputs'))
        os.makedirs(dump_dir, exist_ok=True)
        
        # Create filename with counter for uniqueness
        # phase = "import" if self.is_import else "infer"
        filename = f"inputs_{phase}_{self._run_counter:04d}.npz"
        filepath = os.path.join(dump_dir, filename)
        
        # Convert input_data to numpy arrays if they aren't already
        npz_data = {}
        for key, value in input_data.items():
            if hasattr(value, 'numpy'):  # Handle torch tensors
                npz_data[key] = value.numpy()
            elif isinstance(value, np.ndarray):
                npz_data[key] = value
            else:
                npz_data[key] = np.array(value)
        
        # Save to NPZ file
        np.savez(filepath, **npz_data)

    def _save_output_tensors(self, output_data, phase=None):
        """Dump output data to NPZ file for debugging/analysis purposes."""
        # Create output directory if it doesn't exist
        dump_dir = self.kwargs.get('save_output_tensors_dir', os.path.join(self.run_dir, 'save_tensors', 'outputs'))
        os.makedirs(dump_dir, exist_ok=True)

        # Create filename with counter for uniqueness
        # phase = "import" if self.is_import else "infer"
        filename = f"outputs_{phase}_{self._run_counter:04d}.npz"
        filepath = os.path.join(dump_dir, filename)

        # Convert output_data to numpy arrays if they aren't already
        npz_data = {}
        for key, value in output_data.items():
            if hasattr(value, 'numpy'):  # Handle torch tensors
                npz_data[key] = value.numpy()
            elif isinstance(value, np.ndarray):
                npz_data[key] = value
            else:
                npz_data[key] = np.array(value)

        # Save to NPZ file
        np.savez(filepath, **npz_data)

