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
                if not preprocess_kwargs['name']:
                    preprocess_kwargs['name'] = 'no_preprocess'
                #
                if not postprocess_kwargs['name']:
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
        kwargs_in = kwargs

        upgrade_config = kwargs_in.get('common.upgrade_config', True)
        model_path = kwargs_in.get('session.model_path', None)
        model_ext = os.path.splitext(model_path)[1][1:] if model_path else None
        
        if not upgrade_config:
            kwargs_out = copy.deepcopy(kwargs_in)
        else:
            kwargs_in = copy.deepcopy(kwargs)
            kwargs_out = dict()

            for k, v in kwargs_in.items():
                if k in ('session.target_device',):
                    # options that are not allowed to be None
                    if v is not None:
                        kwargs_out[k] = v
                    #
                elif k.startswith('session.runtime_options.'):
                    # options that are not allowed to be None
                    if v is not None:
                        kwargs_out[k] = v
                    #                
                elif k == 'session.session_name':
                    kwargs_out['session.name'] = v
                elif k == 'dataloader.name':
                    if kwargs_in[k] is not None:
                        kwargs_out[k] = kwargs_in[k]
                    #
                elif k == 'dataloader.path':
                    if kwargs_in[k] is not None:
                        kwargs_out[k] = kwargs_in[k]
                    #
                elif k == 'preprocess.name':
                    if kwargs_in[k] is not None:
                        kwargs_out[k] = kwargs_in[k]
                    #
                elif k == 'postprocess.name':
                    if kwargs_in[k] is not None:
                        kwargs_out[k] = kwargs_in[k]
                    #
                elif k == 'dataset_category':
                    pass
                elif k == 'calibration_dataset':
                    pass
                elif k == 'task_type' or k == 'common.task_type':
                    kwargs_out['common.task_type'] = v
                elif k == 'input_dataset' or k == 'dataloader.input_dataset':
                    kwargs_out['common.input_dataset'] = v
                else:
                    kwargs_out[k] = v
                #
            #

            ###################################################################################
            if not (kwargs_out.get('dataloader.name',None) and kwargs_out.get('dataloader.path',None)):
                input_dataset = kwargs_out.get('common.input_dataset', None)
                if input_dataset == 'imagenet':
                    if kwargs_out.get('dataloader.name', None) is None:
                        kwargs_out['dataloader.name'] = 'image_classification_dataloader'
                    #
                    if kwargs_out.get('dataloader.path', None) is None:
                        kwargs_out['dataloader.path'] = './data/datasets/vision/imagenetv2c/val'
                    #
                elif input_dataset == 'coco':
                    if kwargs_out.get('dataloader.name', None) is None:
                        kwargs_out['dataloader.name'] = 'coco_detection_dataloader'
                    #
                    if kwargs_out.get('dataloader.path', None) is None:
                        kwargs_out['dataloader.path'] = './data/datasets/vision/coco'
                    #
                elif input_dataset == 'cocoseg21':
                    if kwargs_out.get('dataloader.name', None) is None:
                        kwargs_out['dataloader.name'] = 'coco_segmentation_dataloader'
                    #
                    if kwargs_out.get('dataloader.path', None) is None:
                        kwargs_out['dataloader.path'] = './data/datasets/vision/coco'
                    #               
                elif input_dataset == 'cocokpts':
                    if kwargs_out.get('dataloader.name', None) is None:
                        kwargs_out['dataloader.name'] = 'coco_keypoint_detection_dataloader'
                    #
                    if kwargs_out.get('dataloader.path', None) is None:
                        kwargs_out['dataloader.path'] = './data/datasets/vision/coco'
                    #                  
                # else:
                #     print(f'WARNING: {input_dataset} dataset is not supported - please use a supported dataset OR specify both dataloader.name and dataloader.path')  
                # #  
            #
    
            ###################################################################################
            if kwargs_out.get('dataloader.name',None) and kwargs_out.get('dataloader.path',None):
                task_type = kwargs_out.get('common.task_type', None)
                if kwargs_out.get('preprocess.name',None) and kwargs_out.get('postprocess.name',None):
                    pass
                elif task_type == constants.TaskType.TASK_TYPE_CLASSIFICATION:
                    if kwargs_out.get('preprocess.name',None) is None:
                        kwargs_out['preprocess.name'] = 'image_preprocess'
                    #
                elif task_type == constants.TaskType.TASK_TYPE_DETECTION:
                    if kwargs_out.get('preprocess.name',None) is None:
                        kwargs_out['preprocess.name'] = 'image_preprocess'
                    #
                    if kwargs_out.get('postprocess.name',None) is None:
                        kwargs_out['postprocess.name'] = 'object_detection_postprocess'
                    #
                elif task_type == constants.TaskType.TASK_TYPE_SEGMENTATION:
                    if kwargs_out.get('preprocess.name',None) is None:
                        kwargs_out['preprocess.name'] = 'image_preprocess'
                    #
                    if kwargs_out.get('postprocess.name',None) is None:
                        kwargs_out['postprocess.name'] = 'segmentation_postprocess'
                    #   
                elif task_type == constants.TaskType.TASK_TYPE_KEYPOINT_DETECTION:
                    if kwargs_out.get('preprocess.name',None) is None:
                        kwargs_out['preprocess.name'] = 'image_preprocess'
                    #
                    if kwargs_out.get('postprocess.name',None) is None:
                        kwargs_out['postprocess.name'] = 'keypoint_detection_postprocess'
                    #  
                else:
                    print(f'WARNING: task_type {task_type} is not supported - please use a supported task_type OR specify both preprocess.name and postprocess.name')  
                #
            #

            if model_path:
                if kwargs_out.get('preprocess.data_layout', None) is None:
                    data_layout_mapping = {
                        'onnx': presets.DataLayoutType.NCHW,
                        'tflite': presets.DataLayoutType.NHWC,
                    }
                    data_layout = data_layout_mapping.get(model_ext, None)
                    kwargs_out['preprocess.data_layout'] = data_layout
                    kwargs_out['session.data_layout'] = data_layout
                #
            #
        #

        if kwargs_out.get('session.name', None) is None:
            if kwargs_out.get('session.session_name', None) is not None:
                    kwargs_out['session.name'] = kwargs_out['session.session_name']
                #
            #
            kwargs_out.pop('session.session_name', None)
        #
        # override session.name based on model_ext and session_type_dict
        if (kwargs_out.get('session.name', None) is None) and model_path:
            model_ext = os.path.splitext(model_path)[1][1:] if model_path else None
            session_type_dict = kwargs_out.get('common.session_type_dict', None)
            session_type_dict = utils.str_to_literal(session_type_dict)
            session_type_dict = session_type_dict or constants.SESSION_TYPE_DICT_DEFAULT
            if model_ext in session_type_dict:
                kwargs_out['session.name'] = session_type_dict[model_ext]
            else:
                raise RuntimeError(f'ERROR: model extension {model_ext} is not supported - must be one of {list(session_type_dict.keys())}')
            #
        #

        # override calibration parameters with preset_selection
        preset_selection = kwargs_out.get('common.preset_selection', None)
        if preset_selection is not None:
            if preset_selection.lower() == constants.ModelCompilationPreset.PRESET_ACCURACY.lower():
                kwargs_out['session.runtime_options.object_detection:confidence_threshold'] = 0.05
                kwargs_out['session.runtime_options.object_detection:top_k'] = 500
            elif preset_selection.lower() == constants.ModelCompilationPreset.PRESET_SPEED.lower():
                kwargs_out['common.num_frames'] = 10
                kwargs_out['session.runtime_options.object_detection:confidence_threshold'] = 0.3
                kwargs_out['session.runtime_options.object_detection:top_k'] = 200
                kwargs_out['session.runtime_options.advanced_options:calibration_frames'] = 5
                kwargs_out['session.runtime_options.advanced_options:calibration_iterations'] = 5
            #
        #
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
            if cleanup_paths:
                settings['session']['model_path'] = os.path.join(*os.path.normpath(settings['session']['model_path']).split(os.sep)[-2:])    
                settings['session']['artifacts_folder'] = os.path.normpath(settings['session']['artifacts_folder']).split(os.sep)[-1]  
            #   
        #
        super()._write_params(settings, filename, param_template=param_template)

    def get_info_dict(self, input_index):
        if isinstance(self.pipeline_config, dict):
            label_offset_pred = self.pipeline_config.get('metric',{}).get('label_offset_pred',None)
            task_name = self.pipeline_config.get('task_name',{})
        else:
            label_offset_pred = self.kwargs.get('label_offset_pred', None)
            task_name = self.kwargs.get('common.task_name', None)
        #
        dataset_info = self.dataloader.peek_param('dataset_info', None)
        label_offset_pred = self.kwargs.get('metric.label_offset_pred', None)
        info_dict = {'dataset_info': dataset_info,
                     'label_offset_pred': label_offset_pred,
                     'sample_idx': input_index,
                     'task_name': task_name,
                     'run_dir': self.run_dir,
                     'label_offset_pred': label_offset_pred,
                     }
        return info_dict
    

    def _save_input_to_npz(self, input_data):
        """Dump input data to NPZ file for debugging/analysis purposes."""
        try:
            # Create output directory if it doesn't exist
            dump_dir = self.kwargs.get('input_save_dir', os.path.join(self.artifacts_folder, 'inputs'))
            os.makedirs(dump_dir, exist_ok=True)
            
            # Create filename with counter for uniqueness
            phase = "import" if self.is_import else "inference"
            filename = f"onnx_inputs_{phase}_{self._run_counter:04d}.npz"
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
            
            # Increment counter for next run
            self._run_counter += 1
            
            print(f"Dumped ONNX inputs to: {filepath}")
            
        except Exception as e:
            print(f"Warning: Failed to dump inputs to NPZ file: {e}")