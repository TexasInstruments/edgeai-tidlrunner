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

from edgeai_tidlrunner.rtwrapper.core import presets
from ...settings import constants
from ....common import utils


def upgrade_kwargs(**kwargs):
    kwargs_in = kwargs

    upgrade_config = kwargs_in.get('common.upgrade_config', True)
    model_path = kwargs_in.get('session.model_path', None)
    model_ext = os.path.splitext(model_path)[1][1:] if model_path else None
    
    if not upgrade_config:
        kwargs_out = copy.deepcopy(kwargs_in)
        return kwargs_out

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
        # elif k == 'dataloader.name':
        #     if kwargs_in[k] is not None:
        #         kwargs_out[k] = kwargs_in[k]
        #     #
        # elif k == 'dataloader.path':
        #     if kwargs_in[k] is not None:
        #         kwargs_out[k] = kwargs_in[k]
        #     #
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

    # override parameters with preset_selection
    preset_selection = kwargs_out.get('common.preset_selection', None)
    if preset_selection is None or preset_selection.lower() == constants.ModelCompilationPreset.PRESET_DEFAULT.lower():
        # SETTINGS_DEFAULT is already set up for DEFAULT preset - no changes needed
        pass
    elif preset_selection.lower() == constants.ModelCompilationPreset.PRESET_SANITY.lower():
        # very quick calibration and inference for faster compilation and testing - not for accuracy evaluation
        kwargs_out['common.num_frames'] = 1
        kwargs_out['session.runtime_options.advanced_options:calibration_frames'] = 1
        kwargs_out['session.runtime_options.advanced_options:calibration_iterations'] = 1
    elif preset_selection.lower() == constants.ModelCompilationPreset.PRESET_QUICK.lower():
        # quick calibration and inference for faster compilation and testing - not for accuracy evaluation
        kwargs_out['common.num_frames'] = 100
        kwargs_out['session.runtime_options.advanced_options:calibration_frames'] = 5
        kwargs_out['session.runtime_options.advanced_options:calibration_iterations'] = 5
    elif preset_selection.lower() == constants.ModelCompilationPreset.PRESET_ACCURACY.lower():
        # default dataset_type_dict has imagenet mapping to imagenetv2c for quick testing - remove this mapping
        kwargs_out['common.dataset_type_dict'] = None
        kwargs_out['session.runtime_options.object_detection:confidence_threshold'] = 0.05
        kwargs_out['session.runtime_options.object_detection:top_k'] = 500
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

    ###################################################################################
    if not (kwargs_out.get('dataloader.name',None) and kwargs_out.get('dataloader.path',None)):
        input_dataset = kwargs_out.get('common.input_dataset', None)

        if kwargs_out.get('common.dataset_type_dict', None) and input_dataset in kwargs_out['common.dataset_type_dict']:
            input_dataset = kwargs_out['common.dataset_type_dict'][input_dataset]
        #

        if input_dataset == 'imagenet':
            if kwargs_out.get('dataloader.name', None) is None:
                kwargs_out['dataloader.name'] = 'image_classification_dataloader'
            #
            if kwargs_out.get('dataloader.path', None) is None:
                kwargs_out['dataloader.path'] = './data/datasets/imagenet/val'
                kwargs_out['dataloader.label_path'] = './data/datasets/imagenet/val.txt'
            #
        elif input_dataset == 'imagenet_folders':
            if kwargs_out.get('dataloader.name', None) is None:
                kwargs_out['dataloader.name'] = 'image_classification_dataloader'
            #
            if kwargs_out.get('dataloader.path', None) is None:
                kwargs_out['dataloader.path'] = './data/datasets/imagenet/val'
            #
        elif input_dataset == 'imagenetv2c':
            if kwargs_out.get('dataloader.name', None) is None:
                kwargs_out['dataloader.name'] = 'image_classification_dataloader'
            #
            if kwargs_out.get('dataloader.path', None) is None:
                kwargs_out['dataloader.path'] = './data/datasets/imagenetv2c/val'
            #
        elif input_dataset == 'coco':
            if kwargs_out.get('dataloader.name', None) is None:
                kwargs_out['dataloader.name'] = 'coco_detection_dataloader'
            #
            if kwargs_out.get('dataloader.path', None) is None:
                kwargs_out['dataloader.path'] = './data/datasets/coco'
            #
        elif input_dataset == 'cocoseg21':
            if kwargs_out.get('dataloader.name', None) is None:
                kwargs_out['dataloader.name'] = 'coco_segmentation_dataloader'
            #
            if kwargs_out.get('dataloader.path', None) is None:
                kwargs_out['dataloader.path'] = './data/datasets/coco'
            #               
        elif input_dataset == 'cocokpts':
            if kwargs_out.get('dataloader.name', None) is None:
                kwargs_out['dataloader.name'] = 'coco_keypoint_detection_dataloader'
            #
            if kwargs_out.get('dataloader.path', None) is None:
                kwargs_out['dataloader.path'] = './data/datasets/coco'
            #    
        elif input_dataset == 'ade20k':
            if kwargs_out.get('dataloader.name', None) is None:
                kwargs_out['dataloader.name'] = 'ade20k_segmentation_dataloader'
            #
            if kwargs_out.get('dataloader.path', None) is None:
                kwargs_out['dataloader.path'] = './data/datasets/ADEChallengeData2016'
            #   
        elif input_dataset == 'ade20k32':
            if kwargs_out.get('dataloader.name', None) is None:
                kwargs_out['dataloader.name'] = 'ade20k32_segmentation_dataloader'
            #
            if kwargs_out.get('dataloader.path', None) is None:
                kwargs_out['dataloader.path'] = './data/datasets/ADEChallengeData2016'
            #               
        elif input_dataset == 'widerface':
            if kwargs_out.get('dataloader.name', None) is None:
                kwargs_out['dataloader.name'] = 'widerface_detection_dataloader'
            #
            if kwargs_out.get('dataloader.path', None) is None:
                kwargs_out['dataloader.path'] = './data/datasets/widerface'
            #
        elif input_dataset == 'ycbv':
            if kwargs_out.get('dataloader.name', None) is None:
                kwargs_out['dataloader.name'] = 'ycbv_object_6d_pose_dataloader'
            #
            if kwargs_out.get('dataloader.path', None) is None:
                kwargs_out['dataloader.path'] = './data/datasets/ycbv'
            # 
        elif input_dataset == 'nyudepthv2':
            if kwargs_out.get('dataloader.name', None) is None:
                kwargs_out['dataloader.name'] = 'nyudepthv2_dataloader'
            #
            if kwargs_out.get('dataloader.path', None) is None:
                kwargs_out['dataloader.path'] = './data/datasets/nyudepthv2'
            # 
        elif input_dataset == 'ti-robokit_semseg_zed1hd':
            if kwargs_out.get('dataloader.name', None) is None:
                kwargs_out['dataloader.name'] = 'robokit_segmentation_dataloader'
            #
            if kwargs_out.get('dataloader.path', None) is None:
                kwargs_out['dataloader.path'] = './data/datasets/ti-robokit_semseg_zed1hd'
            # 
        elif input_dataset == 'ti-robokit_visloc_zed1hd':
            if kwargs_out.get('dataloader.name', None) is None:
                kwargs_out['dataloader.name'] = 'robokit_visloc_dataloader'
            #
            if kwargs_out.get('dataloader.path', None) is None:
                kwargs_out['dataloader.path'] = './data/datasets/ti-robokit_semseg_zed1hd'
            # 
        # else:
        #     print(f'WARNING: {input_dataset} dataset is not supported - please use a supported dataset OR specify both dataloader.name and dataloader.path')  
        # #  
    #

    ###################################################################################
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
    elif task_type == constants.TaskType.TASK_TYPE_OBJECT_6D_POSE_ESTIMATION:
        if kwargs_out.get('preprocess.name',None) is None:
            kwargs_out['preprocess.name'] = 'image_preprocess'
        #
        if kwargs_out.get('postprocess.name',None) is None:
            kwargs_out['postprocess.name'] = 'yolo_6d_object_pose_postprocess'
        #
    elif task_type == constants.TaskType.TASK_TYPE_AUDIO_CLASSIFICATION:
        if kwargs_out.get('preprocess.name',None) is None:
            kwargs_out['preprocess.name'] = 'audio_classification_preprocess'
        #
        if kwargs_out.get('postprocess.name',None) is None:
            kwargs_out['postprocess.name'] = 'audio_classification_postprocess'
        #
    elif task_type == constants.TaskType.TASK_TYPE_AUDIO_SPEECHENHANCEMENT:
        if kwargs_out.get('preprocess.name',None) is None:
            kwargs_out['preprocess.name'] = 'audio_speechenhancement_preprocess'
        #
        if kwargs_out.get('postprocess.name',None) is None:
            kwargs_out['postprocess.name'] = 'audio_speechenhancement_postprocess'
        #
    # else:
    #     print(f'WARNING: task_type {task_type} is not supported - please use a supported task_type OR specify both preprocess.name and postprocess.name')  
    # #

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
    return kwargs_out