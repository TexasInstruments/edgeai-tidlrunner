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



import warnings

from ....common.utils.config_utils import postprocess_utils
from ....common import utils
from ...settings import constants
from ...settings.constants import presets
from .transforms import *
from .keypoints import *
from .object_6d_pose import *

import warnings
try:
    from .bev_detection import *
except ImportError as e:
    warnings.warn(f'WARNING: bev_detection postprocessing could not be imported - {str(e)}')
    bev_detection = None
    BEVDetNMS = None
    Bbox3d2result = None
    BEVImageSave = None
    MultiClassNMS = None
    MultiClassScaleNMS = None

from . import transforms as postprocess_transforms_types
from .audio_transforms import SoundClassificationPostProcess, SpeechEnhancementPostProcess, GCRNSpeechEnhancementPostProcess
from ....common.bases import transforms_base


class PostProcessTransforms(transforms_base.TransformsCompose):
    def __init__(self, settings, transforms=None, **kwargs):
        assert transforms is not None, 'transforms must be provided'
        super().__init__(transforms, **kwargs)
        self.settings = settings

    def __call__(self, tensor, info_dict):
        tensor, info_dict = super().__call__(tensor, info_dict)
        return tensor, info_dict
    
    ###############################################################
    # post process transforms for none / passthrough
    ###############################################################
    @classmethod
    def create_transforms_none(cls, settings, **kwargs):
        transforms_list = []
        return transforms_list, dict()

    ###############################################################
    # post process transforms for sound classification
    ###############################################################
    @classmethod
    def create_transforms_audio_classification(cls, settings, **kwargs):
        transforms_list = [SoundClassificationPostProcess()]
        return transforms_list, dict(**kwargs)

    ###############################################################
    # post process transforms for speech enhancement
    ###############################################################
    @classmethod
    def create_transforms_audio_speechenhancement(cls, settings, **kwargs):
        audio_model_type = getattr(getattr(settings, 'preprocess', None), 'audio_model_type', None)
        if audio_model_type == 'gcrn':
            transforms_list = [GCRNSpeechEnhancementPostProcess()]
        else:
            transforms_list = [SpeechEnhancementPostProcess()]
        return transforms_list, dict(**kwargs)

    ###############################################################
    # post process transforms for classification
    ###############################################################
    @classmethod
    def create_transforms_classification(cls, settings, save_output=False, save_output_frames=50, **kwargs):
        transforms_list = [SqueezeAxis(), ArgMax(axis=-1)]
        if save_output:
            transforms_list += [ClassificationImageSave(save_output_frames)]
        #
        return transforms_list, dict(**kwargs)

    ###############################################################
    # post process transforms for detection
    ###############################################################
    @classmethod
    def create_transforms_detection_base(cls, settings, formatter=None, resize_with_pad=False, keypoint=False, object6dpose=False, normalized_detections=True,
                                     shuffle_indices=None, squeeze_axis=0, reshape_list=None, ignore_index=None, logits_bbox_to_bbox_ls=False,
                                     detection_threshold=None, detection_top_k=None, detection_keep_top_k=None, save_output=False, save_output_frames=50, **kwargs):

        # detection_threshold = detection_threshold or settings.detection_threshold

        transforms_list = []
        if logits_bbox_to_bbox_ls:
            logits_bbox_to_bbox_kwargs = logits_bbox_to_bbox_ls if isinstance(logits_bbox_to_bbox_ls, dict) else {}
            transforms_list += [LogitsToLabelScore(**logits_bbox_to_bbox_kwargs)]
        #
        transforms_list += [ReshapeList(reshape_list=reshape_list),
                                 ShuffleList(indices=shuffle_indices),
                                 Concat(axis=-1, end_index=3)]
        if squeeze_axis is not None:
            #  TODO make this more generic to squeeze any axis
            transforms_list += [SqueezeAxis()]
        #
        if ignore_index is not None:
            transforms_list += [IgnoreIndex(ignore_index)]
        #
        if formatter is not None:
            if isinstance(formatter, str):
                formatter_name = formatter
                formatter = getattr(postprocess_transforms_types, formatter_name)()
            elif isinstance(formatter, dict):
                if 'type' in formatter:
                    formatter_name = formatter.pop('type')
                elif 'name' in formatter:
                    formatter_name = formatter.pop('name')
                #
                formatter = getattr(postprocess_transforms_types, formatter_name)(**formatter)
            #
            transforms_list += [formatter]
        #
        transforms_list += [DetectionResizePad(resize_with_pad=resize_with_pad, keypoint=keypoint, object6dpose=object6dpose,
                                                    normalized_detections=normalized_detections)]
        if detection_threshold is not None:
            transforms_list += [DetectionFilter(detection_threshold=detection_threshold,
                                                      detection_keep_top_k=detection_keep_top_k)]
        #
        if keypoint:
            transforms_list += [BboxKeypointsConfReformat()]
        if object6dpose:
            transforms_list += [BboxObject6dPoseReformat()]

        if save_output and save_output_frames:
            if keypoint:
                transforms_list += [HumanPoseImageSave(save_output_frames)]
            elif object6dpose:
                transforms_list += [Object6dPoseImageSave(save_output_frames)]
            else:
                transforms_list += [DetectionImageSave(save_output_frames)]
        #
        return transforms_list, dict(reshape_list=reshape_list, detection_threshold=detection_threshold,
                                    formatter=formatter, resize_with_pad=resize_with_pad,
                                    normalized_detections=normalized_detections, shuffle_indices=shuffle_indices,
                                    squeeze_axis=squeeze_axis, ignore_index=ignore_index, logits_bbox_to_bbox_ls=logits_bbox_to_bbox_ls,
                                    keypoint=keypoint, object6dpose=object6dpose)

    @classmethod
    def create_transforms_detection_onnx(self, settings, formatter=None, **kwargs):
        return self.create_transforms_detection_base(settings, formatter=formatter, **kwargs)

    @classmethod
    def create_transforms_detection_mmdet_onnx(self, settings, formatter=None, reshape_list=[(-1,5), (-1,1)], logits_bbox_to_bbox_ls=False, **kwargs):
        return self.create_transforms_detection_base(settings, formatter=formatter, reshape_list=reshape_list, logits_bbox_to_bbox_ls=logits_bbox_to_bbox_ls, **kwargs)

    @classmethod
    def create_transforms_detection_yolov5_onnx(self, settings, formatter=None, reshape_list=[(-1,6)], **kwargs):
        return self.create_transforms_detection_base(settings, formatter=formatter, reshape_list=reshape_list, **kwargs)

    @classmethod
    def create_transforms_detection_yolov5_pose_onnx(self, settings, formatter=None, reshape_list=[(-1,57)], **kwargs):
        return self.create_transforms_detection_base(settings, formatter=formatter, reshape_list=reshape_list, **kwargs)

    @classmethod
    def create_transforms_detection_yolo_6d_object_pose_onnx(self, settings, object6dpose=True, formatter=None, reshape_list=[(-1,15)], **kwargs):
        return self.create_transforms_detection_base(settings, object6dpose=object6dpose, formatter=formatter, reshape_list=reshape_list, **kwargs)

    @classmethod
    def create_transforms_detection_tv_onnx(self, settings, formatter=postprocess_utils.DetectionBoxSL2BoxLS(), reshape_list=[(-1,4), (-1,1), (-1,1)],
            squeeze_axis=None, normalized_detections=True, **kwargs):
        return self.create_transforms_detection_base(settings, reshape_list=reshape_list, formatter=formatter,
            squeeze_axis=squeeze_axis, normalized_detections=normalized_detections, **kwargs)

    @classmethod
    def create_transforms_detection_tflite(self, settings, formatter=postprocess_utils.DetectionYXYX2XYXY(), **kwargs):
        return self.create_transforms_detection_base(settings, formatter=formatter, **kwargs)

    @classmethod
    def create_transforms_detection_mxnet(self, settings, formatter=None, resize_with_pad=False,
                        normalized_detections=False, shuffle_indices=(2,0,1), **kwargs):
        return self.create_transforms_detection_base(settings, formatter=formatter, resize_with_pad=resize_with_pad,
                        normalized_detections=normalized_detections, shuffle_indices=shuffle_indices, **kwargs)

    ###############################################################
    # post process transforms for segmentation
    ###############################################################
    @classmethod
    def create_transforms_segmentation_base(cls, settings, data_layout=None, with_argmax=True, save_output=False, save_output_frames=50, **kwargs):
        transforms_list = [SqueezeAxis()]
        if with_argmax:
            transforms_list += [ArgMax(axis=None, data_layout=data_layout)]
        #
        transforms_list += [NPTensorToImage(data_layout=data_layout),
                                     SegmentationImageResize(),
                                     SegmentationImagetoBytes()]
        if save_output:
            transforms_list += [SegmentationImageSave(save_output_frames)]
        #
        return transforms_list, dict(data_layout=data_layout, with_argmax=with_argmax, **kwargs)

    @classmethod
    def create_transforms_segmentation_onnx(cls, data_layout=presets.DataLayoutType.NCHW, with_argmax=True, **kwargs):
        return cls.create_transforms_segmentation_base(data_layout=data_layout, with_argmax=with_argmax, **kwargs)

    @classmethod
    def create_transforms_segmentation_tflite(cls, data_layout=presets.DataLayoutType.NHWC, with_argmax=True, **kwargs):
        return cls.create_transforms_segmentation_base(data_layout=data_layout, with_argmax=with_argmax, **kwargs)

    ###############################################################
    # post process transforms for human pose estimation
    ###############################################################
    @classmethod
    def create_transforms_human_pose_estimation_base(cls, settings, data_layout=None, with_udp=True, save_output=False, save_output_frames=50, **kwargs):
        # channel_axis = -1 if data_layout == presets.DataLayoutType.NHWC else 1
        # postprocess_human_pose_estimation = [SqueezeAxis()] #just removes the first axis from output list, final size (c,w,h)
        transforms_list = [HumanPoseHeatmapParser(use_udp=with_udp),
                           KeypointsProject2Image(use_udp=with_udp)]

        if save_output:
            transforms_list += [HumanPoseImageSave(save_output_frames)]
        #
        return transforms_list, dict(data_layout=data_layout, with_udp=with_udp, **kwargs)

    @classmethod
    def create_transforms_human_pose_estimation_onnx(cls, settings, data_layout=presets.DataLayoutType.NCHW, **kwargs):
        return cls.create_transforms_human_pose_estimation_base(data_layout=data_layout, with_udp=settings.with_udp, **kwargs)

    ###############################################################
    # post process transforms for depth estimation
    ###############################################################
    @classmethod
    def create_transforms_depth_estimation_base(cls, settings, data_layout=presets.DataLayoutType.NCHW, save_output=False, save_output_frames=50, **kwargs):
        transforms_list = [SqueezeAxis(),
                           NPTensorToImage(data_layout=data_layout),
                           DepthImageResize()]
        if save_output:
            transforms_list += [DepthImageSave(save_output_frames)]
        #
        return transforms_list, dict(data_layout=data_layout, **kwargs)

    @classmethod
    def create_transforms_lidar_base(cls, settings, **kwargs):
        transforms_list = [
            OD3DOutPutPorcess(settings.detection_threshold)
        ]
        return transforms_list, dict(detection_threshold=settings.detection_threshold, **kwargs)

    ###############################################################
    # post process transforms for disparity estimation
    ###############################################################
    @classmethod
    def create_transforms_disparity_estimation_base(cls, settings, data_layout, save_output=False, save_output_frames=50, **kwargs):
        transforms_list = [SqueezeAxis(), 
                           NPTensorToImage(data_layout=data_layout)]
        
        # To REVISIT!
        #if save_output:
        #    transforms_list += [DepthImageSave(save_output_frames)]
        return transforms_list, dict(data_layout=data_layout, **kwargs)

    @classmethod
    def create_transforms_disparity_estimation_onnx(cls, settings, data_layout=presets.DataLayoutType.NCHW, **kwargs):
        return cls.create_transforms_disparity_estimation_base(data_layout=data_layout, **kwargs)

    ###############################################################
    # post process transforms for BEV detection
    ###############################################################
    # To REVISIT
    # Any necessary visualization funtions will be addeed in bev_detection.py
    def create_transforms_bev_detection_base(cls, settings, queue_length=0, data_layout=presets.DataLayoutType.NCHW, save_output=False, save_output_frames=50, **kwargs):
        transforms = None

        try:
            if queue_length > 0:
                postprocess_bev_detection_base = [UpdateTemporalQueue(queue_length=queue_length),
                                                  Bbox3d2result()]
            else:
                postprocess_bev_detection_base = [Bbox3d2result()]
        except Exception as message:
            print(f'BEV postprocess could not be created: {message}')

        if save_output:
            # To be updated
            try:
                postprocess_bev_detection_base += [BEVImageSave(save_output_frames,
                                                                score_threshold=0.5,
                                                                mode='frame')]
            except Exception as message:
                print(f'BEV postprocess could not be created: {message}')

        return postprocess_bev_detection_base, dict(data_layout=data_layout, **kwargs)

    def create_transforms_bev_detection_bevdet(cls, settings, data_layout=presets.DataLayoutType.NCHW, save_output=False, save_output_frames=50, **kwargs):
        transforms = None
        # For bevDet_tiny_256x704_res50_parallel.onnx
        #postprocess_bev_detection_bevdet = [GetBEVDetBBoxes(),
        #                                    BEVDetNMS(),
        #                                    Bbox3d2result()]

        # For bevDet_tiny_256x704_res50_pp_parallel.onnx
        try:
            postprocess_bev_detection_bevdet = [BEVDetNMS(),
                                                Bbox3d2result()]
        except Exception as message:
            print(f'BEV postprocess could not be created: {message}')

        if save_output:
            # To be updated
            try:
                postprocess_bev_detection_bevdet += [BEVImageSave(save_output_frames,
                                                                score_threshold=0.5,
                                                                mode='frame')]
            except Exception as message:
                print(f'BEV postprocess could not be created: {message}')              

        return transforms, dict(data_layout=data_layout, **kwargs)

    def create_transforms_fcos3d(cls, settings, data_layout=presets.DataLayoutType.NCHW, save_output=False, save_output_frames=50, **kwargs):
        transforms = None
        try:
            postprocess_fcos3d = [MultiClassNMS(),
                                  Bbox3d2result()]
        except Exception as message:
            print(f'BEV postprocess could not be created: {message}')

        if save_output:
            # To be updated
            try:
                postprocess_fcos3d += [BEVImageSave(save_output_frames,
                                                    score_threshold=0.2,
                                                    mode='mv_image')]
            except Exception as message:
                print(f'BEV postprocess could not be created: {message}')    

        return postprocess_fcos3d, dict(data_layout=data_layout, **kwargs)

    def create_transforms_bev_detection_fastbev(cls, settings, enable_nms=True, queue_length=0, data_layout=presets.DataLayoutType.NCHW, save_output=False, save_output_frames=50, **kwargs):
        transforms = None

        postprocess_bev_detection_fastbev = []
        if enable_nms:
            try:
                postprocess_bev_detection_fastbev += [MultiClassScaleNMS()]
            except Exception as message:
                print(f'BEV postprocess could not be created: {message}')

        if queue_length > 0:
            try:
                postprocess_bev_detection_fastbev += [UpdateTemporalQueue(queue_length=queue_length)]
            except Exception as message:
                print(f'BEV postprocess could not be created: {message}')

        try:
            postprocess_bev_detection_fastbev += [Bbox3d2result()]
        except Exception as message:
            print(f'BEV postprocess could not be created: {message}')

        if save_output:
            # To be updated
            try:
                postprocess_bev_detection_fastbev += [BEVImageSave(save_output_frames,
                                                                    score_threshold=0.5,
                                                                    mode='frame')]
            except Exception as message:
                print(f'BEV postprocess could not be created: {message}')    

        return postprocess_bev_detection_fastbev, dict(data_layout=data_layout, **kwargs)


def no_postprocess(settings, name='no_postprocess', **kwargs):
    transforms_list, kwargs = PostProcessTransforms.create_transforms_none(settings, name=name, **kwargs)
    return PostProcessTransforms(settings, transforms_list, name=name, **kwargs)


def image_classification_postprocess(settings, name='image_classification_postprocess', **kwargs):
    assert settings.common.task_type == constants.TaskType.TASK_TYPE_CLASSIFICATION, \
        f'image_classification_postprocess can only be used for image classification task type. given task_type is: {settings.common.task_type}'
    transforms_list, kwargs = PostProcessTransforms.create_transforms_classification(settings, **kwargs)
    return PostProcessTransforms(settings, transforms_list, name=name, **kwargs)


def classification_postprocess(settings, *args, **kwargs):
    return image_classification_postprocess(settings, *args, **kwargs)


def object_detection_postprocess(settings, name='object_detection_postprocess', **kwargs):
    assert settings.common.task_type in (constants.TaskType.TASK_TYPE_DETECTION, constants.TaskType.TASK_TYPE_KEYPOINT_DETECTION), \
        f'object_detection_postprocess can only be used for object detection task type. given task_type is: {settings.common.task_type}'
    transforms_list, kwargs = PostProcessTransforms.create_transforms_detection_base(settings, **kwargs)
    return PostProcessTransforms(settings, transforms_list, name=name, **kwargs)


def detection_postprocess(settings, *args, **kwargs):
    return object_detection_postprocess(settings, *args, **kwargs)


def segmentation_postprocess(settings, name='segmentation_postprocess', **kwargs):
    assert settings.common.task_type == constants.TaskType.TASK_TYPE_SEGMENTATION, \
        'segmentation_postprocess can only be used for segmentation task type'
    transforms_list, kwargs = PostProcessTransforms.create_transforms_segmentation_base(settings, **kwargs)
    return PostProcessTransforms(settings, transforms_list, name=name, **kwargs)


def keypoint_detection_postprocess(settings, name='keypoint_detection_postprocess', **kwargs):
    assert settings.common.task_type == constants.TaskType.TASK_TYPE_KEYPOINT_DETECTION, \
        'keypoint_detection_postprocess can only be used for keypoint detection task type'
    transforms_list, kwargs = PostProcessTransforms.create_transforms_detection_yolov5_pose_onnx(settings, **kwargs)
    return PostProcessTransforms(settings, transforms_list, name=name, **kwargs)


def human_pose_estimation_postprocess(settings, name='human_pose_estimation_postprocess', **kwargs):
    assert settings.common.task_type == constants.TaskType.TASK_TYPE_KEYPOINT_DETECTION, \
        'human_pose_estimation_postprocess can only be used for human pose estimation task type'
    transforms_list, kwargs = PostProcessTransforms.create_transforms_disparity_estimation_base(settings, **kwargs)
    return PostProcessTransforms(settings, transforms_list, name=name, **kwargs)


def yolo_6d_object_pose_postprocess(settings, reshape_list=[(-1,15)], name='yolo_6d_object_pose_postprocess', **kwargs):
    assert settings.common.task_type == constants.TaskType.TASK_TYPE_OBJECT_6D_POSE_ESTIMATION, \
        'yolo_6d_object_pose_postprocess can only be used for 6D object pose estimation task type'
    transforms_list, kwargs = PostProcessTransforms.create_transforms_detection_yolo_6d_object_pose_onnx(settings, reshape_list=reshape_list, **kwargs)
    return PostProcessTransforms(settings, transforms_list, name=name, **kwargs)


def audio_classification_postprocess(settings, name='audio_classification_postprocess', **kwargs):
    transforms_list, kwargs = PostProcessTransforms.create_transforms_audio_classification(settings, **kwargs)
    return PostProcessTransforms(settings, transforms_list, name=name, **kwargs)


def audio_speechenhancement_postprocess(settings, name='audio_speechenhancement_postprocess', **kwargs):
    transforms_list, kwargs = PostProcessTransforms.create_transforms_audio_speechenhancement(settings, **kwargs)
    return PostProcessTransforms(settings, transforms_list, name=name, **kwargs)


def bev_detection_postprocess(settings, name='bev_detection_postprocess', **kwargs):
    transforms_list, kwargs =  PostProcessTransforms.create_transforms_bev_detection_base(settings, queue_length=1, **kwargs)
    return PostProcessTransforms(settings, transforms_list, name=name, **kwargs)


def bevdet_model_postprocess(settings, name='bevdet_model_postprocess', **kwargs):
    transforms_list, kwargs = PostProcessTransforms.create_transforms_bev_detection_bevdet(settings, **kwargs)
    return PostProcessTransforms(settings, transforms_list, name=name, **kwargs)


def fcos3d_model_postprocess(settings, name='fcos3d_model_postprocess', **kwargs):
    transforms_list, kwargs = PostProcessTransforms.create_transforms_bev_detection_fcos3d(settings, **kwargs)
    return PostProcessTransforms(settings, transforms_list, name=name, **kwargs)


def fastbev_model_postprocess(settings, name='fastbev_model_postprocess', **kwargs):
    transforms_list, kwargs = PostProcessTransforms.create_transforms_bev_detection_fastbev(settings, **kwargs)
    return PostProcessTransforms(settings, transforms_list, name=name, **kwargs)

def depth_estimation_postprocess(settings, name='depth_estimation_postprocess', **kwargs):
    transforms_list, kwargs = PostProcessTransforms.create_transforms_depth_estimation_base(settings, **kwargs)
    return PostProcessTransforms(settings, transforms_list, name=name, **kwargs)

