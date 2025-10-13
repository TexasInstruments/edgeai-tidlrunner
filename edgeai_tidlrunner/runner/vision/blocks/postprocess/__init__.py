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


from ....common.utils.config_utils import postprocess_utils
from ....common import utils
from ...settings import constants
from ...settings.constants import presets
from .transforms import *
from .keypoints import *
#from .object_6d_pose import *
from . import transforms as postprocess_transforms_types
from ....common.bases import transforms_base


class PostProcessTransforms(transforms_base.TransformsCompose):
    def __init__(self, settings, transforms=None, **kwargs):
        assert transforms is not None, 'transforms must be provided'
        super().__init__(transforms, **kwargs)
        self.settings = settings

    @classmethod
    def from_kwargs(cls, settings, **kwargs):
        if isinstance(kwargs.get('formatter', None), str):
            kwargs['formatter'] = postprocess_utils.get_formatter(settings.common.task_type, kwargs['formatter'])
        #
        if settings.common.task_type == constants.TaskType.TASK_TYPE_CLASSIFICATION:
            transforms, transforms_kwargs = cls.create_transforms_classification(settings, **kwargs)
        elif settings.common.task_type == constants.TaskType.TASK_TYPE_DETECTION:
            transforms, transforms_kwargs = cls.create_transforms_detection_base(settings, **kwargs)
        elif settings.common.task_type == constants.TaskType.TASK_TYPE_SEGMENTATION:
            transforms, transforms_kwargs = cls.create_transforms_segmentation_base(settings, **kwargs)
        elif settings.common.task_type == constants.TaskType.TASK_TYPE_KEYPOINT_DETECTION:
            transforms, transforms_kwargs = cls.create_transforms_human_pose_estimation_base(settings, **kwargs)
        elif settings.common.task_type == constants.TaskType.TASK_TYPE_DEPTH_ESTIMATION:
            transforms, transforms_kwargs = cls.create_transforms_depth_estimation_base(settings, **kwargs)
        elif settings.common.task_type == constants.TaskType.TASK_TYPE_DISPARITY_ESTIMATION:
            transforms, transforms_kwargs = cls.create_transforms_disparity_estimation_base(settings, **kwargs)
        elif settings.common.task_type == constants.TaskType.TASK_TYPE_DETECTION_3DOD:
            transforms, transforms_kwargs = cls.create_transforms_lidar_base(settings, **kwargs)
        elif settings.common.task_type == constants.TaskType.TASK_TYPE_OBJECT_6D_POSE_ESTIMATION:
            transforms, transforms_kwargs = cls.create_transforms_detection_base(settings, object6dpose=True, **kwargs)
        else:
            transforms, transforms_kwargs = cls.create_transforms_none(settings, **kwargs)
        #
        return cls(settings, transforms=transforms, **transforms_kwargs)
        

    ###############################################################
    # post process transforms for classification
    ###############################################################
    @classmethod
    def create_transforms_none(cls, settings, **kwargs):
        transforms_list = []
        return transforms_list

    ###############################################################
    # post process transforms for classification
    ###############################################################
    @classmethod
    def create_transforms_classification(cls, settings, save_output=False, save_output_frames=50, **kwargs):
        transforms_list = [SqueezeAxis(), ArgMax(axis=-1)]
        if save_output:
            transforms_list += [ClassificationImageSave(save_output_frames)]
        #
        return transforms_list, dict()

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
            transforms_list += [LogitsToLabelScore()]
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

        if save_output:
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
        return self.create_transforms_detection_base(settings, formatter=formatter, reshape_list=reshape_list,logits_bbox_to_bbox_ls=logits_bbox_to_bbox_ls, **kwargs)

    @classmethod
    def create_transforms_detection_yolov5_onnx(self, settings, formatter=None, **kwargs):
        return self.create_transforms_detection_base(settings, formatter=formatter, reshape_list=[(-1,6)], **kwargs)

    @classmethod
    def create_transforms_detection_yolov5_pose_onnx(self, settings, formatter=None, **kwargs):
        return self.create_transforms_detection_base(settings, formatter=formatter, reshape_list=[(-1,57)], **kwargs)

    @classmethod
    def create_transforms_detection_yolo_6d_object_pose_onnx(self, settings, formatter=None, **kwargs):
        return self.create_transforms_detection_base(settings, formatter=formatter, reshape_list=[(-1,15)], **kwargs)

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
        return transforms_list, dict(data_layout=data_layout, with_argmax=with_argmax)

    @classmethod
    def create_transforms_segmentation_onnx(cls, data_layout=constants.presets.DataLayoutType.NCHW, with_argmax=True, **kwargs):
        return cls.create_transforms_segmentation_base(data_layout=data_layout, with_argmax=with_argmax, **kwargs)

    @classmethod
    def create_transforms_segmentation_tflite(cls, data_layout=constants.presets.DataLayoutType.NHWC, with_argmax=True, **kwargs):
        return cls.create_transforms_segmentation_base(data_layout=data_layout, with_argmax=with_argmax, **kwargs)

    ###############################################################
    # post process transforms for human pose estimation
    ###############################################################
    @classmethod
    def create_transforms_human_pose_estimation_base(cls, settings, data_layout=None, with_udp=True, save_output=False, save_output_frames=50, **kwargs):
        # channel_axis = -1 if data_layout == constants.presets.NHWC else 1
        # postprocess_human_pose_estimation = [SqueezeAxis()] #just removes the first axis from output list, final size (c,w,h)
        transforms_list = [HumanPoseHeatmapParser(use_udp=with_udp),
                           KeypointsProject2Image(use_udp=with_udp)]

        if save_output:
            transforms_list += [HumanPoseImageSave(save_output_frames)]
        #
        return transforms_list, dict(data_layout=data_layout, with_udp=with_udp)

    @classmethod
    def create_transforms_human_pose_estimation_onnx(cls, settings, data_layout=constants.presets.DataLayoutType.NCHW, **kwargs):
        return cls.create_transforms_human_pose_estimation_base(data_layout=data_layout, with_udp=settings.with_udp, **kwargs)

    ###############################################################
    # post process transforms for depth estimation
    ###############################################################
    @classmethod
    def create_transforms_depth_estimation_base(cls, settings, data_layout=None, save_output=False, save_output_frames=50, **kwargs):
        transforms_list = [SqueezeAxis(),
                           NPTensorToImage(data_layout=data_layout),
                           DepthImageResize()]
        if save_output:
            transforms_list += [DepthImageSave(save_output_frames)]
        #
        return transforms_list, dict(data_layout=data_layout)

    @classmethod
    def create_transforms_depth_estimation_onnx(cls, settings, data_layout=constants.presets.DataLayoutType.NCHW, **kwargs):
        return cls.create_transforms_depth_estimation_base(data_layout=data_layout, **kwargs)

    @classmethod
    def create_transforms_lidar_base(cls, settings, **kwargs):
        transforms_list = [
            OD3DOutPutPorcess(settings.detection_threshold)
        ]
        return transforms_list, dict(detection_threshold=settings.detection_threshold)

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
        return transforms_list, dict(data_layout=data_layout)

    @classmethod
    def create_transforms_disparity_estimation_onnx(cls, settings, data_layout=constants.presets.DataLayoutType.NCHW, **kwargs):
        return cls.create_transforms_disparity_estimation_base(data_layout=data_layout, **kwargs)



def no_postprocess(settings, **kwargs):
    return PostProcessTransforms(settings, transforms=[], **kwargs)


def object_detection_postprocess(settings, name='object_detection_postprocess', **kwargs):
    assert settings.common.task_type == constants.TaskType.TASK_TYPE_DETECTION, \
        'object_detection_postprocess can only be used for object detection task type'
    return PostProcessTransforms.from_kwargs(settings, **kwargs)


def segmentation_postprocess(settings, name='segmentation_postprocess', **kwargs):
    assert settings.common.task_type == constants.TaskType.TASK_TYPE_SEGMENTATION, \
        'segmentation_postprocess can only be used for segmentation task type'
    return PostProcessTransforms.from_kwargs(settings, **kwargs)


def keypoint_detection_postprocess(settings, name='keypoint_detection_postprocess', **kwargs):
    assert settings.common.task_type == constants.TaskType.TASK_TYPE_KEYPOINT_DETECTION, \
        'keypoint_detection_postprocess can only be used for keypoint detection task type'
    return PostProcessTransforms.from_kwargs(settings, **kwargs)
