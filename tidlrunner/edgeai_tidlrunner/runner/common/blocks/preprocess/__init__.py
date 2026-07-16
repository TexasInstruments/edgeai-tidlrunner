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

from ....common import utils
from ....common.bases import transforms_base
from ...settings import constants
from ...settings.constants import presets
from .transforms import *
from .audio_transforms import AudioLoadAndResample, VGGishMelSpectrogram, YAMNetMelSpectrogram, STFTTransform, GCRNSTFTTransform

try:
    from .bev_detection import *
except ImportError as e:
    warnings.warn(f'WARNING: bev_detection postprocessing could not be imported - {str(e)}')


class PreProcessTransforms(transforms_base.TransformsCompose):
    def __init__(self, settings, transforms=None, **kwargs):
        assert transforms is not None, 'transforms must be provided'
        super().__init__(transforms, **kwargs)
        self.settings = settings
        self.input_names = []  # Will be set when we know the input names

    def set_input_names(self, input_names):
        """Set input names to help identify text vs image inputs"""
        self.input_names = input_names

    def _is_text_input(self, tensor, input_name=None):
        """
        Identify if an input is text/tokens vs image based on:
        1. Input name (contains text-related keywords)
        2. Shape (2D tensors are typically text, 4D are images)
        """
        import numpy as np

        # Check 1: Input name contains text-related keywords
        if input_name:
            text_keywords = ['input_ids', 'attention_mask', 'token', 'text', 'input_id', 'attention']
            if any(keyword in input_name.lower() for keyword in text_keywords):
                return True

        # Check 2: Shape-based detection
        # Text inputs are typically 2D: [batch, sequence_length]
        # Image inputs are typically 4D: [batch, channels, height, width] or [batch, height, width, channels]
        if isinstance(tensor, np.ndarray):
            if len(tensor.shape) == 2:
                # 2D tensor - likely text/tokens
                return True
            elif len(tensor.shape) >= 3:
                # 3D or 4D tensor - likely image
                return False

        return False

    def __call__(self, tensor, info_dict):
        """
        Apply preprocessing transforms, but skip text/token inputs in multi-modal models.
        For multi-input models (e.g., CLIP), only preprocess image inputs.
        """
        import numpy as np

        if isinstance(tensor, (list, tuple)):
            # Multiple inputs - process each separately
            processed = []
            for idx, t in enumerate(tensor):
                input_name = self.input_names[idx] if idx < len(self.input_names) else None
                is_text = self._is_text_input(t, input_name)

                if is_text:
                    # Skip preprocessing for text/token inputs
                    processed.append(t)
                else:
                    # Apply preprocessing to image inputs
                    for transform in self.transforms:
                        t, info_dict = transform(t, info_dict)
                    processed.append(t)

            return processed, info_dict
        else:
            # Single input - use parent class behavior
            return super().__call__(tensor, info_dict)

    def __call__(self, tensor, info_dict):
        tensor, info_dict = super().__call__(tensor, info_dict)
        return tensor, info_dict
    
    @classmethod
    def create_transforms_image_preprocess(cls, settings, resize=256, crop=224, data_layout=presets.DataLayoutType.NCHW,
                         reverse_channels=False, backend='cv2', interpolation=None, resize_with_pad=False,
                         add_flip_image=False, pad_color=0, **kwargs):
        # Audio task type dispatch — checked before image logic
        # settings is a nested AttrDict; task_type lives at settings.common.task_type
        task_type = getattr(getattr(settings, 'common', None), 'task_type', None)
        if resize is None:
            transforms_list = [
                # ImageRead(backend=backend),
                # RempoveBatchAxis(),
                ImageCenterCrop(crop),
                ImageToNPTensor4D(data_layout=data_layout)
            ]
        else:
            transforms_list = [
                # ImageRead(backend=backend),
                # RempoveBatchAxis(),
                ImageResize(resize, interpolation=interpolation, resize_with_pad=resize_with_pad, pad_color=pad_color),
                ImageCenterCrop(crop),
                ImageToNPTensor4D(data_layout=data_layout)
            ]

        if reverse_channels:
            transforms_list = transforms_list + [NPTensor4DChanReverse(data_layout=data_layout)]
        if add_flip_image:
            transforms_list += [ImageFlipAdd()]
        #
        transforms_kwargs = dict(resize=resize, crop=crop,
                                    data_layout=data_layout, reverse_channels=reverse_channels,
                                    backend=backend, interpolation=interpolation,
                                    add_flip_image=add_flip_image, resize_with_pad=resize_with_pad, pad_color=pad_color, **kwargs)
        return transforms_list, transforms_kwargs

    @classmethod
    def create_transform_bev_petr(cls, settings, imsize=256, resize=256, crop=224, featsize=(20, 50), queue_length=0,
                        data_layout=presets.DataLayoutType.NCHW, reverse_channels=False,
                        backend='cv2', interpolation=cv2.INTER_AREA, resize_with_pad=False, pad_color=0,
                        name='petr_model_preprocess', **kwargs):
        transforms_list = [
            BEVSensorsRead(imsize, resize, crop),
            ImageRead(backend=backend, bgr_to_rgb=False),
            ImageResize(resize, interpolation=interpolation, resize_with_pad=resize_with_pad, pad_color=pad_color),
            ImageCrop(crop),
            ImageToNPTensor4D(data_layout=data_layout),
        ]

        if queue_length > 0:
            transforms_list += [SetupTemporalQueue(queue_length=queue_length)]
        
        transforms_list += [
            GetPETRGeometry(crop, featsize),
        ]

        transforms_kwargs = dict(imsize=imsize, resize=resize, crop=crop,
                                          data_layout=data_layout, reverse_channels=reverse_channels,
                                          backend=backend, interpolation=interpolation,
                                          resize_with_pad=resize_with_pad, pad_color=pad_color, name=name, **kwargs)
        return transforms_list, transforms_kwargs

    @classmethod
    def create_transform_bev_bevdet(cls, settings, imsize=256, resize=256, crop=224, data_layout=presets.DataLayoutType.NCHW, reverse_channels=False,
                        backend='cv2', interpolation=cv2.INTER_AREA, resize_with_pad=False, pad_color=0,
                        name='bevdet_model_preprocess', **kwargs):
        transforms_list = [
            BEVSensorsRead(imsize, resize, crop),
            ImageRead(backend=backend, bgr_to_rgb=False),
            ImageResize(resize, interpolation=interpolation, resize_with_pad=resize_with_pad, pad_color=pad_color),
            ImageCrop(crop),
            ImageToNPTensor4D(data_layout=data_layout),
            GetBEVDetGeometry(crop)
        ]

        transforms_kwargs = dict(imsize=imsize, resize=resize, crop=crop,
                                          data_layout=data_layout, reverse_channels=reverse_channels,
                                          backend=backend, interpolation=interpolation,
                                          resize_with_pad=resize_with_pad, pad_color=pad_color, name=name, **kwargs)
        return transforms_list, transforms_kwargs

    @classmethod
    def create_transform_bev_bevformer(cls, settings, imsize=256, resize=256, pad=224, bev_size=(50, 50), pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
                        queue_length=0, data_layout=presets.DataLayoutType.NCHW, reverse_channels=False,
                        backend='cv2', interpolation=cv2.INTER_AREA, resize_with_pad=False, pad_color=0,
                        name='bevformer_model_preprocess', **kwargs):
        transforms_list = [
            BEVSensorsRead(imsize, resize, (0, 0, resize[1]+pad[2], resize[0]+pad[3])),
            ImageRead(backend=backend, bgr_to_rgb=True),
            ImageResize(resize, interpolation=interpolation, resize_with_pad=resize_with_pad, pad_color=pad_color),
            ImagePad(pad),
            ImageToNPTensor4D(data_layout=data_layout),
        ]

        if queue_length > 0:
            transforms_list += [SetupTemporalQueue(queue_length=queue_length)]

        transforms_list += [
            GetBEVFormerGeometry(bev_size, pc_range),
        ]

        transforms_kwargs = dict(imsize=imsize, resize=resize, pad=pad,
                                          data_layout=data_layout, reverse_channels=reverse_channels,
                                          backend=backend, interpolation=interpolation,
                                          resize_with_pad=resize_with_pad, pad_color=pad_color, name=name, **kwargs)
        return transforms_list, transforms_kwargs

    @classmethod
    def create_transform_fcos3d(cls, settings, imsize=256, resize=256, pad=224, data_layout=presets.DataLayoutType.NCHW, reverse_channels=False,
                        backend='cv2', interpolation=cv2.INTER_AREA, resize_with_pad=False, pad_color=0,
                        name='fcos3d_model_preprocess', **kwargs):

        transforms_list = [
            BEVSensorsRead(imsize, resize, (0, 0, resize[1]+pad[2], resize[0]+pad[3]), load_type='mv_image_based'),
            ImageRead(backend=backend, bgr_to_rgb=False),
            ImagePad(pad),
            ImageToNPTensor4D(data_layout=data_layout),
            GetFCOS3DGeometry()
        ]

        transforms_kwargs = dict(imsize=imsize, resize=resize, pad=pad,
                                          data_layout=data_layout, reverse_channels=reverse_channels,
                                          backend=backend, interpolation=interpolation,
                                          resize_with_pad=resize_with_pad, pad_color=pad_color, name=name, **kwargs)
        return transforms_list, transforms_kwargs

    @classmethod
    def create_transform_bev_fastbev(cls, settings, imsize=256, resize=256, crop=224, queue_length=0,
                        data_layout=presets.DataLayoutType.NCHW, reverse_channels=False,
                        backend='cv2', interpolation=cv2.INTER_AREA, resize_with_pad=False, pad_color=0,
                        name='fastbev_model_preprocess', **kwargs):
        transforms_list = [
            BEVSensorsRead(imsize, resize, crop),
            ImageRead(backend=backend, bgr_to_rgb=True),
            ImageResize(resize, interpolation=interpolation, resize_with_pad=resize_with_pad, pad_color=pad_color),
            ImageCrop(crop),
            ImageToNPTensor4D(data_layout=data_layout),
        ]

        if queue_length > 0:
            transforms_list += [SetupTemporalQueue(queue_length=queue_length)]

        transforms_list += [
            GetFastBEVGeometry(crop)
        ]

        transforms_kwargs = dict(imsize=imsize, resize=resize, crop=crop,
                                          data_layout=data_layout, reverse_channels=reverse_channels,
                                          backend=backend, interpolation=interpolation,
                                          resize_with_pad=resize_with_pad, pad_color=pad_color, name=name, **kwargs)
        return transforms_list, transforms_kwargs

    @classmethod
    def create_transform_bev_streampetr(cls, settings, imsize=256, resize=256, crop=224, queue_length=0,
                        data_layout=presets.DataLayoutType.NCHW, reverse_channels=False,
                        backend='cv2', interpolation=cv2.INTER_AREA, resize_with_pad=False, pad_color=0,
                        name='streampetr_model_preprocess', **kwargs):
        transforms_list = [
            BEVSensorsRead(imsize, resize, crop),
            ImageRead(backend=backend, bgr_to_rgb=True),
            ImageResize(resize, interpolation=interpolation, resize_with_pad=resize_with_pad, pad_color=pad_color),
            ImageCrop(crop),
            ImageToNPTensor4D(data_layout=data_layout),
        ]

        if queue_length > 0:
            transforms_list += [SetupTemporalQueue(queue_length=queue_length)]

        transforms_list += [
            GetStreamPETRGeometry()
        ]

        transforms_kwargs = dict(imsize=imsize, resize=resize, crop=crop,
                                          data_layout=data_layout, reverse_channels=reverse_channels,
                                          backend=backend, interpolation=interpolation,
                                          resize_with_pad=resize_with_pad, pad_color=pad_color, name=name, **kwargs)
        return transforms_list, transforms_kwargs

    @classmethod
    def create_transform_bev_far3d(cls, settings, imsize=256, resize=256, crop=224, queue_length=0,
                        data_layout=presets.DataLayoutType.NCHW, reverse_channels=False,
                        backend='cv2', interpolation=cv2.INTER_AREA, resize_with_pad=False, pad_color=0,
                        name='far3d_model_preprocess', **kwargs):
        transforms_list = [
            BEVSensorsRead(imsize, resize, crop),
            ImageRead(backend=backend, bgr_to_rgb=False),
            ImageResize(resize, interpolation=interpolation, resize_with_pad=resize_with_pad, pad_color=pad_color),
            ImageCrop(crop),
            ImageToNPTensor4D(data_layout=data_layout),
        ]

        if queue_length > 0:
            transforms_list += [SetupTemporalQueue(queue_length=queue_length)]

        transforms_list += [
            GetFar3DGeometry()
        ]

        transforms_kwargs = dict(imsize=imsize, resize=resize, crop=crop,
                                          data_layout=data_layout, reverse_channels=reverse_channels,
                                          backend=backend, interpolation=interpolation,
                                          resize_with_pad=resize_with_pad, pad_color=pad_color, name=name, **kwargs)
        return transforms_list, transforms_kwargs

    @classmethod
    def create_transform_bev_sparse4d(cls, settings, imsize=256, resize=256, crop=224, queue_length=0,
                        data_layout=presets.DataLayoutType.NCHW, reverse_channels=False,
                        backend='cv2', interpolation=cv2.INTER_AREA, resize_with_pad=False, pad_color=0,
                        name='sparse4d_model_preprocess', **kwargs):
        transforms_list = [
            BEVSensorsRead(imsize, resize, crop),
            ImageRead(backend=backend, bgr_to_rgb=True),
            ImageResize(resize, interpolation=interpolation, resize_with_pad=resize_with_pad, pad_color=pad_color),
            ImageCrop(crop),
            ImageToNPTensor4D(data_layout=data_layout),
        ]

        if queue_length > 0:
            transforms_list += [SetupTemporalQueue(queue_length=queue_length)]

        transforms_list += [
            GetSparse4DHistory()
        ]

        kwargs = dict(imsize=imsize, resize=resize, crop=crop,
                                          data_layout=data_layout, reverse_channels=reverse_channels,
                                          backend=backend, interpolation=interpolation,
                                          resize_with_pad=resize_with_pad, pad_color=pad_color, name=name, **kwargs)
        return transforms_list, kwargs

    ###############################################################
    # audio preprocessing classmethods
    ###############################################################
    @classmethod
    def create_transforms_audio_classification(cls, settings, sample_rate=16000, audio_duration=4.0,
                                                audio_model_type=None, **kwargs):
        if audio_model_type == 'yamnet':
            transforms_list = [YAMNetMelSpectrogram(sample_rate=sample_rate)]
        else:
            transforms_list = [VGGishMelSpectrogram(sample_rate=sample_rate, audio_duration=audio_duration)]
        transforms_kwargs = dict(sample_rate=sample_rate, audio_duration=audio_duration,
                                 audio_model_type=audio_model_type, **kwargs)
        return transforms_list, transforms_kwargs

    @classmethod
    def create_transforms_audio_speechenhancement(cls, settings, audio_model_type=None,
                                              sample_rate=16000, audio_duration=4.0, **kwargs):
        if audio_model_type == 'gcrn':
            transforms_list = [GCRNSTFTTransform(sample_rate=sample_rate, audio_duration=audio_duration)]
        else:
            transforms_list = [STFTTransform()]
        return transforms_list, dict(audio_model_type=audio_model_type,
                                     sample_rate=sample_rate, audio_duration=audio_duration, **kwargs)

    def set_size_details(self, resize, crop):
        for t in self.transforms:
            if isinstance(t, ImageResize):
                t.set_size(resize)
            elif isinstance(t, ImageCenterCrop):
                t.set_size(crop)
            #
        #


def no_preprocess(settings, name='no_preprocess', **kwargs):
    return PreProcessTransforms(settings, transforms=[], name=name, **kwargs)


def image_preprocess(settings, name='image_preprocess', **kwargs):
    transforms_list, kwargs = PreProcessTransforms.create_transforms_image_preprocess(settings, name=name, **kwargs)
    preprocess = PreProcessTransforms(settings, transforms_list, **kwargs)
    return preprocess


def image_classification_preprocess(settings, name='image_classification_preprocess', resize=256, crop=224, **kwargs):
    return image_preprocess(settings, name=name, resize=resize, crop=crop, **kwargs)


def object_detection_preprocess(settings, name='object_detection_preprocess', resize=(512,512), crop=(512,512), **kwargs):
    return image_preprocess(settings, name=name, resize=resize, crop=crop, **kwargs)


def semantic_segmentation_preprocess(settings, name='semantic_segmentation_preprocess', resize=(512,512), crop=(512,512), **kwargs):
    return image_preprocess(settings, name=name, resize=resize, crop=crop, **kwargs)


def fastbev_model_preprocess(settings, name='fastbev_model_preprocess', **kwargs):
    transforms_list, transforms_kwargs = PreProcessTransforms.create_transform_bev_fastbev(settings, name=name, **kwargs)
    return PreProcessTransforms(settings, transforms_list, **transforms_kwargs)


def bevformer_model_preprocess(settings, name='bevformer_model_preprocess', **kwargs):
    transforms_list, transforms_kwargs = PreProcessTransforms.create_transform_bev_bevformer(settings, name=name, **kwargs)
    return PreProcessTransforms(settings, transforms_list, **transforms_kwargs)


def audio_classification_preprocess(settings, name='audio_classification_preprocess', **kwargs):
    transforms_list, transforms_kwargs = PreProcessTransforms.create_transforms_audio_classification(settings, name=name, **kwargs)
    return PreProcessTransforms(settings, transforms_list, **transforms_kwargs)


def audio_speechenhancement_preprocess(settings, name='audio_speechenhancement_preprocess', **kwargs):
    transforms_list, transforms_kwargs = PreProcessTransforms.create_transforms_audio_speechenhancement(settings, name=name, **kwargs)
    return PreProcessTransforms(settings, transforms_list, **transforms_kwargs)

