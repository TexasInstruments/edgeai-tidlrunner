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


from ....common import utils
from ....common.bases import transforms_base
from ...settings import constants
from ...settings.constants import presets
from .transforms import *
from .audio_transforms import AudioLoadAndResample, VGGishMelSpectrogram, YAMNetMelSpectrogram, STFTTransform, GCRNSTFTTransform


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

    @classmethod
    def from_kwargs(cls, settings, resize=256, crop=224, data_layout=presets.DataLayoutType.NCHW,
                         reverse_channels=False, backend='cv2', interpolation=None, resize_with_pad=False,
                         add_flip_image=False, pad_color=0, **extra_kwargs):
        # Audio task type dispatch — checked before image logic
        # settings is a nested AttrDict; task_type lives at settings.common.task_type
        task_type = getattr(getattr(settings, 'common', None), 'task_type', None)
        if task_type == constants.TaskType.TASK_TYPE_AUDIO_CLASSIFICATION:
            transforms, transforms_kwargs = cls.create_transforms_audio_classification(settings, **extra_kwargs)
            return cls(settings, transforms, **transforms_kwargs)
        elif task_type == constants.TaskType.TASK_TYPE_AUDIO_SPEECHENHANCEMENT:
            transforms, transforms_kwargs = cls.create_transforms_audio_speechenhancement(settings, **extra_kwargs)
            return cls(settings, transforms, **transforms_kwargs)
        #
        if resize is None:
            transforms_list = [
                # ImageRead(backend=backend),
                ImageCenterCrop(crop),
                ImageToNPTensor4D(data_layout=data_layout)
            ]
        else:
            transforms_list = [
                # ImageRead(backend=backend),
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
                                    add_flip_image=add_flip_image, resize_with_pad=resize_with_pad, pad_color=pad_color)
        return cls(settings, transforms_list, **transforms_kwargs)

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
                                 audio_model_type=audio_model_type)
        return transforms_list, transforms_kwargs

    @classmethod
    def create_transforms_audio_speechenhancement(cls, settings, audio_model_type=None,
                                              sample_rate=16000, audio_duration=4.0, **kwargs):
        if audio_model_type == 'gcrn':
            transforms_list = [GCRNSTFTTransform(sample_rate=sample_rate, audio_duration=audio_duration)]
        else:
            transforms_list = [STFTTransform()]
        return transforms_list, dict(audio_model_type=audio_model_type,
                                     sample_rate=sample_rate, audio_duration=audio_duration)

    def set_size_details(self, resize, crop):
        for t in self.transforms:
            if isinstance(t, ImageResize):
                t.set_size(resize)
            elif isinstance(t, ImageCenterCrop):
                t.set_size(crop)
            #
        #


def no_preprocess(settings, **kwargs):
    return PreProcessTransforms(settings, transforms=[], **kwargs)


def image_preprocess(settings, name='image_preprocess', **kwargs):
    preprocess = PreProcessTransforms.from_kwargs(settings, **kwargs)
    return preprocess


def image_classification_preprocess(settings, name='image_classification_preprocess', resize=256, crop=224, **kwargs):
    assert settings.task_type == constants.TaskType.TASK_TYPE_CLASSIFICATION, \
        'image_classification_preprocess can only be used for image classification task type'
    return image_preprocess(settings, name=name, resize=resize, crop=crop, **kwargs)


def object_detection_preprocess(settings, name='object_detection_preprocess', resize=(512,512), crop=(512,512), **kwargs):
    assert settings.task_type == constants.TaskType.TASK_TYPE_DETECTION, \
        'object_detection_preprocess can only be used for object detection task type'
    return image_preprocess(settings, name=name, resize=resize, crop=crop, **kwargs)


def semantic_segmentation_preprocess(settings, name='semantic_segmentation_preprocess', resize=(512,512), crop=(512,512), **kwargs):
    assert settings.task_type == constants.TaskType.TASK_TYPE_SEGMENTATION, \
        'semantic_segmentation_preprocess can only be used for segmentation task type'
    return image_preprocess(settings, name=name, resize=resize, crop=crop, **kwargs)


def audio_preprocess(settings, **kwargs):
    return PreProcessTransforms.from_kwargs(settings, **kwargs)
