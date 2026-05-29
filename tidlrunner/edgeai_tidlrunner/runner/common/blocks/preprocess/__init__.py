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
                                    add_flip_image=add_flip_image, resize_with_pad=resize_with_pad, pad_color=pad_color, **kwargs)
        return transforms_list, transforms_kwargs

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


def audio_classification_preprocess(settings, name='audio_classification_preprocess', **kwargs):
    transforms_list, kwargs = PreProcessTransforms.create_transforms_audio_classification(settings, name=name, **kwargs)
    return PreProcessTransforms(settings, transforms_list, **kwargs)


def audio_speechenhancement_preprocess(settings, name='audio_speechenhancement_preprocess', **kwargs):
    transforms_list, kwargs = PreProcessTransforms.create_transforms_audio_speechenhancement(settings, name=name, **kwargs)
    return PreProcessTransforms(settings, transforms_list, **kwargs)

