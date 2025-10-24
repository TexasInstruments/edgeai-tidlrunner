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


class PreProcessTransforms(transforms_base.TransformsCompose):
    def __init__(self, settings, transforms=None, **kwargs):
        assert transforms is not None, 'transforms must be provided'
        super().__init__(transforms, **kwargs)
        self.settings = settings
    
    @classmethod
    def from_kwargs(cls, settings, resize=256, crop=224, data_layout=presets.DataLayoutType.NCHW, 
                         reverse_channels=False, backend='cv2', interpolation=None, resize_with_pad=False,
                         add_flip_image=False, pad_color=0):
        if resize is None:
            transforms_list = [
                ImageRead(backend=backend),
                ImageCenterCrop(crop),
                ImageToNPTensor4D(data_layout=data_layout)
            ]
        else:
            transforms_list = [
                ImageRead(backend=backend),
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
