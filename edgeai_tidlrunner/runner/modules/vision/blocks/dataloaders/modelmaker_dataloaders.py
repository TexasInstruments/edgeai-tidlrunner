# Copyright (c) 2018-2025, Texas Instruments Incorporated
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
import json
import PIL
import numpy as np
import cv2
import random
import tempfile
import copy
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask

from ..... import utils

from . import dataset_base
from . import object_detection
from . import image_classification
from . import semantic_segmentation


class ModelmakerDetectionDataset(object_detection.ObjectDetectionDataLoader):
    pass


def modelmaker_detection_dataloader(name, path, label_path=None):
    is_images_path = 'val' in os.path.split(path)[-1] or 'images' in os.path.split(path)[-1]
    if is_images_path:
        data_path = path
    else:
        data_path = os.path.join(path, 'images')
        label_path = label_path or os.path.join(path, 'annotations', 'instances.json')
    #
    return ModelmakerDetectionDataset(data_path, label_path)


####################################################################################################
class ModelMakerClassificationDataset(image_classification.ImageClassificationDataLoader):
    def __init__(self, img_dir, annotation_file, with_background_class=False):
        super().__init__(img_dir, annotation_file)
        self.image_dir = img_dir

        with open(annotation_file) as afp:
            dataset_store = json.load(afp)
        #
        self.get_dataset_info(dataset_store, with_background_class)
        self.with_background_class = with_background_class
        self.annotations_info = self._find_annotations_info(dataset_store)
        self.dataset_store = dataset_store

    def __getitem__(self, idx, with_label=False, **kwargs):
        image_info = self.dataset_store['images'][idx]
        filename = os.path.join(self.image_dir, image_info['file_name'])
        label = self.annotations_info[idx][0]['category_id']
        if with_label:
            return filename, label
        else:
            return filename

    def get_num_classes(self):
        return self.kwargs['num_classes']

    def __len__(self):
        return self.kwargs['num_images']

    def evaluate(self, predictions, **kwargs):
        metric_tracker = utils.AverageMeter(name='accuracy_top1%')
        num_frames = min(self.num_frames, len(predictions))
        for n in range(num_frames):
            words = self.__getitem__(n, with_label=True)
            gt_label = int(words[1])
            accuracy = self._classification_accuracy(predictions[n], gt_label, **kwargs)
            metric_tracker.update(accuracy)
        #
        return {metric_tracker.name: metric_tracker.avg}

    def _classification_accuracy(self, prediction, target, label_offset_pred=0, label_offset_gt=0,
                                multiplier=100.0, **kwargs):
        prediction = prediction + label_offset_pred
        target = target + label_offset_gt
        accuracy = 1.0 if (prediction == target) else 0.0
        accuracy = accuracy * multiplier
        return accuracy

    def _find_annotations_info(self, dataset_store):
        image_id_to_file_id_dict = dict()
        file_id_to_image_id_dict = dict()
        annotations_info_list = []
        for file_id, image_info in enumerate(dataset_store['images']):
            image_id = image_info['id']
            image_id_to_file_id_dict[image_id] = file_id
            file_id_to_image_id_dict[file_id] = image_id
            annotations_info_list.append([])
        #
        for annotation_info in dataset_store['annotations']:
            if annotation_info:
                image_id = annotation_info['image_id']
                file_id = image_id_to_file_id_dict[image_id]
                annotations_info_list[file_id].append(annotation_info)
            #
        #
        return annotations_info_list


def modelmaker_classification_dataloader(name, path, label_path):
    is_images_path = 'val' in os.path.split(path)[-1] or 'images' in os.path.split(path)[-1]
    if is_images_path:
        data_path = path
    else:
        data_path = os.path.join(path, 'images')
        label_path = label_path or os.path.join(path, 'annotations', 'labels.json')
    #
    return ModelMakerClassificationDataset(data_path, label_path)


####################################################################################################
class ModelMakerSegmentationDataset(semantic_segmentation.SemanticSegmentationDataLoader):
    def __init__(self, img_dir, annotation_file, with_background_class=True, **kwargs):
        super().__init__(img_dir, annotation_file, with_background_class)


def modelmaker_segmentation_dataloader(name, path, label_path):
    is_images_path = 'val' in os.path.split(path)[-1] or 'images' in os.path.split(path)[-1]
    if is_images_path:
        data_path = path
    else:
        data_path = os.path.join(path, 'images')
        label_path = label_path or os.path.join(path, 'annotations', 'instances.json')
    #
    return ModelMakerSegmentationDataset(data_path, label_path)
