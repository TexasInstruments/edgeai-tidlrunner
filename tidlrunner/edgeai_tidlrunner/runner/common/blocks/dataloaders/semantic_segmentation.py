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
import random
import copy
from PIL import Image 
import numpy as np
from PIL import Image
import os
import json
import cv2
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask

from . import dataset_base
from . import dataloader_utils


class SemanticSegmentationDataLoader(dataset_base.DatasetBaseWithUtils):
    def __init__(self, image_dir, annotation_file, categories=None, with_background_class=False, shuffle=True, backend='cv2', bgr_to_rgb=True):
        super().__init__(shuffle=shuffle)
        self.coco = COCO(annotation_file)
        self.image_dir = image_dir
        self.annotation_file = annotation_file
        self.category_map_gt = None  

        self._load_dataset()

        self.get_dataset_info(annotation_file, with_background_class)
        self.with_background_class = with_background_class
        if with_background_class and self.kwargs['dataset_info']['categories'] > len(self.cat_ids):
            self.cat_ids.insert(0, 0)
        #    
        self.categories = categories or range(categories)     
        self.kwargs['num_classes'] = len(self.categories) if categories else len(self.cat_ids)     
        self.image_reader = dataloader_utils.ImageRead(backend=backend, bgr_to_rgb=bgr_to_rgb)

    def _load_dataset(self):
        shuffle = self.kwargs.get('shuffle', False)
        self.coco_dataset = COCO(self.annotation_file)
        filter_imgs = self.kwargs['filter_imgs'] if 'filter_imgs' in self.kwargs else None
        if isinstance(filter_imgs, str):
            # filter images with the given list
            filter_imgs = os.path.join(self.kwargs['path'], filter_imgs)
            with open(filter_imgs) as filter_fp:
                filter = [int(id) for id in list(filter_fp)]
                orig_keys = list(self.coco_dataset.imgs)
                orig_keys = [k for k in orig_keys if k in filter]
                self.coco_dataset.imgs = {k: self.coco_dataset.imgs[k] for k in orig_keys}
            #
        elif filter_imgs:
            # filter and use images with gt only
            sel_keys = []
            for img_key, img_anns in self.coco_dataset.imgToAnns.items():
                if len(img_anns) > 0:
                    sel_keys.append(img_key)
                #
            #
            self.coco_dataset.imgs = {k: self.coco_dataset.imgs[k] for k in sel_keys}
        #

        max_frames = len(self.coco_dataset.imgs)
        num_frames = self.kwargs.get('num_frames', None)
        num_frames = min(num_frames, max_frames) if num_frames is not None else max_frames

        imgs_list = list(self.coco_dataset.imgs.items())
        if shuffle:
            random.seed(int(shuffle))
            random.shuffle(imgs_list)
        #
        self.coco_dataset.imgs = {k:v for k,v in imgs_list[:num_frames]}

        self.cat_ids = self.coco_dataset.getCatIds()
        self.img_ids = self.coco_dataset.getImgIds()
        self.num_frames = self.kwargs['num_frames'] = num_frames

    def __getitem__(self, index, info_dict=None, with_label=False):
        img_id = self.img_ids[index]
        img = self.coco_dataset.loadImgs([img_id])[0]
        image_path = os.path.join(self.image_dir, img['file_name'])     
        image, info_dict = self.image_reader(image_path, info_dict)           
        if with_label:
            ann_ids = self.coco_dataset.getAnnIds(imgIds=img_id, iscrowd=None)
            anno = self.coco_dataset.loadAnns(ann_ids)
            image = Image.open(image_path)
            image, anno = self._filter_and_remap_categories(image, anno)
            image, target = self._convert_polys_to_mask(image, anno)
            return image, info_dict, target
        else:
            return image, info_dict
    
    def __len__(self):
        return self.kwargs['num_frames']

    def get_num_classes(self):
        return self.kwargs['num_classes']
    
    def evaluate(self, run_data, **kwargs):
        predictions = []
        inputs = []
        for data in run_data:
            predictions.append(data['output'])
            inputs.append(data['input'])
        #

        cmatrix = None
        num_frames = min(self.num_frames, len(predictions))
        for n in range(num_frames):
            image, info_dict, label_img = self.__getitem__(n, with_label=True)
            # reshape prediction is needed
            output = predictions[n]
            output = output.astype(np.uint8)
            output = output[0] if (output.ndim > 2 and output.shape[0] == 1) else output
            output = output[:2] if (output.ndim > 2 and output.shape[2] == 1) else output
            # compute metric
            cmatrix = dataloader_utils.confusion_matrix(cmatrix, output, label_img, self.kwargs['num_classes'])
        #
        accuracy = dataloader_utils.segmentation_accuracy(cmatrix)
        return accuracy

    def _remove_images_without_annotations(self, img_ids):
        ids = []
        for ds_idx, img_id in enumerate(img_ids):
            ann_ids = self.coco_dataset.getAnnIds(imgIds=img_id, iscrowd=None)
            anno = self.coco_dataset.loadAnns(ann_ids)
            if self.categories:
                anno = [obj for obj in anno if obj["category_id"] in self.categories]
            if self._has_valid_annotation(anno):
                ids.append(img_id)
            #
        #
        return ids

    def _has_valid_annotation(self, anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        # if more than 1k pixels occupied in the image
        return sum(obj["area"] for obj in anno) > 1000

    def _filter_and_remap_categories(self, image, anno, remap=True):
        anno = [obj for obj in anno if obj["category_id"] in self.categories]
        if not remap:
            return image, anno
        #
        anno = copy.deepcopy(anno)
        for obj in anno:
            obj["category_id"] = self.categories.index(obj["category_id"])
        #
        return image, anno

    def _convert_polys_to_mask(self, image, anno):
        w, h = image.size
        segmentations = [obj["segmentation"] for obj in anno]
        cats = [obj["category_id"] for obj in anno]
        if segmentations:
            masks = self._convert_poly_to_mask(segmentations, h, w)
            cats = np.array(cats, dtype=masks.dtype)
            cats = cats.reshape(-1, 1, 1)
            # merge all instance masks into a single segmentation map
            # with its corresponding categories
            target = (masks * cats).max(axis=0)
            # discard overlapping instances
            target[masks.sum(0) > 1] = 255
        else:
            target = np.zeros((h, w), dtype=np.uint8)
        #
        return image, target

    def _convert_poly_to_mask(self, segmentations, height, width):
        masks = []
        for polygons in segmentations:
            rles = coco_mask.frPyObjects(polygons, height, width)
            mask = coco_mask.decode(rles)
            if len(mask.shape) < 3:
                mask = mask[..., None]
            mask = mask.any(axis=2)
            mask = mask.astype(np.uint8)
            masks.append(mask)
        if masks:
            masks = np.stack(masks, axis=0)
        else:
            masks = np.zeros((0, height, width), dtype=np.uint8)
        return masks
    

class COCOSegmentationDataLoader(SemanticSegmentationDataLoader):
    def __init__(self, *args, **kwargs):
        # coco_to_voc class mapping
        self.category_map_gt = {
            0:0,
            5:1,
            2:2,
            16:3,
            9:4,
            44:5,
            6:6,
            3:7,
            17:8,
            62:9,
            21:10,
            67:11,
            18:12,
            19:13,
            4:14,
            1:15,
            64:16,
            20:17,
            63:18,
            7:19,
            72:20,
        } 
        categories = list(self.category_map_gt.keys())
        super().__init__(*args, categories=categories, **kwargs)


def coco_segmentation_dataloader(settings, name, path, label_path=None, **kwargs):
    if 'val' in os.path.split(path)[-1]:
        data_path = path
    else:
        data_path = os.path.join(path, 'val2017')
        label_path = label_path or os.path.join(path, 'annotations', 'instances_val2017.json')
    #
    return COCOSegmentationDataLoader(data_path, label_path, **kwargs)
