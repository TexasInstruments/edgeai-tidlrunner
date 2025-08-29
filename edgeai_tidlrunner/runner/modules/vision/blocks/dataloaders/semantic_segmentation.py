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
from PIL import Image 
import numpy as np
from PIL import Image
import os
from pycocotools.coco import COCO
import json

from . import dataset_base
from . import dataloader_utils


class SemanticSegmentationDataLoader(dataset_base.DatasetBaseWithUtils):
    def __init__(self, image_dir, annotation_file, with_background_class=False, shuffle=False, backend='cv2', bgr_to_rgb=True):
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
        self.kwargs['num_classes'] = len(self.cat_ids)              
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

    def __getitem__(self, index, info_dict=None):
        img_id = self.img_ids[index]
        img = self.coco_dataset.loadImgs([img_id])[0]
        image_path = os.path.join(self.image_dir, img['file_name'])
        return self.image_reader(image_path, info_dict)

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
        ann_ids = []
        anns = []
        gt_masks = []

        for id,input in zip(self.img_ids,inputs):
            ann_id = self.coco.getAnnIds(imgIds=id)
            ann=self.coco.loadAnns(ann_id)
            ann_ids.append(ann_id)
            anns.append(ann)
            gt_mask = np.zeros((input.shape[-2],input.shape[-1]),dtype=np.uint8)
            gt_mask.fill(255)
            for an in ann:
                if 'segmentation' in an:
                    coco_class = an['category_id']
                    if self.category_map_gt != None:
                        if coco_class in self.category_map_gt:
                            voc_class = self.category_map_gt[coco_class]
                            mask = self.coco.annToMask(an)
                            mask_resized = np.array(Image.fromarray(mask).resize((input.shape[-2],input.shape[-1]), resample=Image.NEAREST))
                            gt_mask[mask_resized == 1] = voc_class
                        #
                    else:
                        voc_class = coco_class
                        mask = self.coco.annToMask(an)
                        mask_resized = np.array(Image.fromarray(mask).resize((input.shape[-2],input.shape[-1]), resample=Image.NEAREST))
                        gt_mask[mask_resized == 1] = voc_class
                    #
                #
            #
            gt_masks.append(gt_mask)
        #
        metric = SegmentationEvaluationMetrics()
        for output, gt_mask, input in zip(predictions,gt_masks,inputs):
            # assuming argmax has already happened in postprocess
            output = output[0] if isinstance(output, list) else output
            output = output[0] if isinstance(output, list) else output            
            output = output.squeeze(0) if hasattr(output, 'shape') and output.shape[0] == 1 else output
            output = output.squeeze(0) if hasattr(output, 'shape') and output.shape[0] == 1 else output
            metric.update(output, gt_mask)
        #
        mean_iou = metric.compute_iou()
        pixel_accuracy = metric.compute_pixel_accuracy()
        accuracy = {'accuracy_mean_iou%':mean_iou*100, 'accuracy_pixel_accuracy%' : pixel_accuracy*100}
        return accuracy


class COCOSegmentationDataLoader(SemanticSegmentationDataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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


def coco_segmentation_dataloader(name, path, label_path=None, shuffle=False):
    if 'val' in os.path.split(path)[-1]:
        data_path = path
    else:
        data_path = os.path.join(path, 'val2017')
        label_path = label_path or os.path.join(path, 'annotations', 'instances_val2017.json')
    #
    return COCOSegmentationDataLoader(data_path,label_path, shuffle=shuffle)


class SegmentationEvaluationMetrics:
    def __init__(self , num_classes = 21 , ignore_index = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.confusion_matrx = np.zeros((self.num_classes,self.num_classes))

    def update(self,pred,target):
        mask = target != self.ignore_index
        pred = pred[mask]
        target = target[mask]

        for t , p in zip(target.flatten(),pred.flatten()):
            self.confusion_matrx[t,p] += 1

    def compute_iou(self):
        intersection = np.diag(self.confusion_matrx)
        ground_truth_set = self.confusion_matrx.sum(axis=1)
        predicted_Set = self.confusion_matrx.sum(axis=0)
        union = ground_truth_set + predicted_Set - intersection
        IoU = intersection / np.maximum(union,1e-10) 
        valid_classes = ground_truth_set > 0
        mean_IoU = np.mean(IoU[valid_classes])
        return mean_IoU
    
    def compute_pixel_accuracy(self):
        correct = np.diag(self.confusion_matrx).sum()
        total = self.confusion_matrx.sum()
        return correct / total if total > 0 else 0

