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
from PIL import Image 
import numpy as np
from PIL import Image
import os
from pycocotools.coco import COCO
import json

from . import dataset_base


class SemanticSegmentationDataLoader(dataset_base.DatasetBase):
    def __init__(self, img_dir, annotation_file, with_background_class=False):
        self.coco = COCO(annotation_file)
        self.img_dir = img_dir
        self.img_ids = self.coco.getImgIds()      
        self.category_map_gt = None       
        with open(annotation_file) as afp:
            self.dataset_store = json.load(afp)
        #
        self.kwargs['dataset_info'] = self.get_dataset_info()

        self.with_background_class = with_background_class
        self.min_class_id = min(self.cat_ids)
        if self.with_background_class and self.min_class_id > 0:
            self.num_classes = len(self.cat_ids) + 1
        else:
            self.num_classes = len(self.cat_ids)
        #

    def __getitem__(self, index):
        img_info = self.coco.loadImgs([self.img_ids[index]])[0]
        img_path = os.path.join(self.img_dir,img_info['file_name'])
        return img_path
    
    def __len__(self):
        num_images = len(self.img_ids)
        return num_images  

    def get_num_classes(self):
        return self.num_classes
    
    def get_dataset_info(self):
        if 'dataset_info' in self.kwargs:
            return self.kwargs['dataset_info']
        #
        # return only info and categories for now as the whole thing could be quite large.
        dataset_store = dict()
        for key in ('info', 'categories'):
            if key in self.dataset_info.keys():
                dataset_store.update({key: self.dataset_info[key]})
            #
        #
        if self.kwargs['num_classes'] is not None:
            dataset_store.update(dict(color_map=self.get_color_map()))
        #
        return dataset_store
    
    def evaluate(self, run_data, **kwargs):
        predictions = []
        inputs = []
        for data in run_data:
            predictions.append(data['output'])
            inputs.append(data['input'])
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
                    else:
                        voc_class = coco_class
                        mask = self.coco.annToMask(an)
                        mask_resized = np.array(Image.fromarray(mask).resize((input.shape[-2],input.shape[-1]), resample=Image.NEAREST))
                        gt_mask[mask_resized == 1] = voc_class
            gt_masks.append(gt_mask)
            
        metric = EvaluationMetrics()
        for output , gt_mask , input in zip(predictions,gt_masks,inputs):
            output_np = output[0]  # shape: (num_classes, H, W)
            pred = output_np.argmax(axis=1).squeeze()
            metric.update(pred,gt_mask)
        mean_iou = metric.compute_iou()
        pixel_accuracy = metric.compute_pixel_accuracy()
        accuracy = {'mean_iou':mean_iou , 'pixel_accuracy' : pixel_accuracy}
        return accuracy
    


def semantic_segmentation_dataloader(name, path, label_path=None):
    return SemanticSegmentationDataLoader(path,label_path)


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


def coco_segmentation_dataloader(name, path, label_path=None):
    return COCOSegmentationDataLoader(path,label_path)


class EvaluationMetrics:
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

