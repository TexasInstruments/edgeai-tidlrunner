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

import json
from pycocotools.coco import COCO
import os
import numbers
import random
import numpy as np
from PIL import Image 
from pycocotools.cocoeval import COCOeval


from .....utils.config_utils import dataset_utils
from . import dataset_base
from . import dataloader_utils


class ObjectDetectionDataLoader(dataset_base.DatasetBaseWithUtils):
    def __init__(self, image_dir, annotation_file, with_background_class=False, shuffle=False, backend='cv2', bgr_to_rgb=True):
        super().__init__(shuffle=shuffle)
        self.image_dir = image_dir
        self.annotation_file = annotation_file

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
        label_offset_pred = kwargs.get('label_offset_pred', 0)

        predictions = []
        inputs = []
        for data in run_data:
            predictions.append(data['output'])
            inputs.append(data['input'])

        num_frames = len(predictions)

        imgs_list = list(self.coco_dataset.imgs.items())
        self.coco_dataset.imgs = {k:v for k,v in imgs_list[:num_frames]}

        #os.makedirs(run_path, exist_ok=True)
        detections_formatted_list = []
        for frame_idx, det_frame in enumerate(predictions):
            for det_id, det in enumerate(det_frame):
                det = self._format_detections(det, frame_idx, label_offset=label_offset_pred)
                category_id = det['category_id'] if isinstance(det, dict) else det[4]
                if category_id >= 1: # final coco categories start from 1
                    detections_formatted_list.append(det)
                #
            #
        #
        coco_ap = 0.0
        coco_ap50 = 0.0
        if len(detections_formatted_list) > 0:
            cocoDet = self.coco_dataset.loadRes(detections_formatted_list)
            cocoEval = COCOeval(self.coco_dataset, cocoDet, iouType='bbox')
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            coco_ap = cocoEval.stats[0]
            coco_ap50 = cocoEval.stats[1]
        #
        accuracy = {'accuracy_ap[.5:.95]%': coco_ap*100.0, 'accuracy_ap50%': coco_ap50*100.0}
        return accuracy

    def _format_detections(self, bbox_label_score, image_id, label_offset=0, class_map=None):
        if class_map is not None:
            assert bbox_label_score[4] in class_map, 'invalid prediction label or class_map'
            bbox_label_score[4] = class_map[bbox_label_score[4]]
        #
        bbox_label_score[4] = self._detection_label_to_catid(bbox_label_score[4], label_offset)
        output_dict = dict()
        image_id = self.img_ids[image_id]
        output_dict['image_id'] = image_id
        det_bbox = bbox_label_score[:4]      # json is not support for ndarray - convert to list
        det_bbox = self._xyxy2xywh(det_bbox) # can also be done in postprocess pipeline
        det_bbox = self._to_list(det_bbox)
        output_dict['bbox'] = det_bbox
        output_dict['category_id'] = int(bbox_label_score[4])
        output_dict['score'] = float(bbox_label_score[5])
        return output_dict

    def _detection_label_to_catid(self, label, label_offset, cat_ids=None):
        if isinstance(label_offset, (list,tuple)):
            label = int(label)
            assert label<len(label_offset), 'label_offset is a list/tuple, but its size is smaller than the detected label'
            label = label_offset[label]
        elif isinstance(label_offset, dict):
            if np.isnan(label) or int(label) not in label_offset.keys():
                #print(utils.log_color('\nWARNING', 'detection incorrect', f'detected label: {label}'
                #                                                          f' is not in label_offset dict'))
                label = 0
            else:
                label = label_offset[int(label)]
            #
        elif isinstance(label_offset, numbers.Number):
            label = int(label + label_offset)
        elif cat_ids:
            label = int(label)
            assert label<len(cat_ids), \
                'the detected label could not be mapped to the 90 COCO categories using the default COCO.getCatIds()'
            label = cat_ids[label]
        #
        return label

    def _to_list(self, bbox):
        bbox = [float(x) for x in bbox]
        return bbox

    def _xyxy2xywh(self, bbox):
        bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
        return bbox


class COCODetectionDataLoader(ObjectDetectionDataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def coco_detection_dataloader(settings, name, path, label_path=None, **kwargs):
    if 'val' in os.path.split(path)[-1]:
        data_path = path
    else:
        data_path = os.path.join(path, 'val2017')
        label_path = label_path or os.path.join(path, 'annotations', 'instances_val2017.json')
    #
    return COCODetectionDataLoader(data_path, label_path, **kwargs)
