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
import numpy as np
from PIL import Image 
from pycocotools.cocoeval import COCOeval


from .....utils.config_utils import dataset_utils
from . import dataset_base


class ObjectDetectionDataLoader(dataset_base.DatasetBase):
    def __init__(self, img_dir, annotation_file, with_background_class=False):
        super().__init__()
        self.img_dir = img_dir
        self.ann_file = annotation_file
        coco = COCO(annotation_file)
        self.img_ids = coco.getImgIds()
        self.img_info = coco.loadImgs(self.img_ids[:])
        self.cat_ids = coco.getCatIds()

        with open(annotation_file) as afp:
            dataset_store = json.load(afp)
        #
        self.get_dataset_info(dataset_store, with_background_class)
        self.with_background_class = with_background_class
        if with_background_class and self.kwargs['dataset_info']['categories'] > len(self.cat_ids):
            self.cat_ids.insert(0, 0)
        #

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_info[index]['file_name'])
        # img = Image.open(img_path).convert('RGB')
        # input_tensor = self.preprocess(img).cpu()  # Add batch dimension
        # batch = torch.stack([input_tensor]).numpy()
        return img_path
    
    def __len__(self):
        return self.kwargs['num_images']
    
    def get_num_classes(self):
        return self.kwargs['num_classes']
    
    def evaluate(self, run_data, **kwargs):
        label_offset_pred = kwargs.get('label_offset_pred', 0)

        predictions = []
        inputs = []
        for data in run_data:
            predictions.append(data['output'])
            inputs.append(data['input'])

        coco = COCO(self.ann_file)
        img_ids = self.img_ids

        detections_formatted_list = []
        length = len(predictions)
        for frame_idx, det_frame in enumerate(predictions):
            for det_id, det in enumerate(det_frame):
                det = self._format_detections(det, frame_idx, label_offset=label_offset_pred)
                category_id = det['category_id'] if isinstance(det, dict) else det[4]
                if category_id >= 1: # final coco categories start from 1
                    detections_formatted_list.append(det)
                #
            #
        #

        coco_dt = coco.loadRes(detections_formatted_list)
        coco_eval = COCOeval(coco, coco_dt, iouType='bbox')
        coco_eval.params.imgIds = img_ids[:length]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        coco_ap = coco_eval.stats[0]
        coco_ap50 = coco_eval.stats[1]
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


def coco_detection_dataloader(name, path, label_path=None):
    if 'val' in os.path.split(path)[-1]:
        data_path = path
    else:
        data_path = os.path.join(path, 'val2017')
        label_path = label_path or os.path.join(path, 'annotations', 'instances_val2017.json')
    #
    return COCODetectionDataLoader(data_path, label_path)
