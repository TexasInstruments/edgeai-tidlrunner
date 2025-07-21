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
        self.img_info = coco.loadImgs(self.img_ids[:])[0]
        self.category_map_gt = None

        with open(annotation_file) as afp:
            dataset_store = json.load(afp)
        #
        self.get_dataset_info(dataset_store, with_background_class)
        self.with_background_class = with_background_class

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_info['file_name'])
        # img = Image.open(img_path).convert('RGB')
        # input_tensor = self.preprocess(img).cpu()  # Add batch dimension
        # batch = torch.stack([input_tensor]).numpy()
        return img_path
    
    def __len__(self):
        return self.kwargs['num_images']
    
    def get_num_classes(self):
        return self.kwargs['num_classes']
    
    def evaluate(self, run_data, label_offset_pred=None, **kwargs):
        predictions = []
        inputs = []
        for data in run_data:
            predictions.append(data['output'])
            inputs.append(data['input'])

        results = []
        coco = COCO(self.ann_file)
        img_ids = self.img_ids
        length = len(self)
        for itr in range(length):
            pred = predictions[itr]
            if len(pred) == 2 and pred[0].shape[-1] == 5:
                boxes = pred[0][:,:4]
                scores = pred[0][:,-1]
                labels = pred[1]
            else:
                boxes = pred[0]
                scores = pred[1]
                labels = pred[2]

            if boxes.ndim == 2 and boxes.shape[0] > 0:
                boxes[:, 2:] -= boxes[:, :2]
                for box, score, label in zip(boxes, scores, labels):
                    results.append({
                        "image_id": img_ids[itr],
                        "category_id": int(label),
                        "bbox": [float(x) for x in box],
                        "score": float(score)
                    })
        coco_dt = coco.loadRes(results)
        coco_eval = COCOeval(coco, coco_dt, iouType='bbox')
        coco_eval.params.imgIds = img_ids[:length]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        coco_ap = coco_eval.stats[0]
        coco_ap50 = coco_eval.stats[1]
        accuracy = {'accuracy_ap[.5:.95]%': coco_ap*100.0, 'accuracy_ap50%': coco_ap50*100.0}

        return accuracy


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
