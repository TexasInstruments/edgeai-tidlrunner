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

from ..... import utils


class DatasetBase(utils.ParamsBase):
    def __init__(self, **kwargs):
        super().__init__()
        # this is required to save the params
        self.kwargs = kwargs
        # call the utils.ParamsBase.initialize()
        super().initialize()

    def get_color_map(self, num_classes):
        color_map = utils.get_color_palette(num_classes)
        return color_map

    def get_dataset_info(self, annotation_file_or_dataset_store, with_background_class=False):
        if isinstance(annotation_file_or_dataset_store, str):
            annotation_file = annotation_file_or_dataset_store
            with open(annotation_file) as afp:
                dataset_store = json.load(afp)
            #
        elif isinstance(annotation_file_or_dataset_store, dict):
            dataset_store = annotation_file_or_dataset_store
        else:
            assert False, 'annotation_file_or_dataset_store is in invalid format'
        #
        if 'dataset_info' in self.kwargs:
            return
        #
        # return only info and categories for now as the whole thing could be quite large.
        dataset_info = dict()
        for key in ('info', 'categories'):
            if key in dataset_store.keys():
                dataset_info.update({key: dataset_store[key]})
            #
        #
        if with_background_class:
            min_category_id = min([category['id'] for category in dataset_info['categories']])
            if min_category_id > 0:
                dataset_info['categories'].append(dict(id=0, category=0, supercategory=0))
            #
        #
        num_classes = len(dataset_info['categories'])
        # classes = dataset_store['categories']
        # class_ids = [class_info['id'] for class_info in classes]
        # class_ids_min = min(class_ids)
        # num_classes = max(class_ids) - class_ids_min + 1

        if num_classes is not None:
            dataset_info.update(dict(color_map=self.get_color_map(num_classes)))
        #

        self.kwargs['dataset_info'] = dataset_info
        self.kwargs['num_classes'] = num_classes
        self.kwargs['num_images'] = len(dataset_store['images'])
