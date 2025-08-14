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


import PIL
import PIL.Image
import cv2
import numpy as np


class ImageRead(object):
    def __init__(self, backend='pil', bgr_to_rgb=True):
        assert backend in ('pil', 'cv2'), f'backend must be one of pil or cv2. got {backend}'
        self.backend = backend
        self.bgr_to_rgb = bgr_to_rgb

    def __call__(self, path, info_dict=None):
        info_dict = info_dict or dict()
        if isinstance(path, str):
            img_data = None
            if self.backend == 'pil':
                img_data = PIL.Image.open(path)
                img_data = img_data.convert('RGB')
                info_dict['data_shape'] = img_data.size[1], img_data.size[0], len(img_data.getbands())
            elif self.backend == 'cv2':
                img_data = cv2.imread(path)
                if img_data.shape[-1] == 1:
                    img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2BGR)
                elif img_data.shape[-1] == 4:
                    img_data = cv2.cvtColor(img_data, cv2.COLOR_BGRA2BGR)

                # Convert to RGB format
                if self.bgr_to_rgb == True:
                    img_data = img_data[:,:,::-1]
                info_dict['data_shape'] = img_data.shape
            #
            info_dict['data'] = img_data
            info_dict['data_path'] = path
        elif isinstance(path, np.ndarray):
            img_data = path
            info_dict['data_shape'] = img_data.shape
            info_dict['data'] = img_data
            info_dict['data_path'] = './'
        elif isinstance(path, tuple):
            img_data = []
            data_path = []
            for i, img_path in enumerate(path):
                data_path.append(img_path)
                if self.backend == 'pil':
                    img_data.append(PIL.Image.open(img_path))
                    img_data[i] = img_data[i].convert('RGB')
                elif self.backend == 'cv2':
                    img_data.append(cv2.imread(img_path))
                    if img_data[i].shape[-1] == 1:
                        img_data[i] = cv2.cvtColor(img_data[i], cv2.COLOR_GRAY2BGR)
                    elif img_data[i].shape[-1] == 4:
                        img_data[i] = cv2.cvtColor(img_data[i], cv2.COLOR_BGRA2BGR)

                    # Convert to RGB format
                    if self.bgr_to_rgb == True:
                        img_data[i] = img_data[i][:,:,::-1]

            info_dict['data_path'] = data_path
            if self.backend == 'pil':
                info_dict['data_shape'] = img_data[0].size[1], img_data[0].size[0], len(img_data[0].getbands())
            else:
                info_dict['data_shape'] = img_data[0].shape
        else:
            assert False, 'invalid input'
        #
        return img_data, info_dict

    def __repr__(self):
        return self.__class__.__name__ + f'(backend={self.backend})'
