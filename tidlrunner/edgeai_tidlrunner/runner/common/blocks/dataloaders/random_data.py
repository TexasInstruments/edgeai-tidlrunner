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


import numpy as np

from . import dataset_base


class RandomDataLoader(dataset_base.DatasetBase):
    def __init__(self, num_frames=1000, size_details=None, **kwargs):
        super().__init__()
        self.num_frames = num_frames
        self.size_details = size_details or [{'shape':[1, 3, 224, 224], 'type':'float'}]
        self.get_dataset_info()

    def __getitem__(self, index, info_dict=None):
        info_dict = info_dict or {}
        # Handle multiple inputs - return a list of tensors
        if len(self.size_details) > 1:
            tensors = []
            for input_detail in self.size_details:
                shape = input_detail['shape']
                type_str = input_detail['type']
                # Determine dtype from type string
                if 'uint8' in type_str:
                    dtype = np.uint8
                elif 'int32' in type_str or 'int64' in type_str:
                    dtype = np.int32
                elif 'int8' in type_str:
                    dtype = np.int8
                else:
                    dtype = np.float32
                #
                # Generate random data with appropriate range
                if dtype in (np.uint8, np.int8, np.int32):
                    tensor = np.random.randint(0, 10, size=shape, dtype=dtype)
                else:
                    tensor = np.random.rand(*shape).astype(dtype=dtype)
                #
                tensors.append(tensor)
            return tensors, info_dict
        else:
            # Single input - return single tensor for backward compatibility
            shape = self.size_details[0]['shape']
            type_str = self.size_details[0]['type']
            # Determine dtype from type string
            if 'uint8' in type_str:
                dtype = np.uint8
            elif 'int32' in type_str or 'int64' in type_str:
                dtype = np.int32
            elif 'int8' in type_str:
                dtype = np.int8
            else:
                dtype = np.float32
            
            # Generate random data with appropriate range
            if dtype in (np.uint8, np.int8, np.int32):
                tensor = np.random.randint(0, 10, size=shape, dtype=dtype)
            else:
                tensor = np.random.rand(*shape).astype(dtype=dtype)
            #
            return tensor, info_dict

    def __len__(self):
        return self.num_frames

    def set_size_details(self, size_details):
        self.size_details = size_details

    def get_dataset_info(self, *args, **kwargs):
        dataset_info = None
        self.kwargs['dataset_info'] = dataset_info
        return dataset_info

def random_dataloader(settings, name, **kwargs):
    return RandomDataLoader(**kwargs)
