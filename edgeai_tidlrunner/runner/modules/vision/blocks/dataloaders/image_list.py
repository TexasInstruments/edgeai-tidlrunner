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
import numpy as np
import os

from . import dataset_base
from . import dataloader_utils


#######################################################################
class ImageListDataLoader(dataset_base.DatasetBase):
    def __init__(self, files, labels=None, file_types=None, backend='pil', bgr_to_rgb=True):
        super().__init__()
        self.files = files
        self.labels = labels
        self.file_types = file_types
        self.image_reader = dataloader_utils.ImageRead(backend=backend, bgr_to_rgb=bgr_to_rgb)

    def __getitem__(self, index, info_dict=None):
        input_img_file = self.files[index]
        img_data, info_dict = self.image_reader(input_img_file, info_dict)
        return img_data, info_dict

    def __len__(self):
        return len(self.files)

    def evaluate(self, run_data, **kwargs):
        raise RuntimeError('ImageFilesDataLoader.evaluate() not implemented.')

    def _read_folder(self, path, file_types=None):
        paths = os.listdir(path)
        paths = [os.path.join(path,f) for f in paths]
        image_files = [f for f in paths if os.path.isfile(f) and (not file_types or os.path.splitext(f)[-1].lower() in file_types)]
        if any(image_files):
            print(f'INFO: found {len(image_files)} image files in {path}')                    
            return image_files, None
        else:
            print(f'INFO: could not find image files in {path}. searching in sub folders...')
            files = []
            labels = []
            for folder in paths:
                paths = os.listdir(folder)
                paths = [os.path.join(folder,f) for f in paths]
                image_files = [f for f in paths if os.path.isfile(f) and (not file_types or os.path.splitext(f)[-1].lower() in file_types)]
                labels_list = [os.path.basename(folder) for f in image_files]
                files.extend(image_files)
                labels.extend(labels_list)
            #
            print(f'INFO: found {len(files)} image files in {path}')
            return files, labels


    def _read_file(self, path, label_path, file_types=None):
        files = []
        labels = None
        with open(label_path, 'r') as fp:
            for line in fp:
                words = line.strip().split()
                files.append(words[0])
                if len(words) > 1:
                    labels = labels or []
                    labels.append(int(words[1]))

        files = [os.path.join(path, f) for f in files if (not file_types or os.path.splitext(f)[-1].lower() in file_types)]
        return files, labels

def image_list_dataloader(name, path):
    return ImageListDataLoader(path)


#######################################################################
class ImageFilesDataLoader(ImageListDataLoader):
    def __init__(self, path, label_path=None, backend='pil', bgr_to_rgb=True, file_types=('.png', '.jpg', '.jpeg')):
        if isinstance(path, list):
            files = path
            labels = label_path if isinstance(label_path, list) else None
        elif isinstance(label_path, str) and label_path.endswith('.txt'):
            files, labels = self._read_file(path, label_path, file_types)
        elif os.path.isdir(path):
            files, labels = self._read_folder(path, file_types)
        else:
            raise RuntimeError(f'ERROR: invalid path: {path}')
        super().__init__(files, labels, backend=backend, bgr_to_rgb=bgr_to_rgb, file_types=file_types)

    def evaluate(self, run_data, **kwargs):
        predictions = []
        inputs = []
        for data in run_data:
            output_dict = data['output']
            predictions.append(list(output_dict.values())[0])
            inputs.append(data['input'])
        #
        correctly_classified = 0
        num_frames = 0
        for prediction, label in zip(predictions, self.labels):
            prediction = prediction[0] if isinstance(prediction, list) else prediction
            pred = np.argmax(prediction, axis=1) if prediction.ndim > 1 else np.argmax(prediction)
            correctly_classified += int(int(pred) == int(label))
            num_frames += len(pred)
        #
        accuracy_percentage = correctly_classified * 100 / num_frames
        return {'accuracy_top1%': accuracy_percentage}


def image_files_dataloader(name, path, label_path=None):
    return ImageFilesDataLoader(path, label_path)
