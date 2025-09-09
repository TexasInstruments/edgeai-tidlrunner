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
import shutil

import edgeai_tidlrunner
from edgeai_tidlrunner.runner import utils


def _get_root(path):
    path = path.rstrip('/')
    root = os.sep.join(os.path.split(path)[:-1])
    return root


def download_imagenetv2(path, split='val', force_download=False):
    notice = f'\nThe ImageNetV2 dataset is small and convenient version of ImageNet.' \
                f'\nNote: The categories in this dataset are same as the original ImageNet Dataset.' \
                f'\n' \
                f'\nThe ImageNetV2 dataset contains new test data for the ImageNet benchmark.' \
                f'\n             It is smaller in size and faster to download - ' \
                f'\n             ImageNetV2c closely matches the accuracy obtained with original ImageNet.' \
                f'\n             So it is a good choice for quick benchmarking.\n' \
                f'\nReference  : Do ImageNet Classifiers Generalize to ImageNet? ' \
                f'\n             Benjamin Recht et.al. https://arxiv.org/abs/1902.10811' \
                f'\nSource Code: https://github.com/modestyachts/ImageNetV2' \
                f'\n'
    
    url = 'https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-top-images.tar.gz'

    print(f'INFO: downloading and preparing dataset: {url} This may take some time.')
    print(notice)

    root = path
    split_file=os.path.join(root, f'{split}.txt')
    download_root = os.path.join(root, 'download')
    extract_root = os.path.join(download_root, 'rawdata')
    extract_path = utils.download_file(url, root=download_root, extract_root=extract_root, mode='r',
                                        force_download=force_download)
    split_path = os.path.join(root, split)

    folders = utils.list_dir(os.path.join(extract_path, 'imagenetv2-top-images-format-val'))
    basename_to_int = lambda f:int(os.path.basename(f))
    folders = sorted(folders, key=basename_to_int)
    lines = []
    for folder_id, folder in enumerate(folders):
        src_files = utils.list_files(folder)
        files = [os.path.join(os.path.basename(folder), os.path.basename(f)) for f in src_files]
        dst_files = [os.path.join(split_path, f) for f in files]
        for src_f, dst_f in zip(src_files, dst_files):
            os.makedirs(os.path.dirname(dst_f), exist_ok=True)
            shutil.copy2(src_f, dst_f)
        #
        folder_lines = [f'{f} {folder_id}' for f in files]
        lines.extend(folder_lines)
    #

    with open(split_file, 'w') as fp:
        fp.write('\n'.join(lines))
    #
    print(f'INFO: dataset ready {path}')
    return extract_path, split_file


def download_coco(path, split='val', force_download=False):
    root = path
    images_folder = os.path.join(path, split)
    annotations_folder = os.path.join(path, 'annotations')
    if (not force_download) and os.path.exists(path) and \
            os.path.exists(images_folder) and os.path.exists(annotations_folder):
        print(f'INFO: dataset exists - will reuse: {path}')
        return path
    #
    print('INFO: downloading and preparing dataset: {path} This may take some time.')
    print(f'\nCOCO Dataset:'
            f'\n    Microsoft COCO: Common Objects in Context, '
            f'\n        Tsung-Yi Lin, et.al. https://arxiv.org/abs/1405.0312\n'
            f'\n    Visit the following url to know more about the COCO dataset. '
            f'\n        https://cocodataset.org/ '
            f'\n')

    dataset_url = 'http://images.cocodataset.org/zips/val2017.zip'
    extra_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    download_root = os.path.join(root, 'download')
    dataset_path = utils.download_file(dataset_url, root=download_root, extract_root=root)
    extra_path = utils.download_file(extra_url, root=download_root, extract_root=root)
    print(f'INFO: dataset ready: {path}')
    return path


def main():
    download_imagenetv2('./data/datasets/vision/imagenetv2c')
    download_coco('./data/datasets/vision/coco')


if __name__ == '__main__':
    main()