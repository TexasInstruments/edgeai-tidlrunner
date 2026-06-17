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


from .random_data import random_dataloader
from .image_list import image_list_dataloader, image_files_dataloader
from .image_cls import image_classification_dataloader
from .coco_det import coco_detection_dataloader
from .coco_seg import coco_segmentation_dataloader
from .coco_kpts import coco_keypoint_detection_dataloader
from .ade20k import ade20k_segmentation_dataloader, ade20k32_segmentation_dataloader

from .image_pix2pix import image_pix2pix_dataloader
from .image_det import image_detection_dataloader
from .image_seg import image_segmentation_dataloader

from .imagenet import imagenet_dataloader
from .imagenetv2 import imagenetv2c_dataloader
from .cityscapes import cityscapes_segmentation_dataloader
from .voc_seg import voc_segmentation_dataloader
from .widerface_det import widerface_detection_dataloader
from .nyudepthv2 import nyudepthv2_dataloader
from .robokit_seg import robokit_segmentation_dataloader
from .robokit_visloc import robokit_visloc_dataloader
from .tiscapes_seg import tiscapes_segmentation_dataloader

from .onnx_backend_dataset import onnx_backend_dataloader
from .tidl_unit_dataset import tidl_unit_dataloader

from .modelmaker_dataloaders import modelmaker_classification_dataloader, \
    modelmaker_detection_dataloader, modelmaker_segmentation_dataloader

from .audio_classification import audio_classification_dataloader
from .speech_enhancement import speech_enhancement_dataloader

import warnings

try:
    from .ycbv import ycbv_object_6d_pose_dataloader
except (ImportError, ModuleNotFoundError) as e:
    # warnings.warn(f'WARNING: ycbv_object_6d_pose_dataloader could not be imported - {str(e)}')
    ycbv_object_6d_pose_dataloader = None

try:
    from .kitti_2015 import kitti_2015_dataloader
except (ImportError, ModuleNotFoundError) as e:
    # warnings.warn(f'WARNING: kitti_2015 dataloader could not be imported - {str(e)}')
    kitti_2015_dataloader = None

try:
    from .kitti_lidar_det import kitti_lidar_det_dataloader
except (ImportError, ModuleNotFoundError) as e:
    # warnings.warn(f'WARNING: kitti_lidar_det dataloader could not be imported - {str(e)}')
    kitti_lidar_det_dataloader = None

try:
    from .nuscenes_dataset import nuscenes_dataloader
except (ImportError, ModuleNotFoundError) as e:
    # warnings.warn(f'WARNING: nuscenes_dataset dataloader could not be imported - {str(e)}')
    nuscenes_dataloader = None

try:
    from .pandaset_dataset import pandaset_frame_dataloader, pandaset_mv_image_dataloader
except (ImportError, ModuleNotFoundError) as e:
    # warnings.warn(f'WARNING: pandaset_dataset dataloader could not be imported - {str(e)}')
    pandaset_frame_dataloader = None
    pandaset_mv_image_dataloader = None
