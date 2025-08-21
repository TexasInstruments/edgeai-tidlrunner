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



from .....rtwrapper.options import presets


# task_type
class TaskType:
    TASK_TYPE_CLASSIFICATION = 'classification'
    TASK_TYPE_DETECTION = 'detection'
    TASK_TYPE_SEGMENTATION = 'segmentation'
    TASK_TYPE_KEYPOINT_DETECTION = 'keypoint_detection'
    TASK_TYPE_DEPTH_ESTIMATION = 'depth_estimation'
    TASK_TYPE_DETECTION_3DOD = 'detection_3d'
    TASK_TYPE_OBJECT_6D_POSE_ESTIMATION = 'object_6d_pose_estimation'
    TASK_TYPE_VISUAL_LOCALIZATION = 'visual_localization'
    TASK_TYPE_DISPARITY_ESTIMATION = 'disparity_estimation'


TaskTypeShortNames = {
    TaskType.TASK_TYPE_CLASSIFICATION: 'cl',
    TaskType.TASK_TYPE_DETECTION: 'od',
    TaskType.TASK_TYPE_SEGMENTATION: 'ss',
    TaskType.TASK_TYPE_KEYPOINT_DETECTION: 'kd',
    TaskType.TASK_TYPE_DEPTH_ESTIMATION: 'de',
    TaskType.TASK_TYPE_DETECTION_3DOD: '3dod',
    TaskType.TASK_TYPE_OBJECT_6D_POSE_ESTIMATION: '6dpose',
    TaskType.TASK_TYPE_VISUAL_LOCALIZATION: 'visloc',
    TaskType.TASK_TYPE_DISPARITY_ESTIMATION: 'sd',
}
