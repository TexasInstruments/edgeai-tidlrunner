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


import cv2
import numpy as np

from . import dataset_base


class VideoCaptureDataLoader(dataset_base.DatasetBase):
    def __init__(self, source=0, num_frames=100, max_num_frames=1000, bgr_to_rgb=True, **kwargs):
        super().__init__(**kwargs)
        self.source = source
        self.bgr_to_rgb = bgr_to_rgb
        self.cap = None
        self.frames = []
        self.frame_index = 0
        self.num_frames = num_frames
        self.max_num_frames = max_num_frames

        self._initialize_capture()
        if num_frames:
            assert num_frames <= self.max_num_frames, f"ERROR: num_frames should not exceed {self.max_num_frames}"
            self._load_frames(num_frames)

    def _initialize_capture(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f'ERROR: Could not open video source: {self.source}')

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def _capture_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None

        if frame.shape[-1] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[-1] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        if self.bgr_to_rgb:
            frame = frame[:, :, ::-1]

        return frame

    def _load_frames(self, num_frames=None):
        frame_count = 0
        while frame_count < num_frames:
            frame = self._capture_frame()
            self.frames.append(frame)
            frame_count += 1

        if len(self.frames) == 0:
            raise RuntimeError(f'ERROR: No frames loaded from video source: {self.source}')

        print(f'INFO: Loaded {len(self.frames)} frames from video source')

    def __getitem__(self, index, info_dict=None):
        info_dict = info_dict or dict()

        if self.num_frames:
            index = index % len(self.frames)
            frame = self.frames[index]
        else:
            frame = self._capture_frame()

        info_dict['data_shape'] = frame.shape
        info_dict['data'] = frame
        info_dict['data_path'] = f'{self.source}:frame_{index}'

        return frame, info_dict

    def __len__(self):
        return self.num_frames or self.max_num_frames

    def evaluate(self, run_data, **kwargs):
        raise RuntimeError('VideoCaptureDataLoader.evaluate() not implemented.')

    def __del__(self):
        if self.cap is not None:
            self.cap.release()


class CameraCaptureDataLoader(VideoCaptureDataLoader):
    def __init__(self, *args, num_frames=None, **kwargs):
        super().__init__(*args, num_frames=num_frames, **kwargs)


class VideoFileDataLoader(VideoCaptureDataLoader):
    def __init__(self, video_path, num_frames=None, bgr_to_rgb=True, **kwargs):
        super().__init__(source=video_path, num_frames=num_frames, bgr_to_rgb=bgr_to_rgb, **kwargs)


def video_capture_dataloader(settings, name, source=0, num_frames=None, **kwargs):
    return VideoCaptureDataLoader(source=source, num_frames=num_frames, **kwargs)


def camera_capture_dataloader(settings, name, source=0, num_frames=None, **kwargs):
    return CameraCaptureDataLoader(source=source, num_frames=num_frames, **kwargs)


def video_file_dataloader(settings, name, video_path, num_frames=None, **kwargs):
    return VideoFileDataLoader(video_path=video_path, num_frames=num_frames, **kwargs)
