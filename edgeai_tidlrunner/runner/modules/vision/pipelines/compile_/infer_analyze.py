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


import sys
import copy
import numpy as np

from ..... import utils
from ..... import bases
from ...settings.settings_default import SETTINGS_DEFAULT, COPY_SETTINGS_DEFAULT
from . import infer_model


class InferAnalyzePipeline(bases.PipelineBase):
    ARGS_DICT=SETTINGS_DEFAULT['infer_analyze']
    COPY_ARGS=COPY_SETTINGS_DEFAULT['infer_analyze']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.infer_pipeline = infer_model.InferModelPipeline(**kwargs)

        kargs_copy = copy.deepcopy(kwargs)
        kargs_copy['tidl_offload'] = False
        self.infer_pipeline_float = infer_model.InferModelPipeline(**kargs_copy)

    def run(self):
        print(f'INFO: starting model analyze')
        print(f'INFO: model inference with offload=False')
        self.infer_pipeline_float.run()
        print(f'INFO: model inference with offload=True')
        self.infer_pipeline.run()

        input_details, output_details = self.infer_pipeline.session.get_input_output_details()
        num_frames = len(self.infer_pipeline.run_data)
        num_outputs = len(output_details)

        for frame_index in range(num_frames):
            for output_index in range(num_outputs):
                output = self.infer_pipeline.run_data[frame_index]['output'][output_index]
                float_output = self.infer_pipeline_float.run_data[frame_index]['output'][output_index]
                diff =  np.mean(np.abs(output - float_output))
                print(f'INFO: analyze: frame_index={frame_index} output_index={output_index} MAE={str(round(diff,5))}')