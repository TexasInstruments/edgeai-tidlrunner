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
import sys
import shutil
import copy

from ...... import rtwrapper
from ......rtwrapper.core import presets
from ...settings.settings_default import SETTINGS_DEFAULT, COPY_SETTINGS_DEFAULT
from ..... import utils
from ...blocks import postprocess
from ...blocks import dataloaders
from ...blocks import sessions
from . import infer_model


class InferAccuracy(infer_model.InferModel):
    ARGS_DICT=SETTINGS_DEFAULT['infer_accuracy']
    COPY_ARGS=COPY_SETTINGS_DEFAULT['infer_accuracy']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run(self):
        print(f'INFO: starting model accuracy evaluation')
        outputs = super()._run()
        run_data = self.get_run_data()

        # now calculate the accuracy
        if hasattr(self.dataloader, 'evaluate'):
            accuracy = self.dataloader.evaluate(run_data)
            print(f'INFO: Accuracy - {accuracy}')
            self.settings['result'].update(accuracy)
            self._write_params('result.yaml')
        else:
            print(f'WARNING: dataloader {self.dataloader.__class__.__name__} does not have evaluate method, skipping accuracy calculation')
        #
        return self.settings['result']
    
