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

from ..... import rtwrapper
from .....rtwrapper.core import presets
from ....common import utils
from ...settings.settings_default import SETTINGS_DEFAULT, COPY_SETTINGS_DEFAULT
from ...blocks import postprocess
from ...blocks import dataloaders
from ...blocks import sessions
from . import infer


class InferAccuracy(infer.InferModel):
    ARGS_DICT=SETTINGS_DEFAULT['accuracy']
    COPY_ARGS=COPY_SETTINGS_DEFAULT['accuracy']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _run(self):
        print(f'INFO: starting model accuracy evaluation')
        common_kwargs = self.settings[self.common_prefix]
        if common_kwargs['incremental']:
            if os.path.exists(self.result_yaml):
                print(f'INFO: incremental {common_kwargs["incremental"]} param.yaml exists: {self.result_yaml}')
                print(f'INFO: skipping infer/accuracy')
                return
            #
        #

        outputs = super()._run()
        run_data = self.get_run_data()

        # now calculate the accuracy
        if hasattr(self.dataloader, 'evaluate'):
            metric_kwargs = self.settings.get('metric', dict())
            accuracy = self.dataloader.evaluate(run_data, **metric_kwargs)
            print(f'INFO: Accuracy - {accuracy}')
            self.settings['result'].update(accuracy)
            self._write_params(self.settings, os.path.join(self.run_dir,'result.yaml'))
        else:
            print(f'WARNING: dataloader {self.dataloader.__class__.__name__} does not have evaluate method, skipping accuracy calculation')
        #
        return self.settings['result']
    
