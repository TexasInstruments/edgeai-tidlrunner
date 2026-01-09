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


from edgeai_tidlrunner import rtwrapper

from ...settings.constants import presets
from ...settings import RuntimeSettings
from .. import preprocess


__all = ['ONNXRuntimeSession', 'SESSION_TYPES_MAPPING']


DataLayoutType = rtwrapper.options.presets.DataLayoutType


def create_input_normalizer(**kwargs):
    input_optimization = kwargs.get('input_optimization', False)
    input_mean = kwargs.get('input_mean', None)
    input_scale = kwargs.get('input_scale', None)
    data_layout = kwargs.get('data_layout', None)
    # input_optimization is set, the input_mean and input_scale are added inside the model
    # in that case, there is not need to normalize here
    if (not input_optimization) and input_mean and input_scale:
        # mean scale could not be absorbed inside the model - do it explicitly
        input_normalizer = preprocess.ImageNormMeanScale(input_mean, input_scale, data_layout)
    else:
        input_normalizer = None
    #
    return input_normalizer


class ONNXRuntimeSession(rtwrapper.core.ONNXRuntimeWrapper):
    def __init__(self, settings, **kwargs):
        if not isinstance(kwargs, RuntimeSettings):
            kwargs = RuntimeSettings(**kwargs)
        #
        super().__init__(**kwargs)
        self.input_normalizer = create_input_normalizer(**self.kwargs)

    def run_import(self, input_data, output_keys=None):
        input_data, info_dict = self.input_normalizer(input_data, info_dict={}) if self.input_normalizer else input_data, {}
        return super().run_import(input_data, output_keys)

    def run_inference(self, input_data, output_keys=None):
        input_data, info_dict = self.input_normalizer(input_data, info_dict={}) if self.input_normalizer else input_data, {}
        return super().run_inference(input_data, output_keys)


class TFLITERuntimeSession(rtwrapper.core.TFLiteRuntimeWrapper):
    def __init__(self, settings, **kwargs):
        if not isinstance(kwargs, RuntimeSettings):
            kwargs = RuntimeSettings(**kwargs)
        #
        super().__init__(**kwargs)
        self.input_normalizer = create_input_normalizer(**self.kwargs)

    def run_import(self, input_data, output_keys=None):
        input_data, info_dict = self.input_normalizer(input_data, info_dict={}) if self.input_normalizer else input_data, {}
        return super().run_import(input_data, output_keys)

    def run_inference(self, input_data, output_keys=None):
        input_data, info_dict = self.input_normalizer(input_data, info_dict={}) if self.input_normalizer else input_data, {}
        return super().run_inference(input_data, output_keys)


class TVMRuntimeSession(rtwrapper.core.TVMRuntimeWrapper):
    def __init__(self, settings, **kwargs):
        if not isinstance(kwargs, RuntimeSettings):
            kwargs = RuntimeSettings(**kwargs)
        #
        super().__init__(**kwargs)
        self.input_normalizer = create_input_normalizer(**self.kwargs)

    def run_import(self, input_data, output_keys=None):
        input_data, info_dict = self.input_normalizer(input_data, info_dict={}) if self.input_normalizer else input_data, {}
        return super().run_import(input_data, output_keys)

    def run_inference(self, input_data, output_keys=None):
        input_data, info_dict = self.input_normalizer(input_data, info_dict={}) if self.input_normalizer else input_data, {}
        return super().run_inference(input_data, output_keys)



SESSION_TYPES_MAPPING = {
    presets.RuntimeType.RUNTIME_TYPE_ONNXRT: ONNXRuntimeSession,
    presets.RuntimeType.RUNTIME_TYPE_TFLITERT: TFLITERuntimeSession,
    presets.RuntimeType.RUNTIME_TYPE_TVMRT: TVMRuntimeSession
}
