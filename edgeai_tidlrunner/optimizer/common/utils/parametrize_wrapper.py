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
import random

import torch
import torch.nn.utils.parametrize as parametrize
import torchao


class ParametrizationBaseModule(torch.nn.Module):
    pass


def _register_parametrizations(module, parametrization_class=None, param_names=None):
    import torch
    import torchao
    from torchao.quantization import pt2e
    import torch.nn.utils.parametrize as parametrize

    for name, child in module.named_children():
        _register_parametrizations(child, parametrization_class=parametrization_class)
    #
    if not isinstance(module, (ParametrizationBaseModule, pt2e.ObserverBase, pt2e.FakeQuantizeBase)):
        for name_p, param in list(module.named_parameters(recurse=False)):
            if param is not None and (param_names is None or name_p in param_names):
                parametrize.register_parametrization(module, name_p, parametrization_class(param))
            #
        #
        for name_p, param in list(module.named_buffers(recurse=False)):
            if param is not None and (param_names is None or name_p in param_names):
                parametrize.register_parametrization(module, name_p, parametrization_class(param))
            #
        #
    #
    return module


def _remove_parametrizations(module):
    import torch
    import torch.nn.utils.parametrize as parametrize
    
    for name, child in module.named_children():
        _remove_parametrizations(child)
    #
    for name_p, param in list(module.named_parameters(recurse=False)):
        if parametrize.is_parametrized(module, name_p):
            parametrize.remove_parametrizations(module, name_p)
        #
    #
    for name_p, param in list(module.named_buffers(recurse=False)):
        if parametrize.is_parametrized(module, name_p):
            parametrize.remove_parametrizations(module, name_p)
        #
    #
    return module


class WeightClipDeltaParametrization(ParametrizationBaseModule):
    '''
    Clip the weights of a layer within a certain delta range.
    '''
    def __init__(self, orig_value, delta_factor = 0.01):
        super().__init__()
        self.delta_factor = delta_factor
        self.with_parametrization = orig_value.dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]
        if self.with_parametrization:
            delta_w = torch.abs(orig_value.detach().data) * self.delta_factor
            self.register_buffer('min_range', orig_value - delta_w)
            self.register_buffer('max_range', orig_value + delta_w)
        #

    def forward(self, w_in):
        w_out = torch.clamp(w_in, min=self.min_range, max=self.max_range) if self.with_parametrization else w_in
        return w_out
    

class WeightClipValueParametrization(ParametrizationBaseModule):
    def __init__(self, orig_value, clip_value = 15.0):
        super().__init__()
        self.clip_value = clip_value
        self.with_parametrization = orig_value.dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]
        if self.with_parametrization:
            self.min_range = -self.clip_value
            self.max_range = +self.clip_value
        #

    def forward(self, w_in):
        w_out = torch.clamp(w_in, min=self.min_range, max=self.max_range) if self.with_parametrization else w_in
        return w_out


PARAMETRIZATION_TYPES_DICT = {
    'weight_clip_delta': WeightClipDeltaParametrization,
    'weight_clip_value': WeightClipValueParametrization,
}


def register_parametrizations(module, parametrization_types=None, param_names=None):
    if parametrization_types:
        for parametrization_type in parametrization_types:
            parametrization_class = PARAMETRIZATION_TYPES_DICT[parametrization_type] if isinstance(parametrization_type, str) else parametrization_type
            _register_parametrizations(module, parametrization_class=parametrization_class, param_names=param_names)
        #
    #


def remove_parametrizations(module):
    return _remove_parametrizations(module)

