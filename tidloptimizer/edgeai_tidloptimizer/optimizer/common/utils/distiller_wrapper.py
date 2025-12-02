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
from torch.ao.quantization import FakeQuantizeBase

from edgeai_tidlrunner.runner.common.utils import print_once

from . import hooks_wrapper, parametrize_wrapper


class DistillerWrapperModule(torch.nn.Module):
    def __init__(self, student_model, teacher_model, **kwargs):
        super().__init__()
        self.student_model = student_model
        self.teacher_model = teacher_model

        self.epochs = kwargs.get('epochs', 10)
        self.warmup_epochs = kwargs.get('warmup_epochs', 3)
        self.warmup_factor = kwargs.get('warmup_factor', 1.0) #1/100.0)
        self.momentum = kwargs.get('momentum', 0.9)
        self.weight_decay = kwargs.get('weight_decay', 1e-4)
        self.temperature = kwargs.get('temperature', 1)
        self.optimizer_step_interval = kwargs.get('optimizer_step_interval', 10)
        self.optimizer_type = kwargs.get('optimizer_type', 'SGD')
        self.lr = kwargs.get('lr', 1e-5)
        self.lr_min = kwargs.get('lr_min', self.lr/100.0)
    
        self.current_epoch = 0
        self.current_iter = 0
        self.hook_handles = []
        self.student_activations_dict = {}
        self.teacher_activations_dict = {}

        lr = self.lr
        if self.optimizer_type == 'SGD':
            self.optimizer = torch.optim.SGD(student_model.parameters(), lr=lr, momentum=self.momentum, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'Adam':
            self.optimizer = torch.optim.Adam(student_model.parameters(), lr=lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f'ERROR: Unsupported optimizer type {self.optimizer_type}')
        #
        self.scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=self.warmup_factor, total_iters=self.warmup_epochs)
        self.optimizer.zero_grad()

    def cleanup(self):
        pass

    def forward(self, *inputs):
        with torch.no_grad():
            teacher_outputs = self.teacher_model(*inputs)
        #
        student_outputs = self.student_model(*inputs)
        return student_outputs, teacher_outputs

    def eval(self):
        self.train(False)
        return self
    
    def train(self, mode: bool=True):
        super().train(mode)
        self.teacher_model.eval()
        self.student_model.train(mode)
        return self
    
    def step_iter(self, outputs, targets):
        loss = self._compute_loss(outputs, targets)
        loss.backward()
        is_optimizer_step = (self.current_iter == 0 or self.current_iter % self.optimizer_step_interval == 0)
        if is_optimizer_step:
            self.optimizer.step()
            self.optimizer.zero_grad()
        #
        lr = round(self._get_last_lr(), 6)
        loss_value = round(loss.item(), 6)
        self.current_iter += 1
        return {'lr':lr, 'loss':loss_value}
    
    def _compute_loss(self, outputs, targets):
        if isinstance(outputs, (list, tuple)) and isinstance(targets, (list, tuple)):
            assert len(outputs) == len(targets), f'number of outputs {len(outputs)} and targets {len(targets)} should be same'
            loss = sum([self._criterion(o, t) for o,t in zip(outputs, targets)])
        else:
            loss = self._criterion(outputs, targets)
        #
        return loss
    
    def _criterion(self, outputs, targets):
        if torch.is_floating_point(outputs):
            loss = torch.nn.functional.smooth_l1_loss(outputs, targets)
        else:
            assert False, "ERROR: Cannot compute loss on integer outputs directly."
        #
        return loss

    def _get_scheduler(self):
        return self.scheduler
    
    def _get_last_lr(self):
        lr = self._get_scheduler().get_last_lr()[0]
        return lr

    def step_epoch(self):
        self._get_scheduler().step()
        self.current_epoch += 1


class DistillerWrapperParametrizeModule(DistillerWrapperModule):
    def __init__(self, student_model, teacher_model, weight_clip_delta=False, layer_activation_loss=True, **kwargs):
        super().__init__(student_model, teacher_model, **kwargs)
        self.weight_clip_delta = weight_clip_delta
        self.layer_activation_loss = layer_activation_loss
        self.layer_activation_loss_scale = 1.0

        if self.layer_activation_loss:
            def student_activation_store_hook_fn(m, input, output):
                self.student_activations_dict[m.__module_name_info__] = output
            #
            self.hook_handles += hooks_wrapper.register_model_forward_hook(self.student_model, student_activation_store_hook_fn, module_type=FakeQuantizeBase)

            def teacher_activation_store_hook_fn(m, input, output):
                output = output.detach() if hasattr(output, 'detach') else output
                self.teacher_activations_dict[m.__module_name_info__] = output
            #
            self.hook_handles += hooks_wrapper.register_model_forward_hook(self.teacher_model, teacher_activation_store_hook_fn, module_type=FakeQuantizeBase)
        #

    def _register_parametrizations(self):
        if self.weight_clip_delta:
            param_names = ('weight', 'bias')
            # clip the range of thes parameters within a certain percentage of the original value
            parametrize_wrapper.register_parametrizations(self.student_model, parametrization_types=('weight_clip_delta',), param_names=param_names)
            # freeze all the other parameters
            # for p_name, p in self.student_model.named_parameters():
            #     if p_name.split('.')[-1] not in param_names:
            #         p.requires_grad = False
            #     #
            # #
        #

    def _remove_parametrizations(self):
        if self.weight_clip_delta:
            parametrize_wrapper.remove_parametrizations(self.student_model)
        #

    def _compute_loss(self, outputs, targets):
        loss = super()._compute_loss(outputs, targets)
        if self.layer_activation_loss:
            activations_match = len(self.student_activations_dict) == len(self.teacher_activations_dict) and all([k1 == k2 for k1, k2 in zip(self.student_activations_dict, self.teacher_activations_dict)])
            if activations_match:
                print_once(f"INFO: can apply layer_activation_loss - student layers:{len(self.student_activations_dict)} vs teacher layers:{len(self.teacher_activations_dict)}")
                act_loss = 0.0
                for k1, k2 in zip(self.student_activations_dict, self.teacher_activations_dict):
                    student_v = self.student_activations_dict[k1]
                    teacher_v = self.teacher_activations_dict[k2]
                    diff = (student_v-teacher_v)
                    act_deviation = diff.abs().mean()
                    act_loss = act_loss + act_deviation * self.layer_activation_loss_scale
                #
                act_loss = act_loss / len(self.student_activations_dict)
                loss += act_loss
            else:
                print_once(f"WARNING: layer_activation_loss cannot be applied - layers mismatch - student layers:{len(self.student_activations_dict)} vs teacher layers:{len(self.teacher_activations_dict)}")
            #
        #
        return loss
    
    def eval(self):
        self._remove_parametrizations()
        super().eval()
    
    def train(self, mode: bool=True):
        m = super().train(mode)
        self._register_parametrizations()
        return m
    
    def cleanup(self):
        hook_handles = self.hook_handles
        hook_handles = list(hook_handles.values()) if isinstance(hook_handles, dict) else hook_handles
        hook_handles = [hook_handles] if not isinstance(hook_handles, list) else hook_handles
        for hook_handle in hook_handles:
            hook_handle.remove()
        #
