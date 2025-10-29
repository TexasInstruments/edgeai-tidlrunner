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

from ..utils import hooks_wrapper, parametrize_wrapper


class DistillWrapperBaseModule(torch.nn.Module):
    def __init__(self, student_model, teacher_model, **kwargs):
        super().__init__()
        self.student_model = student_model
        self.teacher_model = teacher_model

        self.epochs = kwargs.get('epochs', 10)
        self.warmup_epochs = kwargs.get('warmup_epochs', 3)
        self.warmup_factor = kwargs.get('warmup_factor', 1.0) #1/100.0)
        self.lr = kwargs.get('lr', 1e-5)
        self.lr_min = kwargs.get('lr_min', self.lr/100.0)
        self.momentum = kwargs.get('momentum', 0.9)
        self.weight_decay = kwargs.get('weight_decay', 1e-4)
        self.temperature = kwargs.get('temperature', 1)
        self.optimizer_step_interval = kwargs.get('optimizer_step_interval', 10)

        self.current_epoch = 0
        self.current_iter = 0
        self.hook_handles = []
        self.activations_dict = {}

        self.optimizer = torch.optim.SGD(student_model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
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
        # super().eval()
        self.teacher_model.eval()
        self.student_model.eval()
        self.training = False
        return self
    
    def train(self):
        # super().train()
        self.teacher_model.eval()
        self.student_model.train()
        self.training = True
        return self
    
    def step_iter(self, outputs, targets):
        loss = self._compute_loss(outputs, targets)  / self.optimizer_step_interval
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


class DistillWrapperModule(DistillWrapperBaseModule):
    def __init__(self, student_model, teacher_model, weight_clip_delta=True, activation_decay=False, **kwargs):
        super().__init__(student_model, teacher_model, **kwargs)
        self.weight_clip_delta = weight_clip_delta
        self.activation_decay = activation_decay
        if self.activation_decay:
            def activation_store_hook_fn(m, input, output):
                self.activations_dict[m.__module_name_info__] = output.detach()
            #
            self.hook_handles += hooks_wrapper.register_model_forward_hook(self.student_model, activation_store_hook_fn)
        #
        if self.weight_clip_delta:
            param_names = ('weight', 'bias')
            # clip the range of thes parameters within a certain percentage of the original value
            parametrize_wrapper.register_parametrizations(self.student_model, parametrization_types=('weight_clip_delta',), param_names=param_names)
            # freeze all the other parameters
            for p_name, p in self.student_model.named_parameters():
                if p_name.split('.')[-1] not in param_names:
                    p.requires_grad = False
                #
            #
        #

    def cleanup(self):
        super().cleanup()
        hook_handles = self.hook_handles
        hook_handles = list(hook_handles.values()) if isinstance(hook_handles, dict) else hook_handles
        hook_handles = [hook_handles] if not isinstance(hook_handles, list) else hook_handles
        for hook_handle in hook_handles:
            hook_handle.remove()
        #
        if self.weight_clip_delta:
            parametrize_wrapper.remove_parametrizations(self.student_model)
        #

    def _compute_loss(self, outputs, targets):
        loss = super()._compute_loss(outputs, targets)
        if self.activations_dict:
            act_loss = 0.0
            for k, v in self.activations_dict.items():
                # act_deviation = (v-1.0).pow(2).mean()
                # act_deviation = torch.quantile(torch.abs(v[:]), 0.99)
                # act_deviation = torch.kthvalue(v.reshape(-1), int(0.99*torch.numel(v)), dim=0)[0]
                act_deviation = v.abs().mean() + v.std()
                act_loss = act_loss + act_deviation
            #
            act_loss = act_loss / len(self.activations_dict)
            loss += act_loss
        #
        return loss