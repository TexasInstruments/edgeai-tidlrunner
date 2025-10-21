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


class DistillParametrizationBaseModule(torch.nn.Module):
    pass


class DeltaClampParametrization(DistillParametrizationBaseModule):
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

class DistillWrapperModule(torch.nn.Module):
    def __init__(self, student_model, teacher_model, **kwargs):
        super().__init__()
        self.student_model = student_model
        self.teacher_model = teacher_model

        self.criterion = torch.nn.SmoothL1Loss() #torch.nn.MSELoss() #torch.nn.KLDivLoss()
        self.epochs = kwargs.get('epochs', 10)
        self.lr = kwargs.get('lr', 0.0001)
        self.lr_min = kwargs.get('lr_min', self.lr/100)
        self.momentum = kwargs.get('momentum', 0.9)
        self.weight_decay = kwargs.get('weight_decay', 1e-4)
        self.temperature = kwargs.get('temperature', 1)

        self.optimizer = torch.optim.SGD(student_model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=self.lr_min)
    
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
        if isinstance(outputs, (list, tuple)) and isinstance(targets, (list, tuple)):
            assert len(outputs) == len(targets), f'number of outputs {len(outputs)} and targets {len(targets)} should be same'
            loss = sum([self.criterion(o, t) for o,t in zip(outputs, targets)])
        else:
            loss = self.criterion(outputs, targets)
        #
        lr = round(self.scheduler.get_last_lr()[0], 6)
        loss_value = round(loss.item(), 6)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {'lr':lr, 'loss':loss_value}
    
    def step_epoch(self):
        self.scheduler.step()
