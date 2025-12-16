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


def _move_node_kwargs_to_device(model, device):
    import torch
    for node in model.graph.nodes:
        if "device" in node.kwargs and node.kwargs['device'] != torch.device(device):
            with model.graph.inserting_before(node):
                new_kwargs = dict(node.kwargs)
                new_kwargs['device'] = torch.device(device)
                new_node = model.graph.create_node(op=node.op, target=node.target, args=node.args, kwargs=new_kwargs)
                node.replace_all_uses_with(new_node)
                model.graph.erase_node(node)
    model.graph.lint()
    model.recompile()
    return model


def move_model_to_device(model, example_inputs, example_kwargs=None, device=None):
    import torch
    example_kwargs = example_kwargs or {}
    if device:
        return
    
    if isinstance(example_inputs, (list, tuple)):
        for i, inp in enumerate(example_inputs):
            example_inputs[i].to(device=device)
    else:
        example_inputs = [example_inputs.to(device=device)]
    for key, value in example_kwargs.items():
        if isinstance(value, torch.Tensor):
            example_kwargs[key] = value.to(device=device)

    model = model.to(device=device)
    _move_node_kwargs_to_device(model, device)

