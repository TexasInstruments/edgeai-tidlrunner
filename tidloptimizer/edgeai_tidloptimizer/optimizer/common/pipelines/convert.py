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


from __future__ import annotations

import os
import sys
import shutil
import copy
import numpy as np

from edgeai_tidlrunner.runner.common import utils
from edgeai_tidlrunner.runner.common import bases

from edgeai_tidlrunner.runner.common.settings import constants
from edgeai_tidlrunner.runner.common.pipelines.common_ import common_base
from ..settings.settings_default import SETTINGS_DEFAULT, COPY_SETTINGS_DEFAULT


class ConvertModel(common_base.CommonPipelineBase):
    ARGS_DICT=SETTINGS_DEFAULT['convert']
    COPY_ARGS=COPY_SETTINGS_DEFAULT['convert']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _prepare(self):
        super()._prepare()
        os.makedirs(self.run_dir, exist_ok=True)

    def info():
        print(f'INFO: Model convert - {__file__}')

    def _run(self):
        print(f'INFO: starting model convert with parameters: {self.kwargs}')

        common_kwargs = self.settings[self.common_prefix]
        convert_kwargs = common_kwargs.get('convert', {})

        input_model_path = self.model_path
        output_model_path = common_kwargs.get('output_model_path', None)
        if not output_model_path:
            output_model_name = os.path.basename(input_model_path)
            if input_model_path.endswith('.onnx'):
                output_model_name = os.path.splitext(output_model_name)[0] + '.pt2'
            elif input_model_path.endswith('.pt') or input_model_path.endswith('.pt2'):
                output_model_name = os.path.splitext(output_model_name)[0] + '.onnx'
            else:
                raise ValueError(f'ERROR: unsupported model format: {input_model_path}')
            #
            output_model_path = os.path.join(self.run_dir, 'output', output_model_name)
            os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
        #
        output_model_folder = os.path.dirname(output_model_path)

        os.makedirs(self.model_folder, exist_ok=True)
        os.makedirs(output_model_folder, exist_ok=True)

        config_path = os.path.dirname(common_kwargs['config_path']) if common_kwargs['config_path'] else None
        self.download_file(self.model_source, model_folder=self.model_folder, source_dir=config_path)

        output_model = self._run_func(input_model_path, output_model_path, **convert_kwargs)
        return output_model

    @classmethod
    def _run_func(cls, input_model_path, output_model_path=None, example_inputs=None, opset_version=constants.ONNX_OPSET_VERSION_DEFAULT, dynamo=True, **convert_kwargs):
        import torch
        if isinstance(input_model_path, str) and input_model_path.endswith('.onnx'):
            output_model = cls._onnx2torchfile(input_model_path, output_model_path, example_inputs, **convert_kwargs)
        elif isinstance(input_model_path, str) and input_model_path.endswith('.pt'):
            output_model = cls._torch2onnxfile(input_model_path, output_model_path, example_inputs, opset_version=opset_version, dynamo=dynamo, **convert_kwargs)
        elif isinstance(input_model_path, str) and input_model_path.endswith('.pt2'):
            output_model = cls._torch2onnxfile(input_model_path, output_model_path, example_inputs, opset_version=opset_version, dynamo=dynamo, **convert_kwargs)
        elif isinstance(input_model_path, torch.nn.Module) and output_model_path.endswith('.onnx'):
            output_model = cls._torch2onnxfile(input_model_path, output_model_path, example_inputs, opset_version=opset_version, dynamo=dynamo, **convert_kwargs)
        elif isinstance(input_model_path, torch.nn.Module) and output_model_path.endswith('.pt'):
            exported_model = torch.jit.export(input_model_path, example_inputs)
            torch.jit.save(exported_model, output_model_path)
        elif isinstance(input_model_path, torch.nn.Module) and output_model_path.endswith('.pt2'):
            exported_program = torch.export.export(input_model_path, example_inputs)
            torch.export.save(exported_program, output_model_path)
        else:
            raise ValueError(f'ERROR: unsupported model format: {input_model_path}')
        #
        return output_model_path

    @classmethod
    def _get_torch_model(cls, input_model_path, output_model_path=None, example_inputs=None, opset_version=constants.ONNX_OPSET_VERSION_DEFAULT, dynamo=True, training=False, **convert_kwargs):
        import torch
        if isinstance(input_model_path, str) and input_model_path.endswith('.onnx'):
            output_model = cls._onnx2torchfile(input_model_path, output_model_path, example_inputs, for_training=training, **convert_kwargs)
        elif dynamo or (isinstance(input_model_path, str) and input_model_path.endswith('.pt2')):
            output_model = torch.export.load(input_model_path)
        elif isinstance(input_model_path, str) and input_model_path.endswith('.pt'):
            output_model = torch.jit.load(input_model_path)
        elif isinstance(input_model_path, torch.nn.Module):
            output_model = input_model_path
        else:
            raise ValueError(f'ERROR: unsupported model format: {input_model_path}')
        #
        return output_model
    
    @classmethod
    def _get_onnx_input_info(cls, model_path):
        import onnx
        import onnx_graphsurgeon as gs
        model = onnx.load(model_path)
        graph = gs.import_onnx(model)
        input_info = {}
        for input_tensor in graph.inputs:
            # Handle dynamic dimensions (represented as strings)
            shape = []
            for dim in input_tensor.shape:
                if isinstance(dim, str):
                    shape.append(-1)  # Dynamic dimension
                else:
                    shape.append(int(dim))
            
            input_info[input_tensor.name] = {
                'shape': shape,
                'dtype': input_tensor.dtype,
            }
        return input_info

    @classmethod
    def _get_onnx_example_inputs(cls, model_path, to_torch=True):
        import torch
        input_info = cls._get_onnx_input_info(model_path)
        input_shapes = [input_info[name]['shape'] for name in input_info]
        input_dtypes = [input_info[name]['dtype'] for name in input_info]
        input_tensors = [np.random.rand(*shape).astype(dtype) for shape, dtype in zip(input_shapes, input_dtypes)]
        if to_torch:
            input_tensors = [torch.from_numpy(input_tensor) for input_tensor in input_tensors]
        #
        return tuple(input_tensors)

    @classmethod
    def _get_example_inputs(cls, model_path, to_torch=True):
        if isinstance(model_path, str) and model_path.endswith('.onnx'):
            return cls._get_onnx_example_inputs(model_path, to_torch=to_torch)
        else:
            return None

    @classmethod
    def _onnx2torchfile(cls, model_path, output_model_path=None, example_inputs=None, **kwargs):
        if isinstance(model_path, str) and model_path.endswith('.onnx'):
            import edgeai_onnx2torchmodel
            torch_model = edgeai_onnx2torchmodel.convert(model_path, **kwargs)
        else:
            torch_model = model_path
        #
        if output_model_path:
            if not example_inputs:
                example_inputs = cls._get_onnx_example_inputs(model_path, to_torch=True)
            #
            cls._torch2torchfile(torch_model, output_model_path, example_inputs)
        #
        return torch_model

    @classmethod
    def _torch2torchfile(cls, torch_model, output_model_path=None, dynamo=True, example_inputs=None):
        import torch
        if dynamo or output_model_path.endswith('.pt2'):
            exported_program = torch.export.export(torch_model, example_inputs)
            torch.export.save(exported_program, output_model_path)
        elif output_model_path.endswith('.pt'):
            fx_graph = torch.jit.export(torch_model, example_inputs)
            torch.jit.save(fx_graph, output_model_path)
        else:
            raise ValueError(f'ERROR: unsupported student model format: {output_model_path}')
        #

    @classmethod
    def _torch2onnxfile(cls, torch_model, onnx_model_path, example_inputs, opset_version, dynamo=True, simplify=False, onnx_ir_version=None):
        import torch
        if isinstance(torch_model, str) and torch_model.endswith('.pt2'):
            torch_model = torch.export.load(torch_model)
        elif isinstance(torch_model, str) and torch_model.endswith('.pt'):
            torch_model = torch.jit.load(torch_model)
        #
        if dynamo:
            print('INFO: dynamo based onnx export ...')
            custom_translation_table = cls._register_quantized_symbolics(opset_version=opset_version)
            artifacts_dir = os.path.dirname(onnx_model_path)
            onnx_program = torch.onnx.export(torch_model, example_inputs, opset_version=opset_version, dynamo=True, report=True, artifacts_dir=artifacts_dir, custom_translation_table=custom_translation_table)
            onnx_program.save(onnx_model_path)
        else:
            # traditional torchscript based onnx export
            print('INFO: torchscript based onnx export ...')
            torch.onnx.export(torch_model, example_inputs, onnx_model_path, export_params=True, opset_version=opset_version, do_constant_folding=True, training=torch.onnx.TrainingMode.PRESERVE, dynamo=False)
        #
        if simplify:
            print('INFO: simplifying onnx model ...')
            import onnxsim
            onnx_model, onnx_check = onnxsim.simplify(onnx_model_path)
            onnx.save(onnx_model, onnx_model_path)
        #
        if onnx_ir_version:
            print('INFO: converting ONNX IR version ...')
            import onnx
            onnx_model = onnx.load(onnx_model_path)
            onnx_model = onnx.version_converter.convert_version(onnx_model, onnx_ir_version)
            onnx.save(onnx_model, onnx_model_path)
        #

    @classmethod
    def _register_quantized_symbolics(cls, opset_version):
        # import torch
        # import onnxscript
        # from onnxscript.onnx_types import TensorType

        # from edgeai_torchmodelopt.xmodelopt.quantization.v3 import quant_utils 
        # quant_utils.register_onnx_symbolics(opset_version)

        # Copyright (c) Microsoft Corporation.
        # Licensed under the MIT License.
        # mypy: disable-error-code="misc,arg-type,type-arg,valid-type,assignment,return-value"
        # pylint: disable=unused-argument
        """quantized_decomposed ops defined in https://github.com/pytorch/pytorch/blob/main/torch/ao/quantization/fx/_decomposed.py

        - No inplace operators.
        - All functions should not have the script() decorator. This is because
            we want to delay the compilation of the function.
        """

        # from onnxscript.function_libs.torch_lib.ops import common
        # from onnxscript.function_libs.torch_lib.registration import torch_op
        # from onnxscript.onnx_opset import opset18 as op
        # from onnxscript.onnx_types import TensorType


        # @torch_op(
        #     (
        #         "quantized_decomposed::quantize_per_tensor",
        #         "quantized_decomposed::quantize_per_tensor.tensor",
        #         "quantized_decomposed::quantize_per_tensor.tensor2",
        #     ),
        #     trace_only=True,
        # )
        # def quantized_decomposed_quantize_per_tensor(
        #     input: TensorType,
        #     scale: float,
        #     zero_point: int,
        #     quant_min: int,
        #     quant_max: int,
        #     dtype: int,
        # ) -> TensorType:
        #     # TODO(justinchuby): Use dtype when we use opset 21
        #     return op.QuantizeLinear(input, scale, common.constant(zero_point, dtype=dtype))


        # @torch_op(
        #     (
        #         "quantized_decomposed::dequantize_per_tensor",
        #         "quantized_decomposed::dequantize_per_tensor.tensor",
        #         "quantized_decomposed::dequantize_per_tensor.tensor2",
        #     ),
        #     trace_only=True,
        # )
        # def quantized_decomposed_dequantize_per_tensor(
        #     input: TensorType,
        #     scale: float,
        #     zero_point: int,
        #     quant_min: int,
        #     quant_max: int,
        #     dtype: int,
        #     out_dtype: int = -1,
        # ) -> TensorType:
        #     # TODO(justinchuby): Use dtype when we use opset 21
        #     dequantized = op.DequantizeLinear(input, scale, common.constant(zero_point, dtype=dtype))
        #     if out_dtype in (-1, None):
        #         # out_dtype can be None as well
        #         return dequantized
        #     assert out_dtype > 0, f"out_dtype must be -1 or > 0 not {out_dtype}"
        #     return op.Cast(dequantized, to=out_dtype)
        
        # #----
        # @torch_op(
        #     (
        #         "quantized_decomposed::quantize_per_channel",
        #         "quantized_decomposed::quantize_per_channel.tensor",
        #         "quantized_decomposed::quantize_per_channel.tensor2",
        #     ),
        #     trace_only=True,
        # )
        # def quantized_decomposed_quantize_per_channel(
        #     input: TensorType,
        #     scale: float,
        #     zero_point: int,
        #     quant_min: int,
        #     quant_max: int,
        #     dtype: int,
        # ) -> TensorType:
        #     # TODO(justinchuby): Use dtype when we use opset 21
        #     return op.QuantizeLinear(input, scale, common.constant(zero_point, dtype=dtype))


        # @torch_op(
        #     (
        #         "quantized_decomposed::dequantize_per_tensor",
        #         "quantized_decomposed::dequantize_per_tensor.tensor",
        #         "quantized_decomposed::dequantize_per_tensor.tensor2",
        #     ),
        #     trace_only=True,
        # )
        # def quantized_decomposed_dequantize_per_channel(
        #     input: TensorType,
        #     scale: TensorType,
        #     zero_point: TensorType,
        #     unknown: int,
        #     quant_min: int,
        #     quant_max: int,
        #     dtype: int,
        #     out_dtype: int = -1,
        # ) -> TensorType:
        #     # TODO(justinchuby): Use dtype when we use opset 21
        #     dequantized = op.DequantizeLinear(input, scale, common.constant(zero_point, dtype=dtype))
        #     if out_dtype in (-1, None):
        #         # out_dtype can be None as well
        #         return dequantized
        #     assert out_dtype > 0, f"out_dtype must be -1 or > 0 not {out_dtype}"
        #     return op.Cast(dequantized, to=out_dtype)
        
        # custom_translation_table = {
        #     # "quantized_decomposed::quantize_per_tensor": quantized_decomposed_quantize_per_tensor,
        #     # "quantized_decomposed::quantize_per_tensor.tensor": quantized_decomposed_quantize_per_tensor,
        #     # "quantized_decomposed::quantize_per_tensor.tensor2": quantized_decomposed_quantize_per_tensor,
        #     # "quantized_decomposed::dequantize_per_tensor": quantized_decomposed_dequantize_per_tensor,
        #     # "quantized_decomposed::dequantize_per_tensor.tensor": quantized_decomposed_dequantize_per_tensor,
        #     # "quantized_decomposed::dequantize_per_tensor.tensor2": quantized_decomposed_dequantize_per_tensor,
        #     torch.ops.quantized_decomposed.quantize_per_tensor.default: quantized_decomposed_quantize_per_tensor,
        #     torch.ops.quantized_decomposed.dequantize_per_tensor.default: quantized_decomposed_dequantize_per_tensor,
        #     torch.ops.quantized_decomposed.quantize_per_channel.default: quantized_decomposed_quantize_per_channel,
        #     torch.ops.quantized_decomposed.dequantize_per_channel.default: torch.ops.quantized_decomposed.dequantize,
        # }
        # return custom_translation_table

        return None