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


from typing import List, Optional

import torch
from torch.library import Library, impl

from onnxscript.onnx_opset import opset18 as op
from onnxscript.onnx_types import TensorType
from onnxscript.function_libs.torch_lib.registration import torch_op
from onnxscript.function_libs.torch_lib.ops.common import Rank


def get_custom_onnx_translation_table(opset_version):

    # @torch_op(
    #     (
    #         "aten::adaptive_avg_pool2d",
    #     ),
    #     trace_only=True,
    # )
    def custom_adaptive_avg_pool2d(input: TensorType, output_size: List[int]):
        if len(output_size) >= 2 and output_size[-1] == 1 and output_size[-2] == 1:
            avg_pool = op.GlobalAveragePool(input)
            return avg_pool
        else:
            kernel_shape= (input.shape[-1] // output_size[-1], input.shape[-2] // output_size[-2])
            strides= (input.shape[-1] // output_size[-1], input.shape[-2] // output_size[-2])
            avg_pool = op.AveragePool(input, kernel_shape=kernel_shape, strides=strides)
            return avg_pool

    # @torch_op(
    #     (
    #         "quantized_decomposed::quantize_per_channel",
    #         "quantized_decomposed::quantize_per_channel.tensor",
    #         "quantized_decomposed::quantize_per_channel.tensor2",
    #     ),
    #     trace_only=True,
    # )
    def custom_quantize_per_channel(
        input: TensorType,
        scales: TensorType,
        zero_points: TensorType,
        axis: int,
        quant_min: int,
        quant_max: int,
        dtype: int,
    ) -> TensorType:
        """Affine per channel quantization for the Tensor using the same quantization
        parameters for each channel/axis to map from floating point to quantized values.

        Uses ONNX QuantizeLinear with per-axis quantization support.
        """
        # Use opset23 for per-axis quantization support
        return op.QuantizeLinear(input, scales, zero_points, axis=axis, output_dtype=dtype)


    # @torch_op(
    #     (
    #         "quantized_decomposed::dequantize_per_channel",
    #         "quantized_decomposed::dequantize_per_channel.tensor",
    #         "quantized_decomposed::dequantize_per_channel.tensor2",
    #     ),
    #     trace_only=True,
    # )
    def custom_dequantize_per_channel(
        input: TensorType,
        scales: TensorType,
        zero_points: Optional[TensorType],
        axis: int,
        quant_min: int,
        quant_max: int,
        dtype: int,
        out_dtype: int = -1,
    ) -> TensorType:
        """Affine per channel dequantization for the Tensor using the same quantization
        parameters for each channel/axis to map from quantized values to floating point values.

        Uses ONNX DequantizeLinear with per-axis quantization support.
        """
        # onnx and pytorch differ in the type of zero_points
        # in pytorch type of zero_points is int. in onnx, it is the type of input.
        if zero_points is not None:
            zero_points = op.CastLike(zero_points, input)
        #
        # Use opset23 for per-axis quantization support with optional output_dtype
        if out_dtype in (-1, None):
            # Use default output type (same as scales type)
            return op.DequantizeLinear(input, scales, zero_points, axis=axis)
        else:
            assert out_dtype > 0, f"out_dtype must be -1 or > 0 not {out_dtype}"
            return op.DequantizeLinear(
                input, scales, zero_points, axis=axis, output_dtype=out_dtype
            )

    custom_translation_table = {
        torch.ops.aten.adaptive_avg_pool2d.default: custom_adaptive_avg_pool2d,
        torch.ops.quantized_decomposed.quantize_per_channel.default: custom_quantize_per_channel,
        torch.ops.quantized_decomposed.dequantize_per_channel.default: custom_dequantize_per_channel
    }

    return custom_translation_table