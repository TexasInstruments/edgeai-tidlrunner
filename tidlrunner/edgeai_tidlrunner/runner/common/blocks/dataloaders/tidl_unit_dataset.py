# Copyright (c) 2018-2021, Texas Instruments
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
import numpy as np
from . import dataset_base


class TIDLUnitDataset(dataset_base.DatasetBase):
    '''
    Dataset used for tidl unit tests.
    path - base path of test; folder should include model.onnx and
           subfolder test_data_set_0 with inputs/outputs.
    '''

    def __init__(self, path: str, **kwargs):
        super().__init__(**kwargs)

        import onnx
        from onnx.onnx_ml_pb2 import TensorProto
        from onnx import numpy_helper

        self.path = path

        test_data_set_0 = os.path.join(path, "test_data_set_0")
        assert os.path.exists(test_data_set_0), "provided path must have test_data_set_0 subdirectory with protobuff i/o's"

        self.inputs = {}
        self.expected_outputs = {}
        in_counter = 0
        out_counter = 0
        onnx_model = onnx.load(os.path.join(path, "model.onnx"))

        input_npz_path = os.path.join(test_data_set_0, "input.npz")
        output_npz_path = os.path.join(test_data_set_0, "output.npz")
        if os.path.exists(input_npz_path) and os.path.exists(output_npz_path):
            input_npz_data = np.load(input_npz_path)
            initializer_names = [init.name for init in onnx_model.graph.initializer]

            for i, input_info in enumerate(onnx_model.graph.input):
                tensor_name = input_info.name
                if tensor_name in initializer_names:
                    continue
                if tensor_name in input_npz_data:
                    self.inputs[tensor_name] = input_npz_data[tensor_name]
                elif f"input_{i}" in input_npz_data:
                    self.inputs[tensor_name] = input_npz_data[f"input_{i}"]
                elif len(input_npz_data.files) == 1:
                    self.inputs[tensor_name] = input_npz_data[input_npz_data.files[0]]
                elif i < len(input_npz_data.files):
                    self.inputs[tensor_name] = input_npz_data[input_npz_data.files[i]]
                else:
                    raise ValueError(f"Could not find input tensor {tensor_name} in input.npz")

            output_npz_data = np.load(output_npz_path)
            for i, output_info in enumerate(onnx_model.graph.output):
                tensor_name = output_info.name
                if tensor_name in output_npz_data:
                    self.expected_outputs[tensor_name] = output_npz_data[tensor_name]
                elif f"output_{i}" in output_npz_data:
                    self.expected_outputs[tensor_name] = output_npz_data[f"output_{i}"]
                elif len(output_npz_data.files) == 1:
                    self.expected_outputs[tensor_name] = output_npz_data[output_npz_data.files[0]]
                elif i < len(output_npz_data.files):
                    self.expected_outputs[tensor_name] = output_npz_data[output_npz_data.files[i]]
                else:
                    raise ValueError(f"Could not find output tensor {tensor_name} in output.npz")
            return

        for fname in os.listdir(test_data_set_0):
            fpath = os.path.join(test_data_set_0, fname)
            assert (os.path.splitext(fpath)[1] in [".pb", ".bin", ".npz"]), \
                "Invalid file format - Allowed values are, protobuf(.pb), python list(.bin), numpy archive(.npz)"
            file_ext = os.path.splitext(fpath)[1]
            tensor_name = ""
            if file_ext == ".pb":
                file_bytes = open(fpath, mode='rb').read()
                tensor = TensorProto.FromString(file_bytes)
                np_array = numpy_helper.to_array((tensor))
                tensor_name = tensor.name
            elif file_ext == ".bin":
                file_bytes = open(fpath, mode='rb').read()
                if "input_" in fname:
                    tensor_info = onnx_model.graph.input[in_counter]
                elif "output_" in fname:
                    tensor_info = onnx_model.graph.output[out_counter]
                else:
                    assert False, "Incorrect file name, should start with input_ or output_"
                shape_dims = tensor_info.type.tensor_type.shape.dim
                shape = [d.dim_value if d.dim_value > 0 else 1 for d in shape_dims]
                onnx_data_type = tensor_info.type.tensor_type.elem_type
                dtype_map = {
                    TensorProto.FLOAT: np.float32, TensorProto.UINT8: np.uint8,
                    TensorProto.INT8: np.int8, TensorProto.UINT16: np.uint16,
                    TensorProto.INT16: np.int16, TensorProto.INT32: np.int32,
                    TensorProto.INT64: np.int64,
                }
                dtype = dtype_map.get(onnx_data_type, np.float32)
                np_array = np.frombuffer(file_bytes, dtype=dtype).reshape(shape)

            if "input_" in fname:
                if tensor_name == "":
                    tensor_name = onnx_model.graph.input[in_counter].name
                self.inputs[tensor_name] = np_array
                in_counter += 1
            else:
                assert "output_" in fname
                if tensor_name == "":
                    tensor_name = onnx_model.graph.output[out_counter].name
                self.expected_outputs[tensor_name] = np_array
                out_counter += 1

    def __getitem__(self, idx, info_dict=None, **kwargs):
        assert idx == 0
        return self.inputs, info_dict

    def __len__(self):
        return 1

    def __call__(self, index, info_dict=None):
        return self.__getitem__(index, info_dict)

    def evaluate(self, output_list, **kwargs):
        assert isinstance(output_list, list), "Expected output_list is a nested list"

        import onnx
        out_info = onnx.load(os.path.join(self.path, "model.onnx")).graph.output
        output_dict = {}
        for output, info in zip(output_list, out_info):
            output_dict[info.name] = output

        nmse = []
        mse = []
        max_delta = []
        outputs = []
        expected_outputs = []
        epsilon = 1e-10
        for out_name, output in output_dict.items():
            expected_output = self.expected_outputs.get(out_name)
            assert expected_output is not None, f" No expected output for output named {out_name}"

            if output.dtype == object:
                np.testing.assert_array_equal(output, expected_output)
                nmse.append(None)
                mse.append(None)
                continue

            output = np.squeeze(output.astype(float))
            expected_output = np.squeeze(expected_output.astype(float))

            assert expected_output.shape == output.shape, f" Shape mismatch! Expected {expected_output.shape} got {output.shape}"

            curr_mse = np.mean((expected_output - output) ** 2)
            curr_var = np.var(expected_output)
            curr_nmse = None if curr_var < epsilon else curr_mse / curr_var

            mse.append(None if (curr_mse is None or np.isnan(curr_mse)) else curr_mse)
            nmse.append(None if (curr_nmse is None or np.isnan(curr_nmse)) else curr_nmse)

            curr_max_delta = np.max(np.abs(expected_output - output))
            max_delta.append(None if (curr_max_delta is None or np.isnan(curr_max_delta)) else curr_max_delta)

            outputs.append(output)
            expected_outputs.append(expected_output)

            if os.path.basename(os.path.normpath(self.path)).startswith("TopK"):
                break

        return {"outputs": outputs, "expected_outputs": expected_outputs,
                "nmse": nmse, "mse": mse, "delta": max_delta}


def tidl_unit_dataloader(settings, name, path, label_path=None, **kwargs):
    return TIDLUnitDataset(path=path, **kwargs)
