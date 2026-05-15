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


class ONNXBackendDataset(dataset_base.DatasetBase):
    '''
    Dataset used for onnx backend tests.
    path - base path of test; folder should include model.onnx and
           subfolder test_data_set_0 with protobuf inputs/outputs.
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
        for fname in os.listdir(test_data_set_0):
            fpath = os.path.join(test_data_set_0, fname)
            assert os.path.splitext(fpath)[1] == ".pb", "non protobuf file found"

            file_bytes = open(fpath, mode='rb').read()
            tensor = TensorProto.FromString(file_bytes)
            np_array = numpy_helper.to_array((tensor))
            tensor_name = tensor.name

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
        info_dict = info_dict or dict()
        assert idx == 0
        return self.inputs, info_dict

    def __len__(self):
        return 1

    def evaluate(self, output_list, **kwargs):
        assert isinstance(output_list, list) and len(output_list) == 1, \
            "Expected output_list is a nested list with one element"
        output_list = output_list[0]

        import onnx
        out_info = onnx.load(os.path.join(self.path, "model.onnx")).graph.output
        output_dict = {}
        for output, info in zip(output_list, out_info):
            output_dict[info.name] = output

        max_nmse = 0
        for out_name, output in output_dict.items():
            expected_output = self.expected_outputs.get(out_name)
            assert expected_output is not None, f" No expected output for output named {out_name}"

            if output.dtype == object:
                np.testing.assert_array_equal(output, expected_output)
                max_nmse = 0
                continue

            output = np.squeeze(output.astype(float))
            expected_output = np.squeeze(expected_output.astype(float))

            assert expected_output.shape == output.shape, f" Shape mismatch! Expected {expected_output.shape} got {output.shape}"
            max_nmse = max(max_nmse, ((expected_output - output) ** 2 / np.maximum(expected_output, 1 ** -20)).mean())

        return {"max_nmse": max_nmse}


def onnx_backend_dataloader(settings, name, path, label_path=None, **kwargs):
    return ONNXBackendDataset(path=path, **kwargs)
