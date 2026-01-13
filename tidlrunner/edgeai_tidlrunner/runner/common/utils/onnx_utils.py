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

# TI's adaptation of update_model_dims.
# Original code from onnx.tools.update_model_dims() has bug of iterating through all layers.
# But this version has less error checking code.


__all__ = ['onnx_update_model_dims', 'get_all_output_names']


import os
import sys
import logging
from typing import List, Any


def update_dim(tensor=None, new_dim_value=None, dim_idx=None):
    dim_proto = tensor.type.tensor_type.shape.dim[dim_idx]
    if isinstance(new_dim_value, str):
        dim_proto.dim_param = new_dim_value
    elif isinstance(new_dim_value, int):
        if new_dim_value >= 0:
            assert not (dim_proto.HasField('dim_value') and (dim_proto.dim_value != new_dim_value))
        else: #new_dim_value is negative. Not handled currently.
            assert False
    return


def onnx_update_model_dims(model, input_dims, output_dims):
    for input_name, input_dim_arr in input_dims.items():
        input_layer_tensor = [input_tensor for input_tensor in model.graph.input if input_tensor.name == input_name][0]
        for dim_idx, new_dim_value in enumerate(input_dim_arr):
            update_dim(tensor=input_layer_tensor, new_dim_value=new_dim_value, dim_idx=dim_idx)

    for output_name, output_dim_arr in output_dims.items():
        output_layer_tensor = [output_tensor for output_tensor in model.graph.output if output_tensor.name == output_name][0]
        for dim_idx, new_dim_value in enumerate(output_dim_arr):
            update_dim(tensor=output_layer_tensor, new_dim_value=new_dim_value, dim_idx=dim_idx)

    import onnx
    onnx.checker.check_model(model)
    return model


def _find_out_layers (curr_layer) -> List[Any]:
    """
    Return all input nodes to a given node
    """
    out_layers = list(curr_layer.outputs)
    return out_layers


def _is_end_node(node) -> bool:
    """
    Return True if a node is an output node of the model
    """
    if len(_find_out_layers(node)) == 0:
        return True
    return False


def _get_all_child_nodes(node, end_nodes, searched_nodes):
    # recursive function to find all the deny list nodes
    if node not in searched_nodes:
        searched_nodes.append(node)
        logging.debug(f"Adding {node.name} to node list.")

    if node.name in end_nodes or _is_end_node(node):
        return searched_nodes

    for out in node.outputs:
        for n_id in out.outputs:
            if n_id not in searched_nodes:
                searched_nodes = _get_all_child_nodes(n_id, end_nodes, searched_nodes)

    return searched_nodes


def get_all_nodes(model_path, start_end_layers={}, verbose=False, graph=None, **kwargs):
    import onnx_graphsurgeon as gs
    import onnx
    """
    Main function
    ---------------------------------------------------------
    Inputs
    ---------------------------------------------------------
    model_path:             path to input ONNX model
    start_end_layers:       dictionary of the start and end layers, between which (including start
                            and end node) needs to be added to deny list
                            if "None" is passed in the end node (values of dict), then the model output nodes
                            are assumed as the end nodes
     ---------------------------------------------------------------
    Output
    ---------------------------------------------------------------
    nodes:                  list of string of all the nodes that need to be added in the deny list
    """
    logging.getLogger().setLevel(logging.DEBUG if verbose else logging.INFO)

    if graph is None:
        if not os.path.isfile(model_path):
            # check for valid path
            logging.error(f"File {model_path} not found")
            sys.exit(-1)
        #
        model = onnx.load(model_path)
        graph = gs.import_onnx(model)

    model_outputs = [node.inputs[0].name for node in graph.outputs]
    name_to_node_dict = {node.name: node for node in graph.nodes}

    searched_nodes = []
    searched_node_names = []
    for start_name, end_name in start_end_layers.items():
        if start_name not in name_to_node_dict:
            print(f'WARNING: start_name given is not a valid onnx node name: {start_name} - {__file__}')
            continue
        if end_name is None:
            end_name = model_outputs
        elif end_name not in name_to_node_dict:
            print(f'WARNING: end_name given is not a valid onnx node name: {end_name} - {__file__}')
            continue
        elif not isinstance(end_name, (list,tuple)):
            end_name = [end_name]
        #
        start_node = name_to_node_dict[start_name]
        searched_nodes = _get_all_child_nodes(start_node, end_name, searched_nodes)
        # searched_nodes.append(start_node)
        logging.debug(f"Adding {start_name} to node list.")

    return searched_nodes



def get_all_node_names(model_path, start_end_layers={}, match_node_names=True, match_output_names=True, verbose=False, **kwargs):
    import onnx_graphsurgeon as gs
    """
    Main function
    ---------------------------------------------------------
    Inputs
    ---------------------------------------------------------
    model_path:             path to input ONNX model
    start_end_layers:       dictionary of the start and end layers, between which (including start
                            and end node) needs to be added to deny list
                            if "None" is passed in the end node (values of dict), then the model output nodes
                            are assumed as the end nodes
     ---------------------------------------------------------------
    Output
    ---------------------------------------------------------------
    nodes:                  comma separated string of all the nodes that need to be added in the deny list
    """
    logging.getLogger().setLevel(logging.DEBUG if verbose else logging.INFO)

    import onnx
    import onnx_graphsurgeon as gs

    # check for valid path
    if not os.path.isfile(model_path):
        logging.error(f"File {model_path} not found")
        sys.exit(-1)

    model = onnx.load(model_path)
    graph = gs.import_onnx(model)

    start_end_node_names = {}
    for k, v in start_end_layers.items():
        start_node = end_node = None
        for node in graph.nodes:
            for out in node.outputs:
                if match_node_names and k == node.name:
                    start_node = node.name
                elif match_output_names and k == out.name:
                    start_node = node.name
                if match_node_names and v == node.name:
                    end_node = node.name
                elif match_output_names and v == out.name:
                    end_node = node.name
        start_end_node_names.update({start_node: end_node})

    selected_nodes = get_all_nodes(model, start_end_node_names, verbose, graph=graph)

    node_names = [node.name for node in selected_nodes]
    logging.info(f"get_all_output_names with start:end={start_end_layers} returned {len(node_names)} nodes: {node_names}")
    return node_names



def get_all_output_names(model_path, start_end_layers={}, match_node_names=True, match_output_names=True, verbose=False, **kwargs):
    import onnx_graphsurgeon as gs
    """
    Main function
    ---------------------------------------------------------
    Inputs
    ---------------------------------------------------------
    model_path:             path to input ONNX model
    start_end_layers:       dictionary of the start and end layers, between which (including start
                            and end node) needs to be added to deny list
                            if "None" is passed in the end node (values of dict), then the model output nodes
                            are assumed as the end nodes
     ---------------------------------------------------------------
    Output
    ---------------------------------------------------------------
    nodes:                  comma separated string of all the nodes that need to be added in the deny list
    """
    logging.getLogger().setLevel(logging.DEBUG if verbose else logging.INFO)

    import onnx
    import onnx_graphsurgeon as gs

    # check for valid path
    if not os.path.isfile(model_path):
        logging.error(f"File {model_path} not found")
        sys.exit(-1)

    model = onnx.load(model_path)
    graph = gs.import_onnx(model)

    start_end_node_names = {}
    for k, v in start_end_layers.items():
        start_node = end_node = None
        for node in graph.nodes:
            for out in node.outputs:
                if match_node_names and k == node.name:
                    start_node = node.name
                elif match_output_names and k == out.name:
                    start_node = node.name
                if match_node_names and v == node.name:
                    end_node = node.name
                elif match_output_names and v == out.name:
                    end_node = node.name
        start_end_node_names.update({start_node: end_node})

    selected_nodes = get_all_nodes(model, start_end_node_names, verbose, graph=graph)

    output_names = [out.name for node in selected_nodes for out in node.outputs]
    logging.info(f"get_all_output_names with start:end={start_end_layers} returned {len(output_names)} nodes: {output_names}")
    return output_names
