#!/usr/bin/env python3
# Copyright (c) 2018-2025, Texas Instruments
# All Rights Reserved.

"""
HTML Generator - Fuse JSON data into HTML template

This script loads extracted JSON data and fuses it into an HTML template
to create a self-contained, interactive visualization.

Usage:
    python html_generator.py <data.json> <template.html> <output.html> [--activations <activations.json>]

Example:
    python html_generator.py model_data.json.gz template.html output.html
    python html_generator.py model_data.json.gz template.html output.html --activations model_data_activations.json.gz

Output:
    Single self-contained HTML file with all data embedded
"""

import json
import sys
import os
import gzip
import base64
from typing import Dict, Any, List, Optional
from pathlib import Path


def generate_raw_text_from_parameters(layer_index: int, layer_type: str, layer_name: str,
                                      parameters: Dict, gmacs: float) -> str:
    """Generate formatted raw_text from layer parameters for display in analysis tab"""
    lines = []
    lines.append(f"Layer {layer_index}: {layer_type} \"{layer_name}\"")

    # Add key parameters
    for key, value in parameters.items():
        if key == 'outputs':
            lines.append("  Outputs:")
            for i, out in enumerate(value):
                if isinstance(out, dict) and 'dims' in out:
                    dims_str = str(out['dims'])
                    lines.append(f"    [{i}] numDim={out.get('numDim', len(out['dims']))} dims={dims_str}")
                    if 'elementType' in out:
                        lines.append(f"         elementType={out['elementType']}")
        elif key == 'actParams':
            lines.append("  actParams:")
            for k, v in value.items():
                lines.append(f"     {k}={v}")
        elif isinstance(value, dict):
            lines.append(f"  {key}:")
            for k, v in value.items():
                lines.append(f"     {k}={v}")
        else:
            lines.append(f"  {key}={value}")

    # Add GMACS info
    if gmacs > 0:
        lines.append(f"  GMACS: {gmacs:.6f}")

    return "\n".join(lines)


def build_hierarchical_tree(layer_details: Dict[str, Any], edges: List[Dict]) -> Dict[str, Any]:
    """Build hierarchical tree structure from layer details based on node naming"""
    print("Building hierarchical tree structure...")

    node_topo_indices = {name: idx for idx, name in enumerate(layer_details.keys())}

    tree_dict = {}

    for node_name, node_data in layer_details.items():
        node_path = []
        for part in node_name.split('/'):
            node_path.extend(part.split('.'))

        if not node_path:
            continue

        node_info = {
            "name": node_name,
            "op": node_data.get('type', 'Unknown'),
            "is_leaf": True,
            "node_details": {
                "name": node_name,
                "op": node_data.get('type', 'Unknown'),
                "inputs": node_data.get('inputs', []),
                "outputs": node_data.get('outputs', []),
                "input_metadata": node_data.get('input_details', []),
                "output_metadata": node_data.get('output_details', []),
                "attributes": node_data.get('attributes', {})
            },
            "topo_idx": node_topo_indices.get(node_name, 0)
        }

        current_level = tree_dict

        for i, part in enumerate(node_path):
            is_last = (i == len(node_path) - 1)

            if part not in current_level:
                if is_last:
                    current_level[part] = node_info
                else:
                    current_level[part] = {
                        "is_leaf": False,
                        "children": {}
                    }
            elif not is_last:
                current_level[part]["is_leaf"] = False

            if not is_last:
                if "children" not in current_level[part]:
                    current_level[part]["children"] = {}
                current_level = current_level[part]["children"]

    print(f"  Built tree with {len(tree_dict)} top-level modules")

    tree_dict = flatten_single_child_modules(tree_dict)

    return tree_dict


def flatten_single_child_modules(tree: Dict, parent_key: str = '') -> Dict:
    """Recursively flatten modules with only one child module"""
    flattened = {}

    for key, value in tree.items():
        if not value.get("is_leaf", False) and "children" in value:
            value["children"] = flatten_single_child_modules(value["children"], key)
            children = value["children"]

            if len(children) == 1 and not any(child.get("is_leaf", False) for child in children.values()):
                child_key = list(children.keys())[0]
                flattened[f"{key}.{child_key}"] = children[child_key]
            else:
                flattened[key] = value
        else:
            flattened[key] = value

    return flattened


def calculate_node_depths_and_positions(nodes: Dict, edges: List[Dict], width=1200, height=800) -> Dict:
    """
    Calculate optimal x,y positions for nodes using Netron-style layout

    Args:
        nodes: Dict of node_name -> node_data
        edges: List of edge dictionaries
        width: Canvas width
        height: Canvas height

    Returns:
        Updated nodes dict with x, y, depth, horizontal_position
    """
    print("Calculating node positions...")

    node_list = list(nodes.keys())
    node_to_idx = {name: idx for idx, name in enumerate(node_list)}

    children = {name: [] for name in node_list}
    parents = {name: [] for name in node_list}

    for edge in edges:
        src_name = edge.get('source_node_name')
        tgt_name = edge.get('target_node_name')

        if src_name in children and tgt_name in parents:
            children[src_name].append(tgt_name)
            parents[tgt_name].append(src_name)

    roots = [name for name in node_list if len(parents[name]) == 0]
    if not roots:
        roots = [node_list[0]]

    print(f"  Found {len(roots)} root nodes")

    depths = {}

    def assign_depth(node_name, depth):
        current_depth = depths.get(node_name, -1)
        if depth > current_depth:
            depths[node_name] = depth
            for child_name in children[node_name]:
                assign_depth(child_name, depth + 1)

    actual_roots = []
    constant_nodes = []

    for name in roots:
        node_data = nodes.get(name, {})
        node_type = node_data.get('type', '')

        if node_type in ['Constant', 'Initializer'] or 'Constant' in name:
            constant_nodes.append(name)
        else:
            actual_roots.append(name)

    if not actual_roots:
        actual_roots = roots

    print(f"  Actual roots: {len(actual_roots)}")
    print(f"  Constant nodes: {len(constant_nodes)}")

    for root in actual_roots:
        assign_depth(root, 0)

    for const_name in constant_nodes:
        if const_name not in depths:
            child_depths = [depths.get(child, 0) for child in children[const_name]]
            if child_depths:
                depths[const_name] = max(0, min(child_depths) - 1)
            else:
                depths[const_name] = 0

    for name in node_list:
        if name not in depths:
            depths[name] = 0

    max_depth = max(depths.values()) if depths else 0
    print(f"  Max depth: {max_depth}")

    depth_groups = {}
    for name, depth in depths.items():
        if depth not in depth_groups:
            depth_groups[depth] = []
        depth_groups[depth].append(name)

    horizontal_positions = {}

    for depth in sorted(depth_groups.keys()):
        nodes_at_depth = depth_groups[depth]

        if depth == 0:
            for i, name in enumerate(sorted(nodes_at_depth)):
                horizontal_positions[name] = i
        else:
            node_scores = []
            for name in nodes_at_depth:
                parent_positions = [
                    horizontal_positions.get(p, 0)
                    for p in parents[name]
                    if p in horizontal_positions
                ]
                avg_pos = sum(parent_positions) / len(parent_positions) if parent_positions else 0
                node_scores.append((name, avg_pos))

            node_scores.sort(key=lambda x: x[1])

            for i, (name, _) in enumerate(node_scores):
                horizontal_positions[name] = i

    max_width = max(len(nodes) for nodes in depth_groups.values()) if depth_groups else 1
    print(f"  Max width: {max_width}")

    VERTICAL_SPACING = 150
    HORIZONTAL_SPACING = 200
    PADDING = 100

    for name in node_list:
        depth = depths.get(name, 0)
        h_pos = horizontal_positions.get(name, 0)

        layer_width = len(depth_groups.get(depth, [])) * HORIZONTAL_SPACING
        start_x = (width - layer_width) / 2 + HORIZONTAL_SPACING / 2

        nodes[name]['x'] = start_x + h_pos * HORIZONTAL_SPACING
        nodes[name]['y'] = PADDING + depth * VERTICAL_SPACING
        nodes[name]['depth'] = depth
        nodes[name]['horizontal_position'] = h_pos

    print(f"  Calculated positions for {len(nodes)} nodes")

    return nodes


def load_activation_data_from_model_dir(model_dir: str, tidl_data: Dict) -> Dict[str, Any]:
    """
    Load activation data from model directory raw files

    Args:
        model_dir: Path to model directory (e.g., work_dirs/compile/AM62A/8bits/...)
        tidl_data: TIDL data dict with subgraphs and layers

    Returns:
        Dict mapping activation keys (subgraph_layer) to activation plot data
    """
    print(f"\nLoading activation data from model directory: {model_dir}")

    if not os.path.exists(model_dir):
        print(f"  WARNING: Model directory not found: {model_dir}")
        return {}

    try:
        # Import ActivationDataParser from data_extractor
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "data_extractor",
            os.path.join(os.path.dirname(__file__), "data_extractor.py")
        )
        data_extractor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(data_extractor)
        ActivationDataParser = data_extractor.ActivationDataParser

        # Initialize parser
        parser = ActivationDataParser(model_dir=model_dir, frame_idx=0, tidl_data=tidl_data)

        activation_data = {}
        layers_loaded = 0

        # Process each subgraph
        for subgraph_id, subgraph_info in tidl_data.items():
            # Extract numeric subgraph index from key like 'tidl_0' → 0
            sg_num = int(''.join(filter(str.isdigit, str(subgraph_id))) or '0')

            layers = subgraph_info.get('layers', {})

            # Layers can be dict (unified schema) or list
            if isinstance(layers, dict):
                layers_list = list(layers.values())
            else:
                layers_list = layers

            for layer in layers_list:
                layer_idx = layer.get('layer_id', layer.get('layer_index', 0))
                layer_name = layer.get('layer_name', '')
                # Try to get onnx_node_index from either onnx_mapping dict or direct field
                onnx_node_idx = layer.get('onnx_node_index')
                if onnx_node_idx is None:
                    onnx_node_idx = layer.get('onnx_mapping', {}).get('onnx_node_index')

                # Process activation data for this layer (using layer_idx to read file)
                layer_data = parser.process_layer(sg_num, str(layer_idx))

                if layer_data:
                    # Key uses numeric sg_num so template can look up by numeric subgraph ID
                    if onnx_node_idx is not None:
                        activation_key = f"{sg_num}_{onnx_node_idx}"
                    else:
                        activation_key = f"{sg_num}_{layer_idx}"
                    activation_data[activation_key] = layer_data
                    layers_loaded += 1

        print(f"  Loaded activation data for {layers_loaded} layers")
        return activation_data

    except ImportError as e:
        print(f"  WARNING: Could not import ActivationDataParser: {e}")
        return {}
    except Exception as e:
        print(f"  WARNING: Error loading activation data: {e}")
        return {}


def load_metrics_from_model_dir(model_dir: str) -> Dict[str, List[Dict]]:
    """
    Load metrics data from model directory (accuracy_layer_outputs_ref.xlsx)

    Args:
        model_dir: Path to model directory

    Returns:
        Dict mapping subgraph_id to list of metric entries
    """
    print(f"\nLoading metrics from model directory: {model_dir}")

    if not os.path.exists(model_dir):
        print(f"  WARNING: Model directory not found: {model_dir}")
        return {}

    # Look for metrics file (generated by analyze pipeline)
    xlsx_path = os.path.join(model_dir, 'analyze.xlsx')
    if not os.path.exists(xlsx_path):
        print(f"  WARNING: Metrics file not found: {xlsx_path}")
        print(f"  Note: Metrics are only available after running analyze pipeline")
        return {}

    try:
        # Import MetricsParser from data_extractor
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "data_extractor",
            os.path.join(os.path.dirname(__file__), "data_extractor.py")
        )
        data_extractor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(data_extractor)
        MetricsParser = data_extractor.MetricsParser

        # Initialize parser with xlsx_path
        parser = MetricsParser(xlsx_path=xlsx_path)
        metrics_data = parser.get_metrics()

        total_metrics = sum(len(v) for v in metrics_data.values())
        print(f"  Loaded metrics for {total_metrics} layers")
        return metrics_data

    except ImportError as e:
        print(f"  WARNING: Could not import MetricsParser: {e}")
        return {}
    except Exception as e:
        print(f"  WARNING: Error loading metrics: {e}")
        return {}


def load_json_data(json_path: str) -> Dict[str, Any]:
    """Load JSON data (supports both .json and .json.gz)"""
    print(f"Loading JSON data from: {json_path}")

    try:
        if json_path.endswith('.gz'):
            with gzip.open(json_path, 'rt', encoding='utf-8') as f:
                data = json.load(f)
        else:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

        # Check for unified schema format (new) or old format
        if 'model' in data and 'runtime' in data:
            # New unified schema v1.0 with model.onnx and runtime.tidl_rt
            expected_keys = ['metadata', 'model', 'runtime']
            print(f"  Detected: Unified Schema v1.0")
        elif 'onnx' in data and 'tidl' in data:
            # Old unified schema (before restructuring)
            expected_keys = ['metadata', 'onnx', 'tidl']
            print(f"  Detected: Unified Schema v1.0 (legacy paths)")
        else:
            expected_keys = ['metadata', 'model', 'compilation']
            print(f"  Detected: Legacy format")

        missing_keys = [k for k in expected_keys if k not in data]
        if missing_keys:
            print(f"WARNING: JSON missing keys: {missing_keys}")

        print(f"JSON data loaded successfully")
        return data

    except Exception as e:
        print(f"ERROR: Failed to load JSON: {e}")
        raise


def compress_activation_data_per_layer(activation_data: Dict[str, Any]) -> Dict[str, str]:
    """Compress activation data per-layer for lazy loading

    Each layer's data is compressed separately to allow on-demand decompression.

    Args:
        activation_data: Dict mapping layer keys to activation plot data

    Returns:
        Dict mapping layer keys to base64-encoded compressed strings
    """
    print("\nCompressing activation data (per-layer for lazy loading)...")

    compressed_data = {}
    total_original_size = 0
    total_compressed_size = 0

    for layer_key, layer_activation in activation_data.items():
        # Compress each layer separately
        layer_json_str = json.dumps(layer_activation)
        layer_size_before = len(layer_json_str)
        total_original_size += layer_size_before

        layer_compressed = gzip.compress(layer_json_str.encode('utf-8'), compresslevel=9)
        layer_b64 = base64.b64encode(layer_compressed).decode('ascii')
        layer_size_after = len(layer_b64)
        total_compressed_size += layer_size_after

        compressed_data[layer_key] = layer_b64

    print(f"  Original activation data size: {total_original_size / (1024*1024):.2f} MB")
    print(f"  Compressed size (base64): {total_compressed_size / (1024*1024):.2f} MB")
    print(f"  Compression ratio: {total_original_size / total_compressed_size:.2f}x")
    print(f"  Total layers compressed: {len(compressed_data)}")

    return compressed_data


def compress_activation_data(activation_data: Dict[str, Any]) -> str:
    """Compress activation data using gzip+base64 encoding (legacy method)

    Reduces activation data size by 5-10x using maximum compression.

    Args:
        activation_data: Dict containing activation plot data

    Returns:
        Base64-encoded compressed string
    """
    print("\nCompressing activation data...")

    activation_json_str = json.dumps(activation_data)
    activation_size_before = len(activation_json_str) / (1024 * 1024)
    print(f"  Original activation data size: {activation_size_before:.2f} MB")

    activation_compressed = gzip.compress(activation_json_str.encode('utf-8'), compresslevel=9)
    activation_size_compressed = len(activation_compressed) / (1024 * 1024)
    print(f"  Compressed size: {activation_size_compressed:.2f} MB")

    activation_b64 = base64.b64encode(activation_compressed).decode('ascii')
    activation_size_b64 = len(activation_b64) / (1024 * 1024)
    print(f"  Base64 encoded size: {activation_size_b64:.2f} MB")
    print(f"  Compression ratio: {activation_size_before / activation_size_b64:.2f}x")

    return activation_b64


def _has_activation_data(layer: dict) -> bool:
    """Check if a layer has real embedded activation data (histogram or bin_files)."""
    act = layer.get('activation_data') or {}
    hist = act.get('histogram') or {}
    # Schema format: tidl_bins array with data
    if hist.get('tidl_bins') or hist.get('tidl_counts'):
        return True
    # Legacy format: tidl dict with counts
    tidl = hist.get('tidl')
    if isinstance(tidl, dict) and tidl.get('counts'):
        return True
    # bin_files inside activation_data
    bin_files = act.get('bin_files') or {}
    if isinstance(bin_files, dict) and bin_files.get('tidl'):
        return True
    # bin_files as sibling field on layer
    layer_bin = layer.get('bin_files') or {}
    if isinstance(layer_bin, dict) and layer_bin.get('tidl'):
        return True
    return False


def generate_html(json_data: Dict[str, Any], template_path: str, output_path: str, activations_data: Dict[str, Any] = None, json_path: str = None):
    """Generate HTML by fusing unified schema JSON data into template

    Args:
        json_data: Dict containing unified schema data (metadata, onnx, tidl, tvm)
        template_path: Path to HTML template file
        output_path: Path to output HTML file
        activations_data: Optional dict containing activation data (if None, will load from model_dir)
    """
    print(f"\nReading template: {template_path}")

    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read()

    # Validate unified schema format
    # Support both new (model/runtime) and old (onnx/tidl) paths
    has_new_schema = all(key in json_data for key in ['metadata', 'model', 'runtime'])
    has_old_schema = all(key in json_data for key in ['metadata', 'onnx', 'tidl'])

    if not (has_new_schema or has_old_schema):
        raise ValueError("Invalid JSON format. Expected unified schema with 'metadata', 'model', 'runtime' keys")

    print("  Processing: Unified Schema v1.0")

    # Use provided activation data or default to empty dict
    if activations_data is None:
        activations_data = {}

    # Process unified schema format
    if True:  # Always process as unified schema
        # UNIFIED SCHEMA v1.0
        print("  Detected: UNIFIED SCHEMA v1.0")
        is_unified_schema = True

        metadata = json_data['metadata']
        onnx_data = json_data['model']['onnx']
        tidl_data_raw = json_data['runtime']

        # Build model_data from ONNX section
        onnx_layers = onnx_data.get('layers', {})

        # Transform ONNX layers to template format
        transformed_layers = {}
        for layer_name, layer_info in onnx_layers.items():
            # Transform input_details to input_metadata with 'name' field
            input_metadata = []
            for inp in layer_info.get('input_details', []):
                inp_copy = inp.copy()
                if 'tensor_name' in inp_copy:
                    inp_copy['name'] = inp_copy.pop('tensor_name')
                input_metadata.append(inp_copy)

            # Transform output_details to output_metadata with 'name' field
            output_metadata = []
            for out in layer_info.get('output_details', []):
                out_copy = out.copy()
                if 'tensor_name' in out_copy:
                    out_copy['name'] = out_copy.pop('tensor_name')
                output_metadata.append(out_copy)

            # Get all input tensor names (including weights/biases from input_details)
            all_inputs = [inp.get('tensor_name', inp.get('name', ''))
                         for inp in layer_info.get('input_details', [])]

            # Get output tensor names from output_details
            all_outputs = [out.get('tensor_name', out.get('name', ''))
                          for out in layer_info.get('output_details', [])]

            transformed_layers[layer_name] = {
                'layer_name': layer_name,
                'type': layer_info.get('type', 'Unknown'),
                'input': all_inputs,
                'output': all_outputs,
                'input_metadata': input_metadata,
                'output_metadata': output_metadata,
                'attributes': layer_info.get('attributes', {})
            }

        # Add virtual Input and Output nodes so they appear in the graph
        for inp in metadata.get('inputs', []):
            tensor = inp.get('name', '')
            if tensor:
                transformed_layers[f'__input_{tensor}__'] = {
                    'layer_name': tensor,
                    'type': 'Input',
                    'input': [],
                    'output': [tensor],
                    'input_metadata': [],
                    'output_metadata': [{'name': tensor, 'shape': inp.get('shape', []), 'dtype': inp.get('dtype', 'float32'), 'is_constant': False}],
                    'attributes': {'shape': inp.get('shape', []), 'dtype': inp.get('dtype', 'float32')}
                }
        for out in metadata.get('outputs', []):
            tensor = out.get('name', '')
            if tensor:
                transformed_layers[f'__output_{tensor}__'] = {
                    'layer_name': tensor,
                    'type': 'Output',
                    'input': [tensor],
                    'output': [],
                    'input_metadata': [{'name': tensor, 'shape': out.get('shape', []), 'dtype': out.get('dtype', 'float32'), 'is_constant': False}],
                    'output_metadata': [],
                    'attributes': {'shape': out.get('shape', []), 'dtype': out.get('dtype', 'float32')}
                }

        # Transform to template format
        model_data = {
            'model_details': {
                'name': metadata.get('model_name', ''),
                'weights': onnx_data.get('total_weights', 0),
                'no_of_layers': onnx_data.get('num_layers', 0),
                'opset_version': onnx_data.get('opset_version', ''),
                'ir_version': onnx_data.get('ir_version', ''),
                'input_shape': metadata.get('inputs', []),
                'output_shape': metadata.get('outputs', [])
            },
            'layer_details': transformed_layers,
            'edges': []  # Will be built from ONNX layers
        }

        # Build edges from ONNX layer connections
        edges = []
        layer_names = list(transformed_layers.keys())
        for i, layer_name in enumerate(layer_names):
            layer = transformed_layers[layer_name]
            outputs = layer.get('output', [])

            # Find connections to other layers
            for j, target_layer_name in enumerate(layer_names):
                if i == j:
                    continue
                target_layer = transformed_layers[target_layer_name]
                # Use data inputs only (filter out constants)
                target_input_metadata = target_layer.get('input_metadata', [])
                target_inputs = [inp.get('name', '') for inp in target_input_metadata
                               if not inp.get('is_constant', False)]

                # Check if any of our outputs are inputs to target layer
                for output_tensor in outputs:
                    if output_tensor in target_inputs:
                        edges.append({
                            'source_node_id': i,
                            'target_node_id': j,
                            'source_node_name': layer_name,
                            'target_node_name': target_layer_name,
                            'connection_info': {
                                'tensor': output_tensor,
                                'shape': []  # Can extract from output_details if needed
                            }
                        })

        model_data['edges'] = edges

        # Build node_support from ONNX layers
        # Unified schema uses runtime_assignment; legacy uses offload field
        node_support = {}
        for node_idx, (layer_name, layer_info) in enumerate(onnx_layers.items()):
            if is_unified_schema:
                ra = layer_info.get('runtime_assignment', {})
                assigned_runtime = ra.get('assigned_runtime', 'arm')
                # tidl_rt and tvm_rt are both DSP-accelerated; arm is not
                is_supported = assigned_runtime in ('tidl_rt', 'tvm_rt')
                reason_raw = ra.get('reason', '')
                reason = ('; '.join(reason_raw) if isinstance(reason_raw, list) else str(reason_raw or '')) if not is_supported else ''
            else:
                offload = layer_info.get('offload', {})
                runtime_val = offload.get('RunTime')
                assigned_runtime = 'arm' if runtime_val else 'tidl_rt'
                is_supported = runtime_val is None
                reason = offload.get('reason', '') if not is_supported else ''

            node_support[str(node_idx)] = {
                'supported': is_supported,
                'diagInfo': reason,
                'assigned_runtime': assigned_runtime,  # 'tidl_rt', 'tvm_rt', or 'arm'
                'node_name': layer_name,
                'node_type': layer_info.get('type', 'Unknown')
            }

        # Build name→arrayIndex map so SUBGRAPH_DATA.subgraphs[i].nodes contains
        # integer indices matching the arrayIndex used in transformData (template).
        # Must be built BEFORE the subgraph assignment loop below.
        layer_name_to_idx = {name: idx for idx, name in enumerate(transformed_layers.keys())}

        # Identify which layers belong to which subgraph based on TIDL data.
        # node_support is keyed by integer index (str), NOT by layer name, so we
        # look up the index via layer_name_to_idx before writing the subgraph field.
        tidl_subgraphs_raw = tidl_data_raw.get('subgraphs', {})
        for subgraph_id, subgraph_info in tidl_subgraphs_raw.items():
            for tidl_layer in subgraph_info.get('layers', []):
                onnx_mapping = tidl_layer.get('onnx_mapping', {})
                for onnx_name in onnx_mapping.get('onnx_node_names', []):
                    idx_key = str(layer_name_to_idx.get(onnx_name, -1))
                    if idx_key in node_support:
                        node_support[idx_key]['subgraph'] = str(subgraph_id)
                        node_support[idx_key]['supported'] = True
                        # A node present in onnx_mapping is compiled into a TIDL layer
                        # (e.g. TIDL_OdOutputReformatLayer fuses all OD post-processing).
                        # Override any graphvizInfo 'arm' assignment — it runs on C7x DSP.
                        # Also clear the misleading diagInfo message so the UI doesn't
                        # show "will be delegated in post-processing" for an accelerated node.
                        sg_runtime = subgraph_info.get('runtime', 'tidl_rt')
                        node_support[idx_key]['assigned_runtime'] = sg_runtime
                        node_support[idx_key]['diagInfo'] = ''

        # The virtual __input_*__ / __output_*__ sentinel nodes are appended to
        # transformed_layers AFTER node_support is built from onnx_layers, so they
        # have no entry in node_support.  JavaScript then defaults them to
        # supported=false → ARM, showing them as non-accelerated nodes.
        # Fix: register them as transparent IO boundary markers (not real ONNX ops).
        for layer_name, layer_info in transformed_layers.items():
            if layer_name.startswith('__input_') or layer_name.startswith('__output_'):
                idx = layer_name_to_idx.get(layer_name)
                if idx is not None:
                    node_support[str(idx)] = {
                        'supported': True,
                        'diagInfo': '',
                        'assigned_runtime': 'tidl_rt',
                        'node_name': layer_name,
                        'node_type': layer_info.get('type', 'Input')
                    }

        # Build subgraph_data
        subgraph_list = []
        for subgraph_id, subgraph_info in tidl_subgraphs_raw.items():
            onnx_node_indices_in_subgraph = []
            seen = set()
            for tidl_layer in subgraph_info.get('layers', []):
                for onnx_name in tidl_layer.get('onnx_mapping', {}).get('onnx_node_names', []):
                    if onnx_name not in seen:
                        seen.add(onnx_name)
                        idx = layer_name_to_idx.get(onnx_name)
                        if idx is not None:
                            onnx_node_indices_in_subgraph.append(idx)

            subgraph_list.append({
                'id': subgraph_id,
                'nodes': onnx_node_indices_in_subgraph
            })

        subgraph_data = {
            'subgraphs': subgraph_list,
            'node_support': node_support
        }

        # Calculate node positions for graph visualization
        print("\nGenerating graph visualization data...")
        layers_with_positions = calculate_node_depths_and_positions(
            nodes=transformed_layers.copy(),
            edges=edges,
            width=1200,
            height=800
        )
        model_data['layer_details'] = layers_with_positions

        # Build tree structure from ONNX layers
        tree_structure = build_hierarchical_tree(layers_with_positions, edges)

    # Extract TIDL subgraph data from unified schema
    print("\nProcessing TIDL subgraph data...")
    tidl_data = {}
    tidl_subgraphs_raw = tidl_data_raw.get('subgraphs', {})

    for subgraph_id, subgraph_info in tidl_subgraphs_raw.items():
            layers = []
            graph_nodes = []
            graph_edges = []

            for tidl_layer in subgraph_info.get('layers', []):
                layer_index = tidl_layer.get('layer_id', 0)
                layer_type = tidl_layer.get('layer_type', '')
                layer_name = tidl_layer.get('layer_name', '')
                onnx_mapping = tidl_layer.get('onnx_mapping', {})
                onnx_node_indices = onnx_mapping.get('onnx_node_indices', [])
                onnx_node_names = onnx_mapping.get('onnx_node_names', [])
                # Use first index for legacy single-value onnx_node_index
                onnx_node_index = onnx_node_indices[0] if onnx_node_indices else None

                layer_obj = {
                    'layer_index': layer_index,
                    'layer_type': layer_type,
                    'layer_name': layer_name,
                    'parameters': tidl_layer.get('parameters', {}),
                    'macs': 0,
                    'gmacs': tidl_layer.get('gmacs', 0.0),
                    'inputs': tidl_layer.get('inputs', []),
                    'outputs': tidl_layer.get('outputs', []),
                    'onnx_node_index': onnx_node_index,
                    'layer_data': {
                        'onnx_node_indices': onnx_node_indices,
                        'onnx_node_names': onnx_node_names,
                    }
                }
                layers.append(layer_obj)

                outputs = tidl_layer.get('outputs', [])
                output_shape = outputs[0].get('shape', []) if outputs else []
                input_shape = tidl_layer.get('inputs', [{}])[0].get('shape', []) if tidl_layer.get('inputs') else []
                parameters = tidl_layer.get('parameters', {})
                gmacs = tidl_layer.get('gmacs', 0.0)

                raw_text = generate_raw_text_from_parameters(
                    layer_index, layer_type, layer_name, parameters, gmacs
                )

                graph_node = {
                    'id': f'tidl_layer_{layer_index}',
                    'index': layer_index,
                    'name': layer_name,
                    'full_name': layer_name,
                    'type': layer_type,
                    'tidl_supported': True,
                    'inputshape': str(input_shape) if input_shape else 'N/A',
                    'outputshape': str(output_shape),
                    'layer_data': {
                        'raw_text': raw_text,
                        'layer_index': layer_index,
                        'layer_type': layer_type,
                        'layer_name': layer_name,
                        'parameters': parameters,
                        'macs': 0,
                        'gmacs': gmacs,
                        'onnx_node_indices': onnx_node_indices,
                        'onnx_node_names': onnx_node_names,
                    },
                    'onnx_node_index': onnx_node_index,
                    'onnx_name': onnx_node_names[0] if onnx_node_names else ''
                }
                graph_nodes.append(graph_node)

            # Build edges using node_id from inputs (direct node references)
            edges_created = set()
            for target_layer in layers:
                target_idx = target_layer['layer_index']
                for input_item in target_layer.get('inputs', []):
                    source_node_id = input_item.get('node_id')
                    if source_node_id is not None:
                        edge_key = (source_node_id, target_idx)
                        if edge_key not in edges_created:
                            graph_edges.append({
                                'source': f'tidl_layer_{source_node_id}',
                                'target': f'tidl_layer_{target_idx}'
                            })
                            edges_created.add(edge_key)

            # Collect ONNX nodes belonging to this subgraph (for ONNX view in Level 2)
            onnx_nodes_for_subgraph = []
            seen_onnx = set()
            for tidl_layer in subgraph_info.get('layers', []):
                for name in tidl_layer.get('onnx_mapping', {}).get('onnx_node_names', []):
                    if name not in seen_onnx:
                        seen_onnx.add(name)
                        layer_detail = onnx_layers.get(name, {})
                        onnx_nodes_for_subgraph.append({
                            'name': name,
                            'type': layer_detail.get('type', 'Unknown'),
                            'runtime_assignment': layer_detail.get('runtime_assignment', {}),
                        })

            tidl_data[str(subgraph_id)] = {
                'layers': layers,
                'total_gmacs': subgraph_info.get('total_gmacs', 0.0),
                'num_layers': len(layers),
                'tensor_bits': subgraph_info.get('tensor_bits', 8),
                'runtime': subgraph_info.get('runtime', 'tidl_rt'),
                'subgraph_inputs': subgraph_info.get('inputs', []),
                'subgraph_outputs': subgraph_info.get('outputs', []),
                'graph_nodes': graph_nodes,
                'graph_edges': graph_edges,
                'onnx_nodes': onnx_nodes_for_subgraph,
                # Real EVM execution time per frame (None on PC / when not measured)
                'evm_execution_time_ms_per_frame': subgraph_info.get(
                    'evm_execution_time_ms_per_frame', None
                ),
            }

    # Extract performance and metrics data from unified schema
    print("\nExtracting performance and metrics data...")
    if True:  # Always process as unified schema
        # UNIFIED SCHEMA v1.0: Extract from tidl.subgraphs
        # Activation data will be loaded from raw files when has_activation_data is true
        # Use provided activations_data if available, otherwise try to load from model_dir
        if activations_data:
            activation_data = activations_data
        else:
            # Derive model directory from JSON file path: inspector/modelinspector.json -> ../
            _jp = json_path or output_path
            json_dir = os.path.dirname(os.path.abspath(_jp))
            model_dir = os.path.dirname(json_dir)  # one level up from inspector/

            # Check if activation data is already embedded in JSON layers
            # Look for either: (1) histogram data, or (2) bin_files paths indicating extraction was done
            first_sg = next(iter(json_data['runtime'].get('subgraphs', {}).values()), {})
            has_embedded = any(
                _has_activation_data(l)
                for l in first_sg.get('layers', [])
            )

            if has_embedded:
                # Already in JSON — skip raw .bin loading, will be read from layer loop below
                activation_data = {}
                print(f"  Using activation data embedded in JSON")
            elif os.path.exists(model_dir):
                activation_data = load_activation_data_from_model_dir(model_dir, tidl_data)
            else:
                activation_data = {}

            # Always load metrics from analyze.xlsx regardless
            loaded_metrics = load_metrics_from_model_dir(model_dir) if os.path.exists(model_dir) else {}

        metrics_data = {}
        proctime_data = {}
        cycles_data = {}
        memory_data = {}

        tidl_subgraphs_raw = json_data['runtime'].get('subgraphs', {})
        for subgraph_id, subgraph_info in tidl_subgraphs_raw.items():
            sg_id = str(subgraph_id)
            # Numeric-only subgraph ID for activation key (template uses e.g. "0_3" not "tidl_0_3")
            sg_num = ''.join(c for c in str(subgraph_id) if c.isdigit()) or '0'
            metrics_list = []
            proctime_list = []
            cycles_list = []
            memory_list = []

            for tidl_layer in subgraph_info.get('layers', []):
                layer_id = tidl_layer.get('layer_id', 0)
                layer_type = tidl_layer.get('layer_type', '')
                layer_name = tidl_layer.get('layer_name', '')

                # Skip non-computational layers — they have no meaningful activation data
                if layer_type in ('TIDL_DataLayer', 'TIDL_DataConvertLayer'):
                    pass
                # Use activation data embedded in JSON layer (from data_extractor with --act_data)
                # If bin_files paths are present, activation data was extracted and embedded
                layer_activation_data = tidl_layer.get('activation_data')
                if layer_activation_data and layer_type not in ('TIDL_DataLayer', 'TIDL_DataConvertLayer'):
                    hist = layer_activation_data.get('histogram') or {}
                    has_real_data = bool(
                        hist.get('tidl_bins') or hist.get('tidl_counts') or
                        (isinstance(hist.get('tidl'), dict) and hist['tidl'].get('counts'))
                    )
                    # bin_files is now a sibling field of activation_data on the layer
                    layer_bin_files = tidl_layer.get('bin_files') or layer_activation_data.get('bin_files', {})
                    has_bin_files = bool(layer_bin_files.get('tidl') if isinstance(layer_bin_files, dict) else False)

                    if has_real_data or has_bin_files:
                        # Key: {numeric_subgraph}_{layer_id} — matches data_extractor embedding
                        activation_key = f"{sg_num}_{layer_id}"
                        activation_data[activation_key] = layer_activation_data
                        if has_bin_files:
                            tidl_bin = layer_bin_files.get('tidl', 'N/A') if isinstance(layer_bin_files, dict) else 'N/A'
                            notidl_bin = layer_bin_files.get('notidl', 'N/A') if isinstance(layer_bin_files, dict) else 'N/A'
                            print(f"  Layer {layer_id} ({layer_name}): using activation data from bin files")

                # Metrics/Accuracy data (only process if not null)
                metrics = tidl_layer.get('metrics')
                if metrics is not None:
                    metrics_entry = {
                        'subgraph': sg_id,
                        'tidl_layer_id': layer_id,
                        'onnx_layer': layer_name,
                        'mean_abs_diff': metrics.get('mae', 0),
                        'mean_abs_rel_diff': metrics.get('mae_relative', 0),
                        'median_abs_diff': metrics.get('median_abs_diff', 0),
                        'max_abs_diff': metrics.get('max_abs_diff', 0)
                    }
                    if 'snr_db' in metrics:
                        metrics_entry['snr_db'] = metrics['snr_db']
                    metrics_list.append(metrics_entry)

                # Performance data
                if tidl_layer.get('performance'):
                    perf = tidl_layer['performance']

                    # Only add to lists when values are actually present (not None)
                    if perf.get('proctime_us') is not None:
                        proctime_list.append({
                            'layer_num': layer_id,
                            'layer_type': layer_type,
                            'proctime': perf['proctime_us']
                        })

                    if perf.get('layer_cycles') is not None or perf.get('kernel_cycles') is not None:
                        cycles_list.append({
                            'layer_num': layer_id,
                            'layer_type': layer_type,
                            'kernelOnlyCycles': perf.get('kernel_cycles') or 0,
                            'coreLoopCycles':   perf.get('core_loop_cycles') or 0,
                            'layerCycles':      perf.get('layer_cycles') or 0,
                            'ioCycles':         perf.get('io_cycles') or 0,
                        })

                    mem = perf.get('memory') or {}
                    # Only add memory entry when at least one sub-field has a real value
                    if any(mem.get(k) is not None for k in ('l2_kb', 'msmc_kb', 'ddr_kb')):
                        memory_list.append({
                            'layer_num': layer_id,
                            'layer_type': layer_type,
                            'l2_usage':   mem.get('l2_kb') or 0,
                            'msmc_usage': mem.get('msmc_kb') or 0,
                            'ddr_usage':  mem.get('ddr_kb') or 0,
                            'total_usage': mem.get('total_kb') or 0,
                        })

            # Use loaded metrics from Excel if available, otherwise use metrics from JSON
            # loaded_metrics keys are numeric strings like '0'; sg_id is like 'tidl_0'
            sg_num_str = ''.join(filter(str.isdigit, sg_id)) or '0'
            metrics_source = loaded_metrics.get(sg_num_str) or loaded_metrics.get(sg_id)
            if metrics_source:
                # Filter out corrupted entries where metric values are extreme/infinite
                # (can happen for layers like GlobalAveragePool with bad activation data)
                import math
                _MAX_METRIC = 1e6
                clean = []
                for entry in metrics_source:
                    fields = ['max_abs_diff', 'mean_abs_diff', 'median_abs_diff',
                              'mean_abs_rel_diff', 'max_abs_diff_median', 'median_abs_diff_median']
                    corrupted = any(
                        not math.isfinite(float(entry.get(f, 0) or 0)) or
                        abs(float(entry.get(f, 0) or 0)) > _MAX_METRIC
                        for f in fields if entry.get(f) is not None
                    )
                    if not corrupted:
                        clean.append(entry)
                    else:
                        print(f"  Skipping corrupted metrics for layer {entry.get('tidl_layer_id')} "
                              f"({entry.get('onnx_layer')})")
                metrics_data[sg_id] = clean
            elif metrics_list:
                metrics_data[sg_id] = metrics_list

            # On EVM: proctime is computed from inflated debug cycles (unreliable)
            # and memory data is leftover from PC simulation — skip both.
            # EVM is detected by presence of infer_time_subgraph_ms in metadata.
            _is_evm_src = (
                json_data.get('metadata', {}).get('performance_source') == 'evm_hardware' or
                json_data.get('metadata', {}).get('infer_time_subgraph_ms') is not None or
                json_data.get('performance_source') == 'evm_hardware'
            )
            if proctime_list and not _is_evm_src:
                proctime_data[sg_id] = proctime_list
            if cycles_list:
                cycles_data[sg_id] = cycles_list
            if memory_list and not _is_evm_src:
                memory_data[sg_id] = memory_list

        # Config data from metadata and subgraphs
        metadata = json_data.get('metadata', {})
        tidl_subgraphs_raw = json_data['runtime'].get('subgraphs', {})
        # Get target_device and tensor_bits from first subgraph (same across all)
        first_sg = next(iter(tidl_subgraphs_raw.values()), {})
        config_data = {
            'target_device': first_sg.get('target_device', 'Unknown'),
            'task_type': metadata.get('task_type', 'Unknown'),
            'tensor_bits': first_sg.get('tensor_bits', 'Unknown'),
            'accuracy': 'N/A',  # Not in unified schema yet
            'num_frames': 'N/A',
            'num_subgraphs': len(tidl_subgraphs_raw),
            'perfsim_ddr_transfer_mb': 'N/A',
            'perfsim_gmacs': tidl_subgraphs_raw.get('0', {}).get('total_gmacs', 'N/A') if tidl_subgraphs_raw else 'N/A',
            'perfsim_time_ms': tidl_subgraphs_raw.get('0', {}).get('total_time_us', 0) / 1000.0 if tidl_subgraphs_raw else 'N/A'
        }

    # Build overview_data for Level 1 graph (collapsed subgraphs + ARM nodes)
    overview_nodes = []
    overview_edges = []
    subgraph_onnx_names = set()
    onnx_to_overview_id = {}  # Map ONNX node name to overview node id

    for sg_id, sg_info in tidl_data.items():
        # Collect ALL onnx node names for this subgraph:
        # (a) explicitly listed onnx_nodes, AND
        # (b) all names from each TIDL layer's onnx_node_names (includes fused layers
        #     like TIDL_OdOutputReformatLayer that absorb 100+ ONNX nodes but may not
        #     all appear in the explicit onnx_nodes list).
        sg_all_onnx_names = set(n['name'] for n in sg_info.get('onnx_nodes', []))
        for gn in sg_info.get('graph_nodes', []):
            for oname in gn.get('layer_data', {}).get('onnx_node_names', []):
                sg_all_onnx_names.add(oname)
        subgraph_onnx_names.update(sg_all_onnx_names)

        runtime_label = sg_info.get('runtime', 'tidl_rt')
        node_id = f'subgraph_{sg_id}'
        overview_nodes.append({
            'id': node_id,
            'type': 'subgraph',
            'subgraph_id': sg_id,
            'runtime': runtime_label,
            'label': f"{runtime_label} #{sg_id}",
            'total_gmacs': sg_info.get('total_gmacs', 0.0),
            'num_layers': sg_info.get('num_layers', 0),
            'num_onnx_nodes': len(sg_all_onnx_names),
            'tensor_bits': sg_info.get('tensor_bits', 8),
            'inputs': sg_info.get('subgraph_inputs', []),
            'outputs': sg_info.get('subgraph_outputs', []),
        })
        # Map ALL ONNX nodes in this subgraph to the subgraph overview node
        for oname in sg_all_onnx_names:
            onnx_to_overview_id[oname] = node_id

    # Add ARM/unsupported ONNX nodes not covered by any subgraph.
    # A node that appears in subgraph_onnx_names is compiled into a TIDL/TVM layer
    # (even if graphvizInfo.txt / JSON runtime_assignment says 'arm') so it must NOT
    # appear as an ARM overview node.  The subgraph_onnx_names check is the authoritative
    # source; ra.assigned_runtime from the JSON is a secondary hint only.
    if is_unified_schema:
        for layer_name, layer_info in onnx_layers.items():
            ra = layer_info.get('runtime_assignment', {})
            if layer_name not in subgraph_onnx_names and ra.get('assigned_runtime') not in ('tidl_rt', 'tvm_rt'):
                node_id = f'arm_{layer_name}'
                overview_nodes.append({
                    'id': node_id,
                    'type': 'arm',
                    'label': layer_name,
                    'op_type': layer_info.get('type', 'Unknown'),
                    'reason': ra.get('reason', ''),
                })
                onnx_to_overview_id[layer_name] = node_id

    # Build overview edges by following ONNX data flow
    # For each ONNX edge, map source/target ONNX nodes to overview nodes and create edge
    if is_unified_schema:
        seen_overview_edges = set()
        for edge in model_data.get('edges', []):
            src_onnx = edge.get('source_node_name')
            tgt_onnx = edge.get('target_node_name')
            src_overview_id = onnx_to_overview_id.get(src_onnx)
            tgt_overview_id = onnx_to_overview_id.get(tgt_onnx)
            # Only create overview edge if both nodes are in overview and they're different
            if src_overview_id and tgt_overview_id and src_overview_id != tgt_overview_id:
                edge_key = (src_overview_id, tgt_overview_id)
                if edge_key not in seen_overview_edges:
                    seen_overview_edges.add(edge_key)
                    overview_edges.append({
                        'source': src_overview_id,
                        'target': tgt_overview_id
                    })
    overview_data = {'nodes': overview_nodes, 'edges': overview_edges}

    # Convert data to JSON strings for template injection
    print("\nConverting data to JSON strings...")
    model_json = json.dumps(model_data, indent=2)
    subgraph_json = json.dumps(subgraph_data, indent=2)
    tidl_json = json.dumps(tidl_data, indent=2)
    overview_json = json.dumps(overview_data, indent=2)
    metrics_json = json.dumps(metrics_data, indent=2)
    config_json = json.dumps(config_data, indent=2)
    proctime_json = json.dumps(proctime_data, indent=2)
    cycles_json = json.dumps(cycles_data, indent=2)
    memory_json = json.dumps(memory_data, indent=2)
    tree_json = json.dumps(tree_structure, indent=2)

    # Performance source + per-field availability flags for conditional HTML rendering.
    # The template uses these to hide charts / toggle-buttons with no backing data.
    meta = json_data.get('metadata', {})
    # EVM detected by performance_source in metadata (set when /tmp/ CSV was loaded).
    # Also supports: flat infer_time_subgraph_ms (timing present),
    # and legacy root-level performance_source (old JSONs — backward compat).
    is_evm_data = (
        meta.get('performance_source') == 'evm_hardware' or
        meta.get('infer_time_subgraph_ms') is not None or
        json_data.get('performance_source') == 'evm_hardware'
    )
    performance_source = 'evm_hardware' if is_evm_data else 'pc_simulation'

    # Accuracy: flat keys in metadata (accuracy_top1%, accuracy_ap[.5:.95]%, etc.)
    # Also support old nested evm_accuracy key for backward compat
    evm_accuracy = {k: v for k, v in meta.items()
                    if k.lower().startswith('accuracy') and isinstance(v, (int, float))}
    if not evm_accuracy:
        evm_accuracy = meta.get('evm_accuracy', {})

    # Scan cycles_data to find which cycle sub-fields are actually non-zero.
    _has_kernel = _has_core = _has_io = False
    for sg_layers in cycles_data.values():
        for row in sg_layers:
            if row.get('kernelOnlyCycles', 0): _has_kernel = True
            if row.get('coreLoopCycles', 0):   _has_core   = True
            if row.get('ioCycles', 0):          _has_io     = True

    # Scan memory_data to find which memory sub-fields are actually non-zero.
    _has_l2 = _has_msmc = _has_ddr = False
    for sg_layers in memory_data.values():
        for row in sg_layers:
            if row.get('l2_usage', 0):   _has_l2   = True
            if row.get('msmc_usage', 0): _has_msmc = True
            if row.get('ddr_usage', 0):  _has_ddr  = True

    is_evm = (performance_source == 'evm_hardware')
    # On EVM the /tmp/ CSV has no memory data and Layer Cycles at debug_level>1
    # are inflated — proctime_us and IO (dmaPipeupCycles) are unreliable.
    # Force those flags off so the HTML never renders misleading charts.
    perf_availability = {
        'source':        performance_source,
        'has_proctime':  False if is_evm else bool(proctime_data),
        'has_cycles':    bool(cycles_data),
        'has_kernel':    _has_kernel,
        'has_core':      _has_core,
        'has_io':        False if is_evm else _has_io,
        'has_memory':    False if is_evm else bool(memory_data),
        'has_l2':        False if is_evm else _has_l2,
        'has_msmc':      False if is_evm else _has_msmc,
        'has_ddr':       False if is_evm else _has_ddr,
        'evm_accuracy':  evm_accuracy,
        # Timing: flat keys in metadata (new format) OR nested evm_timing (old format)
        'evm_timing':    ({k: meta[k] for k in
                           ('infer_time_subgraph_ms', 'infer_time_core_ms', 'infer_time_invoke_ms')
                           if k in meta}
                          or meta.get('evm_timing', {})),
    }
    perf_availability_json = json.dumps(perf_availability, indent=2)

    print(f"  model_json: {len(model_json) / 1024:.2f} KB")
    print(f"  subgraph_json: {len(subgraph_json) / 1024:.2f} KB")
    print(f"  tidl_json: {len(tidl_json) / 1024:.2f} KB")
    print(f"  metrics_json: {len(metrics_json) / 1024:.2f} KB")
    print(f"  config_json: {len(config_json) / 1024:.2f} KB")
    print(f"  proctime_json: {len(proctime_json) / 1024:.2f} KB")
    print(f"  cycles_json: {len(cycles_json) / 1024:.2f} KB")
    print(f"  memory_json: {len(memory_json) / 1024:.2f} KB")
    print(f"  tree_json: {len(tree_json) / 1024:.2f} KB")
    print(f"  performance_source: {performance_source}")

    # Use per-layer compression for lazy loading (on-demand decompression)
    if activation_data:
        activation_compressed_per_layer = compress_activation_data_per_layer(activation_data)
        activation_json = json.dumps(activation_compressed_per_layer)
        print(f"  Activation data compressed: {len(activation_json) / 1024:.2f} KB")
    else:
        activation_json = json.dumps({})
        print(f"  No activation data provided (was --act_data=false used?)")

    print("\nReplacing template placeholders...")
    compiled_html = template.replace('{{MODEL_DATA}}', model_json)
    compiled_html = compiled_html.replace('{{SUBGRAPH_DATA}}', subgraph_json)
    compiled_html = compiled_html.replace('{{TIDL_LAYER_DATA}}', tidl_json)
    compiled_html = compiled_html.replace('{{ACTIVATION_DATA}}', activation_json)
    compiled_html = compiled_html.replace('{{METRICS_DATA}}', metrics_json)
    compiled_html = compiled_html.replace('{{CONFIG_DATA}}', config_json)
    compiled_html = compiled_html.replace('{{PROCTIME_DATA}}', proctime_json)
    compiled_html = compiled_html.replace('{{CYCLES_DATA}}', cycles_json)
    compiled_html = compiled_html.replace('{{MEMORY_DATA}}', memory_json)
    compiled_html = compiled_html.replace('{{TREE_DATA}}', tree_json)
    compiled_html = compiled_html.replace('{{OVERVIEW_DATA}}', overview_json)
    compiled_html = compiled_html.replace('{{PERF_AVAILABILITY}}', perf_availability_json)

    print(f"\nWriting compiled HTML: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(compiled_html)

    file_size = os.path.getsize(output_path)
    file_size_mb = file_size / (1024 * 1024)

    print(f"Compiled HTML generated successfully")
    print(f"  File size: {file_size_mb:.2f} MB")

    return json_data


def main(json_path, template_path, output_path, activations_json_path=None):

    if not os.path.exists(json_path):
        print(f"ERROR: JSON file not found: {json_path}")
        sys.exit(1)

    if not os.path.exists(template_path):
        print(f"ERROR: Template file not found: {template_path}")
        sys.exit(1)

    print("=" * 70)
    print("HTML Generator - Generating Visualization")
    print("=" * 70)
    print(f"JSON Data:         {json_path}")
    print(f"Activations Data:  {activations_json_path if activations_json_path else 'None (will show message in HTML)'}")
    print(f"Template:          {template_path}")
    print(f"Output HTML:       {output_path}")
    print("=" * 70)

    try:
        json_data = load_json_data(json_path)

        # Load activation data if provided
        activations_data = None
        if activations_json_path and os.path.exists(activations_json_path):
            print(f"\nLoading activation data from: {activations_json_path}")
            activations_data = load_json_data(activations_json_path)
        elif activations_json_path:
            print(f"\nWARNING: Activation data file not found: {activations_json_path}")
        else:
            print(f"\nNo activation data file specified (HTML will show instructions)")

        enriched_json_data = generate_html(json_data, template_path, output_path, activations_data, json_path=json_path)

        # Save the enriched JSON back to inspector.json with all embedded activation data + metadata
        print(f"\nUpdating JSON with activation data and metadata...")
        updated_json_path = json_path
        with open(updated_json_path, 'w', encoding='utf-8') as f:
            json.dump(enriched_json_data, f, indent=2)
        json_size_mb = os.path.getsize(updated_json_path) / (1024 * 1024)
        print(f"  Updated: {updated_json_path} ({json_size_mb:.2f} MB)")

        print("\n" + "=" * 70)
        print("SUCCESS! HTML visualization generated.")
        print("=" * 70)
        print(f"\nOpen this file in your browser:")
        print(f"  {os.path.abspath(output_path)}")
        print("\nFeatures:")
        print("  - Model Performance - View model overview and statistics")
        print("  - Runtime Model - View complete ONNX graph")
        print("  - TIDL Model - View subgraphs with support status")
        print("  - TIDL Layer Details - Enhanced with GMACS, parameters")
        print("  - Activation Analysis - Histogram & Scatter plots")
        print("  - Metrics Analysis - Layer-wise accuracy metrics")
        print("  - Performance Charts - Processing time, cycles, memory")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    """Main function to generate HTML from JSON data"""
    import argparse

    parser = argparse.ArgumentParser(description='Generate HTML visualization from JSON data')
    parser.add_argument('data_json', help='Input JSON file (main data, supports .json or .json.gz)')
    parser.add_argument('template_html', help='HTML template file')
    parser.add_argument('output_html', help='Output HTML file path')
    parser.add_argument('--activations', dest='activations_json', default=None,
                        help='Optional: Activations data JSON file (separate file)')

    args = parser.parse_args()

    # Legacy support: if 4 positional args, treat 2nd as activations
    if len(sys.argv) == 5 and not sys.argv[1].startswith('-'):
        # Old format: data.json activations.json template.html output.html
        print("Detected legacy 4-argument format")
        main(json_path=sys.argv[1], activations_json_path=sys.argv[2],
             template_path=sys.argv[3], output_path=sys.argv[4])
    else:
        main(json_path=args.data_json, template_path=args.template_html,
             output_path=args.output_html, activations_json_path=args.activations_json)