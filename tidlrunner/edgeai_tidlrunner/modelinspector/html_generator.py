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
from typing import Dict, Any


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

        expected_keys = ['metadata', 'model', 'compilation']
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


def generate_html(json_data: Dict[str, Any], template_path: str, output_path: str, activations_data: Dict[str, Any] = None):
    """Generate HTML by fusing JSON data into template

    Args:
        json_data: Dict containing all extracted data
        template_path: Path to HTML template file
        output_path: Path to output HTML file
        activations_data: Optional dict containing activation data (if None, will be empty)
    """
    print(f"\nReading template: {template_path}")

    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read()

    # Use provided activation data or default to empty dict
    if activations_data is None:
        activations_data = {}

    # Detect format: new (3-object) vs old (4-object with metadata/compilation)
    if 'subgraphs' in json_data and isinstance(json_data['subgraphs'], list):
        # NEW FORMAT: model, subgraphs[], performance
        print("  Detected: NEW 3-object structure")
        is_new_format = True

        model_obj = json_data.get('model', {})
        subgraphs = json_data.get('subgraphs', [])
        performance = json_data.get('performance', {})

        # Transform to template format
        model_data = {
            'model_details': {
                'name': model_obj.get('name', ''),
                'weights': model_obj.get('stats', {}).get('total_params', 0),
                'no_of_layers': model_obj.get('stats', {}).get('total_layers', 0),
                'input_shape': model_obj.get('inputs', []),
                'output_shape': model_obj.get('outputs', [])
            },
            'layer_details': model_obj.get('layers', {}),
            'edges': [
                {
                    'source_node_id': i,
                    'target_node_id': i+1,
                    'source_node_name': edge.get('from', ''),
                    'target_node_name': edge.get('to', ''),
                    'connection_info': {
                        'tensor': edge.get('tensor', ''),
                        'shape': edge.get('shape', [])
                    }
                }
                for i, edge in enumerate(model_obj.get('graph', {}).get('edges', []))
            ]
        }

        # Build subgraph_data for template
        node_support = {}
        for sg in subgraphs:
            for node_name, support_info in sg.get('support', {}).items():
                node_support[node_name] = {
                    'supported': support_info.get('accelerated', False),
                    'subgraph': sg['id'],
                    'reason': support_info.get('reason', '')
                }

        subgraph_data = {
            'subgraphs': [{'id': sg['id'], 'nodes': sg.get('onnx_nodes', [])} for sg in subgraphs],
            'node_support': node_support
        }

        tree_structure = model_obj.get('hierarchy', {})

    elif 'model_data' in json_data:
        # LEGACY FORMAT (very old)
        print("  Detected: LEGACY format")
        is_new_format = False
        model_data = json_data['model_data']
        subgraph_data = json_data.get('subgraph_data', {})
        tree_structure = model_data.get('tree_structure', {})
    else:
        # OLD FORMAT: metadata, model, compilation
        print("  Detected: OLD 4-object structure")
        is_new_format = False
        model_data = {
            'model_details': json_data.get('model', {}).get('details', {}),
            'layer_details': json_data.get('model', {}).get('layers', {}),
            'edges': json_data.get('model', {}).get('graph', {}).get('edges', [])
        }
        subgraph_data = {
            'subgraphs': json_data.get('compilation', {}).get('subgraphs', []),
            'node_support': json_data.get('compilation', {}).get('node_support', {})
        }
        tree_structure = json_data.get('model', {}).get('tree_structure', {})

    # Extract TIDL subgraph data
    if is_new_format:
        # NEW FORMAT: Convert from subgraphs[] to tidl_data dict
        tidl_data = {}
        for sg in subgraphs:
            sg_id = str(sg['id'])
            layers = []
            for layer in sg.get('layers', []):
                layer_obj = {
                    'layer_index': layer.get('id', 0),
                    'layer_type': layer.get('type', ''),
                    'layer_name': layer.get('name', ''),
                    'parameters': layer.get('params', {}),
                    'macs': layer.get('ops', {}).get('macs', 0),
                    'gmacs': layer.get('ops', {}).get('gmacs', 0.0)
                }
                if 'onnx_index' in layer:
                    layer_obj['onnx_node_index'] = layer['onnx_index']
                layers.append(layer_obj)

            tidl_data[sg_id] = {
                'layers': layers,
                'total_gmacs': sg.get('summary', {}).get('total_gmacs', 0.0),
                'graph_nodes': sg.get('graph', {}).get('nodes', []),
                'graph_edges': sg.get('graph', {}).get('edges', [])
            }

    elif 'tidl_data' in json_data:
        tidl_data = json_data['tidl_data']
        tidl_subgraphs_new = tidl_data
    else:
        tidl_subgraphs_new = json_data.get('compilation', {}).get('tidl_subgraphs', {})
        tidl_data = {}

        # Transform old structure to template format if needed
        for subgraph_id, subgraph_info in tidl_subgraphs_new.items():
            layers = []
            for layer in subgraph_info.get('layers', []):
                old_layer = {
                    'layer_index': layer.get('index'),
                    'layer_type': layer.get('type'),
                    'layer_name': layer.get('name'),
                    'parameters': layer.get('parameters', {}),
                    'macs': layer.get('macs'),
                    'gmacs': layer.get('gmacs')
                }
                if 'onnx_node_index' in layer:
                    old_layer['onnx_node_index'] = layer['onnx_node_index']
                layers.append(old_layer)

            tidl_data[subgraph_id] = {
                'layers': layers,
                'total_gmacs': subgraph_info.get('total_gmacs', 0.0),
                'graph_nodes': subgraph_info.get('graph', {}).get('nodes', []),
                'graph_edges': subgraph_info.get('graph', {}).get('edges', [])
            }

    # Extract performance and analysis data
    if is_new_format:
        # NEW FORMAT: Extract from subgraphs[] array
        # Use provided activations_data if available, otherwise extract from subgraphs
        if activations_data:
            activation_data = activations_data
        else:
            activation_data = {}
        metrics_data = {}
        proctime_data = {}
        cycles_data = {}
        memory_data = {}

        for sg in subgraphs:
            sg_id = str(sg['id'])
            metrics_list = []
            proctime_list = []
            cycles_list = []
            memory_list = []

            for layer in sg.get('layers', []):
                layer_id = layer.get('index', layer.get('id', 0))  # Try 'index' first, fall back to 'id'

                # Activation data
                if 'activation' in layer:
                    activation_key = f"{sg_id}_{layer_id}"
                    activation_data[activation_key] = layer['activation']

                # Metrics/Accuracy data
                if 'accuracy' in layer:
                    acc = layer['accuracy']
                    metrics_entry = {
                        'subgraph': sg_id,
                        'tidl_layer_id': layer_id,
                        'onnx_layer': layer.get('name', ''),
                        'mean_abs_diff': acc.get('mae', 0),
                        'mean_abs_rel_diff': acc.get('mae_relative', 0),
                        'median_abs_diff': acc.get('median_error', 0),
                        'max_abs_diff': acc.get('max_error', 0)
                    }
                    if 'snr_db' in acc:
                        metrics_entry['snr_db'] = acc['snr_db']
                    metrics_list.append(metrics_entry)

                # Performance data
                if 'perf' in layer:
                    perf = layer['perf']
                    layer_type = layer.get('type', '')

                    if 'time_us' in perf:
                        proctime_list.append({
                            'layer_num': layer_id,
                            'layer_type': layer_type,
                            'proctime': perf['time_us']
                        })

                    if 'cycles' in perf:
                        cycles_list.append({
                            'layer_num': layer_id,
                            'layer_type': layer_type,
                            'kernelOnlyCycles': perf['cycles'].get('kernel', 0),
                            'coreLoopCycles': perf['cycles'].get('core_loop', 0),
                            'layerCycles': perf['cycles'].get('layer', 0),
                            'ioCycles': perf['cycles'].get('io', 0)
                        })

                    if 'memory_kb' in perf:
                        mem = perf['memory_kb']
                        memory_list.append({
                            'layer_num': layer_id,
                            'layer_type': layer_type,
                            'l2_usage': mem.get('l2', 0),
                            'msmc_usage': mem.get('msmc', 0),
                            'ddr_usage': mem.get('ddr', 0),
                            'total_usage': mem.get('total', 0)
                        })

            if metrics_list:
                metrics_data[sg_id] = metrics_list
            if proctime_list:
                proctime_data[sg_id] = proctime_list
            if cycles_list:
                cycles_data[sg_id] = cycles_list
            if memory_list:
                memory_data[sg_id] = memory_list

        # Config data from performance object
        perf_config = performance.get('config', {})
        perf_summary = performance.get('summary', {})
        perf_accuracy = performance.get('accuracy', {})

        accuracy_str = f"{perf_accuracy.get('value', 0)}{perf_accuracy.get('unit', '')}"
        if not perf_accuracy.get('value'):
            accuracy_str = 'N/A'

        config_data = {
            'target_device': perf_config.get('device', 'Unknown'),
            'task_type': perf_config.get('task', 'Unknown'),
            'tensor_bits': perf_config.get('precision', 'Unknown'),
            'accuracy': accuracy_str,
            'num_frames': perf_summary.get('num_frames', 'N/A'),
            'num_subgraphs': perf_summary.get('num_subgraphs', 'N/A'),
            'perfsim_ddr_transfer_mb': perf_summary.get('ddr_transfer_mb', 'N/A'),
            'perfsim_gmacs': perf_summary.get('total_gmacs', 'N/A'),
            'perfsim_time_ms': perf_summary.get('total_time_ms', 'N/A')
        }

    elif 'activation_data' in json_data and 'metrics_data' in json_data:
        # Old format with separate top-level keys
        # Use provided activations_data if available, otherwise use from json_data
        if activations_data:
            activation_data = activations_data
        else:
            activation_data = json_data.get('activation_data', {})
        metrics_data = json_data.get('metrics_data', {})
        config_data = json_data.get('config_data', {})
        proctime_data = json_data.get('proctime_data', {})
        cycles_data = json_data.get('cycles_data', {})
        memory_data = json_data.get('memory_data', {})
    else:
        # New format: activation_data at top level (if exists), metrics/perf from tidl_subgraphs
        # Use provided activations_data if available, otherwise check json_data
        if activations_data:
            activation_data = activations_data
        else:
            activation_data = json_data.get('activation_data', {})

        # If activation_data not at top level, extract from layers (legacy)
        if not activation_data and not activations_data:
            tidl_subgraphs_new = json_data.get('compilation', {}).get('tidl_subgraphs', {})
            for subgraph_id, subgraph_info in tidl_subgraphs_new.items():
                for layer in subgraph_info.get('layers', []):
                    if 'activation' in layer:
                        activation_key = f"{subgraph_id}_{layer.get('index')}"
                        activation_data[activation_key] = layer['activation']

        tidl_subgraphs_new = json_data.get('compilation', {}).get('tidl_subgraphs', {})

        metrics_data = {}
        for subgraph_id, subgraph_info in tidl_subgraphs_new.items():
            metrics_list = []
            for layer in subgraph_info.get('layers', []):
                if 'metrics' in layer:
                    metrics_entry = {
                        'subgraph': subgraph_id,
                        'tidl_layer_id': layer.get('index'),
                        'onnx_layer': layer.get('name', ''),
                        'mean_abs_diff': layer['metrics'].get('mae', 0),
                        'mean_abs_rel_diff': layer['metrics'].get('mean_abs_rel_diff', 0),
                        'median_abs_diff': layer['metrics'].get('median_abs_diff', 0),
                        'max_abs_diff': layer['metrics'].get('max_abs_diff', 0)
                    }
                    if 'snr_db' in layer['metrics']:
                        metrics_entry['snr_db'] = layer['metrics']['snr_db']
                    metrics_list.append(metrics_entry)
            if metrics_list:
                metrics_data[str(subgraph_id)] = metrics_list

        metadata = json_data.get('metadata', {})
        config_data = {
            'target_device': metadata.get('target_device', 'Unknown'),
            'task_type': metadata.get('task_type', 'Unknown'),
            'tensor_bits': metadata.get('tensor_bits', 'Unknown'),
            'accuracy': metadata.get('model_accuracy', 'N/A'),
            'num_frames': metadata.get('num_frames', 'N/A'),
            'num_subgraphs': metadata.get('num_subgraphs', 'N/A'),
            'perfsim_ddr_transfer_mb': metadata.get('perfsim_ddr_transfer_mb', 'N/A'),
            'perfsim_gmacs': metadata.get('perfsim_gmacs', 'N/A'),
            'perfsim_time_ms': metadata.get('perfsim_time_ms', 'N/A')
        }

        proctime_data = {}
        cycles_data = {}
        memory_data = {}

        for subgraph_id, subgraph_info in tidl_subgraphs_new.items():
            proctime_list = []
            cycles_list = []
            memory_list = []

            for layer in subgraph_info.get('layers', []):
                if 'performance' in layer:
                    perf = layer['performance']
                    layer_num = perf.get('layer_num', layer.get('index'))
                    layer_type = perf.get('layer_type', layer.get('type'))

                    if 'proctime_us' in perf:
                        proctime_list.append({
                            'layer_num': layer_num,
                            'layer_type': layer_type,
                            'proctime': perf['proctime_us']
                        })

                    if 'kernel_cycles' in perf or 'core_loop_cycles' in perf:
                        cycles_list.append({
                            'layer_num': layer_num,
                            'layer_type': layer_type,
                            'kernelOnlyCycles': perf.get('kernel_cycles', 0),
                            'coreLoopCycles': perf.get('core_loop_cycles', 0),
                            'layerCycles': perf.get('layer_cycles', 0),
                            'ioCycles': perf.get('io_cycles', 0)
                        })

                    if 'memory' in perf:
                        mem = perf['memory']
                        memory_list.append({
                            'layer_num': layer_num,
                            'layer_type': layer_type,
                            'l2_usage': mem.get('l2_kb', 0),
                            'msmc_usage': mem.get('msmc_kb', 0),
                            'ddr_usage': mem.get('ddr_kb', 0),
                            'total_usage': mem.get('total_kb', 0)
                        })

            if proctime_list:
                proctime_data[subgraph_id] = proctime_list
            if cycles_list:
                cycles_data[subgraph_id] = cycles_list
            if memory_list:
                memory_data[subgraph_id] = memory_list

    print("\nConverting data to JSON strings...")
    model_json = json.dumps(model_data, indent=2)
    subgraph_json = json.dumps(subgraph_data, indent=2)
    tidl_json = json.dumps(tidl_data, indent=2)
    metrics_json = json.dumps(metrics_data, indent=2)
    config_json = json.dumps(config_data, indent=2)
    proctime_json = json.dumps(proctime_data, indent=2)
    cycles_json = json.dumps(cycles_data, indent=2)
    memory_json = json.dumps(memory_data, indent=2)
    tree_json = json.dumps(tree_structure, indent=2)

    print(f"  model_json: {len(model_json) / 1024:.2f} KB")
    print(f"  subgraph_json: {len(subgraph_json) / 1024:.2f} KB")
    print(f"  tidl_json: {len(tidl_json) / 1024:.2f} KB")
    print(f"  metrics_json: {len(metrics_json) / 1024:.2f} KB")
    print(f"  config_json: {len(config_json) / 1024:.2f} KB")
    print(f"  proctime_json: {len(proctime_json) / 1024:.2f} KB")
    print(f"  cycles_json: {len(cycles_json) / 1024:.2f} KB")
    print(f"  memory_json: {len(memory_json) / 1024:.2f} KB")
    print(f"  tree_json: {len(tree_json) / 1024:.2f} KB")

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

    print(f"\nWriting compiled HTML: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(compiled_html)

    file_size = os.path.getsize(output_path)
    file_size_mb = file_size / (1024 * 1024)

    print(f"Compiled HTML generated successfully")
    print(f"  File size: {file_size_mb:.2f} MB")


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

        generate_html(json_data, template_path, output_path, activations_data)

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