#!/usr/bin/env python3
# Copyright (c) 2018-2025, Texas Instruments
# All Rights Reserved.

"""
HTML Generator - Fuse JSON data into HTML template

This script loads extracted JSON data and fuses it into an HTML template
to create a self-contained, interactive visualization.

Usage:
    python html_generator.py <data.json> <template.html> <output.html>

Example:
    python html_generator.py model_data.json.gz template.html output.html

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

        expected_keys = ['metadata', 'model', 'compilation', 'performance']
        missing_keys = [k for k in expected_keys if k not in data]
        if missing_keys:
            print(f"WARNING: JSON missing keys: {missing_keys}")

        print(f"JSON data loaded successfully")
        return data

    except Exception as e:
        print(f"ERROR: Failed to load JSON: {e}")
        raise


def compress_activation_data(activation_data: Dict[str, Any]) -> str:
    """Compress activation data using gzip+base64 encoding

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


def generate_html(json_data: Dict[str, Any], template_path: str, output_path: str):
    """Generate HTML by fusing JSON data into template

    Args:
        json_data: Dict containing all extracted data
        template_path: Path to HTML template file
        output_path: Path to output HTML file
    """
    print(f"\nReading template: {template_path}")

    with open(template_path, 'r', encoding='utf-8') as f:
        template = f.read()

    # Handle both new and old data structure formats
    if 'model_data' in json_data:
        model_data = json_data['model_data']
        subgraph_data = json_data.get('subgraph_data', {})
        tree_structure = model_data.get('tree_structure', {})
    else:
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
    if 'tidl_data' in json_data:
        tidl_data = json_data['tidl_data']
        tidl_subgraphs_new = tidl_data
    else:
        tidl_subgraphs_new = json_data.get('compilation', {}).get('tidl_subgraphs', {})
        tidl_data = {}

    # Transform old structure to template format if needed
    if not ('tidl_data' in json_data):
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
    if 'activation_data' in json_data:
        activation_data = json_data.get('activation_data', {})
        metrics_data = json_data.get('metrics_data', {})
        config_data = json_data.get('config_data', {})
        proctime_data = json_data.get('proctime_data', {})
        cycles_data = json_data.get('cycles_data', {})
        memory_data = json_data.get('memory_data', {})
    else:
        activation_data = {}
        for subgraph_id, subgraph_info in tidl_subgraphs_new.items():
            for layer in subgraph_info.get('layers', []):
                if 'activation' in layer:
                    activation_key = f"{subgraph_id}_{layer.get('index')}"
                    activation_data[activation_key] = layer['activation']

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
                    metrics_list.append(metrics_entry)
            if metrics_list:
                metrics_data[str(subgraph_id)] = metrics_list

        metadata = json_data.get('metadata', {})
        performance = json_data.get('performance', {})
        config_data = {
            'target_device': metadata.get('target_device', 'Unknown'),
            'task_type': metadata.get('task_type', 'Unknown'),
            'tensor_bits': metadata.get('tensor_bits', 'Unknown'),
            'accuracy': metadata.get('model_accuracy', 'N/A'),
            'num_frames': performance.get('num_frames', 'N/A'),
            'num_subgraphs': performance.get('num_subgraphs', 'N/A'),
            'perfsim_ddr_transfer_mb': performance.get('ddr_transfer_mb', 'N/A'),
            'perfsim_gmacs': performance.get('total_gmacs', 'N/A'),
            'perfsim_time_ms': performance.get('total_time_ms', 'N/A')
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
                            'coreLoopCycles': perf.get('core_loop_cycles', 0)
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

    activation_b64 = compress_activation_data(activation_data)
    activation_json = json.dumps(activation_b64)

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


def main(json_path, template_path, output_path):

    if not os.path.exists(json_path):
        print(f"ERROR: JSON file not found: {json_path}")
        sys.exit(1)

    if not os.path.exists(template_path):
        print(f"ERROR: Template file not found: {template_path}")
        sys.exit(1)

    print("=" * 70)
    print("HTML Generator - Generating Visualization")
    print("=" * 70)
    print(f"JSON Data:    {json_path}")
    print(f"Template:     {template_path}")
    print(f"Output HTML:  {output_path}")
    print("=" * 70)

    try:
        json_data = load_json_data(json_path)
        generate_html(json_data, template_path, output_path)

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
    if len(sys.argv) < 4:
        print("=" * 70)
        print("HTML Generator V6 - Fuse JSON Data into HTML Template")
        print("=" * 70)
        print("\nUsage: python html_generator_v6.py <data.json> <template.html> <output.html>")
        print("\nArguments:")
        print("  data.json      - Input JSON file (supports .json or .json.gz)")
        print("  template.html  - HTML template file")
        print("  output.html    - Output HTML file path")
        print("\nExample:")
        print("  python html_generator_v6.py model_data.json.gz template_v6.html output.html")
        print("\nThe script will:")
        print("  1. Load JSON data (compressed or uncompressed)")
        print("  2. Compress activation data using gzip+base64")
        print("  3. Fuse all data into HTML template")
        print("  4. Generate self-contained HTML visualization")
        print("=" * 70)
        sys.exit(1)
        
    main(json_path = sys.argv[1], template_path = sys.argv[2], output_path = sys.argv[3])