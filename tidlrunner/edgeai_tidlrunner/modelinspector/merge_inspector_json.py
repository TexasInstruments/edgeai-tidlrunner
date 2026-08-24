#!/usr/bin/env python3
# Copyright (c) 2018-2025, Texas Instruments
# All Rights Reserved.
"""
merge_inspector_json.py
-----------------------
Merge a TIDL-compiled modelinspector.json with a TVM-compiled
modelinspector.json into a single combined JSON.

Runtime priority ladder:  TIDL → TVM → ARM
  • TIDL JSON:  nodes are  tidl_rt | arm
              (arm = what TIDL couldn't handle)
  • TVM  JSON:  nodes are  tvm_rt  | arm
              (tvm_rt = ARM nodes that TVM CAN handle; arm = what TVM couldn't handle)
  • Merged:     nodes are  tidl_rt | tvm_rt | arm
              (tidl_rt = never demoted; tvm_rt = promoted from arm; arm = truly unaccelerated)

Usage
-----
  python merge_inspector_json.py \\
      --tidl  path/to/tidl_modelinspector.json \\
      --tvm   path/to/tvm_modelinspector.json  \\
      --out   path/to/merged_modelinspector.json

The merged JSON is written to --out.
Then manually: python html_generator.py <out> template.html <out>.html

Example
-------
  python merge_inspector_json.py \\
    --tidl work_dirs/.../tidl/.../modelinspector.json \\
    --tvm work_dirs/.../tvm/.../modelinspector.json \\
    --out merged_modelinspector.json

  python html_generator.py merged_modelinspector.json template.html merged_modelinspector.html
"""

import argparse
import copy
import json
import os
import re
from collections import Counter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(path: str) -> dict:
    """Load JSON file."""
    with open(path, 'r', encoding='utf-8') as fh:
        return json.load(fh)


def _save(data: dict, path: str) -> None:
    """Save JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(data, fh, indent=2)


def _runtime_dist(onnx_layers: dict) -> dict:
    """Count how many nodes per runtime."""
    return dict(Counter(
        v.get('runtime_assignment', {}).get('assigned_runtime', 'unknown')
        for v in onnx_layers.values()
    ))


# ---------------------------------------------------------------------------
# Core merge
# ---------------------------------------------------------------------------

def merge(tidl_json: dict, tvm_json: dict) -> dict:
    """
    Merge TIDL and TVM inspector JSONs respecting the runtime ladder.

    Algorithm
    ---------
    1. Start with TIDL JSON as the base (authoritative source for model structure)
       - If TVM JSON has metadata/model that TIDL lacks, copy those too
    2. For each ONNX node in the TVM JSON:
       - If it's marked tvm_rt in TVM, AND
       - It's marked arm in TIDL (not accelerated by TIDL), THEN
       - Promote it to tvm_rt in the merged JSON (ARM → TVM)
    3. Add all TVM subgraphs to runtime.subgraphs
    4. Merge performance/accuracy metadata from TVM if not in TIDL
    5. Result: tidl_rt > tvm_rt > arm (priority ladder applied)
    """
    merged = copy.deepcopy(tidl_json)

    # ── Ensure merged has model.onnx structure (from TIDL, fallback to TVM) ───
    if 'model' not in merged or 'onnx' not in merged.get('model', {}):
        if 'model' in tvm_json and 'onnx' in tvm_json['model']:
            merged['model'] = copy.deepcopy(tvm_json['model'])
            print(f'  [Merged model] Copied model.onnx from TVM JSON (TIDL had none)')

    # ── Step 1: Identify which nodes can run on TVM (but not on TIDL) ────────
    tvm_onnx_layers = tvm_json.get('model', {}).get('onnx', {}).get('layers', {})
    tvm_capable_nodes = {
        name: info
        for name, info in tvm_onnx_layers.items()
        if info.get('runtime_assignment', {}).get('assigned_runtime') == 'tvm_rt'
    }
    print(f'  [TVM JSON] Found {len(tvm_capable_nodes)} nodes with tvm_rt assignment')

    # ── Step 1b: Auto-detect TVM nodes from subgraph names in runtime ───────
    #            If TVM JSON has no model.onnx.layers, infer from subgraph names.
    #            Also record sg_name -> onnx_node_name so Step 3 can pull the
    #            AUTHORITATIVE tensor name/shape from the TIDL JSON's ONNX
    #            layer info, instead of trusting whatever placeholder name
    #            the TVM JSON itself declares (e.g. a hand-written TVM JSON
    #            may use the producing subgraph's id, like "tidl_1", as a
    #            stand-in for the real ONNX tensor name, like "188").
    sg_name_to_onnx_node = {}
    if not tvm_capable_nodes:
        tvm_subgraphs = tvm_json.get('runtime', {}).get('subgraphs', {})
        tidl_onnx_layers = tidl_json.get('model', {}).get('onnx', {}).get('layers', {})
        arm_nodes_in_tidl = {
            name: info
            for name, info in tidl_onnx_layers.items()
            if info.get('runtime_assignment', {}).get('assigned_runtime') == 'arm'
        }

        # Try to match ARM nodes by inferring from TVM subgraph names
        for sg_name in tvm_subgraphs.keys():
            if sg_name.startswith('tvmgen'):
                # Extract operation hint from subgraph name
                # e.g., "tvmgen_default_fused_nn_global_avg_pool2d" → "global_avg_pool"
                op_hint = sg_name.lower().split('_fused_')[-1] if '_fused_' in sg_name else ''

                # Find matching ARM nodes
                for arm_name, arm_info in arm_nodes_in_tidl.items():
                    node_type = arm_info.get('type', '').lower()
                    # Simple heuristic: match by type name
                    if (op_hint and node_type and
                        any(kw in op_hint for kw in node_type.split('_')) or
                        any(kw in node_type for kw in op_hint.split('_'))):
                        if arm_name not in tvm_capable_nodes:
                            tvm_capable_nodes[arm_name] = arm_info
                            sg_name_to_onnx_node[sg_name] = arm_name
                            print(f'  [Auto-detect] Matched {arm_name} ({node_type}) to TVM subgraph {sg_name}')

    # ── Step 2: Promote ARM → TVM in merged ONNX layers ────────────────────
    #           Only if:  was arm in TIDL AND can be tvm in TVM
    merged_onnx_layers = merged.get('model', {}).get('onnx', {}).get('layers', {})
    promoted_count = 0
    for node_name in tvm_capable_nodes:
        if node_name in merged_onnx_layers:
            current_rt = merged_onnx_layers[node_name].get('runtime_assignment', {}).get('assigned_runtime')
            if current_rt == 'arm':  # ← only promote ARM, NEVER demote TIDL
                merged_onnx_layers[node_name]['runtime_assignment'] = {
                    'assigned_runtime': 'tvm_rt',
                    'reason': 'Promoted from arm: TVM can accelerate this node'
                }
                promoted_count += 1

    print(f'  [Merged ONNX] Promoted {promoted_count} nodes: arm → tvm_rt')

    # ── Step 3: Map each TVM subgraph to its ONNX node name ─────────────────
    #           Preference order:
    #             1. The layer's own onnx_mapping, if already correctly set
    #                (single node, unambiguous)
    #             2. The sg_name -> onnx_node_name match found in Step 1b
    tidl_onnx_layers_full = tidl_json.get('model', {}).get('onnx', {}).get('layers', {})

    def _resolve_onnx_node_name(sg_id, layer):
        onnx_names = (layer.get('onnx_mapping') or {}).get('onnx_node_names') or []
        if len(onnx_names) == 1 and onnx_names[0] in tidl_onnx_layers_full:
            return onnx_names[0]
        return sg_name_to_onnx_node.get(sg_id)

    # ── Step 3b: Add TVM subgraphs to merged runtime ─────────────────────
    tvm_subgraphs = tvm_json.get('runtime', {}).get('subgraphs', {})
    merged_subgraphs = merged.setdefault('runtime', {}).setdefault('subgraphs', {})

    tvm_sg_ids = []
    for sg_id, sg_info in tvm_subgraphs.items():
        if sg_info.get('runtime') == 'tvm_rt':
            sg_copy = copy.deepcopy(sg_info)
            # Normalize to plain tensor-name strings up front — the Tensor
            # fix below overwrites this when resolution succeeds, but for
            # subgraphs where it doesn't (e.g. a duplicate/self-referencing
            # onnx_mapping, as seen elsewhere), sg_copy['inputs']/['outputs']
            # would otherwise stay in the original {'name':...} dict format,
            # which Step 5 below can't hash/compare against subgraph ids.
            # Accept either a plain string (newer, already-clean format) or
            # the older {'name': ..., 'shape': ..., 'dtype': ...} dict.
            def _boundary_name(v):
                return v if isinstance(v, str) else v.get('name')
            sg_copy['inputs'] = [n for n in (_boundary_name(i) for i in sg_copy.get('inputs', [])) if n]
            sg_copy['outputs'] = [n for n in (_boundary_name(o) for o in sg_copy.get('outputs', [])) if n]
            for layer in sg_copy.get('layers', []):
                layer_name = layer.get('layer_name')
                onnx_node_name = _resolve_onnx_node_name(sg_id, layer)
                onnx_info = tidl_onnx_layers_full.get(onnx_node_name, {}) if onnx_node_name else {}

                # Real tensor name/shape/dtype from the authoritative ONNX
                # input_details/output_details — NOT whatever placeholder
                # name the TVM JSON itself declared (e.g. "tidl_1" instead
                # of the real tensor name "188").
                real_inputs = [
                    {'tensor_name': d.get('tensor_name'), 'shape': d.get('shape', []),
                     'dtype': d.get('dtype', 'float32')}
                    for d in onnx_info.get('input_details', []) if not d.get('is_constant', False)
                ]
                real_outputs = [
                    {'tensor_name': d.get('tensor_name'), 'shape': d.get('shape', []),
                     'dtype': d.get('dtype', 'float32')}
                    for d in onnx_info.get('output_details', [])
                ]

                if onnx_node_name:
                    layer['onnx_mapping'] = {
                        'onnx_node_names': [onnx_node_name],
                        'onnx_node_indices': [],
                        'mapping_type': '1-to-1'
                    }
                elif layer.get('onnx_mapping') is None:
                    layer['onnx_mapping'] = {'onnx_node_names': [layer_name], 'onnx_node_indices': []}

                if real_inputs:
                    layer['inputs'] = real_inputs
                    # Subgraph-level inputs/outputs are simple: each entry
                    # is just the tensor name (Step 5 below replaces it
                    # with the neighboring subgraph's id if one produces/
                    # consumes that exact tensor).
                    sg_copy['inputs'] = [i['tensor_name'] for i in real_inputs]
                    print(f'  [Tensor fix] {sg_id} input corrected → '
                          f'{[i["tensor_name"] for i in real_inputs]} {[i["shape"] for i in real_inputs]}')
                if real_outputs:
                    layer['outputs'] = real_outputs
                    sg_copy['outputs'] = [o['tensor_name'] for o in real_outputs]
                    print(f'  [Tensor fix] {sg_id} output corrected → '
                          f'{[o["tensor_name"] for o in real_outputs]} {[o["shape"] for o in real_outputs]}')

                # Sanitize numeric fields
                if layer.get('gmacs') is None:
                    layer['gmacs'] = 0
                if layer.get('parameters') is None:
                    layer['parameters'] = {}
            merged_subgraphs[sg_id] = sg_copy
            tvm_sg_ids.append(sg_id)

    if tvm_sg_ids:
        print(f'  [Merged Subgraphs] Added TVM subgraphs: {tvm_sg_ids}')
    else:
        print(f'  [Merged Subgraphs] No tvm_rt subgraphs found in TVM JSON')

    # ── Step 3c: Fix TIDL subgraph boundary tensor names using the TVM/
    #            Relay-side stub entries ──────────────────────────────────
    #            When the TIDL JSON is freshly extracted from raw TIDL
    #            compiler artifacts (netLog.txt etc.) rather than a TIDL-
    #            only AM6xA compile, its subgraph-level inputs/outputs are
    #            built from netLog's OWN internal buffer names (e.g.
    #            "tidl_0_i0"), not real ONNX tensor names — TIDL's netLog
    #            has no concept of the original ONNX tensor at its
    #            boundary. The TVM JSON's own "tidl_N" stub entries (no
    #            layers, just inputs/outputs) come from the Relay graph's
    #            boundary instead, which usually knows the real ONNX
    #            tensor name — but not always (e.g. it may itself just say
    #            "tidl_4.out0"). Prefer whichever side ISN'T a placeholder;
    #            keep the TIDL side's value if it's already real (e.g. from
    #            data_extractor.py's activation-fusion mapping) rather than
    #            blindly overwriting a good name with a worse placeholder.
    def _is_placeholder_tensor(name):
        # A bare "tidl_4" is a valid, fully-resolved subgraph reference —
        # only flag it as a placeholder when there's an actual unresolved
        # port/buffer suffix after the subgraph id (e.g. "tidl_6_i0",
        # "tidl_9.out0"). The earlier version made every part after the
        # number optional, so it matched valid bare references too and
        # let the stub's worse placeholder overwrite an already-correct
        # value.
        return bool(re.match(r'^tidl_\d+[._](i\d+|o\d+|out\d+)$', name or ''))

    tidl_boundary_fixed = 0
    for sg_id, sg_info in tvm_subgraphs.items():
        if sg_info.get('runtime') != 'tidl_rt':
            continue
        merged_sg = merged_subgraphs.get(sg_id)
        if not merged_sg:
            continue
        # Accept either a plain string (newer, already-clean format) or
        # the older {'name': ..., 'shape': ..., 'dtype': ...} dict.
        def _boundary_name(v):
            return v if isinstance(v, str) else v.get('name')
        stub_inputs = [n for n in (_boundary_name(i) for i in sg_info.get('inputs', [])) if n]
        stub_outputs = [n for n in (_boundary_name(o) for o in sg_info.get('outputs', [])) if n]
        current_inputs = merged_sg.get('inputs', [])
        current_outputs = merged_sg.get('outputs', [])
        if stub_inputs and (not current_inputs or all(_is_placeholder_tensor(t) for t in current_inputs)):
            merged_sg['inputs'] = stub_inputs
            tidl_boundary_fixed += 1
        if stub_outputs and (not current_outputs or all(_is_placeholder_tensor(t) for t in current_outputs)):
            merged_sg['outputs'] = stub_outputs
    if tidl_boundary_fixed:
        print(f'  [Tensor fix] Replaced placeholder boundary names with real '
              f'tensor names for {tidl_boundary_fixed} TIDL subgraphs (from TVM/Relay stub)')

    # ── Step 4: Merge metadata (performance, accuracy) from TVM if absent ───
    tidl_meta = merged.setdefault('metadata', {})
    tvm_meta = tvm_json.get('metadata', {})

    # Copy perf metadata (only if not in TIDL)
    perf_keys = ('performance_source', 'infer_time_subgraph_ms', 'infer_time_core_ms',
                 'infer_time_invoke_ms', 'num_frames')
    for key in perf_keys:
        if key in tvm_meta and key not in tidl_meta:
            tidl_meta[key] = tvm_meta[key]

    # Copy accuracy metrics (all accuracy_* keys not in TIDL)
    for key, val in tvm_meta.items():
        if key.lower().startswith('accuracy') and key not in tidl_meta:
            tidl_meta[key] = val

    # ── Step 4b: Normalize "tidl_N.outM" / "tidl_N_oM" style values ─────────
    # The TVM/Relay-side stub sometimes references another TIDL subgraph's
    # output using ITS OWN shorthand placeholder (e.g. "tidl_9.out0")
    # instead of the real ONNX tensor name — since from TVM's point of
    # view it's a TIDL-internal connection it doesn't need to name. This
    # isn't a tensor name to look up at all; it's already a direct
    # reference to subgraph "tidl_9" and should resolve as one.
    placeholder_ref_re = re.compile(r'^(tidl_\d+)[._]out\d*$')
    placeholder_fixed = 0
    for sg_id, sg_info in merged_subgraphs.items():
        for field in ('inputs', 'outputs'):
            values = sg_info.get(field, []) or []
            new_values = []
            for v in values:
                m = placeholder_ref_re.match(v) if isinstance(v, str) else None
                if m and m.group(1) in merged_subgraphs and m.group(1) != sg_id:
                    new_values.append(m.group(1))
                    placeholder_fixed += 1
                else:
                    new_values.append(v)
            sg_info[field] = new_values
    if placeholder_fixed:
        print(f'  [Tensor fix] Resolved {placeholder_fixed} "tidl_N.outM"-style shorthand references to direct subgraph ids')

    # ── Step 5: Resolve subgraph inputs/outputs across ALL subgraphs ────────
    # Each subgraph's inputs/outputs is a simple list of values — either a
    # neighboring subgraph's id (e.g. "tidl_5") when data_extractor.py
    # already resolved it, or just the raw tensor name when it didn't
    # (TVM didn't exist yet at that point). Now that TVM subgraphs are
    # merged in too, do one global pass matching those remaining tensor
    # names against every subgraph's own inputs/outputs, so the Overview
    # graph can connect subgraphs directly by id instead of re-tracing
    # ownership through the full ONNX node graph.
    tensor_producer, tensor_consumers = {}, {}
    for sg_id, sg_info in merged_subgraphs.items():
        for v in sg_info.get('outputs', []) or []:
            if v not in merged_subgraphs:
                tensor_producer[v] = sg_id
        for v in sg_info.get('inputs', []) or []:
            if v not in merged_subgraphs:
                tensor_consumers.setdefault(v, []).append(sg_id)

    # Boundary values can ALSO be a bare ONNX node name rather than a
    # tensor name — data_extractor.py falls back to the node name when a
    # boundary crosses into a node that isn't part of any subgraph (e.g.
    # an ARM/TVM-bound node), so the Overview graph can name what it
    # crosses into. But a TVM subgraph's OWN inputs/outputs get populated
    # from that same node's real tensor name (via the "Tensor fix" step
    # above), not its node name — so a plain tensor_producer/
    # tensor_consumers lookup (exact string match only) never connects
    # "tidl_0 wants node /foo/Bar" to "tvmgen_..._9 produces tensor
    # /foo/Bar_output_0", even though they're the same boundary. Build a
    # node-name -> owning-subgraph map from every subgraph's own
    # onnx_mapping to bridge the two conventions.
    node_name_to_subgraph = {}
    for sg_id, sg_info in merged_subgraphs.items():
        for layer in sg_info.get('layers', []) or []:
            for onnx_name in (layer.get('onnx_mapping') or {}).get('onnx_node_names') or []:
                node_name_to_subgraph[onnx_name] = sg_id

    resolved_count = 0
    for sg_id, sg_info in merged_subgraphs.items():
        new_inputs = []
        for v in sg_info.get('inputs', []) or []:
            if v in merged_subgraphs:
                producer = None
            else:
                # node_name_to_subgraph first: it's derived from onnx_mapping,
                # which is ground truth for "who owns this ONNX node". A
                # value can ALSO appear in tensor_producer even when it's not
                # a real tensor a subgraph produces — e.g. a subgraph whose
                # OWN boundary output fell back to naming the ARM/TVM node it
                # feeds into (data_extractor.py does this when no OTHER
                # subgraph directly consumes its output) makes tensor_producer
                # think that subgraph "produces" that node name, when really
                # it's just naming where its output goes. If some OTHER
                # subgraph separately references that same node name as ITS
                # OWN input, tensor_producer would incorrectly resolve it to
                # the upstream subgraph instead of the node's true owner.
                producer = node_name_to_subgraph.get(v) or tensor_producer.get(v)
            if producer and producer != sg_id:
                new_inputs.append(producer)
                resolved_count += 1
            else:
                new_inputs.append(v)
        sg_info['inputs'] = list(dict.fromkeys(new_inputs))

        new_outputs = []
        for v in sg_info.get('outputs', []) or []:
            if v in merged_subgraphs:
                consumers = []
            else:
                # Same node_name_to_subgraph-first priority as the inputs
                # loop above, and for the same reason: v can be a node name
                # that some OTHER subgraph's own input fell back to naming
                # (the ARM/TVM node it needs) without actually owning it —
                # onnx_mapping's node_name_to_subgraph is ground truth for
                # who really owns that node.
                node_owner = node_name_to_subgraph.get(v)
                if node_owner and node_owner != sg_id:
                    consumers = [node_owner]
                else:
                    consumers = [c for c in tensor_consumers.get(v, []) if c != sg_id]
            if consumers:
                new_outputs.append(consumers[0])
                resolved_count += 1
            else:
                new_outputs.append(v)
        sg_info['outputs'] = list(dict.fromkeys(new_outputs))
    print(f'  [Connected-subgraph] Resolved {resolved_count} boundary references')

    return merged


def merge_trust_tvm_connections(tvm_json: dict, tidl_json: dict) -> dict:
    """
    Merge (inverted direction from merge()): use the TVM-side JSON as the
    base, trusting its own declared subgraph-level inputs/outputs verbatim
    instead of re-deriving connections ourselves.

    Why this exists: merge()'s Step 5 resolves each boundary independently
    by matching tensor/node names against every OTHER subgraph's own
    boundary values — but a TIDL subgraph's boundary output can itself be
    a fallback node name (naming the ARM/TVM node it feeds into, when no
    OTHER TIDL subgraph directly consumes it), and Step 5 has no way to
    tell that apart from an actual produced tensor. That ambiguity caused
    real misattributions (e.g. tidl_1→2→3→4 each showing a direct jump to
    the next tidl_N instead of routing through the real intermediate TVM
    subgraph). Newer TVM-side model_inspector.json exports already have
    complete, correct subgraph-level connections (including cases our own
    tracing can't fully see, like one TIDL subgraph feeding FOUR separate
    downstream TVM kernels) — so when that's available, trust it outright
    rather than re-deriving it.

    What's taken from where:
      - Subgraph-level inputs/outputs, and full TVM (tvmgen_*) subgraph
        layer detail: from tvm_json, unchanged.
      - Full per-layer detail for each tidl_N subgraph (layers,
        num_layers, total_gmacs, etc.): injected from tidl_json, but its
        inputs/outputs/runtime are NOT used — the tvm_json stub's own
        boundary values for that tidl_N win instead.
      - model.onnx (per-ONNX-node type/shape/runtime_assignment/tree) and
        top-level metadata (model name, overall inputs/outputs): from
        tidl_json wholesale, since tvm_json has neither a 'model' nor a
        'metadata' key at all — then any node still marked 'arm' gets
        promoted to 'tvm_rt' if some tvmgen_* subgraph's own onnx_mapping
        claims it (tidl_json
        has no notion of TVM, so every non-TIDL node defaults to plain
        'arm' there).
    """
    merged = copy.deepcopy(tvm_json)
    merged['model'] = copy.deepcopy(tidl_json.get('model', {}))
    merged['metadata'] = copy.deepcopy(tidl_json.get('metadata', {}))
    # diag_info/diagnostic_summary are TVM-source-specific intermediate
    # artifacts (per-node offload diagnostics) — their information is
    # already folded into model.onnx.layers[].runtime_assignment above,
    # so they'd just be redundant, unvalidated leftovers in the final
    # unified schema (which html_generator.py expects to be exactly
    # metadata/model/runtime).
    merged.pop('diag_info', None)
    merged.pop('diagnostic_summary', None)

    tidl_detail = tidl_json.get('runtime', {}).get('subgraphs', {})
    injected = 0
    missing = []
    for sg_id, stub in merged['runtime']['subgraphs'].items():
        if not sg_id.startswith('tidl_'):
            continue
        detail = tidl_detail.get(sg_id)
        if not detail:
            missing.append(sg_id)
            continue
        for key in ('subgraph_id', 'tidl_tool_version', 'tensor_bits',
                    'total_gmacs', 'num_layers', 'layers', 'target_device'):
            if key in detail:
                stub[key] = detail[key]
        injected += 1
    print(f'  [TIDL detail] Injected layer detail into {injected} tidl_N subgraphs')
    if missing:
        print(f'  WARNING: no TIDL-only detail found for: {missing}')

    node_owner = {}
    for sg_id, info in merged['runtime']['subgraphs'].items():
        if not sg_id.startswith('tvmgen'):
            continue
        for layer in info.get('layers', []) or []:
            for n in (layer.get('onnx_mapping') or {}).get('onnx_node_names') or []:
                node_owner[n] = sg_id

    onnx_layers = merged.get('model', {}).get('onnx', {}).get('layers', {})
    promoted = 0
    for name, info in onnx_layers.items():
        ra = info.get('runtime_assignment', {})
        if ra.get('assigned_runtime') == 'arm' and name in node_owner:
            # Keep the original "why not TIDL" reason (e.g. "Stride must be
            # the same along both horizontal and vertical dimensions") —
            # that's exactly the useful diagnostic here, since the node IS
            # accelerated, just not by TIDL. Wiping it to None would erase
            # the one piece of info that explains the runtime split at all.
            info['runtime_assignment'] = {'assigned_runtime': 'tvm_rt', 'reason': ra.get('reason')}
            promoted += 1
    print(f'  [Merged ONNX] Promoted {promoted} nodes: arm → tvm_rt (reason preserved)')

    return merged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Merge TIDL + TVM modelinspector.json files with runtime ladder: TIDL → TVM → ARM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Merge two compiled inspector JSONs
  python merge_inspector_json.py \\
    --tidl work_dirs/compile/AM62A/*/tidl_*/inspector/modelinspector.json \\
    --tvm  work_dirs/compile/AM62A/*/tvm_*/inspector/modelinspector.json \\
    --out merged_modelinspector.json

  # Then regenerate the HTML
  python html_generator.py merged_modelinspector.json template.html merged.html
        '''
    )
    parser.add_argument('--tidl', required=True, metavar='FILE',
                        help='TIDL-compiled modelinspector.json')
    parser.add_argument('--tvm', required=True, metavar='FILE',
                        help='TVM-compiled modelinspector.json')
    parser.add_argument('--out', required=True, metavar='FILE',
                        help='Output merged modelinspector.json')
    parser.add_argument('--trust-tvm-connections', action='store_true',
                        help='Use the TVM JSON\'s own subgraph-level inputs/outputs '
                             'verbatim instead of re-deriving connections ourselves. '
                             'Use this when the TVM JSON already has complete, correct '
                             'connectivity (see merge_trust_tvm_connections docstring).')
    args = parser.parse_args()

    print('=' * 70)
    print('TIDL + TVM Inspector JSON Merger')
    print('=' * 70)

    print(f'\n[1/4] Loading TIDL JSON: {args.tidl}')
    tidl_json = _load(args.tidl)

    print(f'[2/4] Loading TVM JSON: {args.tvm}')
    tvm_json = _load(args.tvm)

    # Show input distributions
    tidl_sgs = list(tidl_json.get('runtime', {}).get('subgraphs', {}).keys())
    tvm_sgs = list(tvm_json.get('runtime', {}).get('subgraphs', {}).keys())
    tidl_dist = _runtime_dist(tidl_json.get('model', {}).get('onnx', {}).get('layers', {}))
    tvm_dist = _runtime_dist(tvm_json.get('model', {}).get('onnx', {}).get('layers', {}))

    print(f'\n[Input] TIDL:')
    print(f'  Subgraphs: {tidl_sgs}')
    print(f'  ONNX runtime dist: {tidl_dist}')

    print(f'\n[Input] TVM:')
    print(f'  Subgraphs: {tvm_sgs}')
    print(f'  ONNX runtime dist: {tvm_dist}')

    if args.trust_tvm_connections:
        print(f'\n[3/4] Merging (trusting TVM JSON\'s own subgraph connections)...')
        merged = merge_trust_tvm_connections(tvm_json, tidl_json)
    else:
        print(f'\n[3/4] Merging (ladder: TIDL → TVM → ARM)...')
        merged = merge(tidl_json, tvm_json)

    merged_dist = _runtime_dist(merged.get('model', {}).get('onnx', {}).get('layers', {}))
    merged_sgs = list(merged.get('runtime', {}).get('subgraphs', {}).keys())

    print(f'\n[Output] Merged:')
    print(f'  Subgraphs: {merged_sgs}')
    print(f'  ONNX runtime dist: {merged_dist}')

    print(f'\n[4/4] Saving merged JSON → {args.out}')
    _save(merged, args.out)

    print('\n' + '=' * 70)
    print('✓ Merge complete!')
    print('\nNext step: regenerate HTML')
    print('  python html_generator.py \\')
    print(f'    {args.out} \\')
    print(f'    template.html \\')
    print(f'    {os.path.splitext(args.out)[0]}.html')
    print('=' * 70)


if __name__ == '__main__':
    main()
