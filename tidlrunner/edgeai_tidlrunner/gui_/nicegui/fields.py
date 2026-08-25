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

"""Turns the argparse specs in SETTINGS_DEFAULT into GUI field descriptions."""

import argparse
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from edgeai_tidlrunner.rtwrapper.options import enumerations
from edgeai_tidlrunner.runner.common import utils
from edgeai_tidlrunner.runner.common.settings import settings_default  # noqa: F401  (populates SETTINGS_DEFAULT)
from edgeai_tidlrunner.runner.common.settings.constants import SETTINGS_DEFAULT


# order shown in the command selector
COMMAND_ORDER = (
    'compile',
    'infer',
    'evaluate',
    'inspect',
    'analyze',
    'surgery',
    'extract',
    'report',
    'package',
)

# shown at the top of the form, in this order, outside the collapsible groups
PRIMARY_ARGS = (
    'config_path',
    'model_path',
    'target_device',
    # 'target_machine',
    'tensor_bits',
    'calibration_frames',
    'calibration_iterations',
    'num_frames',
    # 'data_name',
    # 'data_path',
    # 'label_path',
    # 'run_label',
    'work_path',
)

# args whose value is a path to an existing file
FILE_ARGS = frozenset({
    'model_path',
    'config_path',
    'label_path',
    'meta_arch_file_path',
    'quant_params_file_path',
    'config_template',
    'param_template',
})

# args whose value is a directory
DIR_ARGS = frozenset({
    'data_path',
    'work_path',
    'run_dir',
    'artifacts_folder',
    'report_path',
    'package_path',
})

_GROUP_BY_DEST_PREFIX = (
    ('model_surgery.', 'Model surgery'),
    ('session.runtime_options.', 'Runtime options'),
    ('session.onnxruntime', 'Runtime options'),
    ('dataloader.', 'Dataset'),
    ('preprocess.', 'Preprocess'),
    ('postprocess.', 'Postprocess'),
    ('session.', 'Session'),
    ('common.', 'General'),
)

GROUP_ORDER = (
    'Session',
    'Dataset',
    'Runtime options',
    'Preprocess',
    'Postprocess',
    'Model surgery',
    'General',
    'Other',
)


# explicit choices for args that argparse leaves as free-form
_EXPLICIT_CHOICES: Dict[str, List[str]] = {
    # 'target_device': utils.enum_to_list(enumerations.TargetDeviceType),
    # 'target_machine': utils.enum_to_list(enumerations.TargetMachineType),
    # 'tensor_bits': utils.enum_to_list(enumerations.TensorBits),
    # 'accuracy_level': utils.enum_to_list(enumerations.AccurcyLevel),
    # 'debug_level': utils.enum_to_list(enumerations.DebugLevel),
    # 'add_data_convert_ops': utils.enum_to_list(enumerations.DataConvertOps),
    # 'data_layout': utils.enum_to_list(enumerations.DataLayoutType),
    # 'runtime_name': utils.enum_to_list(enumerations.RuntimeType),
    # 'enable_tfr_optimization': ['0', '1'],
    # 'pipeline_type': ['compile', 'infer', 'optimize', 'extract', 'package'],
    # 'capture_log': ['adaptive', 'True', 'False'],
    # 'analyze_level': ['0', '1', '2'],
    # 'graph_optimization_level': ['0', '1', '2', '99'],
    # 'simplify_mode': ['pre', 'post', 'all', 'None'],
    # 'shape_inference_mode': ['pre', 'post', 'all', 'None'],
    # 'preset_selection': ['None', 'SPEED', 'ACCURACY', 'BALANCED'],
    # 'audio_model_type': ['vggish11', 'yamnet', 'gtcrn', 'gcrn'],
}

_BOOL_TYPE_NAMES = frozenset({'str_to_bool'})
_INT_TYPE_NAMES = frozenset({'int_or_none', 'str_to_int'})
_FLOAT_TYPE_NAMES = frozenset({'float_or_none'})


@dataclass
class Field:
    name: str
    dest: str
    kind: str                       # bool | int | float | select | text
    default_text: str
    help: str = ''
    group: str = 'Other'
    choices: List[str] = field(default_factory=list)
    nargs: bool = False
    browse: Optional[str] = None    # 'file' | 'dir' | None
    arg_group: Optional[str] = None  # name of the mutually exclusive argparse group, if any


def command_names() -> List[str]:
    available = {k.split('.', 1)[1] for k in SETTINGS_DEFAULT if k.startswith('commands.')}
    ordered = [name for name in COMMAND_ORDER if name in available]
    return ordered + sorted(available - set(ordered))


def _value_to_text(value: Any) -> str:
    if value is None or value is argparse.SUPPRESS:
        return ''
    if isinstance(value, bool):
        return '1' if value else '0'
    if isinstance(value, dict):
        return ', '.join(f'{k}:{v}' for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return ' '.join(str(v) for v in value)
    return str(value)


def _kind_of(name: str, spec: Dict[str, Any]) -> str:
    if name in _EXPLICIT_CHOICES or 'choices' in spec:
        return 'select'
    arg_type = spec.get('type')
    type_name = getattr(arg_type, '__name__', '')
    if type_name in _BOOL_TYPE_NAMES or arg_type is bool:
        return 'bool'
    if spec.get('nargs'):
        return 'text'
    if arg_type is int or type_name in _INT_TYPE_NAMES:
        return 'int'
    if arg_type is float or type_name in _FLOAT_TYPE_NAMES:
        return 'float'
    return 'text'


def _group_of(dest: str) -> str:
    for prefix, group in _GROUP_BY_DEST_PREFIX:
        if dest.startswith(prefix):
            return group
    return 'Other'


def _make_field(name: str, spec: Dict[str, Any]) -> Field:
    gui = spec.get('_gui', False)
    if not gui:
        return None
    
    dest = spec.get('dest', name)
    kind = _kind_of(name, spec)
    choices = [str(c) for c in spec.get('choices', [])] or _EXPLICIT_CHOICES.get(name, [])
    browse = 'file' if name in FILE_ARGS else ('dir' if name in DIR_ARGS else None)
    return Field(
        name=name,
        dest=dest,
        kind=kind,
        default_text=_value_to_text(spec.get('default')),
        help=spec.get('help') or '',
        group=_group_of(dest),
        choices=choices,
        nargs=bool(spec.get('nargs')),
        browse=browse,
        arg_group=spec.get('group'),
    )


def get_fields(command: str) -> List[Field]:
    """All GUI fields for a command, primary ones first, then in declaration order."""
    args_dict = SETTINGS_DEFAULT[f'commands.{command}']
    fields = [_make_field(name, spec) for name, spec in args_dict.items()
              if name != 'command' and not spec.get('_positional')]
    fields = [f for f in fields if f is not None]
    order = {name: index for index, name in enumerate(PRIMARY_ARGS)}
    primary = sorted((f for f in fields if f.name in order), key=lambda f: order[f.name])
    for f in primary:
        f.group = 'Main'
    rest = [f for f in fields if f.name not in order]
    return primary + rest


def grouped_fields(command: str) -> Dict[str, List[Field]]:
    groups: Dict[str, List[Field]] = {}
    for f in get_fields(command):
        groups.setdefault(f.group, []).append(f)
    ordered = {'Main': groups.pop('Main', [])}
    for name in GROUP_ORDER:
        if groups.get(name):
            ordered[name] = groups.pop(name)
    ordered.update(groups)
    return ordered


def default_values(command: str) -> Dict[str, str]:
    return {f.name: f.default_text for f in get_fields(command)}


def arg_groups(command: str) -> Dict[str, List[Field]]:
    """Mutually exclusive argument groups (argparse 'group' key) with 2+ GUI fields,
    in field declaration order."""
    groups: Dict[str, List[Field]] = {}
    for f in get_fields(command):
        if f.arg_group:
            groups.setdefault(f.arg_group, []).append(f)
    return {name: members for name, members in groups.items() if len(members) > 1}


def default_group_selection(command: str) -> Dict[str, str]:
    """Default selected member (first declared) for each mutually exclusive group."""
    return {name: members[0].name for name, members in arg_groups(command).items()}


def build_argv(command: str, values: Dict[str, str],
                group_selected: Optional[Dict[str, str]] = None) -> List[str]:
    """Build CLI tokens for the non-default values only."""
    group_selected = group_selected or {}
    argv = [command]
    for f in get_fields(command):
        if f.arg_group in group_selected and group_selected[f.arg_group] != f.name:
            # only the member selected via the mutually exclusive radio group is emitted
            continue
        text = (values.get(f.name) or '').strip()
        if text == f.default_text.strip() or (not text and not f.default_text):
            continue
        if not text:
            # an emptied field cannot be expressed as a CLI override
            continue
        if f.nargs:
            argv.append(f'--{f.name}')
            argv.extend(re.split(r'[,\s]+', text))
        else:
            argv.append(f'--{f.name}={text}')
    return argv
