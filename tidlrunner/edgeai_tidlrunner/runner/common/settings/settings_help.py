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


import copy
import inspect
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class HelpSettingsEntry:
    name: str
    description: str = ''
    task_type: Optional[str] = None
    required_args: List[str] = field(default_factory=list)
    optional_args: Dict[str, Any] = field(default_factory=dict)
    supports_evaluate: Optional[bool] = None
    availability: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    example: Optional[str] = None


SETTINGS_HELP: Dict[str, Dict[str, Dict[str, Any]]] = {}

SETTINGS_HELP['dataloaders'] : Dict[str, Dict[str, Any]] = {}


def _infer_args_from_signature(fn: Callable[..., Any]) -> Tuple[List[str], Dict[str, Any]]:
    signature = inspect.signature(fn)
    required_args: List[str] = []
    optional_args: Dict[str, Any] = {}

    for param in signature.parameters.values():
        if param.name in ('settings', 'name'):
            continue

        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        if param.default is inspect.Parameter.empty:
            required_args.append(param.name)
        else:
            optional_args[param.name] = param.default

    return required_args, optional_args


def _merge_unique(base: List[str], add: List[str]) -> List[str]:
    result = list(base)
    for item in add:
        if item not in result:
            result.append(item)
    return result


def register_help(
    section: str,
    name: Optional[str] = None,
    description: str = '',
    task_type: Optional[str] = None,
    required_args: Optional[List[str]] = None,
    optional_args: Optional[Dict[str, Any]] = None,
    supports_evaluate: Optional[bool] = None,
    availability: Optional[str] = None,
    tags: Optional[List[str]] = None,
    notes: Optional[str] = None,
    example: Optional[str] = None,
    infer_signature: bool = True,
    overwrite: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to register help metadata for documentation generation."""
    required_args = required_args or []
    optional_args = optional_args or {}
    tags = tags or []

    def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        entry_name = name or fn.__name__

        if section not in SETTINGS_HELP:
            SETTINGS_HELP[section] = {}
        if (entry_name in SETTINGS_HELP[section]) and not overwrite:
            return fn

        inferred_required, inferred_optional = _infer_args_from_signature(fn) if infer_signature else ([], {})
        merged_required = _merge_unique(inferred_required, required_args)
        merged_optional = dict(inferred_optional)
        merged_optional.update(optional_args)

        entry = HelpSettingsEntry(
            name=entry_name,
            description=description.strip(),
            task_type=task_type,
            required_args=merged_required,
            optional_args=merged_optional,
            supports_evaluate=supports_evaluate,
            availability=availability,
            tags=list(tags),
            notes=notes,
            example=example,
        )

        entry_dict = asdict(entry)
        SETTINGS_HELP[section][entry_name] = entry_dict
        setattr(fn, '__settings_help__', copy.deepcopy(entry_dict))
        return fn

    return _decorator


def get_help(section: str, name: Optional[str] = None) -> Dict[str, Any]:
    if name is None:
        return copy.deepcopy(SETTINGS_HELP.get(section, {}))
    return copy.deepcopy(SETTINGS_HELP.get(section, {}).get(name, {}))


def list_help_names(section: str) -> List[str]:
    return sorted(SETTINGS_HELP.get(section, {}).keys())


def export_help_markdown() -> str:
    """Export registered help metadata for a given section as a markdown table."""
    lines = [
        '| Section | Name | Task | Required Args | Optional Args | Description |',
        '|---|---|---|---|---|---|',
    ]

    for section in sorted(SETTINGS_HELP.keys()):
        for name in sorted(SETTINGS_HELP.get(section, {}).keys()):
            entry = SETTINGS_HELP[section][name]
            task = entry.get('task_type') or ''
            required = ', '.join(entry.get('required_args') or [])
            optional = ', '.join((entry.get('optional_args') or {}).keys())
            description = (entry.get('description') or '').replace('|', '\\|')
            lines.append(f'| {section} | {name} | {task} | {required} | {optional} | {description} |')

    return '\n'.join(lines)

