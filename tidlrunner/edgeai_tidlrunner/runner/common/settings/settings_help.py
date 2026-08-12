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

from .constants import SETTINGS_HELP

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
        if section not in SETTINGS_HELP:
            SETTINGS_HELP[section] = {}

        if isinstance(fn, dict):
            entry_name = name
            if (entry_name in SETTINGS_HELP[section]) and not overwrite:
                return fn

            dict_required_args = [f"{k} [{v.get('dest')}] ({v.get('help')})" for k, v in fn.items() if v.get('required') and v.get('dest')]
            dict_optional_args = {f"{k} [{v.get('dest')}] ({v.get('help')})": v.get('default') for k, v in fn.items() if not v.get('required') and v.get('dest')}
            merged_required = _merge_unique(dict_required_args, required_args)
            merged_optional = dict(dict_optional_args)
            merged_optional.update(optional_args)
        else:
            entry_name = name or fn.__name__
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
        if not isinstance(fn, dict):
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
    """Export registered help metadata as improved, readable markdown.

    For the 'command' section: generates a ``### name`` subsection per command
    with a proper ``| Argument | Config Field | Default | Description |`` table
    (one row per argument).

    For other sections (e.g. 'dataloader'): keeps the compact flat table that
    already works well in a markdown renderer.
    """
    lines: list[str] = []

    for section in sorted(SETTINGS_HELP.keys()):
        section_entries = SETTINGS_HELP.get(section, {})
        if not section_entries:
            continue

        lines.append(f'## {section.capitalize()}s')
        lines.append('')

        if section == 'command':
            for name in sorted(section_entries.keys()):
                entry = section_entries[name]
                description = (entry.get('description') or '').strip()
                task_type   = entry.get('task_type') or ''
                notes       = entry.get('notes') or ''
                example     = entry.get('example') or ''

                lines.append(f'### {name}')
                lines.append('')
                if description:
                    lines.append(description)
                    lines.append('')
                if task_type:
                    lines.append(f'**Task type:** {task_type}')
                    lines.append('')
                if notes:
                    lines.append(f'> **Note:** {notes}')
                    lines.append('')
                if example:
                    lines.append(f'**Example:** `{example}`')
                    lines.append('')

                # --- required args table ---
                required_args = entry.get('required_args') or []
                if required_args:
                    lines.append('**Required arguments:**')
                    lines.append('')
                    lines.append('| Argument | Config Field | Description |')
                    lines.append('|---|---|---|')
                    for arg_key in required_args:
                        arg_name, dest, help_text = _parse_arg_key(arg_key)
                        help_text = help_text.replace('|', '\\|')
                        lines.append(f'| `--{arg_name}` | `{dest}` | {help_text} |')
                    lines.append('')

                # --- optional args table ---
                optional_args = entry.get('optional_args') or {}
                if optional_args:
                    lines.append('**Optional arguments:**')
                    lines.append('')
                    lines.append('| Argument | Config Field | Default | Description |')
                    lines.append('|---|---|---|---|')
                    for arg_key, default_val in optional_args.items():
                        arg_name, dest, help_text = _parse_arg_key(arg_key)
                        default_str = str(default_val) if default_val is not None else ''
                        # escape pipe chars that would break the table
                        help_text   = help_text.replace('|', '\\|')
                        default_str = default_str.replace('|', '\\|')
                        lines.append(f'| `--{arg_name}` | `{dest}` | `{default_str}` | {help_text} |')
                    lines.append('')

        else:
            # Non-command sections: flat table (one row per entry, args as comma list)
            lines.append('| Name | Task | Required Args | Optional Args | Description |')
            lines.append('|---|---|---|---|---|')
            for name in sorted(section_entries.keys()):
                entry = section_entries[name]
                task        = entry.get('task_type') or ''
                required    = ', '.join(
                    _parse_arg_key(k)[0] for k in (entry.get('required_args') or [])
                )
                optional    = ', '.join(
                    _parse_arg_key(k)[0] for k in (entry.get('optional_args') or {}).keys()
                )
                description = (entry.get('description') or '').replace('|', '\\|')
                lines.append(f'| {name} | {task} | {required} | {optional} | {description} |')
            lines.append('')

    return '\n'.join(lines)


def _parse_arg_key(key: str):
    """Parse a key of the form 'arg_name [dest] (help text)' into (arg_name, dest, help).
    Falls back gracefully if the format doesn't match.
    Note: help text may itself contain parentheses, so we match up to the last ')'.
    """
    import re
    # Match: word [dest] (anything up to last closing paren)
    m = re.match(r'^([\w.:-]+)\s+\[([^\]]*)\]\s+\((.+)\)$', key)
    if m:
        return m.group(1), m.group(2), m.group(3)
    # Try without help part: 'arg_name [dest]'
    m2 = re.match(r'^([\w.:-]+)\s+\[([^\]]*)\]$', key)
    if m2:
        return m2.group(1), m2.group(2), ''
    return key, '', ''


def export_help_terminal() -> str:
    """Export registered help metadata in a human-readable terminal format.

    For 'command' section: prints each command as a block with a columnar
    argument table (argument | config field | default | description).
    For other sections (e.g. 'dataloader'): prints name, description, and
    required/optional args in a compact list.
    """
    SEP = '─' * 72
    THIN = '·' * 72
    lines = []

    for section in sorted(SETTINGS_HELP.keys()):
        section_entries = SETTINGS_HELP.get(section, {})
        if not section_entries:
            continue

        lines.append('')
        lines.append(SEP)
        lines.append(f'  SECTION: {section.upper()}')
        lines.append(SEP)

        for name in sorted(section_entries.keys()):
            entry = section_entries[name]
            description = (entry.get('description') or '').strip()
            task_type = entry.get('task_type') or ''
            notes = entry.get('notes') or ''
            example = entry.get('example') or ''
            availability = entry.get('availability') or ''
            tags = entry.get('tags') or []

            lines.append('')
            header = f'  {name}'
            if task_type:
                header += f'  [task: {task_type}]'
            if availability:
                header += f'  [availability: {availability}]'
            lines.append(header)
            if description:
                lines.append(f'    {description}')
            if tags:
                lines.append(f'    tags: {", ".join(tags)}')
            if notes:
                lines.append(f'    notes: {notes}')
            if example:
                lines.append(f'    example: {example}')

            # --- required args ---
            required_args = entry.get('required_args') or []
            if required_args:
                lines.append('    Required arguments:')
                for arg_key in required_args:
                    arg_name, dest, help_text = _parse_arg_key(arg_key)
                    dest_str = f'  [{dest}]' if dest else ''
                    help_str = f'  {help_text}' if help_text else ''
                    lines.append(f'      --{arg_name}{dest_str}{help_str}')

            # --- optional args ---
            optional_args = entry.get('optional_args') or {}
            if optional_args:
                if section == 'command':
                    # For commands, render as a table: argument | config field | default | description
                    col_arg = 'Argument'
                    col_dest = 'Config field'
                    col_def = 'Default'
                    col_help = 'Description'

                    rows = []
                    for arg_key, default_val in optional_args.items():
                        arg_name, dest, help_text = _parse_arg_key(arg_key)
                        default_str = str(default_val) if default_val is not None else ''
                        if len(default_str) > 30:
                            default_str = default_str[:27] + '...'
                        rows.append((arg_name, dest, default_str, help_text))

                    if rows:
                        w_arg  = max(len(col_arg),  max(len(r[0]) for r in rows))
                        w_dest = max(len(col_dest), max(len(r[1]) for r in rows))
                        w_def  = max(len(col_def),  max(len(r[2]) for r in rows))
                        # help column: remaining width up to 72 chars total, min 20
                        w_help = max(20, 72 - w_arg - w_dest - w_def - 10)

                        def _fmt_row(a, d, dv, h, _wa=w_arg, _wd=w_dest, _wdv=w_def, _wh=w_help):
                            # wrap help text
                            words = h.split()
                            wrapped, cur = [], ''
                            for word in words:
                                if cur and len(cur) + 1 + len(word) > _wh:
                                    wrapped.append(cur)
                                    cur = word
                                else:
                                    cur = (cur + ' ' + word).strip()
                            if cur:
                                wrapped.append(cur)
                            if not wrapped:
                                wrapped = ['']
                            pad_a  = a.ljust(_wa)
                            pad_d  = d.ljust(_wd)
                            pad_dv = dv.ljust(_wdv)
                            blank_a  = ''.ljust(_wa)
                            blank_d  = ''.ljust(_wd)
                            blank_dv = ''.ljust(_wdv)
                            first = f'    {pad_a}  {pad_d}  {pad_dv}  {wrapped[0]}'
                            rest  = [f'    {blank_a}  {blank_d}  {blank_dv}  {line}' for line in wrapped[1:]]
                            return [first] + rest

                        # header
                        lines.append(f'    Optional arguments:')
                        header_row = f'    {col_arg:<{w_arg}}  {col_dest:<{w_dest}}  {col_def:<{w_def}}  {col_help}'
                        lines.append(header_row)
                        lines.append('    ' + THIN[:len(header_row) - 4])
                        for row in rows:
                            lines.extend(_fmt_row(*row))
                else:
                    # For non-command sections (dataloaders etc.), compact list
                    lines.append('    Optional arguments:')
                    for arg_key, default_val in optional_args.items():
                        arg_name, dest, help_text = _parse_arg_key(arg_key)
                        default_str = f'  (default: {default_val})' if default_val is not None else ''
                        dest_str = f'  [{dest}]' if dest else ''
                        help_str = f'  {help_text}' if help_text else ''
                        lines.append(f'      {arg_name}{dest_str}{default_str}{help_str}')

            lines.append('    ' + THIN[:40])

    lines.append('')
    return '\n'.join(lines)

