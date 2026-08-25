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

"""A file/folder browser dialog for paths on the machine running the server."""

import pathlib
from typing import Optional

from nicegui import ui

from ..common import browse


async def choose_path(start: str = '', dirs_only: bool = False) -> Optional[str]:
    """Open a modal browser and return the selected path, or None if cancelled."""
    state = {'dir': browse.start_directory(start)}

    with ui.dialog() as dialog, ui.card().classes('tidl-picker w-[46rem] max-w-full p-0'):
        with ui.row().classes('tidl-header w-full items-center no-wrap gap-2 px-4 py-3'):
            ui.icon('folder_open').classes('text-xl text-red-5')
            ui.label('Select a folder' if dirs_only else 'Select a file') \
                .classes('tidl-brand text-subtitle1')
            ui.space()
            show_hidden = ui.switch('hidden').props('dense color=white').classes('mr-2')
            ui.button(icon='close', on_click=lambda: dialog.submit(None)) \
                .props('flat round dense color=white')

        breadcrumbs = ui.row().classes(
            'tidl-picker-crumbs w-full items-center no-wrap gap-1 px-3 overflow-auto') \
            .style('border-bottom: 1px solid var(--tidl-line)')
        listing = ui.column().classes('tidl-scroll w-full gap-0 p-2 overflow-auto').style('height: 22rem')
        with ui.row().classes('w-full justify-end items-center gap-2 px-4 py-3') \
                .style('border-top: 1px solid var(--tidl-line)'):
            ui.button('Cancel', on_click=lambda: dialog.submit(None)).props('flat no-caps')
            if dirs_only:
                ui.button('Select folder', icon='check',
                         on_click=lambda: dialog.submit(str(state['dir']))) \
                    .props('unelevated no-caps').classes('tidl-run')

    def navigate(target: pathlib.Path) -> None:
        state['dir'] = target
        render()

    def entry_button(label: str, icon: str, on_click, kind: str) -> None:
        ui.button(label, icon=icon, on_click=on_click) \
            .props('flat no-caps align=left dense') \
            .classes(f'tidl-picker-row tidl-picker-row--{kind} w-full text-left')

    def render_breadcrumbs() -> None:
        breadcrumbs.clear()
        with breadcrumbs:
            parts = state['dir'].parts
            cumulative = pathlib.Path(parts[0])
            ui.button(parts[0], on_click=lambda p=cumulative: navigate(p)) \
                .props('flat dense no-caps').classes('tidl-crumb')
            for part in parts[1:]:
                ui.icon('chevron_right').classes('tidl-dim text-xs')
                cumulative = cumulative / part
                ui.button(part, on_click=lambda p=cumulative: navigate(p)) \
                    .props('flat dense no-caps').classes('tidl-crumb')
        breadcrumbs.run_method('scrollTo', {'left': 999999, 'behavior': 'auto'})

    def render() -> None:
        render_breadcrumbs()
        listing.clear()
        with listing:
            parent = state['dir'].parent
            if parent != state['dir']:
                entry_button('..', 'arrow_upward', lambda: navigate(parent), 'up')
            entries, error = browse.list_directory(state['dir'], show_hidden.value, dirs_only)
            if error:
                ui.label(error).classes('text-negative text-sm p-2')
                return
            if not entries:
                with ui.column().classes('w-full items-center gap-2 p-6'):
                    with ui.element('div').classes('tidl-empty-icon'):
                        ui.icon('folder_off').classes('text-2xl text-primary')
                    ui.label('this folder is empty').classes('text-caption tidl-dim')
                return
            for entry in entries:
                if entry.is_dir():
                    entry_button(entry.name, 'folder', lambda e=entry: navigate(e), 'folder')
                else:
                    entry_button(entry.name, 'description', lambda e=entry: dialog.submit(str(e)), 'file')

    show_hidden.on_value_change(lambda _: render())
    render()
    return await dialog
