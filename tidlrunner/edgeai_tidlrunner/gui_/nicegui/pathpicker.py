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

    with ui.dialog() as dialog, ui.card().classes('w-[46rem] max-w-full'):
        title = ui.label().classes('text-sm font-mono truncate w-full')
        show_hidden = ui.switch('Show hidden').props('dense')
        listing = ui.column().classes('w-full h-96 overflow-auto gap-0')
        with ui.row().classes('w-full justify-end items-center'):
            ui.button('Cancel', on_click=lambda: dialog.submit(None)).props('flat')
            if dirs_only:
                ui.button('Select folder', on_click=lambda: dialog.submit(str(state['dir'])))

    def navigate(target: pathlib.Path) -> None:
        state['dir'] = target
        render()

    def entry_button(label: str, icon: str, on_click) -> None:
        ui.button(label, icon=icon, on_click=on_click) \
            .props('flat no-caps align=left dense') \
            .classes('w-full text-left')

    def render() -> None:
        title.text = str(state['dir'])
        listing.clear()
        with listing:
            parent = state['dir'].parent
            if parent != state['dir']:
                entry_button('..', 'arrow_upward', lambda: navigate(parent))
            entries, error = browse.list_directory(state['dir'], show_hidden.value, dirs_only)
            if error:
                ui.label(error).classes('text-negative text-sm p-2')
                return
            for entry in entries:
                if entry.is_dir():
                    entry_button(entry.name, 'folder', lambda e=entry: navigate(e))
                else:
                    entry_button(entry.name, 'description', lambda e=entry: dialog.submit(str(e)))

    show_hidden.on_value_change(lambda _: render())
    render()
    return await dialog
