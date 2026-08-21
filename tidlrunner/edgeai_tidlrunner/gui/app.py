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

"""NiceGUI frontend for tidlrunner - builds the form from the CLI option specs."""

import argparse
import os
import re
import shlex
from typing import Dict, Optional
from urllib.parse import quote

from fastapi import HTTPException
from fastapi.responses import FileResponse
from nicegui import app as fastapi_app
from nicegui import ui

from edgeai_tidlrunner.version import __version__
from . import fields
from .pathpicker import choose_path
from .process import CommandRunner, cli_prefix

REPORT_ROUTE = '/tidlrunner/modelinspector'

# printed by GenerateModelInspectorHTML when a report is written
_REPORT_LINE = re.compile(r'HTML generation successful\.\s*Output at:\s*(\S+)')

# only paths reported by a run may be served, so the route cannot be used to
# read arbitrary files
_served_reports: set = set()


@fastapi_app.get(REPORT_ROUTE)
def _serve_report(path: str) -> FileResponse:
    if path not in _served_reports or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail='report not found')
    return FileResponse(path, media_type='text/html')


def _is_true(text: str) -> bool:
    return str(text).strip().lower() in ('1', 'true', 'yes')


def _number_to_text(value, kind: str) -> str:
    if value is None or value == '':
        return ''
    if kind == 'int':
        return str(int(value))
    return str(value)


class RunnerPage:
    """Per-client state and widgets."""

    def __init__(self) -> None:
        self.commands = fields.command_names()
        self.command = 'compile' if 'compile' in self.commands else self.commands[0]
        self.values: Dict[str, str] = fields.default_values(self.command)
        self.cwd = os.getcwd()
        self.runner = CommandRunner()
        self.log: ui.log = None
        self.command_preview: ui.label = None
        self.run_button: ui.button = None
        self.stop_button: ui.button = None
        self.status: ui.label = None
        self.report: Optional[str] = None
        self.report_select: ui.select = None
        self.report_view: ui.element = None
        self.report_placeholder: ui.label = None

    # ------------------------------------------------------------------ state

    def argv(self):
        return cli_prefix() + fields.build_argv(self.command, self.values)

    def refresh_preview(self) -> None:
        self.command_preview.text = shlex.join(self.argv())

    def set_value(self, name: str, text: str) -> None:
        self.values[name] = text
        self.refresh_preview()

    def select_command(self, command: str) -> None:
        # keep any value the user explicitly changed, if the new command has that option
        overrides = {name: value for name, value in self.values.items()
                     if value != fields.default_values(self.command).get(name)}
        self.command = command
        self.values = fields.default_values(command)
        for name, value in overrides.items():
            if name in self.values:
                self.values[name] = value
        self.form.refresh()
        self.refresh_preview()

    def reset(self) -> None:
        self.values = fields.default_values(self.command)
        self.form.refresh()
        self.refresh_preview()
        ui.notify('options reset to defaults')

    # ----------------------------------------------------------------- widgets

    def build_field(self, spec: fields.Field) -> None:
        value = self.values.get(spec.name, spec.default_text)
        tooltip = f'{spec.help}\n[{spec.dest}]'.strip()

        if spec.kind == 'bool':
            widget = ui.switch(spec.name, value=_is_true(value),
                               on_change=lambda e, n=spec.name: self.set_value(n, '1' if e.value else '0'))
            widget.classes('w-full')
        elif spec.kind == 'select':
            options = list(spec.choices)
            if value and value not in options:
                options.insert(0, value)
            widget = ui.select(options, value=value or None, label=spec.name, with_input=True,
                               on_change=lambda e, n=spec.name: self.set_value(n, '' if e.value is None else str(e.value)))
            widget.props('dense outlined').classes('w-full')
        elif spec.kind in ('int', 'float'):
            number = None if value == '' else (int(float(value)) if spec.kind == 'int' else float(value))
            widget = ui.number(label=spec.name, value=number,
                               precision=0 if spec.kind == 'int' else None,
                               on_change=lambda e, n=spec.name, k=spec.kind: self.set_value(n, _number_to_text(e.value, k)))
            widget.props('dense outlined').classes('w-full')
        elif spec.browse:
            with ui.row().classes('w-full items-center no-wrap gap-1'):
                widget = ui.input(label=spec.name, value=value,
                                  on_change=lambda e, n=spec.name: self.set_value(n, e.value or ''))
                widget.props('dense outlined').classes('grow')
                ui.button(icon='folder_open',
                          on_click=lambda _, w=widget, s=spec: self.browse(w, s)) \
                    .props('flat dense').tooltip(f'browse for {spec.browse}')
        else:
            widget = ui.input(label=spec.name, value=value,
                              on_change=lambda e, n=spec.name: self.set_value(n, e.value or ''))
            widget.props('dense outlined').classes('w-full')

        if tooltip:
            widget.tooltip(tooltip)

    async def browse(self, widget: ui.input, spec: fields.Field) -> None:
        start = widget.value or self.cwd
        selected = await choose_path(start, dirs_only=(spec.browse == 'dir'))
        if selected:
            widget.value = selected
            self.set_value(spec.name, selected)

    @ui.refreshable_method
    def form(self) -> None:
        groups = fields.grouped_fields(self.command)
        main = groups.pop('Main', [])
        if main:
            with ui.card().classes('w-full'):
                ui.label('Main').classes('text-subtitle2 text-weight-medium')
                with ui.grid(columns=2).classes('w-full gap-2'):
                    for spec in main:
                        self.build_field(spec)
        for group_name, specs in groups.items():
            with ui.expansion(f'{group_name}  ({len(specs)})').classes('w-full border rounded'):
                with ui.grid(columns=2).classes('w-full gap-2 p-2'):
                    for spec in specs:
                        self.build_field(spec)

    # ------------------------------------------------------- model inspector

    def build_inspector(self) -> None:
        with ui.column().classes('w-full h-full gap-2 p-2').style('min-height: 0'):
            with ui.row().classes('w-full items-center no-wrap gap-2'):
                self.report_select = ui.select({}, label='report',
                                               on_change=lambda e: self.show_report(e.value))
                self.report_select.props('dense outlined').classes('grow')
                ui.button(icon='open_in_new', on_click=self.open_report_tab) \
                    .props('flat dense').tooltip('open in a new browser tab')
            self.report_placeholder = ui.label(
                'no model inspector report yet - run compile or inspect first') \
                .classes('text-caption text-grey p-2')
            self.report_view = ui.element('iframe') \
                .classes('w-full grow rounded').style('border: 0; min-height: 0')
        self.show_report(None)

    def report_url(self, path: str) -> str:
        return f'{REPORT_ROUTE}?path={quote(path)}'

    def show_report(self, path: Optional[str]) -> None:
        self.report = path
        if self.report_select.value != path:
            self.report_select.value = path
        self.report_placeholder.set_visibility(not path)
        self.report_view.set_visibility(bool(path))
        if path:
            self.report_view.props(f'src="{self.report_url(path)}"')

    def open_report_tab(self) -> None:
        if not self.report:
            ui.notify('no report selected', type='warning')
            return
        ui.navigate.to(self.report_url(self.report), new_tab=True)

    def register_report(self, path: str) -> None:
        """Add a report a run just announced in the log, and show it."""
        path = os.path.abspath(os.path.join(self.cwd, path))
        if not os.path.isfile(path):
            return
        _served_reports.add(path)
        options = dict(self.report_select.options or {})
        options[path] = os.path.relpath(path, self.cwd)
        self.report_select.options = options
        self.report_select.update()
        self.show_report(path)

    # ------------------------------------------------------------------- run

    def start_run(self) -> None:
        if self.runner.running:
            ui.notify('a run is already in progress', type='warning')
            return
        argv = self.argv()
        self.log.push(f'$ {shlex.join(argv)}')
        try:
            self.runner.start(argv, cwd=self.cwd)
        except OSError as exception:
            self.log.push(f'ERROR: could not start: {exception}')
            ui.notify(f'could not start: {exception}', type='negative')
            return
        self.update_run_state('running')

    def stop_run(self) -> None:
        self.runner.stop()
        self.log.push('INFO: stop requested')

    def update_run_state(self, state: str) -> None:
        running = state == 'running'
        self.run_button.set_enabled(not running)
        self.stop_button.set_enabled(running)
        self.status.text = state

    def poll(self) -> None:
        for kind, payload in self.runner.drain():
            if kind == 'line':
                self.log.push(payload)
                match = _REPORT_LINE.search(payload)
                if match:
                    self.register_report(match.group(1))
            else:
                self.log.push(f'--- finished with exit code {payload} ---')
                self.update_run_state('done' if payload == 0 else f'failed ({payload})')
                ui.notify('run finished' if payload == 0 else f'run failed (exit {payload})',
                          type='positive' if payload == 0 else 'negative')

    # ------------------------------------------------------------------ page

    def build(self) -> None:
        with ui.header().classes('items-center justify-between py-2'):
            ui.label(f'tidlrunner {__version__}').classes('text-h6')
            with ui.row().classes('items-center gap-3'):
                ui.select(self.commands, value=self.command, label='command',
                          on_change=lambda e: self.select_command(e.value)) \
                    .props('dense outlined dark').classes('w-44')
                dark = ui.dark_mode()
                ui.switch('dark', on_change=lambda e: dark.enable() if e.value else dark.disable())

        with ui.splitter(value=52).classes('w-full').style('height: calc(100vh - 6rem)') as splitter:
            with splitter.before:
                with ui.column().classes('w-full h-full overflow-auto p-3 gap-3'):
                    with ui.row().classes('w-full items-center no-wrap gap-1'):
                        cwd_input = ui.input(label='working directory', value=self.cwd,
                                             on_change=lambda e: self.set_cwd(e.value))
                        cwd_input.props('dense outlined').classes('grow')
                        ui.button(icon='folder_open',
                                  on_click=lambda _: self.browse_cwd(cwd_input)).props('flat dense')
                        ui.button('Reset options', icon='restart_alt', on_click=self.reset).props('flat dense no-caps')
                    self.form()

            with splitter.after:
                with ui.column().classes('w-full h-full p-3 gap-2'):
                    with ui.row().classes('w-full items-center gap-2'):
                        self.run_button = ui.button('Run', icon='play_arrow', on_click=self.start_run)
                        self.stop_button = ui.button('Stop', icon='stop', on_click=self.stop_run) \
                            .props('outline color=negative')
                        ui.button('Clear log', icon='delete_sweep',
                                  on_click=lambda: self.log.clear()).props('flat no-caps')
                        ui.space()
                        self.status = ui.label('idle').classes('text-caption text-grey')
                    self.command_preview = ui.label().classes(
                        'w-full font-mono text-xs bg-grey-2 dark:bg-grey-9 rounded p-2 break-all select-all')

                    with ui.tabs().classes('w-full').props('dense align=left') as tabs:
                        log_tab = ui.tab('Log', icon='terminal')
                        inspector_tab = ui.tab('Model Inspector', icon='insights')
                    # quasar's panel wrapper is auto-height, so a percentage-height
                    # iframe collapses unless the chain is forced to 100%
                    ui.add_css('''
                        .tidl-panels { min-height: 0; }
                        .tidl-panels > .q-panel-parent { height: 100%; }
                        .tidl-panels .q-tab-panel { height: 100%; }
                    ''')
                    with ui.tab_panels(tabs, value=log_tab).classes('w-full grow tidl-panels') \
                            .props('keep-alive'):
                        with ui.tab_panel(log_tab).classes('p-0'):
                            self.log = ui.log(max_lines=20000).classes('w-full h-full font-mono text-xs')
                        with ui.tab_panel(inspector_tab).classes('p-0'):
                            self.build_inspector()

        self.stop_button.set_enabled(False)
        self.refresh_preview()
        ui.timer(0.2, self.poll)

    async def browse_cwd(self, widget: ui.input) -> None:
        selected = await choose_path(widget.value or self.cwd, dirs_only=True)
        if selected:
            widget.value = selected
            self.set_cwd(selected)

    def set_cwd(self, path: str) -> None:
        self.cwd = path or os.getcwd()
        self.refresh_preview()


@ui.page('/')
def index() -> None:
    RunnerPage().build()


def main(**kwargs) -> None:
    parser = argparse.ArgumentParser(prog='tidlrunner-gui', description='GUI frontend for tidlrunner')
    parser.add_argument('--host', default='127.0.0.1', help='interface to bind to')
    parser.add_argument('--port', type=int, default=8080, help='port to listen on')
    parser.add_argument('--native', action='store_true', help='open in a desktop window (needs pywebview)')
    parser.add_argument('--no-show', action='store_true', help='do not open a browser automatically')
    args = parser.parse_args()

    ui.run(
        host=args.host,
        port=args.port,
        title='tidlrunner',
        native=args.native,
        show=not (args.no_show or args.native),
        reload=False,
        favicon='\N{HIGH VOLTAGE SIGN}',
        **kwargs,
    )
