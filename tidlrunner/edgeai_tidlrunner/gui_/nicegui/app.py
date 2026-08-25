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
import shlex
import time
from typing import Dict, List, Optional
from urllib.parse import quote

from fastapi import HTTPException
from fastapi.responses import FileResponse
from nicegui import app as fastapi_app
from nicegui import ui

from edgeai_tidlrunner.version import __version__
from ..common import fields
from ..common import icons
from ..common import reports
from ..common.process import CommandRunner, cli_prefix
from . import theme
from .pathpicker import choose_path

REPORT_ROUTE = '/tidlrunner/modelinspector'

# only paths reported by a run may be served, so the route cannot be used to
# read arbitrary files
_served_reports: set = set()


@fastapi_app.get(REPORT_ROUTE)
def _serve_report(path: str) -> FileResponse:
    if path not in _served_reports or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail='report not found')
    return FileResponse(path, media_type='text/html')


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f'{seconds:.1f}s'
    minutes, remainder = divmod(int(seconds), 60)
    return f'{minutes}m {remainder:02d}s'


class RunnerPage:
    """Per-client state and widgets."""

    def __init__(self) -> None:
        self.commands = fields.command_names()
        self.command = 'compile' if 'compile' in self.commands else self.commands[0]
        self.values: Dict[str, str] = fields.default_values(self.command)
        self.group_selected: Dict[str, str] = fields.default_group_selection(self.command)
        self.cwd = os.getcwd()
        self.runner = CommandRunner()
        self.log: ui.log = None
        self.progress_label: Optional[ui.label] = None
        self.command_preview: ui.label = None
        self.run_button: ui.button = None
        self.stop_button: ui.button = None
        self.status: ui.badge = None
        self.status_dot: ui.element = None
        self.spinner: ui.spinner = None
        self.dark: ui.dark_mode = None
        self.report: Optional[str] = None
        self.report_select: ui.select = None
        self.report_view: ui.element = None
        self.report_placeholder: ui.element = None
        self.run_started_at: Optional[float] = None
        self.stat_exit: ui.chip = None
        self.stat_duration: ui.chip = None
        self.stat_reports: ui.chip = None

    # ------------------------------------------------------------------ state

    def argv(self):
        return cli_prefix() + fields.build_argv(self.command, self.values, self.group_selected)

    def refresh_preview(self) -> None:
        self.command_preview.text = shlex.join(self.argv())

    def set_value(self, name: str, text: str) -> None:
        self.values[name] = text
        self.refresh_preview()

    def set_group_selection(self, group_name: str, name: str) -> None:
        self.group_selected[group_name] = name
        self.form.refresh()
        self.refresh_preview()

    def select_command(self, command: str) -> None:
        # keep any value the user explicitly changed, if the new command has that option
        overrides = {name: value for name, value in self.values.items()
                     if value != fields.default_values(self.command).get(name)}
        prev_group_selected = self.group_selected
        self.command = command
        self.values = fields.default_values(command)
        for name, value in overrides.items():
            if name in self.values:
                self.values[name] = value
        new_groups = fields.arg_groups(command)
        self.group_selected = fields.default_group_selection(command)
        for group_name, selected_name in prev_group_selected.items():
            if group_name in new_groups and any(m.name == selected_name for m in new_groups[group_name]):
                self.group_selected[group_name] = selected_name
        self.form.refresh()
        self.refresh_preview()

    def reset(self) -> None:
        self.values = fields.default_values(self.command)
        self.group_selected = fields.default_group_selection(self.command)
        self.form.refresh()
        self.refresh_preview()
        ui.notify('options reset to defaults')

    # ----------------------------------------------------------------- widgets

    def build_field(self, spec: fields.Field) -> None:
        value = self.values.get(spec.name, spec.default_text)
        label = f'{spec.name} ({spec.help})' if spec.help else spec.name
        tooltip = f'{spec.help}\n[{spec.dest}]'.strip()

        if spec.kind == 'bool':
            widget = ui.switch(label, value=fields.is_true(value),
                               on_change=lambda e, n=spec.name: self.set_value(n, '1' if e.value else '0'))
            widget.classes('w-full')
        elif spec.kind == 'select':
            options = list(spec.choices)
            if value and value not in options:
                options.insert(0, value)
            widget = ui.select(options, value=value or None, label=label, with_input=True,
                               on_change=lambda e, n=spec.name: self.set_value(n, '' if e.value is None else str(e.value)))
            widget.props('dense outlined').classes('w-full')
        elif spec.kind in ('int', 'float'):
            number = None if value == '' else (int(float(value)) if spec.kind == 'int' else float(value))
            widget = ui.number(label=label, value=number,
                               precision=0 if spec.kind == 'int' else None,
                               on_change=lambda e, n=spec.name, k=spec.kind: self.set_value(n, fields.number_to_text(e.value, k)))
            widget.props('dense outlined').classes('w-full')
        elif spec.browse:
            with ui.row().classes('w-full items-center no-wrap gap-1'):
                widget = ui.input(label=label, value=value,
                                  on_change=lambda e, n=spec.name: self.set_value(n, e.value or ''))
                widget.props('dense outlined').classes('grow')
                ui.button(icon='folder_open',
                          on_click=lambda _, w=widget, s=spec: self.browse(w, s)) \
                    .props('flat dense').tooltip(f'browse for {spec.browse}')
        else:
            widget = ui.input(label=label, value=value,
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

    def build_arg_group(self, members: List[fields.Field]) -> None:
        """Radio selector for a set of mutually exclusive arguments (argparse 'group')."""
        group_name = members[0].arg_group
        selected = self.group_selected.get(group_name, members[0].name)
        options = {m.name: (f'{m.name} ({m.help})' if m.help else m.name) for m in members}
        with ui.column().classes('w-full gap-1'):
            ui.radio(options, value=selected,
                     on_change=lambda e, g=group_name: self.set_group_selection(g, e.value)) \
                .props('inline dense')
            for m in members:
                if m.name == selected:
                    self.build_field(m)

    def build_fields(self, specs: List[fields.Field]) -> None:
        arg_groups = fields.arg_groups(self.command)
        rendered = set()
        for spec in specs:
            if spec.arg_group in arg_groups:
                if spec.arg_group in rendered:
                    continue
                rendered.add(spec.arg_group)
                self.build_arg_group(arg_groups[spec.arg_group])
            else:
                self.build_field(spec)

    @ui.refreshable_method
    def form(self) -> None:
        groups = fields.grouped_fields(self.command)
        main = groups.pop('Main', [])
        if main:
            accent = icons.command_color(self.command)
            with ui.element('div').classes('tidl-card w-full p-3').style(f'border-left-color: {accent}'):
                with ui.row().classes('items-center gap-2 q-mb-sm'):
                    ui.icon(icons.command_icon(self.command)).style(f'color: {accent}')
                    ui.label(self.command).classes('text-subtitle2 text-weight-bold')
                    ui.badge(f'{len(main)} main options').props('outline color=grey-7')
                ui.separator().classes('q-mb-sm')
                with ui.column().classes('w-full gap-2'):
                    self.build_fields(main)
        for group_name, specs in groups.items():
            with ui.expansion(group_name, icon=icons.group_icon(group_name)) \
                    .classes('tidl-group w-full').props('dense-toggle expand-separator') \
                    .style(f'--tidl-group-accent: {icons.group_color(group_name)}'):
                with ui.column().classes('w-full gap-2 p-3'):
                    self.build_fields(specs)

    # ------------------------------------------------------- model inspector

    def build_inspector(self) -> None:
        with ui.column().classes('w-full h-full gap-2 p-2').style('min-height: 0'):
            with ui.row().classes('w-full items-center no-wrap gap-2'):
                ui.icon('description').classes('tidl-dim')
                self.report_select = ui.select({}, label='report',
                                               on_change=lambda e: self.show_report(e.value))
                self.report_select.props('dense outlined options-dense').classes('grow')
                ui.button(icon='open_in_new', on_click=self.open_report_tab) \
                    .props('flat dense round').tooltip('open in a new browser tab')
            with ui.column().classes('w-full items-center gap-3 p-8') as self.report_placeholder:
                with ui.element('div').classes('tidl-empty-icon'):
                    ui.icon('insights').classes('text-3xl text-primary')
                ui.label('No model inspector report yet').classes('text-subtitle2')
                ui.label('Run compile or inspect to generate one').classes('text-caption tidl-dim')
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
        _served_reports.add(path)
        options = dict(self.report_select.options or {})
        options[path] = os.path.relpath(path, self.cwd)
        self.report_select.options = options
        self.report_select.update()
        self.show_report(path)
        self.stat_reports.text = f'reports: {len(options)}'

    # ------------------------------------------------------------------- run

    def start_run(self) -> None:
        if self.runner.running:
            ui.notify('a run is already in progress', type='warning')
            return
        argv = self.argv()
        self.push_log_line(f'$ {shlex.join(argv)}')
        try:
            self.runner.start(argv, cwd=self.cwd)
        except OSError as exception:
            self.push_log_line(f'ERROR: could not start: {exception}')
            ui.notify(f'could not start: {exception}', type='negative')
            return
        self.run_started_at = time.time()
        self.stat_exit.text = 'exit: …'
        self.stat_exit.props('outline color=grey-7')
        self.stat_duration.text = 'time: …'
        self.stat_duration.props('outline color=grey-7')
        self.update_run_state('running')

    def stop_run(self) -> None:
        self.runner.stop()
        self.push_log_line('INFO: stop requested')

    def update_run_state(self, state: str) -> None:
        running = state == 'running'
        self.run_button.set_enabled(not running)
        self.stop_button.set_enabled(running)
        self.spinner.set_visibility(running)
        colour = {'running': 'primary', 'idle': 'grey-7', 'done': 'positive'}.get(state, 'negative')
        self.status.text = state
        self.status.props(f'outline color={colour}')
        dot_class = {'running': 'tidl-dot--running', 'idle': '', 'done': 'tidl-dot--done'}.get(state, 'tidl-dot--failed')
        self.status_dot.classes(remove='tidl-dot--running tidl-dot--done tidl-dot--failed', add=dot_class)

    def clear_log(self) -> None:
        self.log.clear()
        self.progress_label = None

    def push_log_line(self, text: str) -> None:
        """Commit a real line to the log, ending any in-place progress update."""
        self.log.push(text)
        self.progress_label = None

    def push_progress(self, text: str) -> None:
        """Show a carriage-return-terminated update (e.g. a tqdm bar) in place,
        overwriting the previous one instead of growing the log."""
        if self.progress_label is None or self.progress_label.is_deleted:
            with self.log:
                self.progress_label = ui.label(text)
        else:
            self.progress_label.text = text

    def poll(self) -> None:
        for kind, payload in self.runner.drain():
            if kind == 'line':
                self.push_log_line(payload)
                report_path = reports.resolve_report_line(payload, self.cwd)
                if report_path:
                    self.register_report(report_path)
            elif kind == 'progress':
                self.push_progress(payload)
            else:
                self.push_log_line(f'--- finished with exit code {payload} ---')
                self.update_run_state('done' if payload == 0 else f'failed ({payload})')
                duration = time.time() - self.run_started_at if self.run_started_at else None
                self.stat_exit.text = f'exit: {payload}'
                self.stat_exit.props(f'outline color={"positive" if payload == 0 else "negative"}')
                if duration is not None:
                    self.stat_duration.text = f'time: {_format_duration(duration)}'
                    self.stat_duration.props('outline color=grey-7')
                ui.notify('run finished' if payload == 0 else f'run failed (exit {payload})',
                          type='positive' if payload == 0 else 'negative')

    # ------------------------------------------------------------------ page

    def build(self) -> None:
        theme.apply()

        with ui.header().classes('tidl-header items-center justify-between px-4 py-2'):
            with ui.row().classes('items-center gap-3 no-wrap'):
                ui.icon('developer_board').classes('text-3xl text-red-5')
                with ui.column().classes('gap-0'):
                    ui.label('tidlrunner').classes('tidl-brand text-h6 text-weight-bold leading-none')
                    ui.label('TIDL model compilation and evaluation') \
                        .classes('text-caption text-grey-5 leading-none')
                ui.badge(__version__).props('outline color=white').classes('self-center')
                with ui.row().classes('tidl-status-chip self-center'):
                    self.status_dot = ui.element('div').classes('tidl-dot').tooltip('run status')
            with ui.row().classes('items-center gap-3 no-wrap'):
                self.dark = ui.dark_mode(False)
                ui.button(icon='dark_mode', on_click=self.toggle_dark) \
                    .props('flat round dense color=white').tooltip('toggle dark mode')

        with ui.splitter(value=52).classes('w-full').style('height: calc(100vh - 5.5rem)') as splitter:
            with splitter.before:
                with ui.column().classes('tidl-scroll w-full h-full overflow-auto p-4 gap-3'):
                    with ui.element('div').classes('tidl-card w-full p-3'):
                        with ui.row().classes('w-full items-center no-wrap gap-2'):
                            ui.icon('folder').classes('tidl-dim')
                            cwd_input = ui.input(label='working directory', value=self.cwd,
                                                 on_change=lambda e: self.set_cwd(e.value))
                            cwd_input.props('dense outlined').classes('grow')
                            ui.button(icon='folder_open',
                                      on_click=lambda _: self.browse_cwd(cwd_input)) \
                                .props('flat dense round').tooltip('browse')
                            ui.button(icon='restart_alt', on_click=self.reset) \
                                .props('flat dense round').tooltip('reset options to defaults')
                    with ui.row().classes('w-full items-center no-wrap gap-2'):
                        ui.select(self.commands, value=self.command, label='command',
                                  on_change=lambda e: self.select_command(e.value)) \
                            .props('dense outlined options-dense').classes('grow')
                        self.run_button = ui.button('Run', icon='play_arrow', on_click=self.start_run) \
                            .props('unelevated no-caps').classes('tidl-run px-4')
                        self.stop_button = ui.button('Stop', icon='stop', on_click=self.stop_run) \
                            .props('outline no-caps color=negative').classes('tidl-stop')
                    self.form()

            with splitter.after:
                with ui.column().classes('w-full h-full p-4 gap-3').style('min-height: 0'):
                    with ui.row().classes('tidl-preview tidl-scroll grow items-start no-wrap gap-1 text-xs p-3'):
                        ui.label('$').classes('tidl-prompt')
                        self.command_preview = ui.label().classes('break-all select-all grow')
                        ui.button(icon='content_copy', on_click=self.copy_command) \
                            .props('flat dense round color=grey-4').tooltip('copy command')

                    with ui.row().classes('tidl-tabs w-full items-center no-wrap gap-2'):
                        with ui.tabs().props('dense align=left inline-label') as tabs:
                            log_tab = ui.tab('Log', icon='terminal')
                            inspector_tab = ui.tab('Model Inspector', icon='insights')
                        ui.space()
                        self.stat_exit = ui.chip('exit: -', icon='flag').props('outline color=grey-7')
                        self.stat_duration = ui.chip('time: -', icon='schedule').props('outline color=grey-7')
                        self.stat_reports = ui.chip('reports: 0', icon='insights').props('outline color=grey-7')
                        self.spinner = ui.spinner('dots', size='1.4rem').classes('text-primary')
                        self.status = ui.badge('idle').props('outline color=grey-7')
                        ui.button(icon='delete_sweep', on_click=self.clear_log) \
                            .props('flat dense round').tooltip('clear log')

                    with ui.tab_panels(tabs, value=log_tab).classes('w-full grow tidl-panels') \
                            .props('keep-alive'):
                        with ui.tab_panel(log_tab).classes('p-0'):
                            self.log = ui.log(max_lines=20000) \
                                .classes('tidl-log tidl-scroll w-full h-full font-mono text-xs p-2')
                        with ui.tab_panel(inspector_tab).classes('p-0'):
                            self.build_inspector()

        self.stop_button.set_enabled(False)
        self.spinner.set_visibility(False)
        self.refresh_preview()
        ui.timer(0.2, self.poll)

    def toggle_dark(self) -> None:
        self.dark.toggle()

    def copy_command(self) -> None:
        ui.clipboard.write(self.command_preview.text)
        ui.notify('command copied', type='positive')

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
