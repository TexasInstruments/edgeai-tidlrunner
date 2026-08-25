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

"""Colours, icons and stylesheet shared by the tidlrunner GUI."""

from nicegui import ui

BRAND = '#cc0000'

COMMAND_ICONS = {
    'compile': 'memory',
    'infer': 'bolt',
    'evaluate': 'assessment',
    'inspect': 'insights',
    'analyze': 'query_stats',
    'surgery': 'healing',
    'extract': 'unarchive',
    'report': 'summarize',
    'package': 'inventory_2',
}

GROUP_ICONS = {
    'Main': 'tune',
    'Model surgery': 'healing',
    'Runtime options': 'memory',
    'Dataset': 'dataset',
    'Preprocess': 'filter_center_focus',
    'Postprocess': 'auto_fix_high',
    'Session': 'lan',
    'General': 'settings',
    'Other': 'more_horiz',
}

_CSS = '''
:root {
    --tidl-radius: 12px;
    --tidl-line: rgba(15, 23, 42, .10);
    --tidl-surface: #ffffff;
    --tidl-muted: #64748b;
}
.body--dark {
    --tidl-line: rgba(255, 255, 255, .12);
    --tidl-surface: #1a1f2b;
    --tidl-muted: #94a3b8;
}
body { background: #eef1f6; }
.body--dark body, .body--dark { background: #10141c; }

/* header: dark slate with a brand-red edge */
.tidl-header {
    background: linear-gradient(100deg, #171c26 0%, #232b3b 60%, #3a2230 100%);
    border-bottom: 2px solid var(--q-primary);
    box-shadow: 0 2px 14px rgba(0, 0, 0, .22);
}
.tidl-brand {
    letter-spacing: .5px;
    background: linear-gradient(90deg, #ffffff 0%, #ffb4b4 130%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}

.tidl-card {
    background: var(--tidl-surface);
    border: 1px solid var(--tidl-line);
    border-radius: var(--tidl-radius);
    box-shadow: 0 1px 2px rgba(15, 23, 42, .05);
}

/* collapsible option groups */
.tidl-group {
    background: var(--tidl-surface);
    border: 1px solid var(--tidl-line);
    border-radius: var(--tidl-radius);
    overflow: hidden;
    transition: border-color .15s ease, box-shadow .15s ease;
}
.tidl-group:hover { border-color: rgba(204, 0, 0, .35); }
.tidl-group .q-expansion-item__content { border-top: 1px solid var(--tidl-line); }

/* command preview reads as a shell line */
.tidl-preview {
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    background: #0e1420;
    color: #cbd5e1;
    border: 1px solid rgba(255, 255, 255, .08);
    border-radius: var(--tidl-radius);
    line-height: 1.45;
    max-height: 5.5rem;
    overflow: auto;
}

/* log keeps default colours - only the frame is themed */
.tidl-log {
    border: 1px solid var(--tidl-line);
    border-radius: var(--tidl-radius);
}

/* quasar's panel wrapper is auto-height, so a percentage-height iframe
   collapses unless the whole chain is forced to 100% */
.tidl-panels { min-height: 0; }
.tidl-panels > .q-panel-parent { height: 100%; }
.tidl-panels .q-tab-panel { height: 100%; }

.tidl-tabs { border-bottom: 1px solid var(--tidl-line); }

.tidl-scroll::-webkit-scrollbar { width: 10px; height: 10px; }
.tidl-scroll::-webkit-scrollbar-thumb {
    background: rgba(100, 116, 139, .38);
    border-radius: 8px;
    border: 2px solid transparent;
    background-clip: content-box;
}
.tidl-scroll::-webkit-scrollbar-thumb:hover { background-color: rgba(100, 116, 139, .6); }
.tidl-scroll::-webkit-scrollbar-track { background: transparent; }

.tidl-dim { color: var(--tidl-muted); }
'''


def apply() -> None:
    ui.colors(primary=BRAND, secondary='#334155', accent='#0ea5e9',
              positive='#16a34a', negative='#dc2626', warning='#d97706', info='#0284c7',
              dark='#1a1f2b', dark_page='#10141c')
    ui.add_css(_CSS)
