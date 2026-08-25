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

"""Colours and stylesheet of the nicegui frontend - bold TI-branded, dark-first."""

from nicegui import ui

from ..common.icons import BRAND

_FONTS_HEAD = '''
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
'''

_CSS = '''
:root {
    --tidl-radius: 12px;
    --tidl-line: rgba(15, 23, 42, .10);
    --tidl-surface: #ffffff;
    --tidl-muted: #64748b;
    --tidl-sans: 'Sora', ui-sans-serif, system-ui, -apple-system, sans-serif;
    --tidl-mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
.body--dark {
    --tidl-line: rgba(255, 255, 255, .12);
    --tidl-surface: #1a1f2b;
    --tidl-muted: #94a3b8;
}
body {
    background:
        radial-gradient(circle at 10% 5%, rgba(204, 0, 0, .06), transparent 40%),
        radial-gradient(circle at 90% 10%, rgba(38, 166, 154, .07), transparent 42%),
        radial-gradient(circle at 50% 100%, rgba(156, 39, 176, .05), transparent 45%),
        #eef1f6;
    font-family: var(--tidl-sans);
}
.body--dark body, .body--dark {
    background:
        radial-gradient(circle at 15% -10%, rgba(204, 0, 0, .16), transparent 45%),
        radial-gradient(circle at 85% 8%, rgba(38, 166, 154, .10), transparent 40%),
        radial-gradient(circle at 50% 105%, rgba(156, 39, 176, .09), transparent 50%),
        #0c0f16;
}

/* header: dark slate with a glowing brand-red edge */
.tidl-header {
    background: linear-gradient(100deg, #10131a 0%, #1b2230 55%, #341620 100%);
    position: relative;
    box-shadow: 0 2px 20px rgba(0, 0, 0, .35);
}
.tidl-header::after {
    content: '';
    position: absolute;
    left: 0; right: 0; bottom: -2px;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--q-primary) 20%, #ff8a8a 50%, var(--q-primary) 80%, transparent);
    box-shadow: 0 0 14px 1px rgba(204, 0, 0, .65);
}
.tidl-brand {
    font-family: var(--tidl-sans);
    font-weight: 700;
    letter-spacing: .5px;
    background: linear-gradient(90deg, #ffffff 0%, #ffb4b4 130%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}

/* header status dot next to the version badge */
.tidl-status-chip { display: flex; align-items: center; gap: .4rem; }
.tidl-dot {
    width: .6rem; height: .6rem; border-radius: 50%;
    background: #64748b;
    transition: background-color .2s ease;
}
.tidl-dot--running { background: var(--q-primary); animation: tidl-pulse 1.1s ease-in-out infinite; }
.tidl-dot--done { background: #16a34a; }
.tidl-dot--failed { background: #dc2626; }
@keyframes tidl-pulse {
    0%   { box-shadow: 0 0 0 0 rgba(204, 0, 0, .55); }
    70%  { box-shadow: 0 0 0 7px rgba(204, 0, 0, 0); }
    100% { box-shadow: 0 0 0 0 rgba(204, 0, 0, 0); }
}

/* cards + collapsible option groups */
.tidl-card {
    background: var(--tidl-surface);
    border: 1px solid var(--tidl-line);
    border-left: 3px solid var(--q-primary);
    border-radius: var(--tidl-radius);
    box-shadow: 0 1px 3px rgba(15, 23, 42, .08), 0 10px 26px -16px rgba(15, 23, 42, .18);
}
.tidl-group {
    background: var(--tidl-surface);
    border: 1px solid var(--tidl-line);
    border-left: 3px solid transparent;
    border-radius: var(--tidl-radius);
    overflow: hidden;
    transition: border-color .18s ease, box-shadow .18s ease, transform .15s ease;
}
.tidl-group:hover {
    border-left-color: var(--tidl-group-accent, var(--q-primary));
    box-shadow: 0 14px 28px -18px rgba(204, 0, 0, .4);
    box-shadow: 0 14px 28px -18px color-mix(in srgb, var(--tidl-group-accent, var(--q-primary)) 55%, transparent);
    transform: translateY(-1px);
}
.tidl-group .q-icon:not(.q-expansion-item__toggle-icon) { color: var(--tidl-group-accent, var(--q-primary)); }
.tidl-group .q-expansion-item__content { border-top: 1px solid var(--tidl-line); }

/* run / stop actions */
.tidl-run {
    background: linear-gradient(135deg, #ff5252 0%, #cc0000 55%, #7a0000 100%) !important;
    box-shadow: 0 4px 14px rgba(204, 0, 0, .4);
    transition: transform .15s ease, box-shadow .15s ease;
}
.tidl-run:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(204, 0, 0, .55); }
.tidl-run:disabled { opacity: .4; box-shadow: none; }
.tidl-stop { transition: box-shadow .15s ease; }
.tidl-stop:hover:not(:disabled) { box-shadow: 0 0 0 3px rgba(220, 38, 38, .22); }

/* command preview reads as a glowing terminal chip */
.tidl-preview {
    font-family: var(--tidl-mono);
    background: #0a0e16;
    color: #d7dee8;
    border: 1px solid rgba(204, 0, 0, .28);
    border-radius: var(--tidl-radius);
    line-height: 1.5;
    max-height: 5.5rem;
    overflow: auto;
    box-shadow: inset 0 0 0 1px rgba(255, 255, 255, .02), 0 0 20px -8px rgba(204, 0, 0, .45);
}
.tidl-prompt { color: #ff6b6b; font-weight: 600; margin-right: .4em; }

/* log keeps default colours - only the frame and font are themed */
.tidl-log {
    border: 1px solid var(--tidl-line);
    border-radius: var(--tidl-radius);
    font-family: var(--tidl-mono);
}

/* quasar's panel wrapper is auto-height, so a percentage-height iframe
   collapses unless the whole chain is forced to 100% */
.tidl-panels { min-height: 0; }
.tidl-panels > .q-panel-parent { height: 100%; }
.tidl-panels .q-tab-panel { height: 100%; }

.tidl-tabs { border-bottom: 1px solid var(--tidl-line); }

/* model inspector empty state */
.tidl-empty-icon {
    width: 4.5rem; height: 4.5rem; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    background: radial-gradient(circle at 30% 30%, rgba(204, 0, 0, .2), rgba(204, 0, 0, .04));
    box-shadow: inset 0 0 0 1px rgba(204, 0, 0, .18);
}

/* path picker dialog */
.tidl-picker {
    border-radius: var(--tidl-radius);
    overflow: hidden;
}
.tidl-picker-crumbs { min-height: 2rem; }
.tidl-crumb {
    color: var(--tidl-muted);
    min-height: 0;
    padding: 2px 8px;
    font-size: .8rem;
    white-space: nowrap;
    flex-shrink: 0;
}
.tidl-crumb:last-child { color: inherit; font-weight: 600; }
.tidl-picker-row {
    border-radius: 8px;
    transition: background-color .15s ease;
}
.tidl-picker-row:hover { background: rgba(204, 0, 0, .08); }
.tidl-picker-row--folder .q-icon { color: var(--q-primary); }
.tidl-picker-row--file .q-icon, .tidl-picker-row--up .q-icon { color: var(--tidl-muted); }

.tidl-scroll::-webkit-scrollbar { width: 10px; height: 10px; }
.tidl-scroll::-webkit-scrollbar-thumb {
    background: rgba(100, 116, 139, .38);
    border-radius: 8px;
    border: 2px solid transparent;
    background-clip: content-box;
}
.tidl-scroll::-webkit-scrollbar-thumb:hover { background-color: rgba(204, 0, 0, .55); }
.tidl-scroll::-webkit-scrollbar-track { background: transparent; }

.tidl-dim { color: var(--tidl-muted); }
'''


def apply() -> None:
    ui.add_head_html(_FONTS_HEAD)
    ui.colors(primary=BRAND, secondary='#334155', accent='#0ea5e9',
              positive='#16a34a', negative='#dc2626', warning='#d97706', info='#0284c7',
              dark='#1a1f2b', dark_page='#0c0f16')
    ui.add_css(_CSS)
