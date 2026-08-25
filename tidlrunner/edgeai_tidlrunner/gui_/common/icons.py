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

"""Brand colour and material icon names shared by the GUI frontends."""

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

# per-command accent hues, drawn from Quasar's classic multi-hue semantic
# palette (cyan/green/purple/teal/amber/indigo/blue/pink) - compile keeps the
# TI brand red since it's the flagship command
COMMAND_COLORS = {
    'compile': BRAND,
    'infer': '#31ccec',
    'evaluate': '#21ba45',
    'inspect': '#9c27b0',
    'analyze': '#26a69a',
    'surgery': '#f2c037',
    'extract': '#3f51b5',
    'report': '#1e88e5',
    'package': '#e91e63',
}

# per-group accent hues; 'Main' is excluded - it follows the current
# command's colour dynamically instead of a fixed one
GROUP_COLORS = {
    'Model surgery': '#f4511e',
    'Runtime options': '#f2c037',
    'Dataset': '#26a69a',
    'Preprocess': '#9c27b0',
    'Postprocess': '#e91e63',
    'Session': '#31ccec',
    'General': '#3f51b5',
    'Other': '#94a3b8',
}


def command_icon(command: str) -> str:
    return COMMAND_ICONS.get(command, 'tune')


def group_icon(group: str) -> str:
    return GROUP_ICONS.get(group, 'settings')


def command_color(command: str) -> str:
    return COMMAND_COLORS.get(command, BRAND)


def group_color(group: str) -> str:
    return GROUP_COLORS.get(group, BRAND)
