#!/usr/bin/env python3
"""Generate tidlrunner/docs/options.md from registered help metadata.

Run from the repo root:
    python tidlrunner/docs/generate_options_md.py
"""

import os
import sys

# Make sure the tidlrunner package is importable when run from the repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_TIDLRUNNER_PKG = os.path.join(_HERE, '..') 
if _TIDLRUNNER_PKG not in sys.path:
    sys.path.insert(0, _TIDLRUNNER_PKG)

# Importing settings_default triggers all register_help() decorators.
from edgeai_tidlrunner.runner.common.settings import settings_default  # noqa: F401
from edgeai_tidlrunner.runner.common.settings.settings_help import export_help_markdown

PREAMBLE = """\
# Options

These options have a short form that is easy to use on the command line and an
equivalent long form (the *Config Field*) that can be used in a config file.
To understand how short options map to structured config fields see the
**Config Field** column in the tables below.

Also see the
[default settings](../edgeai_tidlrunner/runner/common/settings/settings_default.py)
where these are defined and the
[example config files](../../data/configs/).

"""

OUTPUT_PATH = os.path.join(_HERE, 'options.md')

def main():
    body = export_help_markdown()
    content = PREAMBLE + body + '\n'
    with open(OUTPUT_PATH, 'w') as fh:
        fh.write(content)
    print(f'Written: {OUTPUT_PATH}')

if __name__ == '__main__':
    main()
