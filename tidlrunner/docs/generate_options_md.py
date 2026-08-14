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

## Commandline and configfile options
* These options have a short form that is easy to use on the commandline and an equivalent long form (the *Config Field*) that can be used in a config file.
To understand how short options map to the config fields see the **Config Field** column in the tables below.
* The **Default** column shows the default value for each option.
* For boolean options, recommended values of flags are 0 or 1. (Other values such as True and False are also accepted, but can cause confusion due to the differences in how they are interpreted by argparse and by yaml loading.)
* Details and usage of Model Surgery can be seen in [tidl-onnx-model-optimizer](https://github.com/TexasInstruments/edgeai-tidl-tools/blob/master/model-tools/tidl-onnx-model-optimizer/README.md#usage)
* See [example config files](../../data/configs/) for a a variety of examples.
* For more details, check [default settings](../edgeai_tidlrunner/runner/common/settings/settings_default.py)


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
