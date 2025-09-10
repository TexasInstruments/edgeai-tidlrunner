# edgeai-tidl-runner

Bring Your Edge AI Models For Compilation and Inference (BYOM)

This package provides a wrapper over the core [TIDL model compilation and runtimes](https://github.com/TexasInstruments/edgeai-tidl-tools) to make  model compilation and inference interface easy to use. 

This will be installed as **edgeai_tidlrunner** Python package.

edgeai_tidlrunner package has two parts:

* **edgeai_tidlrunner.runner** (high level interface) - runner has additional pipeline functionalities such as data loaders and preprocess required to run the entire pipeline correctly. This is a high level interface that hides most of the details and provides a Pythonic and command line APIs. (Recommended for beginners)

* **edgeai_tidlrunner.rtwrapper** (advanced interface) - rtwrapper is a thin wrapper over the core OSRT and TIDL-RT runtimes - the wrapper is provided for ease of use and also to make the usage of various runtimes consistent. This is an advanced wrapper does not impose much restrictions on the usage and the full flexibility and functionality of the underlying runtimes are available to the user. 

<hr>
<hr>

## Setup

[Setup instructions](docs/setup.md)

<hr>
<hr>

[Getting started](docs/getting_started.md)

<hr>
<hr>

[Usage](docs/usage.md)

<hr>
<hr>

[Advanced Usage](docs/advanced_usage.md)

<hr>
<hr>
