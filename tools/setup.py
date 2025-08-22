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


import sys
import os
import importlib
from setuptools import setup, Extension, find_packages

from setuptools.command.develop import develop
from setuptools.command.install import install
from setuptools.command.build_py import build_py


###############################################################################
def import_file_folder(file_or_folder_name):
    if file_or_folder_name.endswith(os.sep):
        file_or_folder_name = file_or_folder_name[:-1]
    #
    parent_folder = os.path.dirname(file_or_folder_name)
    basename = os.path.splitext(os.path.basename(file_or_folder_name))[0]
    sys.path.insert(0, parent_folder)
    imported_module = importlib.import_module(basename, __name__)
    sys.path.pop(0)
    return imported_module


download_py = import_file_folder(os.path.join(os.path.dirname(__file__), 'tidl_tools_package', 'download.py'))

###############################################################################
# Custom command classes to trigger TIDL tools download
class PostBuildCommand(build_py):
    """Post-build command for both regular and editable installs."""

    def run(self):
        print("DEBUG: PostBuildCommand.run() called")
        build_py.run(self)
        self.download_tidl_tools()


class PostDevelopCommand(develop):
    """Post-installation for development mode."""

    def run(self):
        print("DEBUG: PostDevelopCommand.run() called")
        develop.run(self)
        self.download_tidl_tools()


class PostInstallCommand(install):
    """Post-installation for installation mode."""

    def run(self):
        print("DEBUG: PostInstallCommand.run() called")
        install.run(self)
        self.download_tidl_tools()


def download_tidl_tools_hook():
    """Hook to download TIDL tools after installation"""
    tools_version = os.environ.get("TIDL_TOOLS_VERSION", download_py.TIDL_TOOLS_VERSION_DEFAULT)
    tools_type = os.environ.get("TIDL_TOOLS_TYPE", download_py.TIDL_TOOLS_TYPE_DEFAULT)

    print(f"INFO: Starting TIDL tools download (version: {tools_version}, type: {tools_type})")

    # Get install path
    try:
        import tidl_tools_package
        install_path = os.path.dirname(tidl_tools_package.__file__)
        print(f"INFO: Using installed package path: {install_path}")
    except ImportError:
        # Fallback to tidl_tools_package subdirectory of current directory
        install_path = os.path.join(os.path.dirname(__file__), 'tidl_tools_package')
        print(f"INFO: Using package source path: {install_path}")
        # raise RuntimeError("ERROR: tidl_tools_package is not installed. Please install it first.")

    try:
        download_py.setup_tidl_tools(install_path, tools_version, tools_type)
        print("INFO: TIDL tools download completed successfully!")
    except Exception as e:
        print(f"WARNING: TIDL tools download failed: {e}")
        print("You can manually download later by running:")
        print(
            f"TIDL_TOOLS_VERSION={tools_version} python -c \"from setup import setup_tidl_tools; setup_tidl_tools('{install_path}', '{tools_version}', '{tools_type}')\"")


# Add the hook to all command classes
PostBuildCommand.download_tidl_tools = staticmethod(download_tidl_tools_hook)
PostDevelopCommand.download_tidl_tools = staticmethod(download_tidl_tools_hook)
PostInstallCommand.download_tidl_tools = staticmethod(download_tidl_tools_hook)


def main():
    # this is a bare minimum setup to enable PostBuild Commands for download of tidl_tools
    # assuming that rest of the details are in pyproject.toml
    setup(
        cmdclass={
            'build_py': PostBuildCommand,
            'develop': PostDevelopCommand,
            'install': PostInstallCommand,
        },
    )


if __name__ == '__main__':
    main()
