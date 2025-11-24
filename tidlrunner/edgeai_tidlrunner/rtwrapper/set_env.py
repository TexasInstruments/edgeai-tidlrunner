#!/usr/bin/env bash

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


import os
import sys
import subprocess
import glob


def set_env(**kwargs):
  import tidl_tools_package

  print("INFO: Setting environment variables for TIDL Runner: {}".format(kwargs))

  if 'TIDL_TOOLS_PATH' not in os.environ or 'LD_LIBRARY_PATH' not in os.environ:
    try:
      tidl_tools_path = tidl_tools_package.get_tidl_tools_path(kwargs['target_device'])
      tidl_tools_path = os.path.abspath(tidl_tools_path)
      os.environ['TIDL_TOOLS_PATH'] = tidl_tools_path
      print('INFO: TIDL_TOOLS_PATH is set to:', os.environ['TIDL_TOOLS_PATH'])
      os.environ['LD_LIBRARY_PATH'] = ":" + tidl_tools_path + ":" + os.environ.get('LD_LIBRARY_PATH', '')        
      print('INFO: LD_LIBRARY_PATH is set to:', os.environ['LD_LIBRARY_PATH'])      
    except:
      raise RuntimeError("tidl_tools_package not found. Please ensure it is installed correctly.")

  if 'TIDL_RT_ONNX_VARDIM' not in os.environ:
    print("INFO: setting TIDL_RT_ONNX_VARDIM to 1")
    # This is needed for ONNX models with variable input dimensions
    # It allows the TIDL runtime to handle variable dimensions in ONNX models
    os.environ['TIDL_RT_ONNX_VARDIM'] = "1"
  
  if 'TIDL_RT_PERFSTATS' not in os.environ:
    print("INFO: setting TIDL_RT_PERFSTATS to 1")
    # This is needed for performance statistics collection
    # It allows the TIDL runtime to collect performance statistics
    os.environ['TIDL_RT_PERFSTATS'] = "1"

  if 'TIDL_RT_DDR_STATS' not in os.environ:
    print("INFO: setting TIDL_RT_DDR_STATS to 1")
    # This is needed for DDR memory statistics collection
    # It allows the TIDL runtime to collect DDR memory statistics
    # This is useful for debugging and performance analysis
    os.environ['TIDL_RT_DDR_STATS'] = "1"

  ####################################################################
  # optional: check if AVX instructions are available in the machine
  # by default AVX is enabled - setting this TIDL_RT_AVX_REF flag to "0" wil disable AVX    
  if 'TIDL_RT_AVX_REF' not in os.environ:
    try:
        print(f"INFO: model compilation in PC can use AVX instructions (if it is available)")
        proc_cpuinfo_commmand = "cat /proc/cpuinfo|grep avx|wc|tail -n 1|awk '{print $1;}'"
        proc = subprocess.Popen([proc_cpuinfo_commmand], stdout=subprocess.PIPE, shell=True)
        out_ret, err_ret = proc.communicate()
        num_avx_cores = int(out_ret)
        os.environ['TIDL_RT_AVX_REF'] = '1' if num_avx_cores > 0 else '0'
        print(f'INFO: CPU cores with AVX support: {num_avx_cores}')
    except:
        print("INFO: could not find AVX instructions - AVX will not be used.")
    #
  
  if 'ARM64_GCC_PATH' not in os.environ:
    # Set the ARM64 GCC path for TVM compilation
    # This is needed for compiling models for ARM64 architecture
    os.environ['ARM64_GCC_PATH'] = os.path.join(os.environ['TIDL_TOOLS_PATH'], 'arm-gnu-toolchain-13.2.Rel1-x86_64-aarch64-none-linux-gnu')
    print('INFO: ARM64_GCC_PATH is set to:', os.environ['ARM64_GCC_PATH'])

  # if 'PYTHONPATH' not in os.environ:
  #   # Make sure the current directory is included in PYTHONPATH      
  #   os.environ['PYTHONPATH'] = ":" + os.environ.get('PYTHONPATH', '')
  #   # Print the PYTHONPATH to verify
  #   print('INFO: PYTHONPATH is set to:', os.environ['PYTHONPATH'])

  return kwargs

def update_tvm_artifacts(**kwargs):
    # tvmdlr artifacts are different for pc and evm device
    # point to the right artifact before this script executes
    print("INFO: settings the correct symlinks in tvmdlr compiled artifacts")

    target_device = kwargs['target_device']

    if 'ARTIFACTS_BASE_PATH' not in os.environ:
        # Set the default artifacts base path if not already set
        # This can be overridden by environment variables
        os.environ['ARTIFACTS_BASE_PATH'] = f'./work_dirs/compile/{target_device}/8bits'
    #
    print(f"INFO: ARTIFACTS_BASE_PATH is set to: {os.environ['ARTIFACTS_BASE_PATH']}")
    ARTIFACTS_BASE_PATH = os.environ['ARTIFACTS_BASE_PATH']

    artifacts_folders = glob.glob(f'{ARTIFACTS_BASE_PATH}/*_tvmdlr_*')
    cur_dir=os.getcwd()

    artifact_files = ["deploy_lib.so", "deploy_graph.json", "deploy_params.params"]

    for artifact_folder in artifacts_folders:
      print('INFO: Entering: ${artifact_folder}')
      os.chdir(f'{artifact_folder}/artifacts')
      for artifact_file in os.listdir('.'):
        # Check if the artifact file matches the expected names
        if artifact_file not in artifact_files:
          continue

        # Create symbolic links for each artifact file for the target machine
        print(f"INFO: Creating symbolic link for {artifact_file} for TARGET_MACHINE: {os.environ['TARGET_MACHINE']}")
        # Remove any existing symlink or file with the same name
        if os.path.islink(artifact_file) or os.path.exists(artifact_file):
          os.remove(artifact_file)

        # Create a symlink to the specific artifact for the target machine
        os.symlink(f'{artifact_file}.{os.environ["TARGET_MACHINE"]}', artifact_file)
      #
      os.chdir(cur_dir)


    # TIDL_ARTIFACT_SYMLINKS is used to indicate that the symlinks have been set to evm
    # This flag is not used by TIDL, but it can be used by other scripts or tools
    os.environ['TIDL_ARTIFACT_SYMLINKS']="1"


def set_environment(update_artifacts=True, **kwargs):
    # Set the environment variables for TIDL Runner
    kwargs = set_env(**kwargs)

    # Update the TVM artifacts symlinks based on the target device and machine
    if update_artifacts:
      update_tvm_artifacts(**kwargs)

    print("INFO: Environment variables for TIDL Runner have been set successfully.")


def restart_with_proper_environment(**kwargs):
    """
    Restart the process with correct environment.
    This should be called only if TIDL_TOOLS_PATH or LD_LIBRARY_PATH is not properly set
    """
    set_environment(**kwargs)

    # Prepare the new environment
    new_env = os.environ.copy()
    
    # Restart the current script with the new environment
    print("INFO: Restarting script with updated environment...")
    result = subprocess.run([sys.executable] + sys.argv, env=new_env, check=True)
    
    if result.returncode != 0:
        print("ERROR: Failed to restart the script with updated environment.")
        sys.exit(result.returncode)
    else:
        sys.exit(0)


def main():
    set_environment()


if __name__ == "__main__":
  # This script sets up the environment variables for TIDL Runner
  # It configures paths for TIDL tools, artifacts, and other necessary settings
  # It also updates the TVM artifacts symlinks based on the target device and machine
  # This is essential for running TIDL models on different platforms
  # Ensure that the TIDL tools package is installed and accessible in the environment
  # The script can be run directly or imported as a module in other scripts
  # It is designed to be run in a Linux environment with Python 3.x
  # The script assumes that the TIDL tools package is available in the specified path
  # It also assumes that the necessary environment variables are set correctly
  # If any required environment variable is not set, it will use default values  
  main()
  