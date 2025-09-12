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
import argparse
import shutil
import PIL
import PIL.Image
import numpy as np
import onnx
import glob

import edgeai_tidlrunner
from edgeai_tidlrunner.rtwrapper.options import presets


def preprocess_input(input_img_file):
    '''
    A simple example input image preprocessing function
    Args:
        input_img_file:

    Returns:
        resized, mean substracted and scaled image tencor
        Tensor size is in the format: 1, C, H, W
    '''
    width = 224
    height = 224
    input_mean=[123.675, 116.28, 103.53]
    input_scale=[0.017125, 0.017507, 0.017429]
    input_img = PIL.Image.open(input_img_file).convert("RGB").resize((width, height), PIL.Image.BILINEAR)
    input_data = np.expand_dims(input_img, axis=0)
    input_data = np.transpose(input_data, (0, 3, 1, 2))
    normalized_data = np.zeros(input_data.shape, dtype=np.float32)
    for mean, scale, ch in zip(input_mean, input_scale, range(input_data.shape[1])):
        normalized_data[:, ch, :, :] = (input_data[:, ch, :, :] - mean) * scale
    #
    return normalized_data


def run_compile(args, runtime_type, runtime_settings, model_path, artifacts_folder):
    '''
    Run the model compilation
    Args:
        runtime_settings:
        model_path:
        artifacts_folder:

    Returns:
        None
    '''

    calib_dataset = glob.glob(f'{args.data_path}/*.*')

    runtime_wrapper = runtime_type(
            model_path=model_path,
            **runtime_settings)

    for input_index in range(runtime_wrapper.get_runtime_options()['advanced_options:calibration_frames']):
        input_data = preprocess_input(calib_dataset[input_index])
        runtime_wrapper.run_import(input_data)
    print(f'INFO: model import done')


def run_infer(args, runtime_type, runtime_settings, model_path, artifacts_folder, num_frames):
    '''
    Run the model inference. Requires compilation to be run before this is invoked.
    Args:
        runtime_settings:
        model_path:
        artifacts_folder:
        num_frames:

    Returns:
        None
    '''
    # dataset parameters for actual inference
    val_dataset = glob.glob(f'{args.data_path}/*.*')

    runtime_wrapper = runtime_type(
            model_path=model_path,
            **runtime_settings)

    for input_index in range(num_frames):
        input_data = preprocess_input(val_dataset[input_index])
        outputs = runtime_wrapper.run_inference(input_data)
        print(outputs)

    print(f'INFO: model inference done')


def main(args):
    # the cwd must be the root of the respository
    if os.path.split(os.getcwd())[-1] in ('scripts', 'tutorials'):
        os.chdir('../')
    #

    #########################################################################
    # print environment variables
    assert ('TIDL_TOOLS_PATH' in os.environ and 'LD_LIBRARY_PATH' in os.environ), \
        "Check the environment variables, TIDL_TOOLS_PATH, LD_LIBRARY_PATH"
    print("TIDL_TOOLS_PATH=", os.environ['TIDL_TOOLS_PATH'])
    print("LD_LIBRARY_PATH=", os.environ['LD_LIBRARY_PATH'])
    print("TARGET_SOC=", args.target_device)
    print(f"INFO: current dir is: {os.getcwd()}")

    if not os.path.exists(os.environ['TIDL_TOOLS_PATH']):
        print(f"ERROR: TIDL_TOOLS_PATH: {os.environ['TIDL_TOOLS_PATH']} not found")
    else:
        print(f'INFO: TIDL_TOOLS_PATH contents: {os.listdir(os.environ["TIDL_TOOLS_PATH"])}')
    #

    #########################################################################
    if not os.path.exists(args.model_path):
        raise RuntimeError(f'incorrect model_path - doest not exist: {args.model_path}')

    if not os.path.exists(args.data_path):
        raise RuntimeError(f'incorrect data_path - doest not exist: {args.data_path}')

    #########################################################################
    # high level settings
    num_frames = 1
    modelartifacts_path = './work_dirs/rtwrapper'

    #########################################################################
    # prepare the model and artifacts folders
    model_name = os.path.basename(args.model_path)
    run_dir = os.path.join(modelartifacts_path, os.path.splitext(model_name)[0])

    if args.command=='compile_model' and os.path.exists(run_dir):
        print(f'INFO: clearing run_dir folder before compile: {run_dir}')
        shutil.rmtree(run_dir, ignore_errors=True)
    #

    model_folder = os.path.join(run_dir, 'model')
    model_path = os.path.join(model_folder, os.path.basename(args.model_path))
    os.makedirs(model_folder, exist_ok=True)
    shutil.copy2(args.model_path, model_path)
    onnx.shape_inference.infer_shapes_path(model_path, model_path)
    print(f'INFO: model_path - {model_path}')

    artifacts_folder = os.path.join(run_dir, 'artifacts')
    os.makedirs(artifacts_folder, exist_ok=True)
    print(f'INFO: artifacts_folder - {artifacts_folder}')

    #########################################################################
    # low level runtime_settings and runtime_options
    runtime_settings = {
        # add any runtime_settings overrides here
        'target_device': args.target_device,
        'runtime_options': {
            # add any runtime_options overrides here
            'tidl_tools_path': os.environ['TIDL_TOOLS_PATH'],
            'artifacts_folder': artifacts_folder,
        }
    }
    print(f'INFO: settings - {runtime_settings}')

    #########################################################################
    if args.runtime_name is None:
        model_ext = os.path.splitext(args.model_path)[-1]
        runtime_types_mapping = {
            '.onnx': presets.RuntimeType.RUNTIME_TYPE_ONNXRT,
            '.tflite': presets.RuntimeType.RUNTIME_TYPE_TFLITERT,
        }
        args.runtime_name = runtime_types_mapping[model_ext]
    #
    runtime_type = edgeai_tidlrunner.rtwrapper.RUNTIME_TYPES_MAPPING[args.runtime_name]

    #########################################################################
    # import and inference can be run in single call if separat3 process is used for them
    # otherwise one would have to choose between either import or inference in one call of this script.,
    if args.command == "compile":
        run_compile(args, runtime_type, runtime_settings, model_path, artifacts_folder)
    elif args.command == "infer":
        run_infer(args, runtime_type, runtime_settings, model_path, artifacts_folder, num_frames)
    else:
        assert False, f"ERROR: please specify a valid command - got: {args.command}"


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', type=str, choices=('compile', 'infer'))
    parser.add_argument('--model_path', type=str, default='./data/models/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv.onnx')
    parser.add_argument('--data_path', type=str, default='./data/datasets/vision/imagenetv2c/val')
    parser.add_argument('--target_device', type=str, default='AM68A')
    parser.add_argument('--runtime_name', type=str, default=None)
    return parser


if __name__ == '__main__':
    print(f'argv: {sys.argv}')

    parser = get_arg_parser()
    args = parser.parse_args()

    main(args)
