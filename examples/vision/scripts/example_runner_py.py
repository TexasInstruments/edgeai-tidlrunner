#!/usr/bin/env python

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
import argparse

import edgeai_tidlrunner


def main(args):
    # can give a nested dictionary or a flat dictionary
    # for example instead of above, one could also give fields such as
    # 'session.model_path': args.model_path
    # 'runtime_settings.runtime_options.advanced_options:calibration_frames': 5
    kwargs = {
        'session' : {
            'model_path': args.model_path,
        },
        'dataloader': {
            'path': args.data_path,
        },
        'runtime_settings': {
            # add any runtime_settings overrides here
            'target_device': args.target_device,
            'runtime_options': {
                # add any runtime_options override here
                'advanced_options:calibration_frames': 12,
                'advanced_options:calibration_iterations': 12
            }
        }
    }

    #########################################################################
    # import and inference can be run in single call if separat3 process is used for them
    # otherwise one would have to choose between either import or inference in one call of this script.,
    if args.command == "compile":
        edgeai_tidlrunner.run('compile', **kwargs)
    elif args.command == "infer":
        edgeai_tidlrunner.run('infer', **kwargs)
    else:
        assert False, f"ERROR: please specify a valid command - got: {args.command}"


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', type=str, choices=('compile', 'infer'))
    parser.add_argument('--model_path', type=str, default='./data/models/vision/classification/imagenet1k/torchvision/mobilenet_v2_tv.onnx')
    parser.add_argument('--data_path', type=str, default='./data/datasets/vision/imagenetv2c/val')
    parser.add_argument('--target_device', type=str, default='AM68A')
    return parser


if __name__ == '__main__':
    print(f'argv: {sys.argv}')

    parser = get_arg_parser()
    args = parser.parse_args()

    main(args)
