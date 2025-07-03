# Copyright (c) 2018-2021, Texas Instruments
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
import copy
import argparse
import json
import wurlitzer

from ...rtwrapper.options import presets
from ...rtwrapper.options import attr_dict
from .. import utils
from . import settings_base


class PipelineBase():
    ARGS_DICT = settings_base.SETTINGS_TARGET_MODULE_ARGS_DICT
    COPY_ARGS = {}

    def __init__(self, **kwargs):
        super().__init__()
        kwargs_default = self._set_default_args()
        kwargs_in = self._flatten_dict(**kwargs)
        kwargs_in = self._expand_short_args(**kwargs_in)
        kwargs_in = self._upgrade_kwargs(**kwargs_in)
        kwargs_cmd = copy.deepcopy(kwargs_default)
        kwargs_cmd.update(kwargs_in)
        kwargs_cmd = self._copy_args(**kwargs_cmd)
        self.kwargs = kwargs_cmd

        settings = self._parse_to_dict(**self.kwargs)
        self.settings = attr_dict.AttrDict(settings)

        self.run_data = None # last run data, can be used by other pipelines

        self.common_prefix = 'common'
        self.dataloader_prefix = 'dataloader'
        self.session_prefix = 'session'
        self.preprocess_prefix = 'preprocess'
        self.postprocess_prefix = 'postprocess'

    def info(self):
        print(f'INFO: running - {__file__}')

    def run(self):
        pass

    def get_run_data(self):
        """
        Returns the input, output details of the last run.
        """
        return self.run_data

    def _flatten_dict_fields(self, kwargs_flat, prefix, override_dict_fields=False, **kwargs):
        dict_keys = [k for k, v in kwargs.items() if isinstance(v, dict) and len(v)>0 and isinstance(list(v.keys())[0], str)]
        nondict_keys = [k for k, v in kwargs.items() if k not in dict_keys]
        if override_dict_fields:
            for k in dict_keys:
                key_prefix = prefix + '.' + k if prefix else k
                self._flatten_dict_fields(kwargs_flat, key_prefix, **kwargs[k])
            #
            for k in nondict_keys:
                key_prefix = prefix + '.' + k if prefix else k
                kwargs_flat[key_prefix] = kwargs[k]
            #
        else:
            for k in nondict_keys:
                key_prefix = prefix + '.' + k if prefix else k
                kwargs_flat[key_prefix] = kwargs[k]
            #
            for k in dict_keys:
                key_prefix = prefix + '.' + k if prefix else k
                self._flatten_dict_fields(kwargs_flat, key_prefix, **kwargs[k])
            #

    def _flatten_dict(self, **kwargs):
        kwargs_flat = {}
        prefix = None
        self._flatten_dict_fields(kwargs_flat, prefix, **kwargs)
        return kwargs_flat

    def _parse_dict_fields(self, prefix_keys, kwargs):
        prefix_dict = {}
        for k in prefix_keys:
            v = kwargs.pop(k)
            prefix_dict[k.split('.')[-1]] = v
        #
        return prefix_dict

    def _parse_to_dict(self, **kwargs):
        kwargs = copy.deepcopy(kwargs)
        keys = kwargs.keys()
        prefixes = set(['.'.join(k.split('.')[:-1]) for k in keys])
        prefixes = sorted(prefixes, key=lambda x:len(x.split('.')), reverse=True)
        # put the '' entry last
        prefixes = [k for k in prefixes if k != '']
        for prefix in prefixes:
            prefix_keys = [k for k in keys if k.startswith(prefix)]
            prefix_dict = self._parse_dict_fields(prefix_keys, kwargs)
            kwargs[prefix] = prefix_dict
        #
        return kwargs
        
    def _copy_args(self, **kwargs_cmd):
        # copy entries if required
        for k in self.COPY_ARGS:
            src_key = self.COPY_ARGS[k]
            if src_key in kwargs_cmd:
                kwargs_cmd[k] = kwargs_cmd[src_key]
            #
        #
        return kwargs_cmd

    def _set_default_args(self, **kwargs):
        kwargs_cmd = {}
        for k_name, v_dict in self.ARGS_DICT.items():
            kwargs_cmd[v_dict['dest']] = v_dict['default']
        kwargs_cmd.update(kwargs)
        return kwargs_cmd

    def _expand_short_args(self, **kwargs):
        model_id = kwargs.get('session.model_id', None)
        kwargs_cmd = {}
        for k, v in kwargs.items():
            if '.' not in k:
                if k in self.ARGS_DICT:
                    v_dict = self.ARGS_DICT[k]
                    kwargs_cmd[v_dict['dest']] = v
                else:
                    print(f'WARNING: unrecognized argument - config {model_id} may need upgrade: {k}')
                    kwargs_cmd[k] = v
                #
            else:
                kwargs_cmd[k] = v
            #
        #
        return kwargs_cmd

    def _upgrade_kwargs(self, **kwargs):
        return kwargs

    @classmethod
    def _arg_parser_info(cls):
        default_entries = {k:v['default'] for k,v in cls.ARGS_DICT.items()}
        output = json.dumps(default_entries, indent=4)
        return f'defaults: {output}'

    @classmethod
    def _add_argument(cls, parser, name, **kwargs):
        if 'dest' in kwargs:
            alternate_arg_name = kwargs['dest']
            parser.add_argument(f'--{name}', f'--{alternate_arg_name}', **kwargs)
        else:
            parser.add_argument(f'--{name}', **kwargs)
        #

    @classmethod
    def get_arg_parser(cls):
        args_dict = copy.deepcopy(cls.ARGS_DICT)
        group_keys = args_dict.keys()
        group_dict = {}
        for k in group_keys:
            v = args_dict[k]
            group_name = v.pop('group', None)
            if group_name not in group_dict:
                group_dict[group_name] = []
            #
            group_dict[group_name].append((k,v))

        parser = argparse.ArgumentParser(
            description='Runner commandline arguments',
            epilog=cls._arg_parser_info()
        )

        for group_k, group_entry in group_dict.items():
            if group_k is not None:
                group = parser.add_mutually_exclusive_group(required=False)
                for k, v in group_entry:
                    cls._add_argument(group, k, **v)
                #
            else:
                for k, v in group_entry:
                    cls._add_argument(parser, k, **v)
                #
            #
        #
        return parser

    def run(self):
        capture_log = self.settings['common']['capture_log']
        log_file = self.settings['common']['log_file']
        if capture_log and self.run_dir and log_file:
            if not log_file.startswith('/') and not log_file.startswith('.'):
                log_file =  os.path.join(self.run_dir, log_file)
            #
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            with open(log_file, 'a') as log_fp:
                with wurlitzer.pipes(stdout=log_fp, stderr=log_fp):
                    return self._run()
                #
            #
        else:
            return self._run()