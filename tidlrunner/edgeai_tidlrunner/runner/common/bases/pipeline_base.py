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
import copy
import argparse
import json
import wurlitzer

from edgeai_tidlrunner.rtwrapper.options import presets
from edgeai_tidlrunner.rtwrapper.options import attr_dict

from .. import utils
from . import settings_base


class _TrackProvidedAction(argparse._StoreAction):
    def __call__(self, parser, namespace, values, option_string=None):
        if not hasattr(namespace, '_provided_args'):
            namespace._provided_args = set()
        namespace._provided_args.add(self.dest)
        super().__call__(parser, namespace, values, option_string)


class PipelineBase():
    ARGS_DICT = {}
    COPY_ARGS = {}

    def __init__(self, **kwargs):
        super().__init__()

        self.pipeline_config = kwargs.pop('common.pipeline_config', None)

        kwargs_in = self._flatten_dict(**kwargs)
        kwargs_in = self._expand_short_args(**kwargs_in)
        kwargs_in = self._upgrade_kwargs(**kwargs_in)

        kwargs_cmd = self._set_default_args()        
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

    def get_run_data(self):
        """
        Returns the input, output details of the last run.
        """
        return self.run_data

    @classmethod
    def _flatten_dict_fields(cls, kwargs_flat, prefix, override_dict_fields=False, **kwargs):
        dict_keys = [k for k, v in kwargs.items() if isinstance(v, dict) and len(v)>0 and isinstance(list(v.keys())[0], str)]
        nondict_keys = [k for k, v in kwargs.items() if k not in dict_keys]
        if override_dict_fields:
            for k in dict_keys:
                key_prefix = prefix + '.' + k if prefix else k
                cls._flatten_dict_fields(kwargs_flat, key_prefix, **kwargs[k])
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
                cls._flatten_dict_fields(kwargs_flat, key_prefix, **kwargs[k])
            #

    @classmethod
    def _flatten_dict(cls, **kwargs):
        kwargs_flat = {}
        prefix = None
        cls._flatten_dict_fields(kwargs_flat, prefix, **kwargs)
        return kwargs_flat

    @classmethod
    def _parse_dict_fields(cls, prefix_keys, kwargs):
        prefix_dict = {}
        for k in prefix_keys:
            v = kwargs.pop(k)
            prefix_dict[cls._split_fields(k)[-1]] = v
        #
        return prefix_dict

    @classmethod
    def _parse_to_dict(cls, **kwargs):
        kwargs = copy.deepcopy(kwargs)
        keys = kwargs.keys()
        prefixes = set(['.'.join(cls._split_fields(k)[:-1]) for k in keys])
        prefixes = sorted(prefixes, key=lambda x:len(cls._split_fields(x)), reverse=True)
        # put the '' entry last
        prefixes = [k for k in prefixes if k != '']
        for prefix in prefixes:
            prefix_keys = [k for k in keys if k.startswith(prefix) and k != prefix]
            prefix_dict = cls._parse_dict_fields(prefix_keys, kwargs)
            kwargs[prefix] = prefix_dict
        #
        return kwargs

    @classmethod
    def _split_fields(cls, key, separator='.'):
        """
        Split a key on '.' but keep numeric parts joined.
        Examples:
        'common.model_path' -> ['common', 'model_path']
        'session.model.0' -> ['session', 'model.0']
        'preprocess.resize.224' -> ['preprocess', 'resize.224']
        """
        parts = key.split(separator)
        if len(parts) <= 1:
            return parts
        #
        result = [parts[0]]  # Start with first part
        for part in parts[1:]:
            if part and part[0].isdigit():
                # Numeric part - join with previous
                result[-1] += separator + part
            else:
                # Non-numeric part - add as new element
                result.append(part)
            #
        #
        return result
    
    @classmethod
    def _set_default_args(cls, **kwargs):
        kwargs_cmd = {}
        for k_name, v_dict in cls.ARGS_DICT.items():
            if v_dict['default'] != argparse.SUPPRESS:
                kwargs_cmd[v_dict['dest']] = v_dict['default']
        kwargs_cmd.update(kwargs)
        return kwargs_cmd

    @classmethod
    def _expand_short_args(cls, **kwargs):
        model_id = kwargs.get('session.model_id', None)
        kwargs_cmd = {}
        for k, v in kwargs.items():
            if '.' not in k:
                if k in cls.ARGS_DICT:
                    v_dict = cls.ARGS_DICT[k]
                    kwargs_cmd[v_dict['dest']] = v
                else:
                    #print(f'WARNING: unrecognized argument - config {model_id} may need upgrade: {k}')
                    kwargs_cmd[k] = v
                #
            elif k not in kwargs_cmd:
                kwargs_cmd[k] = v
            #
        #
        return kwargs_cmd

    @classmethod
    def _upgrade_kwargs(cls, **kwargs):
        return kwargs

    @classmethod
    def _arg_parser_info(cls):
        default_entries = {k:v['default'] for k,v in cls.ARGS_DICT.items()}
        output = json.dumps(default_entries, indent=4)
        return f'defaults: {output}'

    @classmethod
    def _add_argument(cls, parser, name, **kwargs):
        positional = kwargs.pop('positional', False)
        prefix_dash = '' if positional else '--'
        if 'dest' in kwargs:
            alternate_arg_name = kwargs['dest']
            parser.add_argument(f'{prefix_dash}{name}', f'{prefix_dash}{alternate_arg_name}', action=_TrackProvidedAction, **kwargs)
        else:
            parser.add_argument(f'{prefix_dash}{name}', action=_TrackProvidedAction, **kwargs)
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
            allow_abbrev=False,
            argument_default=argparse.SUPPRESS,
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

    def _copy_args(self, **kwargs_cmd):
        # copy entries if required
        for k in self.COPY_ARGS:
            src_key = self.COPY_ARGS[k]
            if src_key in kwargs_cmd:
                kwargs_cmd[k] = kwargs_cmd[src_key]
            #
        #
        return kwargs_cmd
    
    def prepare(self):
        capture_log = self.settings['common']['capture_log']
        log_file = self.settings['common']['log_file']
        if capture_log and log_file and hasattr(self, 'run_dir') and self.run_dir:
            if not log_file.startswith('/') and not log_file.startswith('.'):
                log_file =  os.path.join(self.run_dir, log_file)
            #
            os.makedirs(os.path.dirname(log_file), exist_ok=True)       
            with open(log_file, 'a') as log_fp:
                if capture_log == settings_base.CaptureLogModes.CAPTURE_LOG_MODE_TEE:
                    tee_stdout = utils.TeeLogWriter([sys.stdout, log_fp])
                    tee_stderr = utils.TeeLogWriter([sys.stderr, log_fp])  
                else:
                    tee_stdout = log_fp
                    tee_stderr = log_fp
                #
                with wurlitzer.pipes(stdout=tee_stdout, stderr=tee_stderr):
                    return self._prepare()
                #
            #
        else:
            return self._prepare()
        
    def run(self):
        capture_log = self.settings['common']['capture_log']
        log_file = self.settings['common']['log_file']
        if capture_log != settings_base.CaptureLogModes.CAPTURE_LOG_MODE_OFF and log_file and hasattr(self, 'run_dir') and self.run_dir:
            if not log_file.startswith('/') and not log_file.startswith('.'):
                log_file =  os.path.join(self.run_dir, log_file)
            #        
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            with open(log_file, 'a') as log_fp:
                if capture_log == settings_base.CaptureLogModes.CAPTURE_LOG_MODE_TEE:
                    tee_stdout = utils.TeeLogWriter([sys.stdout, log_fp])
                    tee_stderr = utils.TeeLogWriter([sys.stderr, log_fp])  
                else:
                    tee_stdout = log_fp
                    tee_stderr = log_fp
                #
                with wurlitzer.pipes(stdout=tee_stdout, stderr=tee_stderr):
                    return self._run()
                #
            #        
        else:
            return self._run()
        
    def _prepare(self):
        raise RuntimeError(f'PipelineBase._prepare() is not implemented in {self.__class__.__name__}. Please implement it in the derived class.')

    def _run(self):
        raise RuntimeError(f'PipelineBase._run() is not implemented in {self.__class__.__name__}. Please implement it in the derived class.')