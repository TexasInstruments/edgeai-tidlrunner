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
import shutil
import copy
import ast
import yaml

from ....common import utils
from ....common import bases
from ...settings.settings_default import SETTINGS_DEFAULT, COPY_SETTINGS_DEFAULT
from ...settings import constants


class CommonPipelineBase(bases.PipelineBase):
    ARGS_DICT = SETTINGS_DEFAULT['basic']
    COPY_ARGS = COPY_SETTINGS_DEFAULT['basic']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.work_path = None
        self.model_source = None
        self.model_folder = None
        self.model_path = None
        self.run_dir = None

        if 'work_path' in self.settings[self.common_prefix]:
            work_path = self.settings[self.common_prefix]['work_path']
            self.work_path = self._build_run_dir(work_path)
        #
        if self.session_prefix in self.settings and self.settings[self.session_prefix].get('model_path', None):
            common_kwargs = self.settings[self.common_prefix]                 
            config_path = os.path.dirname(common_kwargs['config_path']) if common_kwargs['config_path'] else None            
            model_source = self.settings[self.session_prefix]['model_path']       
            self.model_source = self._get_file_path(model_source, source_dir=config_path)

            run_dir = self.settings[self.session_prefix]['run_dir']
            self.run_dir = self._build_run_dir(run_dir)

            model_basename = os.path.basename(self.model_source)
            self.model_folder = os.path.join(self.run_dir, 'model')
            self.model_path = os.path.join(self.model_folder, model_basename)
            self.settings[self.session_prefix]['model_path'] = self.model_path
        #

    def _prepare(self):
        pass

    def _get_file_path(self, model_source, source_dir=None):
        is_web_link = model_source.startswith('http')
        is_simple_path = os.sep not in os.path.normpath(model_source)
        if is_web_link:
            return model_source
        else:
            model_source = os.path.join(source_dir, model_source) if source_dir and is_simple_path else model_source
            return model_source

    def download_file(self, model_source, model_folder, source_dir=None):
        is_web_link = model_source.startswith('http')
        is_simple_path = os.sep not in os.path.normpath(model_source)
        if is_web_link:
            utils.download_file(model_source, model_folder)
        else:
            model_source = os.path.join(source_dir, model_source) if source_dir and is_simple_path else model_source
            if not os.path.exists(model_source) and os.path.exists(model_source + '.link'):
                model_download_folder = os.path.dirname(model_source)
                utils.download_file(model_source, model_download_folder)
            #
            shutil.copy2(model_source, model_folder)
        #

    def _build_run_dir(self, run_dir):
        pipeline_type = self.kwargs.get('common.pipeline_type', 'compile')
        task_type = self.kwargs.get('common.task_type', None) or None
    
        model_basename = os.path.basename(self.model_source) if self.model_source else ''
        model_id = self.settings[self.session_prefix].get('model_id', '')
        if not model_id:
            unique_id = utils.generate_unique_id(model_basename, num_characters=8)            
            if task_type in constants.TaskTypeShortNames:
                task_type_short_name = constants.TaskTypeShortNames[task_type]
                model_id = task_type_short_name + '-' + unique_id                
            else:
                pipeline_type = self.kwargs.get('common.pipeline_type', 'x') or 'x'
                model_id = pipeline_type + '-' + unique_id
            #
        #
        model_id_underscore = model_id + '_'

        tensor_bits = self.kwargs.get('session.runtime_options.tensor_bits', '') or ''
        tensor_bits_str = f'{str(tensor_bits)}' if tensor_bits else ''
        tensor_bits_slash = f'{str(tensor_bits)}' + os.sep if tensor_bits else ''

        target_device = self.kwargs.get('session.target_device', 'NONE') or 'NONE'
        target_device_str = target_device if target_device else ''
        target_device_slash = target_device + os.sep if target_device else ''

        model_basename_wo_ext, model_ext = os.path.splitext(model_basename)
        model_ext = model_ext[1:] if len(model_ext)>0 else model_ext

        runtime_name = self.kwargs.get('session.name', '') or ''

        run_dir = run_dir.replace('{work_path}', self.work_path) if self.work_path else run_dir
        run_dir = run_dir.replace('{pipeline_type}', pipeline_type)
        run_dir = run_dir.replace('{model_id}_', model_id_underscore)
        run_dir = run_dir.replace('{model_id}', model_id)
        run_dir = run_dir.replace('{runtime_name}', runtime_name)        
        run_dir = run_dir.replace('{model_ext}', model_ext)        
        run_dir = run_dir.replace('{tensor_bits}/', tensor_bits_slash)
        run_dir = run_dir.replace('{tensor_bits}', tensor_bits_str)
        run_dir = run_dir.replace('{target_device}/', target_device_slash)
        run_dir = run_dir.replace('{target_device}', target_device_str)
        run_dir = run_dir.replace('{model_name}', model_basename_wo_ext)
        if self.model_source:
            run_dir = self._replace_model_path(run_dir, '{model_path}', run_dir_tree_depth=3)
        #
        return run_dir

    def _replace_model_path(self, run_dir, model_path_str, run_dir_tree_depth):
        run_dir_tree_depth = 3
        model_path_tree = os.path.abspath(os.path.splitext(self.model_source)[0]).split(os.sep)
        if len(model_path_tree) > run_dir_tree_depth:
            model_path_tree = model_path_tree[-run_dir_tree_depth:]
        #
        run_dir = run_dir.replace(model_path_str, '_'.join(model_path_tree))    
        return run_dir    
    
    def info(self):
        print(f'INFO: Common pipeline base - {__file__}')

    def _run(self):
        return None

    def _write_params(self, settings, filename, param_template=None):
        params = utils.pretty_object(settings)
        if isinstance(param_template, str):
            with open(param_template, 'r') as fp:
                param_template = yaml.safe_load(fp)
            #
        #
        params = utils.cleanup_dict(params, param_template)
        with open(filename, 'w') as fp:
            yaml.dump(params, fp)
        #