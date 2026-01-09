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
#################################################################################


import os
from . import logger_utils

try:
    from .model_infos import MODEL_INFOS_DICT
except Exception as e:
    MODEL_INFOS_DICT = {}
    # print(logger_utils.log_color("WARNING", "MODEL_INFOS_DICT is not found", str(e)))
    

model_id_artifacts_pair = {v['model_id']+'_'+v['session_name']:v['artifact_name']
                           for k, v in MODEL_INFOS_DICT.items()
                           }

shortlisted_model_list = {v['model_id']+'_'+v['session_name']:v['artifact_name']
                          for k, v in MODEL_INFOS_DICT.items()
                          if v['shortlisted']}

recommended_model_list = {v['model_id']+'_'+v['session_name']:v['artifact_name']
                          for k, v in MODEL_INFOS_DICT.items()
                          if v['recommended']}

super_set = list(model_id_artifacts_pair.keys())


def get_selected_models(selected_task=None):
    selected_models_list = [key for key in model_id_artifacts_pair if key in shortlisted_model_list]
    selected_models_for_a_task = [model for model in selected_models_list if model.split('-')[0] == selected_task]
    return selected_models_for_a_task


def get_artifact_name(model_id_or_artifact_id, session_name=None, guess_names=False):
    # artifact_id is model_id followed by session_name
    # either pass a model_id and a session_name
    # or directly pass the artifact_id and don't pass session_name
    if session_name is None:
        artifact_id = model_id_or_artifact_id
    else:
        model_id = model_id_or_artifact_id
        artifact_id = f'{model_id}_{session_name}'
    #

    artifact_name = None
    if artifact_id in model_id_artifacts_pair:
        artifact_name = model_id_artifacts_pair[artifact_id]
    elif guess_names:
        model_id, runtime_name = artifact_id.split('_')
        # create mapping dictionaries
        model_id_to_model_name_dict = {k.split('_')[0]:'-'.join(v.split('-')[1:]) \
                for k,v in model_id_artifacts_pair.items()}
        short_runtime_name_dict = {'tvmrt':'TVM', 'tflitert':'TFL', 'onnxrt':'ONR'}
        # finally for the artifact name
        if runtime_name in short_runtime_name_dict and model_id in model_id_to_model_name_dict:
            artifact_name = f'{short_runtime_name_dict[runtime_name]}-{model_id_to_model_name_dict[model_id]}'

    return artifact_name


def get_name_key_pair_list(model_ids, session_name, remove_models=True):
    shortlisted_model_list_entries = shortlisted_model_list.keys()
    name_key_pair_list = []
    for model_id in model_ids:
        artifact_id = f'{model_id}_{session_name}'
        artifact_name =  model_id_artifacts_pair[artifact_id] if artifact_id in model_id_artifacts_pair else None
        if artifact_name is not None and \
                (not remove_models or artifact_id in shortlisted_model_list_entries):
            name_key_pair_list.append((artifact_name, model_id))
        #
    #
    return name_key_pair_list


def is_shortlisted_model(artifact_id):
    shortlisted_model_list_entries = shortlisted_model_list.keys()
    is_shortlisted = (artifact_id in shortlisted_model_list_entries)
    return is_shortlisted


def is_recommended_model(artifact_id):
    recommended_model_entries = recommended_model_list.keys()
    is_recommended = (artifact_id in recommended_model_entries)
    return is_recommended


