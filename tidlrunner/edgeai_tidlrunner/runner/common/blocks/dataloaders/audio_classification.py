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
import csv
import copy
import random
import numpy as np

from .dataset_base import DatasetBase


# UrbanSound8K class names, indexed by classID (0-9)
# Source: https://urbansounddataset.weebly.com/urbansound8k.html
URBANSOUND8K_CLASSES = [
    'air_conditioner',   # classID 0
    'car_horn',          # classID 1
    'children_playing',  # classID 2
    'dog_bark',          # classID 3
    'drilling',          # classID 4
    'engine_idling',     # classID 5
    'gun_shot',          # classID 6
    'jackhammer',        # classID 7
    'siren',             # classID 8
    'street_music',      # classID 9
]

NUM_CLASSES = len(URBANSOUND8K_CLASSES)

# Mapping from AudioSet class indices (521-class YAMNet output) to UrbanSound8K classIDs (0-9).
# Derived from repos/audioai-modelzoo/inference/yamnet_sc/yamnet_class_map.yml (0-indexed).
# Only AudioSet classes with a clear US8K equivalent are mapped; others are ignored.
# Used in evaluate() to aggregate 521-class probabilities into 10 US8K bins.
AUDIOSET_TO_US8K = {
    # US8K 0: air_conditioner
    406: 0,  # Mechanical fan
    407: 0,  # Air conditioning
    # US8K 1: car_horn
    302: 1,  # Vehicle horn, car horn, honking
    303: 1,  # Toot
    312: 1,  # Air horn, truck horn
    # US8K 2: children_playing
    10:  2,  # Children shouting
    66:  2,  # Children playing
    # US8K 3: dog_bark
    69:  3,  # Dog
    70:  3,  # Bark
    71:  3,  # Yip
    72:  3,  # Howl
    73:  3,  # Bow-wow
    74:  3,  # Growling
    75:  3,  # Whimper (dog)
    # US8K 4: drilling
    339: 4,  # Dental drill, dentist's drill
    418: 4,  # Power tool
    419: 4,  # Drill
    # US8K 5: engine_idling
    337: 5,  # Engine
    338: 5,  # Light engine (high frequency)
    342: 5,  # Medium engine (mid frequency)
    343: 5,  # Heavy engine (low frequency)
    344: 5,  # Engine knocking
    345: 5,  # Engine starting
    346: 5,  # Idling
    347: 5,  # Accelerating, revving, vroom
    # US8K 6: gun_shot
    421: 6,  # Gunshot, gunfire
    422: 6,  # Machine gun
    423: 6,  # Fusillade
    424: 6,  # Artillery fire
    425: 6,  # Cap gun
    # US8K 7: jackhammer
    414: 7,  # Jackhammer
    # US8K 8: siren
    317: 8,  # Police car (siren)
    318: 8,  # Ambulance (siren)
    319: 8,  # Fire engine, fire truck (siren)
    390: 8,  # Siren
    391: 8,  # Civil defense siren
    # US8K 9: street_music — Music + all instrument/genre subcategories (indices 132-276)
    **{i: 9 for i in range(132, 277)},
}


class UrbanSound8KDataLoader(DatasetBase):
    """Dataloader for the UrbanSound8K dataset.

    Expects the dataset at:
      <path>/metadata/UrbanSound8K.csv
      <path>/audio/fold{N}/<slice_file_name>

    Args:
        path: Root directory of the UrbanSound8K dataset.
        fold: Fold number (1-10) or list of fold numbers to load.
              If None, all 10 folds are loaded.
        shuffle: If True, shuffle file list with this value as seed.
        **kwargs: Additional kwargs stored in self.kwargs via DatasetBase.
    """

    def __init__(self, path, fold=None, shuffle=True, **kwargs):
        super().__init__(path=path, fold=fold, shuffle=shuffle, **kwargs)
        self.path = path
        self.sample_rate = kwargs.get('sample_rate', 16000)

        # Normalize fold argument to a set of ints (or None for all folds)
        if fold is None:
            self._folds = None  # all folds
        elif isinstance(fold, (list, tuple)):
            self._folds = {int(f) for f in fold}
        else:
            self._folds = {int(fold)}

        self.files = []
        self.labels = []
        self.folds_per_file = []

        self._load_metadata()

        if shuffle:
            rng_state = random.getstate()
            random.seed(int(shuffle))
            combined = list(zip(self.files, self.labels, self.folds_per_file))
            random.shuffle(combined)
            random.setstate(rng_state)
            self.files, self.labels, self.folds_per_file = map(list, zip(*combined))

    def _load_metadata(self):
        metadata_path = os.path.join(self.path, 'metadata', 'UrbanSound8K.csv')
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f'UrbanSound8K metadata CSV not found: {metadata_path}\n'
                f'Download the dataset first with examples/audio/scripts/download_urbansound8k.sh'
            )

        with open(metadata_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                fold_num = int(row['fold'])
                if self._folds is not None and fold_num not in self._folds:
                    continue
                audio_file = os.path.join(
                    self.path, 'audio', f'fold{fold_num}', row['slice_file_name']
                )
                self.files.append(audio_file)
                self.labels.append(int(row['classID']))
                self.folds_per_file.append(fold_num)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index, info_dict=None):
        import soundfile as sf

        audio_file = self.files[index]
        label = self.labels[index]
        fold = self.folds_per_file[index]

        data, file_sr = sf.read(audio_file, dtype='float32', always_2d=False)

        # Stereo to mono
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        # Ensure float32 (soundfile returns float32 when dtype='float32')
        data = data.astype(np.float32)

        if info_dict is None:
            info_dict = {}
        info_dict = copy.copy(info_dict)
        info_dict['label'] = label
        info_dict['filename'] = os.path.basename(audio_file)
        info_dict['sample_rate'] = file_sr
        info_dict['fold'] = fold

        return data, info_dict

    def evaluate(self, run_data, **kwargs):
        from sklearn.metrics import f1_score, accuracy_score

        y_true = []
        y_pred_top1 = []
        y_pred_top5 = []

        for i, data in enumerate(run_data):
            output = data['output']

            # Handle both list output (with postprocess) and dict output (without)
            output = output[0] if isinstance(output, list) else output
            output = list(output.values())[0] if isinstance(output, dict) else output

            # Flatten to 1D logits: handles (1, num_classes), (num_classes,), etc.
            output = np.squeeze(output)
            if output.ndim > 1:
                output = output.flatten()

            # YAMNet outputs 521 AudioSet classes — aggregate to 10 US8K bins.
            if output.shape[0] == 521:
                aggregated = np.zeros(NUM_CLASSES, dtype=np.float32)
                for audioset_idx, us8k_idx in AUDIOSET_TO_US8K.items():
                    aggregated[us8k_idx] += output[audioset_idx]
                output = aggregated

            top1 = int(np.argmax(output))
            top5 = set(np.argsort(output)[-5:].tolist())

            y_true.append(self.labels[i])
            y_pred_top1.append(top1)
            y_pred_top5.append(1 if self.labels[i] in top5 else 0)

        num_samples = len(y_true)
        accuracy_top1 = accuracy_score(y_true, y_pred_top1) * 100.0
        accuracy_top5 = sum(y_pred_top5) * 100.0 / num_samples
        f1_macro = f1_score(y_true, y_pred_top1, average='macro', zero_division=0) * 100.0

        return {
            'accuracy_top1%': accuracy_top1,
            'accuracy_top5%': accuracy_top5,
            'f1_macro%': f1_macro,
        }


def audio_classification_dataloader(settings, name, path, **kwargs):
    return UrbanSound8KDataLoader(path=path, **kwargs)
