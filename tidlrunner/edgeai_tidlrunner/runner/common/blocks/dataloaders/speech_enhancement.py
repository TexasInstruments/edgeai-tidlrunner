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
import glob
import copy
import random
import numpy as np

from .dataset_base import DatasetBase


class VoiceBankDemandDataLoader(DatasetBase):
    """Dataloader for the VoiceBank-DEMAND-16k dataset (speech enhancement).

    Loads paired noisy/clean WAV files from:
      <path>/<split>/noisy/<filename>.wav
      <path>/<split>/clean/<filename>.wav

    Files are matched by sorted filename — the same filename must exist in
    both noisy/ and clean/ subdirectories.

    Args:
        path: Root directory of VoiceBank-DEMAND-16k dataset.
        split: 'test' or 'train'. Default 'test' (standard evaluation split).
        shuffle: If True (or a non-zero int), shuffle file list using value as seed.
        **kwargs: Additional kwargs stored in self.kwargs via DatasetBase.
    """

    def __init__(self, path, split='test', shuffle=False, **kwargs):
        super().__init__(path=path, split=split, shuffle=shuffle, **kwargs)
        self.path = path
        self.split = split
        self.sample_rate = kwargs.get('sample_rate', 16000)

        self.noisy_files = []
        self.clean_files = []

        self._load_file_pairs()

        if shuffle:
            rng_state = random.getstate()
            random.seed(int(shuffle))
            combined = list(zip(self.noisy_files, self.clean_files))
            random.shuffle(combined)
            random.setstate(rng_state)
            self.noisy_files, self.clean_files = map(list, zip(*combined))

    def _load_file_pairs(self):
        noisy_dir = os.path.join(self.path, self.split, 'noisy')
        clean_dir = os.path.join(self.path, self.split, 'clean')

        if not os.path.isdir(noisy_dir):
            raise FileNotFoundError(
                f'VoiceBank-DEMAND noisy dir not found: {noisy_dir}\n'
                f'Download the dataset first with examples/audio/scripts/download_voicebank_demand.py'
            )
        if not os.path.isdir(clean_dir):
            raise FileNotFoundError(
                f'VoiceBank-DEMAND clean dir not found: {clean_dir}\n'
                f'Download the dataset first with examples/audio/scripts/download_voicebank_demand.py'
            )

        noisy_paths = sorted(glob.glob(os.path.join(noisy_dir, '*.wav')))
        clean_paths = sorted(glob.glob(os.path.join(clean_dir, '*.wav')))

        noisy_names = [os.path.basename(p) for p in noisy_paths]
        clean_names = [os.path.basename(p) for p in clean_paths]

        if noisy_names != clean_names:
            raise ValueError(
                f'Noisy and clean file lists do not match in {self.path}/{self.split}/.\n'
                f'Noisy: {len(noisy_names)} files, Clean: {len(clean_names)} files.'
            )

        self.noisy_files = noisy_paths
        self.clean_files = clean_paths

    def __len__(self):
        return len(self.noisy_files)

    def __getitem__(self, index, info_dict=None):
        import soundfile as sf

        noisy_file = self.noisy_files[index]
        clean_file = self.clean_files[index]

        noisy, file_sr = sf.read(noisy_file, dtype='float32', always_2d=False)
        clean, _ = sf.read(clean_file, dtype='float32', always_2d=False)

        # Stereo to mono (average channels)
        if noisy.ndim > 1:
            noisy = np.mean(noisy, axis=1)
        if clean.ndim > 1:
            clean = np.mean(clean, axis=1)

        noisy = noisy.astype(np.float32)
        clean = clean.astype(np.float32)

        if info_dict is None:
            info_dict = {}
        info_dict = copy.copy(info_dict)
        info_dict['clean_waveform'] = clean
        info_dict['filename'] = os.path.basename(noisy_file)
        info_dict['sample_rate'] = file_sr

        return noisy, info_dict

    @staticmethod
    def _compute_si_sdr(ref, est):
        """Scale-Invariant Signal-to-Distortion Ratio (pure NumPy).

        Reference: repos/GCRN-complex-qat/src/evaluation/evaluate.py
        """
        ref = np.asarray(ref, dtype=np.float64).ravel()
        est = np.asarray(est, dtype=np.float64).ravel()
        eps = np.finfo(np.float64).eps
        alpha = (eps + np.dot(ref, est)) / (np.dot(ref, ref) + eps)
        s_target = alpha * ref
        e_noise = est - s_target
        return float(10.0 * np.log10((np.sum(s_target ** 2) + eps) / (np.sum(e_noise ** 2) + eps)))

    def evaluate(self, run_data, **kwargs):
        from pesq import pesq as pesq_fn
        from pystoi import stoi as stoi_fn

        pesq_scores = []
        stoi_scores = []
        sisdr_scores = []

        sr = self.sample_rate

        for data in run_data:
            info = data.get('info_dict', {})

            clean = info.get('clean_waveform')
            enhanced = info.get('enhanced_waveform')

            if clean is None or enhanced is None:
                # No postprocess path — metrics require waveform domain; skip entry
                continue

            clean = np.asarray(clean, dtype=np.float32).ravel()
            enhanced = np.asarray(enhanced, dtype=np.float32).ravel()

            # Trim or pad to same length
            min_len = min(len(clean), len(enhanced))
            clean = clean[:min_len]
            enhanced = enhanced[:min_len]

            # PESQ wideband (ref first, est second) — uses float64 internally
            try:
                p = pesq_fn(sr, clean.astype(np.float64), enhanced.astype(np.float64), 'wb')
                pesq_scores.append(float(p))
            except Exception:
                pass

            # STOI (ref first, est second)
            try:
                s = stoi_fn(clean.astype(np.float64), enhanced.astype(np.float64), sr, extended=False)
                stoi_scores.append(float(s))
            except Exception:
                pass

            sisdr_scores.append(self._compute_si_sdr(clean, enhanced))

        def _mean(lst):
            return float(np.mean(lst)) if lst else float('nan')

        return {
            'pesq': _mean(pesq_scores),
            'stoi': _mean(stoi_scores),
            'si_sdr_db': _mean(sisdr_scores),
        }


def speech_enhancement_dataloader(settings, name, path, **kwargs):
    return VoiceBankDemandDataLoader(path=path, **kwargs)
