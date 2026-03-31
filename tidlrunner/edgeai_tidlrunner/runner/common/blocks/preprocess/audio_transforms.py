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

"""Audio preprocessing transforms for audio model pipelines.

Pitfall notes (validated in WI-006 / agent_ws/tests/validate_audio_preprocessing.py):
  1. PITFALL #1 — Mel scale: librosa defaults to Slaney; torchaudio defaults to HTK.
     Always pass htk=True in librosa.feature.melspectrogram() calls.
  2. PITFALL #2 — center=True edge frames: PyTorch and librosa compute the first/last
     frames differently when center=True. Non-edge frames agree at ~130 dB SNR.
     Production uses center=True for VGGish11/YAMNet (matches training); center=False
     for GTCRN (matches postprocess ISTFT in SpeechEnhancementPostProcess).
  3. PITFALL #3 — sqrt-Hann periodicity: torch.hann_window(N) is periodic (sym=False).
     Must use scipy.signal.windows.hann(N, sym=False) ** 0.5 for GTCRN.
  4. PITFALL #4 — YAMNet power path: torchaudio VGGishLogMelSpectrogram does
     power_spec -> sqrt -> mel_scale, which equals mel @ |STFT|. Use power=1.0 in
     librosa (NOT power=2.0) to match.
  5. PITFALL #5 — YAMNet log offset is 0.001, NOT 0.01 or 1e-9.
"""

import numpy as np


class AudioLoadAndResample:
    """Load audio from a file path and resample to target sample rate.

    Only active when `data` is a string (file path). Passes through numpy arrays
    unchanged. Used for --input_files CLI paths; dataloaders return waveforms directly.

    Returns a float32 1-D mono numpy waveform at target_sample_rate Hz.
    """

    def __init__(self, target_sample_rate=16000):
        self.target_sample_rate = target_sample_rate

    def __call__(self, data, info_dict):
        if not isinstance(data, str):
            return data, info_dict

        import soundfile as sf
        import librosa

        waveform, sr = sf.read(data, always_2d=False)
        waveform = waveform.astype(np.float32)

        # Stereo → mono
        if waveform.ndim == 2:
            waveform = np.mean(waveform, axis=1)

        # Resample if needed
        if sr != self.target_sample_rate:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.target_sample_rate)

        info_dict['sample_rate'] = self.target_sample_rate
        return waveform, info_dict


class VGGishMelSpectrogram:
    """Log-mel spectrogram for VGGish11.

    Produces shape (1, 1, 64, 126) float32 for 4 s / 16 kHz audio.

    PITFALL #1: htk=True — librosa defaults to Slaney; torchaudio uses HTK.
    PITFALL #2: center=True — matches training setup; edge-frame differences are
                acceptable and do not affect classification accuracy in practice.
    """

    def __init__(
        self,
        sample_rate=16000,
        n_fft=1024,
        hop_length=512,
        n_mels=64,
        fmin=0.0,
        fmax=8000.0,
        center=True,
        log_offset=1e-9,
        audio_duration=4.0,
        target_frames=126,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.center = center
        self.log_offset = log_offset
        self.audio_duration = audio_duration
        self.target_frames = target_frames

    def __call__(self, data, info_dict):
        import librosa

        waveform = np.asarray(data, dtype=np.float32)
        if waveform.ndim != 1:
            waveform = waveform.ravel()

        # Pad or crop waveform to fixed duration
        target_samples = int(self.sample_rate * self.audio_duration)
        if len(waveform) < target_samples:
            waveform = np.pad(waveform, (0, target_samples - len(waveform)))
        else:
            waveform = waveform[:target_samples]

        # Log-mel spectrogram — htk=True matches torchaudio's mel_scale="htk" default
        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            center=self.center,
            htk=True,
            power=2.0,
            norm=None,
        )
        log_mel = np.log(mel + self.log_offset)  # (n_mels, T)

        # Pad or crop to target_frames
        if log_mel.shape[1] < self.target_frames:
            pad_width = self.target_frames - log_mel.shape[1]
            log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)))
        else:
            log_mel = log_mel[:, :self.target_frames]

        # (n_mels, target_frames) → (1, 1, n_mels, target_frames)
        output = log_mel[np.newaxis, np.newaxis].astype(np.float32)
        return output, info_dict


class YAMNetMelSpectrogram:
    """Log-mel spectrogram for YAMNet.

    Produces shape (1, 1, 96, 64) float32 (time-first, mel-second) from the
    first 96-frame patch of the audio.

    PITFALL #1: htk=True — match torchaudio HTK scale.
    PITFALL #4: power=1.0 — YAMNet's VGGishLogMelSpectrogram does
                power_spec -> sqrt -> mel_scale; librosa power=1.0 is equivalent.
    PITFALL #5: log_offset=0.001 — NOT 0.01 or 1e-9.
    PITFALL #2: center=True — matches training setup.
    """

    def __init__(
        self,
        sample_rate=16000,
        n_fft=512,
        win_length=400,
        hop_length=160,
        n_mels=64,
        fmin=125.0,
        fmax=7500.0,
        center=True,
        log_offset=0.001,
        num_frames=96,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.center = center
        self.log_offset = log_offset
        self.num_frames = num_frames

    def __call__(self, data, info_dict):
        import librosa

        waveform = np.asarray(data, dtype=np.float32)
        if waveform.ndim != 1:
            waveform = waveform.ravel()

        # Mel spectrogram — power=1.0 matches torchaudio's power_spec -> sqrt path
        mel = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            center=self.center,
            htk=True,
            power=1.0,
            norm=None,
        )
        log_mel = np.log(mel + self.log_offset)  # (n_mels, T)

        # Extract first num_frames patch; zero-pad if shorter
        if log_mel.shape[1] < self.num_frames:
            pad_width = self.num_frames - log_mel.shape[1]
            log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)))
        else:
            log_mel = log_mel[:, :self.num_frames]

        # Transpose to (num_frames, n_mels) — YAMNet expects time-first
        log_mel = log_mel.T  # (96, 64)

        # (num_frames, n_mels) → (1, 1, num_frames, n_mels)
        output = log_mel[np.newaxis, np.newaxis].astype(np.float32)
        return output, info_dict


class STFTTransform:
    """Complex STFT for GTCRN speech enhancement.

    Produces shape (1, 257, T, 2) float32 — (batch, freq, time, real/imag).

    PITFALL #3: sqrt-Hann window must use sym=False (periodic) to match
                torch.hann_window(N) used in GTCRN training.
    PITFALL #2: center=False — matches SpeechEnhancementPostProcess ISTFT
                (center=False avoids edge-frame discrepancies with PyTorch).
    """

    def __init__(self, n_fft=512, hop_length=256, win_length=512, center=False):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.center = center

    def __call__(self, data, info_dict):
        import librosa
        import scipy.signal

        waveform = np.asarray(data, dtype=np.float32)
        if waveform.ndim != 1:
            waveform = waveform.ravel()

        # sqrt-Hann window — periodic (sym=False) matches torch.hann_window default
        window = scipy.signal.windows.hann(self.win_length, sym=False) ** 0.5

        stft = librosa.stft(
            y=waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.center,
        )
        # stft: (257, T) complex64

        # Stack real/imag along last axis → (257, T, 2); add batch → (1, 257, T, 2)
        stft_ri = np.stack([stft.real, stft.imag], axis=-1)  # (257, T, 2)
        output = stft_ri[np.newaxis].astype(np.float32)       # (1, 257, T, 2)
        return output, info_dict
