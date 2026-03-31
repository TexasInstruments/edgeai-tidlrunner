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


import numpy as np


class SoundClassificationPostProcess:
    """Postprocess for sound classification models (VGGish11, YAMNet).

    Applies numerically-stable softmax over the model output logits,
    then stores the top-1 and top-5 class indices in info_dict.

    Input tensor: list of numpy arrays; first element has shape
    (1, num_classes) or (num_classes,).
    Returns softmax probabilities (1D, shape (num_classes,)) as the tensor.
    """

    def __call__(self, tensor, info_dict):
        # Extract model output from the list returned by the session
        output = tensor[0] if isinstance(tensor, list) else tensor
        logits = np.squeeze(output)  # (num_classes,)

        # Numerically stable softmax
        shifted = logits - np.max(logits)
        exp_x = np.exp(shifted)
        probs = exp_x / exp_x.sum()

        # Top-1 and top-5 (descending by probability)
        sorted_indices = np.argsort(probs)[::-1]
        info_dict['predicted_class'] = int(sorted_indices[0])
        info_dict['top5_indices'] = sorted_indices[:5].tolist()

        return probs, info_dict


class SpeechEnhancementPostProcess:
    """Postprocess for speech enhancement models (GTCRN).

    Reconstructs the enhanced waveform from the GTCRN complex STFT output
    via librosa.istft with a sqrt-Hann window (matching the preprocessing
    in audio_transforms.py / WI-007).

    Input tensor: list of numpy arrays; first element has shape
    (1, 257, T, 2) — (batch, freq=257, time, real/imag).
    Stores the reconstructed waveform in info_dict['enhanced_waveform'].
    Returns the original tensor unchanged (evaluate() uses info_dict).

    PITFALL: window must use sym=False (periodic Hann) and **0.5 to match
    the analysis window used in STFTTransform (WI-007).  center=False avoids
    edge-frame discrepancies between librosa and PyTorch (see WI-006).
    """

    def __init__(self, n_fft=512, hop_length=256, win_length=512, center=False):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.center = center

    def __call__(self, tensor, info_dict):
        import librosa
        import scipy.signal

        # Extract GTCRN output from session result list
        output = tensor[0] if isinstance(tensor, list) else tensor
        output = np.squeeze(output, axis=0)  # (257, T, 2)

        # Reconstruct complex spectrogram: real + j*imag
        enhanced_complex = output[..., 0] + 1j * output[..., 1]  # (257, T)

        # Inverse STFT with sqrt-Hann window (periodic, matching analysis)
        window = scipy.signal.windows.hann(self.win_length, sym=False) ** 0.5
        waveform = librosa.istft(
            enhanced_complex,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.center,
        )

        info_dict['enhanced_waveform'] = waveform.astype(np.float32)

        # Pass original tensor through — evaluate() reads from info_dict
        return tensor, info_dict


class GCRNSpeechEnhancementPostProcess:
    """Postprocess for GCRN speech enhancement model.

    Reconstructs the enhanced waveform from GCRN complex STFT output
    via librosa.istft with a Hamming window (matching GCRNSTFTTransform).

    Input tensor: list of numpy arrays; first element has shape
    (1, 2, T, 161) — (batch, real/imag, time, freq=161).
    Stores the reconstructed waveform in info_dict['enhanced_waveform'].
    Returns the original tensor unchanged (evaluate() reads from info_dict).

    Note: tensor layout (B, RI, T, F) differs from GTCRN's (B, F, T, RI).
    The transpose step (real + 1j*imag).T converts (T, 161) → (161, T) for librosa.
    """

    def __init__(self, n_fft=320, hop_length=160, win_length=320, center=True):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.center = center

    def __call__(self, tensor, info_dict):
        import librosa
        import scipy.signal

        # Extract GCRN output from session result list
        output = tensor[0] if isinstance(tensor, list) else tensor
        output = np.squeeze(output, axis=0)  # (2, T, 161)

        # Separate real and imaginary channels
        real = output[0]  # (T, 161)
        imag = output[1]  # (T, 161)

        # Build complex spectrogram; transpose to (161, T) for librosa (freq, time)
        enhanced_complex = (real + 1j * imag).T  # (161, T)

        # Inverse STFT with Hamming window (sym=True, matching GCRNSTFTTransform)
        window = scipy.signal.windows.hamming(self.win_length, sym=True)
        waveform = librosa.istft(
            enhanced_complex,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=self.center,
        )

        info_dict['enhanced_waveform'] = waveform.astype(np.float32)

        # Pass original tensor through — evaluate() reads from info_dict
        return tensor, info_dict
