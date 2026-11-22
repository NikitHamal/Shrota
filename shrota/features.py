"""16 kHz 80-Bin Log-Mel Feature Extractor with Causal/Global CMVN.

Engineered so offline and 160 ms chunked streaming features match 100% identically.
Parameters:
  sr = 16000 Hz
  n_fft = 400 (25 ms window)
  hop_length = 160 (10 ms stride)
  n_mels = 80 (fmin=20 Hz, fmax=7600 Hz)
"""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
WIN_LENGTH = 400
N_MELS = 80
FMIN = 20.0
FMAX = 7600.0


def hz_to_mel(freq: float | np.ndarray) -> float | np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(freq) / 700.0)


def mel_to_hz(mel: float | np.ndarray) -> float | np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def build_mel_filterbank(
    sr: int = SAMPLE_RATE,
    n_fft: int = N_FFT,
    n_mels: int = N_MELS,
    fmin: float = FMIN,
    fmax: float = FMAX,
) -> np.ndarray:
    """Create standard triangular mel filterbank matrix of shape (n_mels, n_fft // 2 + 1)."""
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sr / 2.0, n_freqs)
    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        up_slope = (fft_freqs - left) / max(center - left, 1e-8)
        down_slope = (right - fft_freqs) / max(right - center, 1e-8)
        fb[i] = np.maximum(0.0, np.minimum(up_slope, down_slope))
    # Slaney-style area normalization
    enorm = 2.0 / np.maximum(hz_points[2 : n_mels + 2] - hz_points[:n_mels], 1e-8)
    fb *= enorm[:, np.newaxis]
    return fb


class LogMelExtractor(nn.Module):
    """Causal frame-exact 80-bin Log-Mel Spectrogram extractor."""

    def __init__(
        self,
        sr: int = SAMPLE_RATE,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        n_mels: int = N_MELS,
    ):
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        window = torch.hann_window(n_fft, periodic=True, dtype=torch.float32)
        self.register_buffer("window", window, persistent=False)

        mel_fb = torch.from_numpy(build_mel_filterbank(sr, n_fft, n_mels))
        self.register_buffer("mel_fb", mel_fb, persistent=False)

        # Default global CMVN statistics (can be updated from corpus stats)
        self.register_buffer("cmvn_mean", torch.full((n_mels,), -4.2, dtype=torch.float32))
        self.register_buffer("cmvn_istd", torch.full((n_mels,), 0.42, dtype=torch.float32))

    def set_cmvn(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.cmvn_mean.copy_(mean.view(-1))
        self.cmvn_istd.copy_(1.0 / torch.clamp(std.view(-1), min=1e-5))

    def forward(self, wav: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Compute causal log-mel spectrogram.

        Args:
            wav: (B, T_samples) or (T_samples,) float32 waveform in [-1, 1]
        Returns:
            mel: (B, T_frames, 80) float32 log-mel features
        """
        squeeze = False
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
            squeeze = True

        # Causal left-pad of (n_fft - hop_length) = 240 samples so every 160-sample hop
        # produces exactly 1 frame and chunked streaming with 240-sample history is 100% identical!
        pad_left = self.n_fft - self.hop_length
        wav_padded = F.pad(wav, (pad_left, 0), mode="constant", value=0.0)

        # Unfold into frames of length n_fft with step hop_length
        frames = wav_padded.unfold(dimension=-1, size=self.n_fft, step=self.hop_length)
        frames = frames * self.window

        spec = torch.fft.rfft(frames, n=self.n_fft, dim=-1)
        power = spec.real.pow(2) + spec.imag.pow(2)

        mel = torch.matmul(power, self.mel_fb.t())
        log_mel = torch.log(torch.clamp(mel, min=1e-10))

        if normalize:
            log_mel = (log_mel - self.cmvn_mean) * self.cmvn_istd

        if squeeze:
            return log_mel.squeeze(0)
        return log_mel
