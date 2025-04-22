import math
import os
import random
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import librosa
import librosa.util as librosa_util
from librosa.util import normalize, pad_center, tiny
from scipy.signal import get_window
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output


mel_basis = {}
hann_window = {}


def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    # collapse any channel dimension (e.g. stereo → mono)
    if y.dim() == 3 and y.size(2) > 1:
        y = y.mean(dim=2)
    elif y.dim() == 2 and y.size(1) > 1:
        y = y.mean(dim=1)
    
    # ensure [batch, time]
    if y.dim() == 1:
        y = y.unsqueeze(0)
    # pad along time axis only
    pad_amount = (n_fft - hop_size) // 2
    y = F.pad(y, (pad_amount, pad_amount), mode='constant', value=0)

    # compute STFT
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window.setdefault(f"{win_size}_{y.dtype}_{y.device}",
                                      torch.hann_window(win_size, dtype=y.dtype, device=y.device)),
        center=center,
        pad_mode='reflect',
        normalized=False,
        return_complex=False,
        onesided=True
    )
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec


def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    global mel_basis
    dtype_device = str(spec.dtype) + '_' + str(spec.device)
    fmax_dtype_device = str(fmax) + '_' + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=spec.dtype, device=spec.device)
    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = spectral_normalize_torch(spec)
    return spec


def mel_spectrogram_torch(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    # collapse any channel dimension
    if y.dim() == 3 and y.size(2) > 1:
        y = y.mean(dim=2)
    elif y.dim() == 2 and y.size(1) > 1:
        y = y.mean(dim=1)
    
    # ensure [batch, time]
    if y.dim() == 1:
        y = y.unsqueeze(0)
    
    # ensure signal is at least n_fft long (required by torch.stft)
    length = y.size(-1)
    if length < n_fft:
        y = F.pad(y, (0, n_fft - length), mode='constant', value=0)

    # center‐pad for STFT windowing
    pad_amount = (n_fft - hop_size) // 2
    y = F.pad(y, (pad_amount, pad_amount), mode='constant', value=0)
    
    # compute STFT
    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=hann_window.setdefault(f"{win_size}_{y.dtype}_{y.device}",
                                      torch.hann_window(win_size, dtype=y.dtype, device=y.device)),
        center=center,
        pad_mode='reflect',
        normalized=False,
        return_complex=False,
        onesided=True
    )
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    
    # apply Mel filterbank
    key = f"{fmax}_{spec.dtype}_{spec.device}"
    if key not in mel_basis:
        m = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        mel_basis[key] = torch.from_numpy(m).to(dtype=spec.dtype, device=spec.device)
    spec = torch.matmul(mel_basis[key], spec)
    
    return spectral_normalize_torch(spec)