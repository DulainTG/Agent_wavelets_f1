"""Battle-Lemarié-inspired wavelet utilities."""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, Iterable

import numpy as np

Array = np.ndarray


@lru_cache(maxsize=1)
def _wavelet_fourier(size: int) -> Array:
    freqs = np.fft.fftfreq(size)
    omega = 2 * np.pi * freqs
    psi_hat = (1j * omega) ** 4 * np.exp(-(omega / 2) ** 2)
    psi_hat[freqs < 0] = 0.0
    return psi_hat


def _scaling_fourier(size: int) -> Array:
    freqs = np.fft.fftfreq(size)
    omega = 2 * np.pi * freqs
    phi_hat = np.exp(-(omega / 2) ** 2)
    return phi_hat


def dyadic_scales(j_min: int, j_max: int) -> Array:
    return 2.0 ** np.arange(j_min, j_max + 1)


def wavelet_transform(x: Array, scales: Iterable[float]) -> Dict[float, Array]:
    n = len(x)
    fft_x = np.fft.fft(x)
    result: Dict[float, Array] = {}
    psi_hat = _wavelet_fourier(n)
    for scale in scales:
        filt = psi_hat * scale
        filt = np.roll(filt, int(scale) % n)
        Wx = np.fft.ifft(fft_x * filt)
        result[scale] = Wx
    return result


def modulus_transform(coeffs: Dict[float, Array]) -> Dict[float, Array]:
    return {scale: np.abs(values) for scale, values in coeffs.items()}


def time_average(x: Array) -> float:
    return float(np.mean(x))


def variance(x: Array) -> float:
    return float(np.mean(np.abs(x - np.mean(x)) ** 2))


def correlation_matrix(coeffs: Dict[float, Array]) -> Array:
    scales = list(coeffs.keys())
    n = len(scales)
    mat = np.zeros((n, n))
    for i, si in enumerate(scales):
        xi = coeffs[si]
        xi = xi - np.mean(xi)
        denom_i = np.sqrt(np.mean(np.abs(xi) ** 2)) + 1e-12
        for j, sj in enumerate(scales):
            xj = coeffs[sj]
            xj = xj - np.mean(xj)
            denom_j = np.sqrt(np.mean(np.abs(xj) ** 2)) + 1e-12
            mat[i, j] = np.mean(np.real(xi * np.conj(xj))) / (denom_i * denom_j)
    return mat


def cross_phase_modulus(phase_coeffs: Dict[float, Array], modulus_coeffs: Dict[float, Array]) -> Array:
    scales = list(phase_coeffs.keys())
    n = len(scales)
    mat = np.zeros((n, n))
    for i, si in enumerate(scales):
        xi = phase_coeffs[si]
        xi = xi - np.mean(xi)
        denom_i = np.sqrt(np.mean(np.abs(xi) ** 2)) + 1e-12
        for j, sj in enumerate(scales):
            mj = modulus_coeffs[sj]
            mj = mj - np.mean(mj)
            denom_j = np.sqrt(np.mean(np.abs(mj) ** 2)) + 1e-12
            mat[i, j] = np.mean(np.real(xi) * mj) / (denom_i * denom_j)
    return mat


def scale_average(features: Dict[float, float]) -> float:
    values = np.array(list(features.values()))
    return float(values.mean())


def resample_to_length(signal: Array, length: int) -> Array:
    if len(signal) == length:
        return signal
    freqs = np.fft.rfft(signal)
    resampled = np.fft.irfft(freqs, n=length)
    return resampled

