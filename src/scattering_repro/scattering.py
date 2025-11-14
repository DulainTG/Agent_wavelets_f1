"""Scattering statistics computation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from . import wavelet

Array = np.ndarray


@dataclass
class ScatteringStats:
    scales: List[float]
    wavelet_power: Dict[float, float]
    sparsity: Dict[float, float]
    modulus_power: Dict[float, float]
    coeffs: Dict[float, Array]
    modulus_coeffs: Dict[float, Array]
    phase_modulus_corr: np.ndarray
    modulus_corr: np.ndarray
    phase_corr: np.ndarray
    scale_invariant_phase: float
    scale_invariant_modulus: float


def _energy(x: Array) -> float:
    return float(np.mean(np.abs(x) ** 2))


def _sparsity(x: Array) -> float:
    return float(np.mean(np.abs(x))) / (np.sqrt(_energy(x)) + 1e-12)


def compute_scattering(x: Array, j_min: int = 1, j_max: int | None = None) -> ScatteringStats:
    n = len(x)
    if j_max is None:
        j_max = int(np.log2(n)) - 2
    scales = list(wavelet.dyadic_scales(j_min, j_max))
    coeffs = wavelet.wavelet_transform(x, scales)
    modulus = wavelet.modulus_transform(coeffs)
    wavelet_power = {s: _energy(coeffs[s]) for s in scales}
    sparsity = {s: _sparsity(coeffs[s]) for s in scales}
    modulus_power = {s: _energy(modulus[s]) for s in scales}
    phase_corr = wavelet.correlation_matrix(coeffs)
    modulus_corr = wavelet.correlation_matrix({s: modulus[s] for s in scales})
    phase_modulus_corr = wavelet.cross_phase_modulus(coeffs, modulus)
    scale_invariant_phase = wavelet.scale_average(wavelet_power)
    scale_invariant_modulus = wavelet.scale_average(modulus_power)
    return ScatteringStats(
        scales=scales,
        wavelet_power=wavelet_power,
        sparsity=sparsity,
        modulus_power=modulus_power,
        coeffs=coeffs,
        modulus_coeffs=modulus,
        phase_modulus_corr=phase_modulus_corr,
        modulus_corr=modulus_corr,
        phase_corr=phase_corr,
        scale_invariant_phase=scale_invariant_phase,
        scale_invariant_modulus=scale_invariant_modulus,
    )


def summarize_stats(stats: ScatteringStats) -> Dict[str, float]:
    return {
        "scale_invariant_phase": stats.scale_invariant_phase,
        "scale_invariant_modulus": stats.scale_invariant_modulus,
        "avg_sparsity": float(np.mean(list(stats.sparsity.values()))),
    }


@dataclass
class MomentDiagnostics:
    cdf_x: Array
    cdf_y: Array
    moments: Dict[float, float]
    leverage: float
    zumbach: float


def compute_moment_diagnostics(x: Array, scales: Iterable[float], q_values: Iterable[float]) -> MomentDiagnostics:
    increments = np.diff(x)
    sorted_inc = np.sort(increments)
    cdf_y = np.linspace(0, 1, increments.size, endpoint=False)
    q_values = list(q_values)
    moments = {q: float(np.mean(np.abs(increments) ** q)) for q in q_values}
    leverage_scale = int(scales[len(scales) // 2])
    shift = min(leverage_scale, increments.size - 1)
    leverage = float(np.mean(increments[:-shift] * np.abs(increments[shift:])))
    zumbach = float(np.mean(np.sign(increments[:-shift]) * np.abs(increments[shift:])))
    return MomentDiagnostics(
        cdf_x=sorted_inc,
        cdf_y=cdf_y,
        moments=moments,
        leverage=leverage,
        zumbach=zumbach,
    )


@dataclass
class SynthesisResult:
    original: Array
    synthesized: Array


def microcanonical_synthesis(target: Array, iterations: int = 50, seed: int = 0) -> SynthesisResult:
    rng = np.random.default_rng(seed)
    synth = rng.normal(size=target.size)
    synth -= synth.mean()
    synth *= np.std(target) / (np.std(synth) + 1e-12)
    for _ in range(iterations):
        target_fft = np.fft.fft(target)
        synth_fft = np.fft.fft(synth)
        synth = np.fft.ifft(np.sqrt(np.abs(target_fft) ** 2) * np.exp(1j * np.angle(synth_fft))).real
        synth -= synth.mean()
        synth *= np.std(target) / (np.std(synth) + 1e-12)
    return SynthesisResult(original=target, synthesized=synth)

