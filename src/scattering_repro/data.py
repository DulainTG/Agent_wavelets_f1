"""Data generation and loading utilities for reproducing wavelet scattering experiments."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable

import numpy as np

Array = np.ndarray


@dataclass
class ProcessSpec:
    """Specification describing a stochastic process to simulate."""

    name: str
    generator: Callable[[int, int], Array]
    length: int
    seed: int

    def generate(self) -> Array:
        rng = np.random.default_rng(self.seed)
        samples = self.generator(self.length, rng)
        return np.asarray(samples, dtype=float)


def fractional_brownian_motion(length: int, rng: np.random.Generator, hurst: float = 0.5) -> Array:
    """Simulate fractional Brownian motion via simple Gaussian increments."""
    scale = 1.0
    if hurst != 0.5:
        # approximate by rescaling increments to mimic Hurst exponent
        scale = length ** (hurst - 0.5)
    increments = rng.normal(scale=scale, size=length)
    return np.cumsum(increments)


def signed_poisson_jumps(length: int, rng: np.random.Generator, intensity: float = 0.1) -> Array:
    """Simulate a signed compound Poisson process."""
    times = rng.exponential(scale=1 / intensity, size=length * 2)
    arrival_times = np.cumsum(times)
    max_time = length
    arrival_times = arrival_times[arrival_times < max_time]
    increments = np.zeros(length)
    jumps = rng.choice([-1.0, 1.0], size=arrival_times.size)
    idx = np.minimum(arrival_times.astype(int), length - 1)
    np.add.at(increments, idx, jumps)
    return np.cumsum(increments)


def multifractal_random_walk(length: int, rng: np.random.Generator, lamb: float = 0.05) -> Array:
    """Simulate a multifractal random walk with log-normal cascade."""
    gaussian = rng.normal(size=length)
    freqs = np.fft.rfftfreq(length)
    freqs[0] = freqs[1]
    spectrum = (np.abs(freqs) + 1e-6) ** (-1.0)
    phase = rng.normal(size=freqs.size) * math.sqrt(lamb)
    omega = np.fft.irfft(spectrum * phase, n=length)
    increments = gaussian * np.exp(omega - np.var(omega) / 2)
    return np.cumsum(increments)


def skewed_mrw(length: int, rng: np.random.Generator, lamb: float = 0.05, skew: float = 0.2) -> Array:
    base = multifractal_random_walk(length, rng, lamb=lamb)
    kernel_size = min(1024, length)
    t = np.arange(kernel_size)
    kernel = (t + 1) ** (-1.5)
    kernel /= kernel.sum()
    asym = np.convolve(np.diff(base, prepend=base[0]), kernel, mode="same")
    return base + skew * asym


def quadratic_hawkes(length: int, rng: np.random.Generator, baseline: float = 0.05) -> Array:
    """Discrete-time approximation of a quadratic Hawkes process."""
    kernel_len = 1024
    t = np.arange(kernel_len)
    h = (t + 1) ** (-1.2)
    h /= h.sum()
    g = np.exp(-0.01 * t)
    g /= g.sum()
    k = np.exp(-0.03 * t)
    k /= np.linalg.norm(k)
    x = np.zeros(length)
    intensity = np.full(length, baseline)
    for i in range(1, length):
        start = max(0, i - kernel_len)
        window = slice(start, i)
        jumps = x[window]
        excitation = np.dot(h[: jumps.size][::-1], jumps)
        quad = np.dot(k[: jumps.size][::-1], jumps) ** 2
        intensity[i] = baseline + 0.2 * excitation + 0.1 * quad
        prob = max(intensity[i], 1e-6)
        event = rng.uniform() < prob
        sign = rng.choice([-1.0, 1.0])
        x[i] = x[i - 1] + event * sign
    return x


def turbulence_surrogate(length: int, rng: np.random.Generator) -> Array:
    """Generate a surrogate turbulence signal via filtered log-normal cascade."""
    increments = rng.normal(size=length)
    freqs = np.fft.rfftfreq(length)
    spectrum = (freqs + 1e-6) ** (-5.0 / 3.0)
    phase = np.exp(1j * rng.uniform(0, 2 * np.pi, size=freqs.size))
    signal = np.fft.irfft(np.sqrt(spectrum) * phase, n=length)
    return signal


def sp500_surrogate(length: int, rng: np.random.Generator) -> Array:
    """Generate a surrogate S&P 500 log-price series with intraday seasonality."""
    dt = 1 / (24 * 12)
    t = np.arange(length) * dt
    daily_cycle = 1 + 0.3 * np.sin(2 * np.pi * t * 24)
    noise = rng.normal(scale=0.01, size=length)
    increments = daily_cycle * noise
    return np.cumsum(increments)


def predefined_processes() -> Dict[str, ProcessSpec]:
    length_medium = 2 ** 14
    length_long = 2 ** 17
    return {
        "brownian": ProcessSpec("Brownian", fractional_brownian_motion, length_medium, seed=1234),
        "poisson": ProcessSpec("SignedPoisson", signed_poisson_jumps, length_medium, seed=5678),
        "mrw": ProcessSpec("MRW", multifractal_random_walk, length_medium, seed=1357),
        "skewed_mrw": ProcessSpec("SkewedMRW", skewed_mrw, length_medium, seed=2468),
        "hawkes": ProcessSpec("QuadraticHawkes", quadratic_hawkes, length_medium, seed=9876),
        "sp500": ProcessSpec("SP500", sp500_surrogate, length_long, seed=1122),
        "turbulence": ProcessSpec("Turbulence", turbulence_surrogate, length_long, seed=3344),
    }


def generate_all(processes: Iterable[ProcessSpec]) -> Dict[str, Array]:
    series: Dict[str, Array] = {}
    for spec in processes:
        series[spec.name] = spec.generate()
    return series

