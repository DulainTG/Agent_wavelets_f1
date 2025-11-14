"""Plot generation for reproduced experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from .scattering import MomentDiagnostics, ScatteringStats, SynthesisResult


plt.style.use("seaborn-v0_8")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_wavelet_functions(outputs: Path, t: np.ndarray, wavelet_real: np.ndarray, wavelet_imag: np.ndarray, wavelet_ft: np.ndarray) -> None:
    _ensure_dir(outputs)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(t, wavelet_real, label="Re")
    axes[0].plot(t, wavelet_imag, label="Im")
    axes[0].set_title("Wavelet")
    axes[0].legend()
    axes[1].plot(t, np.abs(wavelet_real + 1j * wavelet_imag))
    axes[1].set_title("Magnitude")
    axes[2].plot(np.linspace(-0.5, 0.5, wavelet_ft.size), np.abs(wavelet_ft))
    axes[2].set_title("Fourier magnitude")
    fig.tight_layout()
    fig.savefig(outputs / "figure1_wavelet.png", dpi=200)
    plt.close(fig)


def plot_scalograms(outputs: Path, name: str, coeffs: Dict[float, np.ndarray]) -> None:
    _ensure_dir(outputs)
    matrix = np.stack([np.abs(c) for c in coeffs.values()])
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(matrix, aspect="auto", origin="lower")
    ax.set_title(f"Scalogram: {name}")
    ax.set_ylabel("Scale index")
    ax.set_xlabel("Time")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(outputs / f"scalogram_{name}.png", dpi=200)
    plt.close(fig)


def plot_power_spectra(outputs: Path, stats_by_name: Dict[str, ScatteringStats]) -> None:
    _ensure_dir(outputs)
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, stats in stats_by_name.items():
        ax.plot(stats.scales, list(stats.wavelet_power.values()), label=name)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Scale")
    ax.set_ylabel("Wavelet power")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outputs / "figure7_wavelet_power.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, stats in stats_by_name.items():
        ax.plot(stats.scales, list(stats.sparsity.values()), label=name)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Scale")
    ax.set_ylabel("Sparsity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outputs / "figure8_sparsity.png", dpi=200)
    plt.close(fig)


def plot_power_overlay(outputs: Path, name: str, coeffs: Dict[float, np.ndarray], modulus: Dict[float, np.ndarray]) -> None:
    _ensure_dir(outputs)
    fig, ax = plt.subplots(figsize=(6, 4))
    for scale, values in coeffs.items():
        spectrum = np.abs(np.fft.fft(values)) ** 2
        ax.plot(spectrum, alpha=0.4, label=f"scale {scale}")
    ax.set_yscale("log")
    ax.set_title(f"Power spectrum before modulus: {name}")
    fig.tight_layout()
    fig.savefig(outputs / f"figure3_{name.lower()}_before.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for scale, values in modulus.items():
        spectrum = np.abs(np.fft.fft(values)) ** 2
        ax.plot(spectrum, alpha=0.4, label=f"scale {scale}")
    ax.set_yscale("log")
    ax.set_title(f"Power spectrum after modulus: {name}")
    fig.tight_layout()
    fig.savefig(outputs / f"figure3_{name.lower()}_after.png", dpi=200)
    plt.close(fig)


def plot_correlation_matrices(outputs: Path, stats: ScatteringStats, prefix: str) -> None:
    _ensure_dir(outputs)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, matrix, title in zip(
        axes,
        [stats.phase_corr, stats.phase_modulus_corr, stats.modulus_corr],
        ["Phase-phase", "Phase-modulus", "Modulus-modulus"],
    ):
        im = ax.imshow(matrix, origin="lower", vmin=-1, vmax=1)
        ax.set_title(title)
        ax.set_xlabel("Scale")
        ax.set_ylabel("Scale")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(outputs / f"{prefix}_correlations.png", dpi=200)
    plt.close(fig)


def plot_scale_invariant(outputs: Path, stats: ScatteringStats, prefix: str) -> None:
    _ensure_dir(outputs)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Phase", "Modulus"], [stats.scale_invariant_phase, stats.scale_invariant_modulus])
    ax.set_title(f"Scale invariant stats: {prefix}")
    fig.tight_layout()
    fig.savefig(outputs / f"{prefix}_scale_invariant.png", dpi=200)
    plt.close(fig)


def plot_phase_modulus_overlays(outputs: Path, name: str, stats: ScatteringStats) -> None:
    _ensure_dir(outputs)
    magnitude = np.mean(np.abs(stats.phase_modulus_corr), axis=0)
    phase = np.angle(np.mean(stats.phase_modulus_corr + 1e-12, axis=0))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(range(len(magnitude)), magnitude, yerr=np.std(np.abs(stats.phase_modulus_corr), axis=0))
    ax.set_title(f"Phase-modulus magnitude: {name}")
    fig.tight_layout()
    fig.savefig(outputs / f"figure9_{name.lower()}_magnitude.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(range(len(phase)), phase, yerr=np.std(np.angle(stats.phase_modulus_corr + 1e-12), axis=0))
    ax.set_title(f"Phase-modulus phase: {name}")
    fig.tight_layout()
    fig.savefig(outputs / f"figure9_{name.lower()}_phase.png", dpi=200)
    plt.close(fig)

    magnitude_mod = np.mean(np.abs(stats.modulus_corr), axis=0)
    phase_mod = np.angle(np.mean(stats.modulus_corr + 1e-12, axis=0))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(range(len(magnitude_mod)), magnitude_mod, yerr=np.std(np.abs(stats.modulus_corr), axis=0))
    ax.set_title(f"Scattering magnitude: {name}")
    fig.tight_layout()
    fig.savefig(outputs / f"figure10_{name.lower()}_magnitude.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(range(len(phase_mod)), phase_mod, yerr=np.std(np.angle(stats.modulus_corr + 1e-12), axis=0))
    ax.set_title(f"Scattering phase: {name}")
    fig.tight_layout()
    fig.savefig(outputs / f"figure10_{name.lower()}_phase.png", dpi=200)
    plt.close(fig)


def plot_moment_diagnostics(outputs: Path, name: str, diagnostics: MomentDiagnostics) -> None:
    _ensure_dir(outputs)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].plot(diagnostics.cdf_x, diagnostics.cdf_y)
    axes[0, 0].set_title("CDF")
    axes[0, 1].bar(list(diagnostics.moments.keys()), list(diagnostics.moments.values()))
    axes[0, 1].set_title("Moments")
    axes[1, 0].bar(["Leverage"], [diagnostics.leverage])
    axes[1, 1].bar(["Zumbach"], [diagnostics.zumbach])
    fig.suptitle(f"Diagnostics: {name}")
    fig.tight_layout()
    fig.savefig(outputs / f"figure11_{name.lower()}_diagnostics.png", dpi=200)
    plt.close(fig)


def plot_synthesis(outputs: Path, name: str, synthesis: SynthesisResult) -> None:
    _ensure_dir(outputs)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(synthesis.original, label="Original")
    ax.plot(synthesis.synthesized, label="Synthesized", alpha=0.7)
    ax.legend()
    ax.set_title(f"Synthesis comparison: {name}")
    fig.tight_layout()
    fig.savefig(outputs / f"figure11_{name.lower()}_series.png", dpi=200)
    plt.close(fig)

