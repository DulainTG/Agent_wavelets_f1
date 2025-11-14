"""End-to-end workflow reproduction."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np

from . import data, plots, scattering, wavelet


def build_wavelet_demo(n: int = 512) -> Dict[str, np.ndarray]:
    t = np.linspace(-4, 4, n)
    omega = 2 * np.pi
    wavelet_real = (t ** 4 - 6 * t ** 2 + 3) * np.exp(-t ** 2 / 2)
    wavelet_imag = (t ** 3 - 3 * t) * np.exp(-t ** 2 / 2)
    psi = wavelet_real + 1j * wavelet_imag
    wavelet_ft = np.fft.fftshift(np.fft.fft(psi))
    return {"t": t, "real": wavelet_real, "imag": wavelet_imag, "ft": wavelet_ft}


def run(outputs: Path) -> None:
    outputs.mkdir(parents=True, exist_ok=True)

    demo = build_wavelet_demo()
    plots.plot_wavelet_functions(outputs, demo["t"], demo["real"], demo["imag"], demo["ft"])

    process_specs = data.predefined_processes()
    series = data.generate_all(process_specs.values())

    stats_by_name: Dict[str, scattering.ScatteringStats] = {}

    for name, signal in series.items():
        stats = scattering.compute_scattering(signal)
        stats_by_name[name] = stats
        plots.plot_scalograms(outputs, name, stats.coeffs)
        plots.plot_correlation_matrices(outputs, stats, prefix=f"{name.lower()}")
        plots.plot_scale_invariant(outputs, stats, prefix=f"{name.lower()}")
        diagnostics = scattering.compute_moment_diagnostics(signal, stats.scales, np.linspace(0.3, 3.0, 6))
        plots.plot_moment_diagnostics(outputs, name, diagnostics)
        synthesis = scattering.microcanonical_synthesis(signal)
        plots.plot_synthesis(outputs, name, synthesis)
        if name in {"SP500", "SignedPoisson"}:
            plots.plot_power_overlay(outputs, name, stats.coeffs, stats.modulus_coeffs)
        if name in {"SkewedMRW", "QuadraticHawkes"}:
            plots.plot_phase_modulus_overlays(outputs, name, stats)

    plots.plot_power_spectra(outputs, stats_by_name)

    summary_lines = []
    for name, stats in stats_by_name.items():
        summary = scattering.summarize_stats(stats)
        summary_lines.append(
            f"{name}: phase={summary['scale_invariant_phase']:.4f}, modulus={summary['scale_invariant_modulus']:.4f}, sparsity={summary['avg_sparsity']:.4f}"
        )
    (outputs / "table1_summary.txt").write_text("\n".join(summary_lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce wavelet scattering experiments")
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args.outputs)


if __name__ == "__main__":
    main()

