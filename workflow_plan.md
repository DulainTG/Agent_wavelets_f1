# Experimental Workflow Extraction for "Scale Dependencies and Self-Similar Models with Wavelet Scattering Spectra"

## Data Acquisition or Simulation
- **Simulated processes for benchmarking**
  - Generate fractional Brownian motion realizations with stationary increments and scaling exponent \(\zeta_2 = 1\) (standard Brownian motion); ensure wavelet sparsity parameters satisfy \(c_s = \pi/4\) and \(\zeta_s = 0\).【F:main.tex†L1116-L1121】
  - Simulate signed Poisson jump processes with intensity \(\lambda\), assigning each jump a random \(\pm 1\) sign; capture both regimes \(2^j \ll \lambda^{-1}\) and \(2^j \gg \lambda^{-1}\) to expose non self-similar scaling.【F:main.tex†L1133-L1146】
  - Implement multifractal random walks (MRW) where increments \(\delta_j X(t) = \delta_j B(t) e^{\Omega_j(t)}\) and \(\Omega_j\) is a Gaussian log-correlated field with logarithmic covariance governed by \(\lambda^2\); reproduce skewed MRW variant using causal power-law kernel \(k(t)\).【F:main.tex†L1175-L1203】
  - Simulate quadratic Hawkes point processes with kernels \(h(t) \sim |t|^{-1.2}\), \(g(t) \sim e^{-0.01|t|}\), and quadratic kernel \(k(t) \sim e^{-0.03|t|}\) normalized so \(\int(|h|+|k|^2)=0.9\); include random \(\pm 1\) jump signs.【F:main.tex†L1205-L1227】
- **Financial dataset**
  - Collect S&P 500 log-price series sampled every 5 minutes from 3 Jan 2000 to 10 Oct 2018 (\(N=7.5\times10^5\) samples) following preprocessing described in Appendix.【F:main.tex†L1244-L1272】【F:main.tex†L1658-L1666】
- **Turbulence dataset**
  - Acquire helium jet velocity measurements (Reynolds number 929) with \(N = 3.5\times10^7\) samples for analysis.【F:main.tex†L1279-L1299】
- **Additional references**
  - Source code reference implementation available at https://github.com/RudyMorel/scattering_spectra for cross-checking parameterizations.【F:main.tex†L279-L286】

## Preprocessing Pipeline
- **Wavelet framework**: Use the complex Battle-Lemarié wavelet restricted to positive frequencies, with exponential decay and \(m=4\) vanishing moments; compute dyadic-scale wavelet transforms only for \(2^j>1\) (\(j\geq1\)).【F:main.tex†L431-L436】
- **S&P preprocessing**:
  - Replace 17,956 missing 5-minute increments with Gaussian noise using time-of-day statistics.【F:main.tex†L1658-L1661】
  - Remove intraday seasonality by normalizing with average volatility profiles.【F:main.tex†L1662-L1662】
  - Normalize overnight bins by average overnight volatility to reduce non-stationarity.【F:main.tex†L1663-L1664】
  - Mitigate tick-size discreteness via a 15-minute moving-average low-pass filter.【F:main.tex†L1665-L1666】
- **Scale selection**: Set maximum wavelet scale \(2^J\) at or below the process integral scale; ensure \(2^J < N\) when the integral scale is unknown to maintain estimator reliability.【F:main.tex†L1060-L1071】

## Models and Algorithms
- **Scattering spectra components**
  - Wavelet spectrum \(\sigma_W^2(j)\) per (4); estimate via time averaging as in (5).【F:main.tex†L469-L481】
  - Sparsity factor \(s_W^2(j)\) normalization formula (27).【F:main.tex†L1360-L1364】
  - Phase-modulus cross-spectrum \(\Phi_3\) coefficients normalized by \(\widetilde\sigma_W\).【F:main.tex†L1366-L1371】
  - Scattering cross-spectrum \(\Phi_4\) via second wavelet transform of the modulus; normalization as in (30).【F:main.tex†L1373-L1378】
- **Joint correlation analysis**
  - Compute joint phase-modulus correlation matrix (Fig. 4–6) using normalized coefficients and Toeplitz structure assumptions.【F:main.tex†L622-L676】
  - Derive scale-invariant statistics \(\overline C_{W|W|}(a)\) and \(\overline C_S(a,b)\) via scale averaging (14, 16).【F:main.tex†L1018-L1038】
- **Process-specific modeling**
  - For MRW, enforce \(\zeta_s = \lambda^2\) matching Table parameters.【F:main.tex†L1183-L1189】【F:main.tex†L1099-L1105】
  - For skewed MRW and quadratic Hawkes, incorporate time-asymmetry through specified causal kernels to induce non-zero phases.【F:main.tex†L1197-L1203】【F:main.tex†L1215-L1228】

## Training and Generation Procedures
- **Estimator computation**: Use time averages over single long realizations to estimate scattering spectra; ensure sufficient blocks \(N/(2\alpha s)\) given wavelet support \(\alpha\) and integral scale \(s\) for convergence.【F:main.tex†L1062-L1067】
- **Scale-invariant aggregation**: Average scattering coefficients across scales for self-similar processes as in (13)–(16).【F:main.tex†L1010-L1038】
- **Microcanonical maximum-entropy synthesis**:
  - Define energy vector \(\Phi(x)\) with up to \(J^3\) features (\(J\leq \log_2 N\)).【F:main.tex†L1379-L1380】
  - Form microcanonical set \(\Omega_\epsilon\) with \(\epsilon = 10^{-3}\|\Phi(\widetilde{x})\|_2\).【F:main.tex†L1674-L1679】
  - Initialize synthesis from Gaussian white noise and apply L-BFGS-B gradient descent on \(\ell(x)=\|\Phi(x)-\Phi(\bar x)\|_2^2\) until convergence.【F:main.tex†L1686-L1694】
  - For financial and turbulence data, synthesize sequences of length \(N=7\times10^5\) using \(J=11\) scales (375 parameters).【F:main.tex†L1450-L1456】

## Evaluation Metrics and Methodology
- **Visual diagnostics**: Replicate scalograms and power spectra overlays to visualize cross-scale dependencies (Figures 2–3).【F:main.tex†L514-L603】
- **Scattering spectra plots**: Plot \(\sigma_W^2(j)\), \(s_W^2(j)\), \(\overline C_{W|W|}(a)\), and \(\overline C_S(a,b)\) with error bars showing scale-variance to assess self-similarity and asymmetry.【F:main.tex†L1073-L1171】
- **Self-similarity assessment**: Verify that scale-averaged coefficients are independent of \(j\); interpret error bars relative to estimator variance.【F:main.tex†L1255-L1272】【F:main.tex†L1293-L1299】
- **Test moments for model validation**:
  - Empirical CDFs of finest-scale increments.【F:main.tex†L1416-L1416】
  - Marginal moments for \(q\in[0.3,3]\) across scales.【F:main.tex†L1418-L1419】
  - Leverage correlations at scale \(2^j=159\) (day) for all processes except jet (\(2^j=1\)).【F:main.tex†L1421-L1429】
  - Zumbach integrals using cumulative asymmetry of volatility for the same scales.【F:main.tex†L1430-L1443】
  - Display estimator variance as error bars, emphasizing high-variance fourth-order statistics.【F:main.tex†L1455-L1459】

## Empirical Results to Reproduce
1. **Figure 1** – Real/imaginary Battle-Lemarié wavelet and Fourier magnitude.【F:main.tex†L431-L444】
2. **Figure 2** – Scalograms and wavelet coefficient phases for signed Poisson vs. S&P daily increments (2000–2018).【F:main.tex†L514-L522】【F:main.tex†L538-L544】
3. **Figure 3** – Power spectrum overlap before/after modulus for S&P data.【F:main.tex†L587-L604】
4. **Figure 4** – Wavelet auto-correlation matrix magnitude \(\E\{WXWX^T\}\) for S&P.【F:main.tex†L651-L657】
5. **Figure 5** – Phase-modulus correlation \(\E\{WX|WX|^T\}\) for S&P.【F:main.tex†L651-L663】
6. **Figure 6** – Modulus auto-correlation \(\E\{|WX||WX|^T\}\) for S&P.【F:main.tex†L651-L669】
7. **Figure 7** – Wavelet power spectrum \(\sigma_W^2(j)\) across processes.【F:main.tex†L1073-L1089】
8. **Figure 8** – Sparsity factor \(s_W^2(j)\) across processes.【F:main.tex†L1073-L1089】
9. **Table 1** – Self-similarity parameters \(\zeta_2, c_s, \zeta_s\) for all processes including jet and S&P.【F:main.tex†L1091-L1112】
10. **Figure 9** – Scale-invariant phase-modulus cross-spectrum magnitude/phase with error bars for skewed MRW and quadratic Hawkes.【F:main.tex†L1151-L1159】
11. **Figure 10** – Scale-invariant scattering cross-spectrum magnitude/phase with error bars for skewed MRW and quadratic Hawkes.【F:main.tex†L1163-L1172】
12. **Figure 11** – Original vs. synthesized increments plus CDF, marginal moments, leverage, and Zumbach comparisons for all processes; include error bars for higher-order statistics.【F:main.tex†L1387-L1459】

## Dependencies and Environment Requirements
- Ability to perform complex wavelet transforms with Battle-Lemarié filters satisfying Littlewood–Paley condition; support for dyadic scales and positive-frequency restriction.【F:main.tex†L431-L436】【F:main.tex†L1484-L1497】
- Numerical routines for modulus wavelet transforms, second-layer wavelet analysis, and normalization by empirical \(\widetilde\sigma_W\).【F:main.tex†L469-L481】【F:main.tex†L1360-L1378】
- Facilities to compute large Toeplitz-like correlation matrices and to average over scales/time for single long realizations.【F:main.tex†L622-L676】【F:main.tex†L1010-L1067】
- Optimization toolkit supporting L-BFGS-B for gradient descent-based microcanonical sampling, with capacity to initialize from Gaussian noise and enforce \(\epsilon=10^{-3}\|\Phi(\widetilde x)\|_2\) tolerance.【F:main.tex†L1674-L1694】
- Computational resources to handle long sequences (up to \(3.5\times10^7\) samples) and multiple scales (e.g., \(J=11\)) when estimating scattering spectra and synthesizing sequences.【F:main.tex†L1279-L1299】【F:main.tex†L1450-L1456】
- External dataset access for S&P 5-minute data and helium jet turbulence measurements; ensure preprocessing pipeline can be executed before wavelet analysis.【F:main.tex†L1248-L1272】【F:main.tex†L1658-L1666】
- Optional: reference implementation from the public repository for validation and reproducibility checks.【F:main.tex†L279-L286】
