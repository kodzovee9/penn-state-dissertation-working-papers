# Replication package

**Paper:** "Ability, Credit Constraints, and the Observed Returns to Child Work"
**Author:** Kodzovi Senu Abalo (World Bank), kabalo@worldbank.org

## Requirements

Python 3.9+ with `numpy`, `scipy`, `matplotlib` (see `requirements.txt`).
All scripts are deterministic (fixed random seeds). Total runtime is roughly
10–15 minutes on a standard laptop. Each script is self-contained (no imports
across scripts) and writes its figures to the working directory.

## Scripts and the exhibits they generate

Run in numerical order to regenerate every exhibit:

| Script | Paper exhibits | Approx. runtime |
|---|---|---|
| `01_population_economies.py` | Table 2 (economies E1–E4); Figure 2 (`naive_coefficient_sweeps.png`); decomposition checks | ~2 min |
| `02_specification_ladder.py` | Figure 3 (`specification_ladder.png`), noise-free, N = 200,000 | ~1 min |
| `03_thresholds_correlation_welfare.py` | Figure 1 (`regime_map_kappa.png`); Figure 4 (`rho_decomposition.png`); Table 3 (welfare); threshold verifications reported in Appendix B | ~4 min |
| `04_policy_experiments.py` | Figures 5–6 (`policy_credit.png`, `policy_subsidy.png`); Section 5 policy numbers | ~2 min |
| `05_reversal_boundary.py` | Appendix B: exact constrained reversal boundary verification and the illustrative constrained-reversal example (φ/κ ≈ 10.8) | ~2 min |
| `06_constrained_share_and_drift.py` | Figure 7 (`conshare_robustness.png`); Appendix B: negative-selection drift check (economy E1) | ~2 min |
| `07_delta_robustness.py` | Section 4.5: skill-building channel (δ > 0) robustness numbers | ~2 min |

## Key numerical claims and where they are produced

- Table 2 naive coefficients (−2.83, −0.10, +0.68, +0.68) and their
  displacement/selection components: script 01.
- Full-specification exact recovery (0, α, κ): script 02.
- Zero sign mismatches for the selection thresholds of Propositions 1–2 and the
  reversal boundaries of Proposition 3; σ* = 1 knife-edge at d̄ = 0: scripts 03
  and 05.
- Consumption-equivalent welfare gains (mean +11.0%; +20.5% poorest quintile to
  +2.8% richest at d̄ = 0.6): script 03, part D.
- Policy drift +(−0.07 → +0.79) in E2 and (+0.71 → +1.42) in E3: script 04;
  opposite drift (−2.83 → −4.18) in the negative-selection economy E1: script 06.
- Positive naive coefficients at constrained shares from 26% to 100%: script 06.
- Interior-margin lemma with δ up to 0.45 (min interior m = 0.147): script 07.

## Model summary (for orientation)

Two-period unitary household; CRRA utility (curvature σ = inverse EIS); child
time split between work l and schooling s = 1 − l; child wage w·a^φ; human
capital a^κ·s^α + δ·l; borrowing d ≤ d̄ at gross rate R; schooling cost θ.
Baseline illustration parameters: α = 0.5, w = 0.3, w₂ = 3.5, θ = 0.25,
R = 1.4, β = 0.8, κ = 1, δ = 0, d̄ = 0. The solvers use the closed-form
unconstrained optimum and vectorized bisection on the constrained first-order
condition; corner solutions at l = 0 are handled explicitly.
