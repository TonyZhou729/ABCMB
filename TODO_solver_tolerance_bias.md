# TODO: investigate `rtol_large_k_PE` high-ℓ amplitude bias (scalar sector)

**Filed 2026-06-12 from the B-modes work (branch `bmodes`).** Conspicuous
note on `main` because this is a *potential latent bias in the production
scalar pipeline*, not just a tensor-branch detail.

## The finding (tensor sector, confirmed)

The `PerturbationEvolver` solves each k-mode with `Kvaerno5` + a
`PIDController` whose large-k relative tolerance is
`rtol_large_k_PE = 1e-4` (for `k > k_split_PE = 0.01`; see
`model_specs.py`).

While building tensor B-modes I found that **this exact tolerance biases the
high-ℓ power spectrum LOW**, by an amount that grows with ℓ (≡ grows with k):

| ℓ | tensor BB: (rtol 1e-4) / (converged) |
|---|---|
| 237 | -0.02% |
| 296 | -0.04% |
| 450 | -0.7%  |
| 490 | -0.8%  |

Mechanism: accumulated ODE amplitude error over the long `lna` integration,
which grows with k. Tightening to `rtol 1e-5 / atol 1e-9` removed essentially
all of it (within 1e-4 of the fully converged answer). Evidence on the
`bmodes` branch: `diag_bb_tol.py` (CPU A/B), `diag_bb_gpu_tol.py` (GPU
A/B + cost), and `design_bmodes.md` (convergence section). The tensor solver
there was given its own tighter `rtol_ten=1e-5 / atol_ten=1e-9` to fix it.

## Why this matters for `main` (UNTESTED — the actual TODO)

The **scalar** TT/TE/EE/Pk solve uses the *same* `rtol_large_k_PE = 1e-4`.
So the scalar spectra plausibly carry a similar sub-percent low-bias at high
ℓ. Circumstantial support: in `pytests/accuracy_test.py` the largest
residuals vs CLASS sit at the **very highest ℓ** —

- TT: max rel err 2.6e-3 at ℓ≈2493
- EE: max rel err 4.2e-3 at ℓ≈2491

— exactly where a k-accumulating solver bias would manifest. Some of that
tail error may be solver tolerance rather than CLASS-side or interpolation.

## Suggested investigation

1. A/B the scalar pipeline: rerun `accuracy_test.py` (or a transfer-level
   probe) with `rtol_large_k_PE=1e-5`, `atol_large_k_PE=1e-8` and compare
   TT/EE at ℓ ~ 1500–2500 against the 1e-4 default and against CLASS.
2. Measure the wall-clock / memory cost of the tighter tolerance on the full
   scalar solve (the tensor sector showed it's cheap there, but the scalar
   system is Ny~45 over ~550 modes — verify the cost is acceptable).
3. If it materially improves high-ℓ accuracy at acceptable cost, consider
   tightening the `rtol_large_k_PE` / `atol_large_k_PE` defaults in
   `model_specs.py` (currently 1e-4 / 1e-6).

Note: this interacts with the reverse-AD path and the OLE training cost, so
weigh the accuracy gain against the per-call cost before changing the default.

— see branch `bmodes` and its `HANDOFF_bmodes.md` for full context.
