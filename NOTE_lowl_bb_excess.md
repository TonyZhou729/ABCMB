# OPEN: low-ℓ raw tensor-BB ~0.4% excess vs converged CLASS (2026-06-15)

For a follow-up session to chase down. **Not** the §5 ℓ=477 item — that one
is resolved (it was CubicSpline interpolation; see bottom).

## The symptom

ABCMB raw tensor BB (lensing off, r=0.1, fiducial LCDM) is systematically
**~0.4% HIGH at low ℓ** vs a high-precision tensor-only CLASS reference
(`class_tensor_hp_reference()` in `pytests/accuracy_test_bb.py`), decaying
smoothly to the recomb-tail floor:

| ℓ | 2 | 10 (peak) | 64 | 100 | 152 | 237 | 331 | 490 |
|---|---|---|---|---|---|---|---|---|
| ABCMB/CLASS−1 | +3.3e-3 | **+4.2e-3** | +2.6e-3 | +1.8e-3 | +9e-4 | +1.3e-3 | +5e-4 | ~0 |

It is at the spline **nodes** (ABCMB computes those ℓ exactly), so it is a
real transfer-level difference, not interpolation. It is smooth, single-
signed (ABCMB high), broad (ℓ=2 → ~300), peaks at ℓ≈10.

Pre-existing: the original handoff only spot-checked ℓ≥237 (saw ~1e-3, called
it "sub-permille, correct") and never characterized ℓ<200, where the same
bias is ~4× larger. It was masked in the old test because the ℓ=477
interpolation spike (6.9e-3) was the reported max.

## Ruled OUT (with evidence) — do not re-investigate these

1. **CubicSpline interpolation.** The excess is *at* nodes. Interpolation
   only explains the separate ℓ=477 spike. (`diag_bb_dense_hp.py`
   CLASS-internal self-interp test: ~7e-3 mid-gap, 0 at nodes.)
2. **`tensor_method` mismatch.** CLASS's *default* `tensor_method` is
   `tm_massless_approximation` (`class_EDE/source/input.c:3650`), the same
   method ABCMB uses. The reference is already apples-to-apples on neutrino
   treatment.
3. **`n_t` / tilt.** ABCMB's n_t (`main.py:423`,
   `-r/8*(2-r/8-n_s)`) is byte-identical to the test's `n_t_scc`. A tilt-fit
   to the shape was coincidental.
4. **Solver tolerance.** Tolerance ladder `diag_bb_tol_ladder.py` (A=1e-5/1e-9
   default, B=1e-6/1e-10, C=1e-7/1e-11): **max |A/C−1| over ℓ≤100 = 1.4e-5**.
   The default `rtol_ten=1e-5` is fully converged at low ℓ. (Tightening only
   nudges the *high*-ℓ tail by ~1e-4 — the landmine-#5 effect — so the
   default is fine; do not tighten it for this.)

## Remaining candidates (where to look)

A **converged, structural ABCMB-vs-CLASS difference**, concentrated at low ℓ
(reion-bump + large-scale region). Most likely, cheapest first:

- **Reionization-bump tensor source.** The excess peaks at ℓ≈10 and is broad.
  First test: set `tau_reion ≈ 0` in BOTH codes and re-measure low-ℓ nodes.
  If the excess collapses → it's the reion contribution to the tensor source
  (visibility g into `S_T2 = -hdot·e^{-κ} + g·Π`, `S_E = √6·g·Π`); compare
  ABCMB's reion handling in the tensor source vs CLASS. If it persists →
  not reion, go to the next item.
- **Low-k tensor transfer quadrature.** `get_tensor_k_axes` (tensors.py:29)
  is the *scalar* k-grid truncated at the tensor k_max — tuned for scalars,
  not for the tensor BB transfer at low k. Low ℓ ↔ low k. Test: densify the
  tensor k grid at low k and see if low-ℓ moves. Compare low-k node density
  vs CLASS's tensor q-sampling.
- **Large-scale / SW-analog source term.** The `-hdot·e^{-κ}` (ISW-analog)
  integral dominates low ℓ; a small difference in how it is integrated over
  lna at large scales vs CLASS.

## Diagnostics already built (in worktree root)

- `diag_bb_lowl_nodes.py` — node-by-node ABCMB vs hp-CLASS at low ℓ (the
  characterization above). **Start here.**
- `diag_bb_tol_ladder.py` — the tolerance ladder that ruled out solver tol.
- `diag_bb_lowl_3way.py` — ABCMB vs default-CLASS vs hp-CLASS (run it to also
  confirm default and hp CLASS agree at low ℓ → the difference is robustly on
  the ABCMB side, not an hp-reference artifact; this was set up but superseded
  by the tol ladder, which already settled the convergence question).
- `diag_bb_dense_hp.py` — the resolved §5 interpolation diagnostic.

All run on GPU (~10 min each): `salloc ... --gpus=1`; `JAX_PLATFORM_NAME=gpu`.

## How the test currently handles it (`pytests/accuracy_test_bb.py::test_bb_raw`)

- recomb nodes ℓ≥100: held to **2.5e-3** (actual max ~1.6e-3 — clean).
- low-ℓ nodes 3≤ℓ<100: held to **5e-3** (actual max 4.2e-3 — this excess).
- full band 3≤ℓ≤490: **1e-2** (interp-limited; the ℓ=477 spline residual).
- ℓ=2 exempt; 491–500 cutoff sliver excluded.

When the low-ℓ cause is found and fixed, tighten the low-ℓ node bar back to
2.5e-3 (or merge it with the ℓ≥100 band).

---

### Resolved this session (2026-06-15): §5 ℓ=477 interpolation residual

The original open item — raw-BB 6.9e-3 at ℓ=477 — was **CubicSpline
interpolation error over the 40-wide node gap (450→490)**, not transfer
physics. Proven by the CLASS-internal self-interpolation test
(`diag_bb_dense_hp.py`): re-splining CLASS's OWN node values through the
identical `interpax.CubicSpline` reproduces ~7e-3 at ℓ=477 with zero error at
the nodes, matching the observed residual. Nodes are already maximally dense
(`bessel_l_tab`; the phi-tables exist only there), so finer tensor nodes would
need new Bessel tables (cf. the separate Bessel-recurrence effort). For now
the test holds the interp-limited band to 1%, the same floor the scalar
spectra use.
