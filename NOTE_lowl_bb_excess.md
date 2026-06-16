# low-ℓ raw tensor-BB ~0.4% excess vs CLASS — ROOT (half) FOUND 2026-06-16

**Not** the §5 ℓ=477 item (resolved: CubicSpline interpolation; see bottom).

## ROOT CAUSE (≈half the excess): tensor integration STARTS TOO LATE.

`TensorPerturbationEvolver.evolution_one_k` (tensors.py:376) caps the start at
lna=-10 (z~22000) — too late for the photon polarization quadrupole Π to settle
before recomb. Fixed-start sweep (`diag_bb_starttime.py`): BB is start-converged
only for t0 ≤ -13 (ℓ=10: cap -10 → +4.20e-3; -11 → +2.98e-3; -13/-16/-22 →
+2.00e-3). Fix = cap at -14 (`proposed_diff_tensor_starttime.patch`).
End-to-end verify (`diag_bb_starttime_verify.py`): improves EVERY node ℓ=2-450,
**zero wall-clock cost** (6.84→6.87s); low-ℓ max 4.20e-3→3.09e-3, recomb max
1.59e-3→0.85e-3. Earlier start is the more-correct answer (closer to CLASS,
which is start/TCA-independent). The SAME -10 cap is in the scalar evolver
(perturbations.py:279) — NOT changed (scalars pass + are far less sensitive;
re-validate before touching).

**After the fix, two residuals remain:** (a) ℓ=2-4 ~+3.0e-3 (reion-bump region,
start-insensitive — likely the ~0.5% HyRex-vs-C-HYREC-2 visibility-wing diff,
see #9); (b) a converged ~1-1.7e-3 recomb-region residual (the genuine
inter-code Π difference below, now ~halved). Both are within inter-code scatter
for raw tensor BB; (a) is the next target if pursued.

## The symptom

ABCMB raw tensor BB (lensing off, r=0.1, fiducial LCDM) is systematically
**~0.4% HIGH at low ℓ** vs high-precision tensor-only CLASS
(`class_tensor_hp_reference()`), decaying smoothly to the recomb-tail floor:

| ℓ | 2 | 10 (peak) | 64 | 100 | 152 | 237 | 331 | 490 |
|---|---|---|---|---|---|---|---|---|
| ABCMB/CLASS−1 | +3.3e-3 | **+4.2e-3** | +2.6e-3 | +1.8e-3 | +9e-4 | +1.3e-3 | +5e-4 | ~0 |

At the spline **nodes** (computed exactly) — a real transfer-level difference,
not interpolation. Smooth, single-signed (ABCMB high), broad, peaks at ℓ≈10,
→0 by ℓ≈490 (so it is k-dependent, largest for low-k / super-horizon-at-recomb
modes).

## CONCLUSION (this session): genuine inter-code difference in the tensor
## moment evolution, established in the tight-coupling epoch. NOT a TCA effect,
## NOT recombination, NOT a localizable bug.

The excess lives in the **polarization source** `source_E = √6·g·Π`. Perturbation-
level ABCMB-vs-CLASS at recomb (`diag_bb_pi_pert_xcheck.py`,
`diag_bb_tca_origin.py`): GW amplitude **h is exact (+0.004% at ℓ=10)**, but the
**polarization quadrupole Π is +0.1% high**, which squares up to +0.4% in
`Cl_BB ∝ |∫√6·g·Π·radB|²`. EE carries the identical excess (shared source); TT
carries ~⅔ (its `−hdot·e^{−κ}` term is cleaner). The Π residual is established
by z≈1400 (tight-coupling era) and *decreases* toward recomb (z=1400 +0.16% →
z=950 +0.08%) — the flavor of an early-time IC transient. It is in the larger
moments (δ_g, G0, G2), NOT the tiny frozen shear_g.

**Who is right (h is exact, so this is the polarization quadrupole only):**
undetermined, but it is NOT a TCA artifact on either side. Pushing CLASS's TCA
triggers 1.5e-3 → 1e-7 (CLASS runs full-hierarchy for ~all the evolution)
moves CLASS BB by **+0.006%** — i.e. TCA accounts for only ~1.4% of the 0.4%
gap (`diag_class_minimal_tca.py`). CLASS is TCA-independent/converged; ABCMB's
full-hierarchy result differs by +0.4% for a non-TCA reason. For raw low-ℓ
tensor BB (a tiny observable) a 0.4% inter-code difference is within normal
CLASS-vs-CAMB-class scatter.

## Ruled OUT this session (with evidence; do NOT re-investigate)

1. **Reionization.** `diag_bb_srcgrid_reion.py` part C: reion fraction of BB
   swings 99%→0% from ℓ=2→12, excess stays flat +4e-3. Decisive.
2. **k-transfer grid** (`diag_bb_kgrid_conv.py` A): x1/x2/x4 → ℓ=10 stable 1e-5.
3. **k-source/perturbation grid** (`diag_bb_srcgrid_reion.py` B): source
   x1/x2/x4 re-solve, no trend.
4. **lna/time grid** (`diag_bb_lna_conv.py`): Nlna 500→4000 changes ℓ=10 by <1e-8.
5. **CLASS under-convergence at low ℓ** (`diag_class_lowl_ladder.py`): CLASS
   converged to <0.01% across k/q/tol/time/TCA; converging it *widens* the gap.
6. **Equations / ICs / radials / source-formula / GW-eq / truncations.**
   Line-by-line vs CLASS perturbations.c + transfer.c, flat limit: all verbatim
   (P2, F/G/U hierarchies, l_max closures, radB=½(j'+2j/x), GW eq, IC=gw_ini/√6).
7. **GW-source densities** (`diag_gw_source_norm.py`): ρ_unit·ρ_g, ρ_unit·ρ_ν,
   and ρ_ν/ρ_g all match CLASS to ~2e-6 (ppm). Not a normalization bug.
8. **GW amplitude h**: exact (+0.004% at ℓ=10, perturbation level).
9. **Recombination is NOT the driver.** Both ABCMB (HyRex) and default CLASS
   use HYREC-2 (CLASS 3.3.4 default `recombination=hyrec`, input.c:5772). ABCMB
   xe(z) is only +0.07% off C-HYREC-2 (`diag_recomb_hyrec_recfast.py`); the
   HyRec↔RecFast spread moves BB only ~0.07%; ABCMB BB sits +0.42% above HyRec
   AND +0.35% above RecFast (above *both*). Recomb explains ≲0.07% of the 0.4%.
   (The visibility g wings differ ~0.5% antisymmetrically but that mostly
   cancels in the BB integral — proven by the small HyRec↔RecFast BB spread.)
10. **Solver under-resolution of the tight-coupling quadrupole**
    (`diag_bb_dtmax_test.py`): shear_g is frozen at ~3e-12 (below atol_ten=1e-9)
    through tight coupling, BUT forcing resolution (dtmax 0.005, atol 1e-13)
    changes BB by <2e-5. shear_g is real-but-irrelevant (negligible Π weight).
11. **Tensor TCA (CLASS-side) / gw_source photon-drop during TCA.** See
    conclusion: CLASS TCA-independent → not it. The photon gw_source term CLASS
    drops during TCA is negligible (δ_g, shear_g suppressed in tight coupling).
12. **Scalar EE cross-check** (`diag_scalar_ee_xcheck.py`): scalar EE is clean
    (mean −3e-4 over ℓ=100–1500), so this is tensor-specific, not a shared
    visibility/recomb bug at the 0.4% level.

## Only untested structural difference left

**Early-time start / IC transient.** ABCMB starts every tensor mode at
lna ≤ −15 (z > 3e6) with exactly-zero moments (h=1/√6); CLASS starts later,
per-mode. The Π bias being largest early and decaying toward recomb is
consistent with an IC transient that depends on the start epoch. Test: sweep
ABCMB's tensor t0 (force earlier/later, check BB start-time convergence). If
insensitive → irreducible accumulated inter-code difference; if sensitive →
the start criterion. Not yet run.

## How the test handles it (`pytests/accuracy_test_bb.py::test_bb_raw`)

- recomb nodes ℓ≥100: held 2.5e-3 (actual ~1.6e-3).
- low-ℓ nodes 3≤ℓ<100: held 5e-3 (actual 4.2e-3 — this excess).
- full band: 1e-2 (interp-limited ℓ=477 residual). ℓ=2 exempt; 491–500 excluded.

Since the excess is NOT a fixable bug (verbatim-correct tensor code; within
inter-code scatter for raw tensor BB), the 5e-3 low-ℓ bar should STAY, with the
comment updated to "converged tensor-quadrupole inter-code difference, not
reion/grid/recomb/TCA" rather than "unknown structural diff." Do NOT tighten to
2.5e-3.

Diagnostics in worktree root: diag_bb_kgrid_conv.py, diag_bb_srcgrid_reion.py,
diag_bb_eett_xcheck.py, diag_class_lowl_ladder.py, diag_bb_lna_conv.py,
diag_bb_visibility_xcheck.py, diag_bb_pi_pert_xcheck.py, diag_scalar_ee_xcheck.py,
diag_recomb_hyrec_recfast.py, diag_gw_source_norm.py, diag_bb_tca_origin.py,
diag_bb_dtmax_test.py, diag_class_minimal_tca.py (+ logs).

---

### Resolved earlier (2026-06-15): §5 ℓ=477 interpolation residual

raw-BB 6.9e-3 at ℓ=477 = CubicSpline interpolation error over the 40-wide node
gap (450→490), not transfer physics. Proven by `diag_bb_dense_hp.py`
(CLASS-internal self-interp reproduces ~7e-3 at ℓ=477, 0 at nodes). Nodes are
maximally dense (bessel_l_tab); finer needs new Bessel tables.
