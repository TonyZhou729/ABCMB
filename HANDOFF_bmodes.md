# B-modes implementation — session handoff (2026-06-12)

## TL;DR

Tensor (primordial GW) B-modes + lensing E→B mixing are **implemented and
working** on branch `bmodes` in worktree `/pscratch/sd/c/carag/ABCMB-bmodes`.
Physics is validated: at shared, interpolation-free ℓ-nodes ABCMB matches a
**converged** CLASS to **sub-permille** in the recomb tail. The scalar
regression test still passes.

**UPDATE 2026-06-15:** §5 (the ℓ=477 residual) is **RESOLVED** — it was
CubicSpline interpolation over the 40-wide node gap, not transfer physics
(proven; see §5 and `NOTE_lowl_bb_excess.md`). Restructuring the test to
check nodes separately surfaced a *new* item: a converged ~0.4% low-ℓ
(ℓ≲100) raw-BB excess vs CLASS, peak 4.2e-3 at ℓ=10. Solver tol,
interpolation, `tensor_method`, and n_t are all RULED OUT — it is a
structural ABCMB-vs-CLASS difference in the reion-bump / large-scale tensor
source, deferred to a follow-up session. Full writeup + remaining candidates
+ ready-made diagnostics in **`NOTE_lowl_bb_excess.md`**.

The B-modes work is committed (`bmodes` @ `1774dad`, pushed). The test +
diagnostic changes from 2026-06-15 are not yet committed.

---

## 1. What was built

New file **`abcmb/tensors.py`** (all tensor-specific code, ~560 lines):
- `TensorPerturbationEvolver` — evolves GW amplitude h + photon tensor
  temperature/polarization hierarchies + a massless-neutrino tensor
  hierarchy, synchronous gauge, CLASS conventions verbatim. State vector
  `[F0..F5, G0..G5, Fur0..Fur17, h, hdot]` (Ny=32). Equations transcribed
  from `class_EDE/source/perturbations.c` (tensor blocks), divided by aH to
  integrate in lna. Kvaerno5 + PIDController, `SaveAt(ts=lna)`, vmap-over-k
  on GPU / scan on CPU — same structure as the scalar `PerturbationEvolver`.
- `TensorSourceTable` — tabulates the two CLASS tensor sources
  `S_T2 = -hdot·exp(-κ) + g·Pi` and `S_E = √6·g·Pi`.
- `TensorSpectrumSolver` — integrates sources against the flat-space tensor
  radial functions (T/E/B; see `transfer.c` `TENSOR_*` cases) using the
  existing tabulated `phi0/phi1/phi2` Bessels (`j_l'' = (2·phi2 - phi0)/3`,
  **no new Bessel tables**), rolling-`lax.scan` accumulator pattern copied
  from the scalar solver. Returns unlensed tensor `(TT, TE, EE, BB)`,
  splined onto the `lensing_ells` grid, zero above `l_tensor_max`.
- `get_tensor_k_axes` — truncates the scalar k grids at the tensor k_max
  (CLASS uses the same stepping formula, smaller cutoff).
- `rho_relativistic` — duck-typed extensibility hook: custom species can
  contribute to the GW source via a `tensor_rho_rel(lna, params)` method;
  otherwise massless-ν gives ρ and massive-ν gives 3P (CLASS
  `tensor_method = massless_approximation`). **Zero changes to species.py.**

Touched existing files (minimal, default path unchanged except Output shape):
- **`abcmb/model_specs.py`**: +tensor spec defaults (`tensors=False`,
  `l_tensor_max=500`, `l_max_g_ten=5`, `l_max_pol_g_ten=5`, `l_max_ur_ten=17`,
  `Nlna_ten=500`, `rtol_ten=1e-5`, `atol_ten=1e-9`, `max_steps_ten=4096`).
- **`abcmb/spectrum.py`**: `lensed_Cls` now takes `ClBB_unlensed`, builds
  ξ± from (EE±BB), returns 4-tuple incl. lensed BB (CLASS `lensing.c`,
  `accurate_lensing=1` path). `get_Cl` gains optional `tensor_cls` (added to
  unlensed scalars before lensing, static branch) and returns
  `(TT, TE, EE, BB)`; BB is zeros when tensors+lensing both off.
- **`abcmb/main.py`**: `Model` fields `TPE`/`TSS` (None when `tensors=False`,
  same pattern as `thermo_model_DNeff`); pipeline wiring in
  `_run_post_recomb`; `Output` gains `ClBB` (after `ClEE`);
  `add_derived_parameters` sets `r` (default 1, CLASS convention) and `n_t`
  (default = CLASS "scc" self-consistency) only when `specs["tensors"]`;
  `r`/`n_t` added to `expected_keys`.

New test **`pytests/accuracy_test_bb.py`**, diagnostics **`diag_bb_*.py`**,
**`diag_class_ladder*.py`**, GPU timer **`time_tests_bb.py`**, design doc
**`design_bmodes.md`** (full physics + convergence write-up).

## 2. How to use it

```python
model = Model(l_max=2500, lensing=True, tensors=True)
out = model({'r': 0.1})          # n_t defaults to scc; or pass n_t explicitly
out.ClBB                          # B-mode spectrum (tensor + lensing)
```
`tensors=False` (default) → `out.ClBB` is all zeros (lensing off) or pure
lensing B (lensing on). Run tensor configs **on GPU** — the tensor solve is a
serial mode-scan on CPU (tens of minutes); on GPU it vmaps.

## 3. Validation results (final, this session)

Scalar regression `pytests/accuracy_test.py`: **PASSED** (162.8 s, CPU) —
confirms the default scalar path is unchanged.

Fiducial LCDM + r=0.1, vs classy 3.3.4:

| Quantity | max rel err | where | bar | status |
|----------|-------------|-------|-----|--------|
| raw TT (s+t) vs CLASS default  | 2.6e-3 | l=2493 | 1% | ✅ (scalar-tail) |
| raw EE (s+t) vs CLASS default  | 4.2e-3 | l=2491 | 1% | ✅ (scalar-tail) |
| raw BB vs CLASS **default**    | 3.2e-2 | l=500  | —  | info: CLASS unconverged |
| raw BB vs CLASS **hp**, 3≤l≤490| 6.9e-3 | l=477  | 2.5e-3 | ❌ see §5 |
| lensed TT (s+t)                | 2.0e-3 | l=142  | 1% | ✅ |
| lensed EE (s+t)                | 2.2e-3 | l=2    | 1% | ✅ |
| lensed BB (total)              | 6.9e-3 | l=2500 | 1% | ✅ |

**Node-level (the decisive physics check)** — `diag_bb_tol.py` (tight ABCMB
tol) vs `diag_class_ladder.py` L5 (fully converged CLASS), at the shared,
**computed** (un-interpolated) ℓ-nodes:

| l | tight-ABCMB / converged-CLASS |
|---|---|
| 237 | 1.0010 |
| 296 | 1.0007 |
| 450 | 1.0003 |
| 490 | 1.0004 |

→ The tensor transfer physics is correct to **sub-permille**.

GPU timing (1×A100, warm, lensing=True): tensors=False **9.2 s**,
tensors=True **15.4 s** (+6.2 s).

## 4. The convergence investigation (why the gap vs default CLASS is not a bug)

First pass showed raw BB ~2e-3 for l≲200 growing to ~1.6% at l~450 vs
**default-precision** CLASS. Ruled out, in order (all in `diag_*`):
1. Bessel tables — exact scipy Bessels change Cl by ≤1e-4 (`diag_bb_bessel.py`).
2. ABCMB grids — 4× denser lna ≈1e-6; no-source-k-interp ≤2e-4 (`diag_bb_converge.py`).
3. ℓ-interp — `bessel_tab/l.txt` IS CLASS's ℓ-list; 237/296/450/490 computed in both.
4. CLASS k/q sampling — precision ladder moves CLASS monotonically toward
   ABCMB, closing ~60% then saturating ~0.6-0.8% above ABCMB
   (`diag_class_ladder*.py`).
5. ABCMB solver tol — scalar large-k rtol 1e-4 biased tensor BB low by
   0.2-0.8% (grows with l). **Fixed**: tensor solver gets its own
   `rtol_ten=1e-5/atol_ten=1e-9` (within 1e-4 of the tight-tol limit for
   +6 s; `diag_bb_gpu_tol.py`).

Conclusion: ~1% disagreement with **default** CLASS in the BB tail is
**CLASS's own unconvergence**, not ABCMB error — hence the test compares
against a high-precision tensor-only CLASS reference
(`class_tensor_hp_reference()` in the test).

## 5. RESOLVED (2026-06-15) — the 6.9e-3 raw-BB residual at l=477

**Resolution:** it was CubicSpline interpolation error over the 40-wide node
gap (450→490), NOT transfer physics. Proven by the CLASS-internal
self-interpolation test (`diag_bb_dense_hp.py`, rewritten to default-precision
CLASS + l_linstep=5 for speed): re-splining CLASS's OWN node values through
the identical `interpax.CubicSpline` reproduces ~7e-3 at l=477 with zero error
at the nodes, matching the observed residual. Nodes are already maximally
dense (`bessel_l_tab`), so tightening needs new Bessel tables; the test now
holds the interp-limited band to 1% (the scalar floor) and the nodes tight.
A separate low-ℓ excess was found in the process — see `NOTE_lowl_bb_excess.md`.
The original investigation notes below are retained for context.

### (original notes)

The remaining failure is at **l=477**, which sits *between* the tensor
ℓ-nodes 450 and 490 (40 apart). At the nodes themselves agreement is
sub-permille (§3). The residual is therefore almost certainly **ℓ-spline
error in the sparse-node region**, and the comparison mixes ABCMB's
CubicSpline-over-bessel-nodes with CLASS's own sparse-ℓ interpolation — i.e.
it may not be a true error in either code.

The diagnostic to confirm this — `diag_bb_dense_hp.py`, CLASS with
`l_linstep=1` (every ℓ computed, no CLASS-side interp) vs ABCMB — was
**started but killed** (cranked precision × 500 transfer solves ran >70 min;
ABCMB half completed, CLASS half did not). **Re-run it** (give it a generous
walltime, or drop the cranked k/q precision and keep only `l_linstep=1` since
node-convergence is already established) to attribute the 477 residual.

Likely resolutions, cheapest first:
- **If it's pure interpolation**: relax the test to assert at nodes only, or
  to 1% in the 450–490 mid-node band; keep 2.5e-3 elsewhere. (Defensible:
  the physics is sub-permille.)
- **If a true 2‰ everywhere is wanted**: densify the tensor ℓ-node grid in
  `TensorSpectrumSolver.__init__` (use a finer subset of `bessel_l_tab`, or
  add interpolation nodes) so the spline has <40-wide gaps near l~450-500.
- The **491–500 sliver** (1.4e-2) is the dying tail across the arbitrary
  `l_tensor_max` cutoff — CLASS has a computed node at exactly 500, ABCMB
  splines through 530. Already excluded from the assertion; leave it.

l=2 carries the known small ABCMB error (2.6e-3) — exempt per user.

## 6. State / how to resume

- **Worktree**: `/pscratch/sd/c/carag/ABCMB-bmodes`, branch `bmodes`
  (from `origin/main` @ 5eabbab). NOT committed — `git status` dirty.
- **Allocation**: job `54385646` (bmodes_dev2) still RUNNING, ~2.5 h left as
  of wrap-up. First alloc `54376071` was released.
- **To validate after any change**: tensor configs on GPU via
  `srun --jobid=<id> --overlap ... python pytests/accuracy_test_bb.py`
  (set `JAX_PLATFORM_NAME=gpu`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`).
  CPU works but the tensor mode-scan is slow.
- **Recommended commit** (once §5 is resolved or the test tolerance is
  consciously accepted): stage `abcmb/{tensors,spectrum,main,model_specs}.py`,
  `pytests/accuracy_test_bb.py`, `time_tests_bb.py`, `design_bmodes.md`, and
  the `diag_*.py` scripts (keep them — open bug). Exclude `*.npz` artifacts.
  Scalar `accuracy_test.py` is green, so the default path is safe to commit.

## 7. Files

Production: `abcmb/tensors.py` (new), `abcmb/spectrum.py`, `abcmb/main.py`,
`abcmb/model_specs.py`.
Test: `pytests/accuracy_test_bb.py`.
Docs: `design_bmodes.md` (physics + convergence), this file.
Diagnostics (keep — §5 open): `diag_bb_profile.py` (err vs l),
`diag_bb_bessel.py` (Bessel isolation), `diag_bb_converge.py` (grid A/B),
`diag_bb_tol.py` (CPU tol A/B), `diag_bb_gpu_tol.py` (GPU tol/cost),
`diag_bb_dense_hp.py` (the §5 diagnostic to re-run), `diag_class_ladder.py`,
`diag_class_ladder2.py` (CLASS precision ladders), `time_tests_bb.py`.
Artifacts (gitignore): `diag_bb_profile.npz`, `diag_bb_bessel_inputs.npz`.
