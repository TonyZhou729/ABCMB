# Scope: bring the curvature-branch Bessel recurrence into `bmodes`

**Goal (user):** integrate the curvature branch's changes so the B-mode path
uses the every-ℓ hyperspherical-Bessel recurrence instead of the phi-tables +
sparse-ℓ CubicSpline. This kills the §5 ℓ=477 raw-BB residual (pure spline
error over the 40-wide node gap) and drops the tensor solver's table
dependence. **`origin/curvature` (== local `worktree-curvature`, @ `3174b25`)
is authoritative and read-only; only `bmodes` is modified.**

Both branches are siblings off `5eabbab`. Verified `origin/curvature` and
`worktree-curvature` are the *same commit*.

---

## 1. What each branch changed (vs base `5eabbab`, `abcmb/` only)

| File | curvature | bmodes | overlap? |
|------|-----------|--------|----------|
| `spectrum.py` | rewrote `get_Cl` → recurrence; **deleted** `Cl_one_ell`, sparse-ℓ spline, `ells_indices`, module-level phi-tables + `j`, the ℓ-cap | added `tensor_cls` arg to `get_Cl`; `lensed_Cls` → 4-tuple EE↔BB mixing | **YES — the real merge** |
| `main.py` | `+curvature=` kwarg to SS; `omega_k`/`K` derived params; `omega_Lambda −= omega_k`; `expected_keys += omega_k,K` | TPE/TSS fields+init; tensor wiring in `_run_post_recomb`; `r`/`n_t` derived; `Output.ClBB`; `expected_keys += r,n_t` | trivial (only `expected_keys` literal collides) |
| `model_specs.py` | curvature specs; `species.Curvature`; curved k-grids (`omega_k_ref`, `closed_integer_nu`, `K_ref`) | tensor specs block | none (disjoint regions) |
| `ABCMBTools.py` | +109: `sin_K`, `cot_K`, `_curv_f`, `_curv_g`, `_curv_g_diff` | — | clean (curvature-only) |
| `species.py` | +179: `Curvature` species + curved ρ/ICs | — | clean |
| `background.py`, `perturbations.py` | curved distances / scalar ICs | — | clean |
| `bessel_tab/*.txt` | **deleted** (79 MB) | — (still imported by `tensors.py`!) | see §3 |
| `tensors.py` | — (does not exist on curvature) | +750 new | — |

**Net:** a `git merge origin/curvature` into `bmodes` applies cleanly
everywhere except `spectrum.py` (substantial, expected) and a one-line
`expected_keys` collision in `main.py` (keep both: `omega_k,K` + `r,n_t`).
The bulk of the curvature feature (new species, curved background, curved
k-grids, ABCMBTools helpers, table deletions) lands without conflict.

---

## 2. The recurrence is a clean fit for the tensor radials

`spectrum.py:_Cl_all_ells_curved` (curvature) walks ℓ via the three-term
hyperspherical recurrence, and its `step` already computes, per (Nlna, Nk):

- `Phi_l`            = Φ_ℓ           → at K=0 = **jₗ(kχ)**
- `dPhi`  = dΦ/dχ    → `dPhi/k`      → at K=0 = **jₗ′(kχ)**
- `d2Phi` = d²Φ/dχ²  → `d2Phi/k²`    → at K=0 = **jₗ″(kχ)**
- `x_eff` = q·S_K(χ) → at K=0 = **kχ = x**

The tensor radial functions (`tensors.py:Cl_one_ell`, CLASS `transfer.c`) are
*exactly* combinations of those quantities (flat limit):

```
radT = sqrt(3/8 (l+2)(l+1)l(l-1)) · jₗ / x²
radE = 1/4 [ jₗ″ + 4 jₗ′/x − (1 − 2/x²) jₗ ]
radB = 1/2 [ jₗ′ + 2 jₗ/x ]
```

So the port is: inside a recurrence `step`, form `radT/radE/radB` from
`Phi_l, dPhi/k, d2Phi/k², x_eff`, contract against the tensor sources
`source_T2, source_E` (and primordial `P_h = r A_s (k/k_pivot)^{n_t}`),
accumulate `ClTT_t/ClTE_t/ClEE_t/ClBB_t` at **every** integer ℓ in
`2..l_tensor_max`, zero-pad to `out_ells`. No tables, no spline → the ℓ=477
residual is gone by construction.

**Tensors stay flat (K=0).** The scalar Φ recurrence is *not* the correct
spin-2 radial basis in curved space, so the tensor port is validated only at
`omega_k = 0`. `omega_k ≠ 0` AND `tensors=True` must be guarded/documented as
unsupported (both are opt-in, default off, so no regression). Carrying `K`
through the tensor recurrence (reusing `tools.sin_K/cot_K`) is fine because it
is exact at K=0; just don't claim the curved-tensor result.

---

## 3. The hard dependency that forces the choice

`tensors.py:11` imports `bessel_l_tab, xphi0_tab, phi0_tab, … , j` from
`spectrum.py`. The curvature branch **deleted all of those**. So after merging
curvature, `tensors.py` will not import until the recurrence port is done —
the port is *mandatory*, not optional, once curvature is merged.

---

## 4. Two ways to scope it

### Option A — tensor-only recurrence, NO curvature merge (minimal)
Write a flat (K=0) spherical-Bessel recurrence self-contained in `tensors.py`;
replace `TensorSpectrumSolver.{Cl_one_ell, get_Cl}` sparse-node+spline with the
every-ℓ recurrence; drop the table imports. Leave bmodes' scalar `spectrum.py`
and the tables untouched (scalars keep tables+spline; they pass at 1%).
- **Touches:** `tensors.py` only (+ maybe copy `sin_K`/`cot_K` or hardcode flat forms).
- **Fixes:** ℓ=477 raw-BB residual; tensor table dependence.
- **Does NOT:** reconcile bmodes with curvature (bmodes keeps tables for
  scalars; curvature deleted them → the two still conflict on a future main
  merge). Does NOT bring omega_k.
- **Risk:** low. Smallest, most surgical.

### Option C — full `git merge origin/curvature` into bmodes, then port (alignment)
Merge curvature wholesale (resolving spectrum.py + the trivial main.py
collision), then port the tensor recurrence in `tensors.py`. Scalars **and**
tensors use the recurrence; tables deleted; bmodes gains omega_k; the two
branches are reconciled (what "bring bmodes into alignment with curvature"
literally asks).
- **Touches:** `spectrum.py` (merge: adopt curvature's recurrence `get_Cl`,
  graft bmodes' `tensor_cls` add-in + 4-tuple `lensed_Cls`), `main.py`
  (trivial), `tensors.py` (port), + accept the rest of curvature clean.
- **Fixes:** ℓ=477 for BB *and* removes the scalar sparse-ℓ spline; full
  table deletion; mergeable-with-curvature.
- **Risk:** medium. Biggest piece is the `spectrum.py` 3-way reconciliation
  and re-validating the scalar path (curvature's `get_Cl` must still produce
  the bmodes 4-tuple incl. tensor add-in + lensed BB).

**Recommendation:** Option C — it is what the user asked ("integrate the
changes in the curvature branch … bring it into alignment"), and once
curvature is merged the tensor port is required anyway (§3). Option A is the
fallback if the spectrum.py reconciliation proves larger than expected.

---

## 5. Execution plan for Option C

1. **Baseline first:** confirm reverse-AD through `tensors=True` is finite on
   the *current* bmodes (in flight) and record scalar `accuracy_test.py` +
   `accuracy_test_bb.py` + tensors=True warm time as the pre-merge reference.
2. `git merge origin/curvature` on `bmodes`. Resolve:
   - `main.py`: union `expected_keys` (`omega_k,K` + `r,n_t`); keep both
     add_derived blocks; keep both SS-init kwargs (curvature= AND the TPE/TSS
     init is separate).
   - `model_specs.py`: should auto-merge (disjoint); verify tensor specs +
     curvature specs both present.
   - `spectrum.py`: take curvature's recurrence `get_Cl`; re-introduce the
     `tensor_cls=None` arg, the "add tensor to unlensed totals + bb_unlensed"
     block, and switch `get_lensed_Cls` to the 4-tuple `lensed_Cls`; keep
     bmodes' 4-tuple `lensed_Cls` body (curvature didn't touch it).
3. **Port `tensors.py:TensorSpectrumSolver`** to the recurrence (§2): new
   `_Cl_all_ells_tensor`, drop table imports + spline + `Cl_one_ell`. Keep
   `TensorPerturbationEvolver` untouched (it is table-independent).
4. **Validate:** `accuracy_test.py` (scalar path unchanged), `accuracy_test_bb.py`
   (ℓ=477 band bar can tighten now — every ℓ computed; nodes already
   sub-permille), reverse-AD `tensors=True` finite both lensings, tensors=True
   warm time. All on GPU.
5. The §5 / band-tolerance test assertions in `accuracy_test_bb.py` can be
   tightened from 1e-2 once the spline is gone (the interp floor was the only
   reason for the loose 1e-2). The low-ℓ 4e-3 bar STAYS (inter-code scatter,
   not interp).

## 6. Open risks
- **curved + tensors** unsupported (flat-only tensor radials) — guard it.
- **reverse-AD**: curvature's recurrence is revAD-validated at K=0 for scalars;
  the tensor port reuses the same `lax.scan`/`jax.checkpoint` chunked pattern,
  so it should inherit that — but must be re-tested (ties to task #1).
- **k-grid**: tensor transfer currently truncates the scalar grid at the tensor
  k_max (`get_tensor_k_axes`). The recurrence consumes `self.k_axis_transfer`;
  confirm the tensor k-axis still feeds the recurrence correctly (the
  contraction is over the tensor k-grid, not the full scalar one).
