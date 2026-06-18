# Proposed: dual scalar spectrum path (flat→table fast-path, curved→recurrence)

**Goal:** recover the intrinsic +0.64 s (lensing=False) / +0.78 s (lensing=True)
the every-ℓ recurrence adds to the scalar forward, WITHOUT losing curvature
(omega_k) support or the tensor ℓ=477 fix. Gate the scalar `get_Cl` on the
static `self.curvature` flag: flat → fast sparse-ℓ table+spline (the long-
validated `origin/main` path); curved → the hyperspherical recurrence. The
tensor sector (`tensors.py`) is unchanged — it always uses its own recurrence,
so ℓ=477 stays fixed. Flat is the common case (all B-mode runs are flat —
the `tensors`+`curvature` guard enforces it — and OLE training is flat), so the
default path returns to ~9.2/9.8 s.

## Why this is correct/safe
- Scalar Cls are smooth; the sparse-node + CubicSpline path meets the 1% scalar
  accuracy bar (it's `origin/main`'s path, validated by `accuracy_test.py` for
  years). Every-ℓ scalar exactness was never needed — only the tensor BB was.
- `curvature=False` ⇔ flat ⇔ fast table; `curvature=True` ⇔ curved ⇔ recurrence
  (which is *required* for omega_k≠0, per the curvature spec). Branch is on a
  `static=True` field, so it resolves at trace time — no runtime/JIT cost.
- Tensor path untouched (`tensors.py` recurrence) → ℓ=477 fix preserved.

## Files touched (`bmodes`, on top of merge 7dcf6b2)

1. **Restore the deleted flat table machinery, verbatim from `5eabbab`** (the
   pre-curvature `origin/main` scalar path — already reviewed/validated code;
   curvature deleted it):
   - `abcmb/bessel_tab/{l,xphi0,phi0,xphi1,phi1,xphi2,phi2}.txt`
     → `git checkout 5eabbab -- abcmb/bessel_tab/` (restore deleted tracked files).
   - `abcmb/spectrum.py` module-level (5eabbab L21–110): `bessel_l_tab`,
     `xphi*_tab`, `phi*_tab` loaders + the `j`, `phi0`, `phi1`, `phi2` helpers.
   - `SpectrumSolver` fields `ells_indices`, `lensing_ells_indices`
     (+ their `__init__` setup, 5eabbab L222–236) — added alongside the
     existing `curv_ells`/`curv_xmin`.
   - the scalar `Cl_one_ell` method (5eabbab L616+, the rolling-`lax.scan`
     accumulator version).
   - `setup.cfg`: re-add `bessel_tab/*.txt` to `package_data`.

2. **`abcmb/spectrum.py` `get_Cl` — the only NOVEL logic** (branch the unlensed
   scalar computation; the tensor graft + lensing tail are unchanged/shared):

```python
    def get_Cl(self, PT, BG, params, tensor_cls=None):
        """...docstring unchanged..."""
        if self.curvature:
            # Curved geometry: exact every-ℓ hyperspherical-Bessel recurrence
            # (required for omega_k != 0; reduces to j_l at K=0).
            sources = self._transfer_sources(PT, BG, params)
            tt_all, te_all, ee_all = self._Cl_all_ells_curved(sources, params)
            off = self.curv_ells.shape[0] - self.lensing_ells.shape[0]
            tt_unlensed = tt_all[off:]
            te_unlensed = te_all[off:]
            ee_unlensed = ee_all[off:]
        else:
            # Flat geometry: fast sparse-ℓ table transfer + CubicSpline. The
            # scalar Cls are smooth (1% spline floor = the scalar accuracy bar),
            # so this avoids the every-ℓ recurrence walk on the common path.
            tt_raw, te_raw, ee_raw = vmap(
                self.Cl_one_ell, in_axes=(0, None, None, None)
            )(self.lensing_ells_indices, PT, BG, params)
            node_ells = bessel_l_tab[self.lensing_ells_indices]
            tt_unlensed = CubicSpline(node_ells, tt_raw, check=False)(self.lensing_ells)
            te_unlensed = CubicSpline(node_ells, te_raw, check=False)(self.lensing_ells)
            ee_unlensed = CubicSpline(node_ells, ee_raw, check=False)(self.lensing_ells)

        # ---- shared: tensor contributions + lensing (UNCHANGED) ----
        if tensor_cls is not None:
            tt_unlensed = tt_unlensed + tensor_cls[0]
            te_unlensed = te_unlensed + tensor_cls[1]
            ee_unlensed = ee_unlensed + tensor_cls[2]
            bb_unlensed = tensor_cls[3]
        else:
            bb_unlensed = jnp.zeros_like(ee_unlensed)

        def get_lensed_Cls():
            tt_l, te_l, ee_l, bb_l = self.lensed_Cls(
                self.lensing_ells, tt_unlensed, te_unlensed, ee_unlensed,
                bb_unlensed, PT, BG, params)
            return (tt_l[self.ells-2], te_l[self.ells-2],
                    ee_l[self.ells-2], bb_l[self.ells-2])

        def get_unlensed_Cls():
            return (tt_unlensed[self.ells-2], te_unlensed[self.ells-2],
                    ee_unlensed[self.ells-2], bb_unlensed[self.ells-2])

        return lax.cond(self.lensing, get_lensed_Cls, get_unlensed_Cls)
```

(`Cl_one_ell` references `bessel_l_tab`, `xphi*_tab`, `phi*_tab`, `j` — all
restored in step 1. `_transfer_sources`/`_Cl_all_ells_curved` stay for the
curved branch. Nothing in `tensors.py` changes.)

## Validation plan (GPU)
- **Timing** (`prof_default_path.py`): tensors=False flat → expect ~9.2/9.8 s
  (back to the table baseline; regression gone). curvature=True → still
  recurrence (~9.8/10.6, unchanged).
- **accuracy_test.py** (flat scalar, now the table path): PASS (it's main's path).
- **accuracy_test_bb.py** (flat, tensors=True: table scalar + recurrence tensor):
  raw/lensed BB unchanged (tensor recurrence untouched → ℓ=477 still fixed),
  raw TT/EE unchanged (table scalar, as pre-curvature-merge).
- **reverse-AD** tensors=True: table scalar path is main's (revAD-known-finite);
  tensor recurrence revAD already validated → expect finite both lensings.
- (Optional) a curved omega_k≠0 spot-check to confirm the recurrence branch
  still runs.

## Trade-off acknowledged
`spectrum.py` becomes a table+recurrence hybrid (more surface than either pure
path), and re-adds the 79 MB Bessel tables that curvature deleted. This is the
cost of keeping omega_k support AND a fast flat path. Committed as a follow-up
on `bmodes` (after 7dcf6b2), not a force-push.
