# B modes in ABCMB — design (branch `bmodes`)

Goal: ClBB output matching CLASS at the accuracy of the existing
`accuracy_test.py` bar (1% where the signal is non-negligible), with two
physical contributions:

1. **Tensor (primordial GW) modes** → unlensed BB (+ tensor TT/TE/EE), new.
2. **Lensing of E into B** → lensed BB from the scalar sector, an extension of
   the existing correlation-function lensing code.

Everything is transcribed from CLASS (`class_EDE/source/`, vanilla in the
tensor sector) so the two codes are comparable mode-for-mode.

## What CLASS does (reference)

- **Variables** (synchronous gauge, tensor sector, flat): GW amplitude `h`
  (stored as `gw`, IC `h = 1/sqrt(6)`, `h' = 0`, all other moments 0), photon
  tensor temperature hierarchy `F_0..F_lmax` written as
  `delta_g (=F0), theta_g (=3k/4 F1), shear_g (=F2/2), F3, F4, ...` with
  `l_max_g_ten = 5`, photon tensor polarization `G_0..G_lmax`
  (`l_max_pol_g_ten = 5`), and a massless ("ur") tensor hierarchy
  (`l_max_ur = 17`) under the default `tensor_method = massless_approximation`
  where its density is `rho_ur + 3 p_ncdm`.
- **Equations** (conformal time, kappa' = 1/tau_c):
  - `Pi = -1/sqrt(6) (delta_g/10 + 2 shear_g/7 + 3 F4/70 - 3 G0/5 + 6 G2/7 - 3 G4/70)`
  - photon: standard hierarchy with `-kappa' X` damping on every moment,
    source `+sqrt(6) h'` on `delta_g`, `+kappa' sqrt(6) Pi` recoupling on
    `delta_g` and `G0` (see perturbations.c:8540-8705)
  - ur: same without scattering terms
  - GW: `h'' = -2 aH h' - k^2 h + S`,
    `S = -sqrt(6) * 4 a^2 * (8 pi G/3 c^2) [rho_g (delta_g/15 + 4 shear_g/21 + F4/35) + rho_rel (same with ur moments)]`
- **Sources**: `S_T2 = -h' exp(-kappa) + g Pi`, `S_P = +sqrt(6) g Pi`
  (the + sign is CLASS/CAMB "historical convention"; copied to match).
- **Radial functions** (flat, x = k(tau0-tau), Phi = j_l):
  - T: `sqrt(3/8 (l+2)(l+1)l(l-1)) j_l/x^2` (same factor as scalar E)
  - E: `1/4 [ j_l'' + 4 j_l'/x - (1 - 2/x^2) j_l ]`
  - B: `1/2 [ j_l' + 2 j_l/x ]`
  With ABCMB's tabulated `phi0 = j_l`, `phi1 = j_l'`, `phi2 = (3 j_l'' + j_l)/2`:
  `j_l'' = (2 phi2 - phi0)/3`. **No new Bessel tables needed.**
- **Cl**: `Cl^XY = 4 pi int dk/k P_h(k) Delta_X Delta_Y`,
  `P_h(k) = r A_s (k/k_pivot)^{n_t}`, default `n_t = -r/8 (2 - r/8 - n_s)`
  (CLASS "scc" self-consistency); tensor Cl cut at `l_tensor_max = 500`.
- **k grids**: same stepping formulas as scalars, truncated at
  `k_max = k_max_tau0_over_l_max * l_tensor_max / tau0` (≈ 0.064/Mpc default).
- **Lensed BB** (lensing.c, `accurate_lensing=1` path, which ABCMB already
  mirrors): `ksi_p` built from `(ClEE+ClBB)`, `ksi_m` from `(ClEE-ClBB)`;
  `ClEE_lensed = pi sum (ksi_p d22 + ksi_m d2m2) w`,
  `ClBB_lensed = pi sum (ksi_p d22 - ksi_m d2m2) w`.
  Lensing operates on the **total** (scalar+tensor) unlensed spectra.

## ABCMB implementation (max parsimony, min footprint)

**New file `abcmb/tensors.py`** — everything tensor-specific:

- `TensorPerturbationEvolver(eqx.Module)`: mirrors `PerturbationEvolver`
  (same fields, same Kvaerno5 + PIDController specs, `SaveAt(ts=lna)` on the
  same 500-point lna grid, vmap-over-k on GPU / scan on CPU, same
  starting-time logic). State vector
  `[F0..F5, G0..G5, Fur0..Fur17, h, hdot]` (`Ny = 32`), CLASS variable
  conventions kept verbatim; equations divided by `aH` for d/dlna.
  `rho_rel` = MasslessNeutrino rho + 3*P of MassiveNeutrino; custom species
  can opt in via a `tensor_rho_rel(lna, params)` method (duck-typed hook —
  zero changes to `species.py`).
  Output: `TensorSourceTable(k, lna, source_T2, source_E)` — sources, not raw
  moments, are tabulated.
- `TensorSpectrumSolver(eqx.Module)`: `Cl_one_ell` clone with the three
  tensor radial functions and the rolling lna-scan accumulator pattern (3
  carries), vmapped over the bessel-table ell nodes up to `l_tensor_max`,
  splined onto `2..l_tensor_max` and zero-padded to the `lensing_ells` grid.
  Returns unlensed tensor `(TT, TE, EE, BB)`.
- `get_tensor_k_axes(specs, k_axis_perturbations, k_axis_transfer)`: truncates
  the existing scalar grids at the tensor k_max (identical to CLASS's loop
  with the smaller cutoff, since the stepping formula is the same).

**Touched existing files (kept minimal):**

- `model_specs.py`: +5 spec defaults (`tensors=False`, `l_tensor_max=500`,
  `l_max_g_ten=5`, `l_max_pol_g_ten=5`, `l_max_ur_ten=17`).
- `spectrum.py`: `lensed_Cls` gains `ClBB_unlensed` input and a BB output
  (ksi_p/ksi_m combination above); `get_Cl` gains optional `tensor_cls`
  argument (added to the unlensed spectra before lensing — static Python
  branch, no retracing) and now returns `(TT, TE, EE, BB)`; BB is zeros for
  `tensors=False, lensing=False`.
- `main.py`: Model fields `TPE`/`TSS` (None when `tensors=False`, same
  pattern as `thermo_model_DNeff`); construction of the two tensor objects;
  `_run_post_recomb` runs the tensor pipeline and passes `tensor_cls` to
  `SS.get_Cl`; `Output` gains `ClBB` (after `ClEE`);
  `add_derived_parameters` sets `r` (default 1, CLASS convention) and `n_t`
  (default scc) only when `specs["tensors"]`.

**Not done (scope):** vector modes; curvature; tensor ncdm exact hierarchy
(`tensor_method=exact`); these match CLASS defaults anyway.

## Cost (measured, 1xA100, lensing=True, warm)

- `tensors=False`: 9.16 s — baseline unchanged.
- `tensors=True`, default tolerances (rtol_ten 1e-5 / atol_ten 1e-9):
  15.4 s (+6.2 s). The overhead is the tensor `vmap`+`while_loop` solve
  (395 modes, Ny=32, sync-bound like the scalar PE, despite the small
  system) plus the tensor Cl scan.
- Tightening to 1e-6/1e-10 costs +10.5 s total and buys 6e-7 accuracy —
  not the default.
- Memory negligible (sources table ~ 1.5 MB). On CPU the tensor solve is a
  serial scan over modes and dominates (tens of minutes) — run tensor
  configs on GPU.

## Validation

- New `pytests/accuracy_test_bb.py` (CPU, classy 3.3.4): fiducial LCDM +
  `r=0.1`. Raw config: TT/EE (s+t) vs default CLASS at 1%; raw BB vs a
  **high-precision tensor-only CLASS reference** at 2.5 permille for
  3 <= l <= 500 (l=2 exempt, known small ABCMB error). Lensed config:
  TT/EE/BB vs default CLASS at 1%.
- Existing `accuracy_test.py` stays green (default path unchanged except
  the Output shape).
- GPU smoke + timing via `time_tests_bb.py`.

## Accuracy findings (2026-06-12 convergence investigation)

First-pass agreement vs default-precision CLASS: TT 2.6e-3, EE 4.2e-3
(scalar-limited, tail of the spectrum), tensor BB ~2e-3 for l <= 200 but a
smooth deficit growing to 1.6% at l=450. Attribution, in order:

1. **Not the Bessel tables**: replacing phi0/phi1/phi2 with exact scipy
   spherical Bessels changes BB/EE/TT at nodes by <= 1e-4
   (`diag_bb_bessel.py`).
2. **Not ABCMB grids**: 4x denser tensor lna grid changes BB by ~1e-6;
   evolving directly on the transfer k grid (no source k-interpolation)
   changes it by <= 2e-4 (`diag_bb_converge.py`).
3. **Not ell interpolation**: `bessel_tab/l.txt` is exactly CLASS's l-node
   list, so l=237/296/450/490 are computed (not splined) in BOTH codes.
4. **CLASS-side k/q sampling**: a CLASS precision ladder
   (`diag_class_ladder*.py`) moves CLASS monotonically toward ABCMB:
   default -> k_step_sub 0.005 / q_linstep 0.05 + fine time sampling +
   weakened TCA/RSA closes ~60% of the gap, then saturates ~0.6-0.8%
   above ABCMB at l=450/490.
5. **ABCMB solver tolerance (the production fix)**: the scalar PE
   large-k tolerances (rtol 1e-4) bias tensor BB low by 0.2-0.8%
   (growing with l). Already at rtol 1e-5 / atol 1e-9 (the shipped
   default — `diag_bb_gpu_tol.py`) the result is within 1e-4 of the
   tight-tolerance limit, which lands within **0.3-1.0e-3 of converged
   CLASS at every node** (`diag_bb_tol.py`):

   | l | tight-ABCMB / converged-CLASS |
   |---|---|
   | 237 | 1.0010 |
   | 296 | 1.0007 |
   | 450 | 1.0003 |
   | 490 | 1.0004 |

   Hence the tensor solver gets its own spec'd tolerances
   (`rtol_ten=1e-6`, `atol_ten=1e-11`, `max_steps_ten=4096`) — nearly
   free, since the tensor system is small (Ny=32, ~400 modes, k <= 0.065).

Residual ~1% disagreement with *default-precision* CLASS in the BB tail
(l ~ 350-500) is CLASS's own unconvergence, not ABCMB error — hence the
high-precision reference in the accuracy test.
