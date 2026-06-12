# ABCMB flat-geometry audit (Omega_k = 0 / K = 0 assumptions)

Audit of `/pscratch/sd/c/carag/ABCMB/.claude/worktrees/curvature/` (clean checkout of `origin/main`),
preparing for curved (open/closed) FLRW support. All line numbers refer to files under `abcmb/` in that worktree.
Read-only audit; no code executed.

Convention used below: `K` is the curvature constant in the FLRW metric (Mpc^-2 in the code's units),
`omega_k = Omega_k h^2 = -K c^2 / H_100^2` style closure parameter. "Flat closure" = the assumption
`Omega_r + Omega_m + Omega_Lambda = 1`.

---

## 0. Executive inventory (where flatness is baked in)

| # | Location | What assumes flatness | Curvature change |
|---|----------|----------------------|------------------|
| 1 | `main.py:594` | `omega_Lambda = h^2 - omega_r - omega_m` (flat closure) | subtract `omega_k` |
| 2 | `background.py:166` | `H = sqrt(8πG ρ_tot/3)` — no `-K/a²` term | curvature term (or curvature pseudo-fluid, see §2.3) |
| 3 | `background.py:208` | `aH' ∝ (ρ+3P)` — consistent only if curvature enters as fluid w=−1/3 or explicit term | see §2.3 |
| 4 | `background.py:559` | `rA_rec = tau0 - tau(lna_rec)` — identifies χ with comoving angular-diameter distance | `r_A = S_K(χ)` |
| 5 | `perturbations.py:238-239`, `387-391` | Einstein constraints with bare `k²` | `k² → k²−3K` (η-constraint), shear eq. gets `(k²−3K)` |
| 6 | `perturbations.py:389` | `α = aH(h'+6η')/(2k²)` | `2k²` → curved generalization |
| 7 | `perturbations.py:194`, species `y_ini` | Flat radiation-era adiabatic ICs in powers of `kτ` | O(K/k²) corrections; `k` itself becomes `ν`-label |
| 8 | `species.py` hierarchies (§3) | l-recurrence coefficients `l/(2l+1)·k`, `(l+1)/(2l+1)·k` | acquire curvature factors `s_l = sqrt(1−(l²−s²... )K/k²)`-type (Hu et al. 1998 / CLASS) |
| 9 | `species.py:751,972,1440,1447` | Free-streaming truncation `F'_lmax = k F_{lmax−1} − (lmax+1)/τ F_lmax` | flat `j_l` recursion at lmax; curved version uses hyperspherical cot-functions |
| 10 | `spectrum.py:49-51, 53-109, 719-753, 763` | Radial functions = spherical Bessel `j_l(kχ)`; LOS argument `x = k(τ0−τ)` | hyperspherical Bessel `Φ_l^ν(χ)`; single-variable `x` tabulation breaks |
| 11 | `spectrum.py:266` | `P_R(k) = A_s (k/k_p)^{n_s−1} · 2π²/k³` | `k³` measure → `ν(ν²−K)`-type (convention-dependent); closed: discrete k |
| 12 | `spectrum.py:789-791` | Cl integrand `∝ T²/k dk` | dk-measure changes; closed universe → sum over discrete modes |
| 13 | `spectrum.py:412-435` | Limber lensing: `chi = tau0−tau`, `k=(l+½)/chi`, `window=(χ*−χ)/(χ*χ)` | all distances → `S_K(χ)` ratios |
| 14 | `spectrum.py:376-380` | `Om(z) = Ωm(1+z)³/(Ωm(1+z)³+Ω_Λ)` — flat (also ignores radiation) | add `Ω_k(1+z)²` to denominator |
| 15 | `model_specs.py:120-193` | k-grids built from *fiducial flat* `tau0_fid`, `rs_rec_fid` | fiducials only (static grid); closed universe needs discrete `k` grid |
| 16 | `species.py:432-490` (`DarkEnergy`) | consumes `params['omega_Lambda']` from the flat closure | inherits fix #1 |

Things that are **already geometry-clean** given a corrected `H(a)`: conformal-time tabulation
(`dτ = dlna/aH`), optical depth κ, baryon-drag κ_d, sound horizon r_s (radial distance), HyRex
(consumes only H, T, nH arrays), LINX, the reionization model, and the lensed-Cl correlation-function
machinery (operates on the sphere; geometry enters only through `Clpp`).

---

## 1. `main.py` — parameters, derived parameters, orchestration

### 1.1 Parameter defaults (`add_derived_parameters`, main.py:368-612)

```python
374  params['h']             = jnp.array(params.get('h', 0.6736))
375  params['H0']            = jnp.array(params['h'] * cnst.H0_over_h)
376  params['omega_cdm']     = jnp.array(params.get('omega_cdm', 0.120))
377  params['omega_b']       = jnp.array(params.get("omega_b", 0.02237))
378  params['A_s']           = jnp.array(params.get('A_s', 2.1e-9))
379  params['n_s']           = jnp.array(params.get('n_s', 0.9649))
380  params['TCMB0']         = jnp.array(params.get('TCMB0', 2.34865418e-4))
```
Reionization (383-390): `tau_reion=0.0544` (if `specs["input_tau_reion"]`, the default) else `z_reion=7.67`;
`Delta_z_reion=0.5`, `z_reion_He=3.5`, `Delta_z_reion_He=0.5`, `exp_reion=1.5`.
Massive neutrinos (398-400): `T_nu_massive=0.71611`, `N_nu_massive=0`, `m_nu_massive=0.06`.
`T_nu_massless=0.71636856` (428). BBN: `YHe=0.245` default, or table interp, or LINX (461-545).

### 1.2 The flat closure — THE central line

```python
392  # Here we fill in a fake omega_Lambda just so that the DE energy density can be computed in a loop.
395  params['omega_Lambda'] = 0.
...
569  # Loop over matter fluids to compute total matter density today.
570  rho_m = 0.
571  for s in self.species_list:
572      if s.is_matter:
573          rho_m += s.rho(0., params)
574  params['omega_m'] = rho_m / (3 * cnst.H0_over_h**2/8/jnp.pi/cnst.G)
...
578  a_early = jnp.exp(-23.)
579  rho_r  = 0.
581  for s in self.species_list:
582      rho_r += s.rho(jnp.log(a_early), params)
...
586  params['omega_r'] = rho_r * a_early**4 / (3 * cnst.H0_over_h**2/8/jnp.pi/cnst.G)
587  params['R_nu']    = rho_nu / rho_r
591  params['om']      = params['omega_m'] / jnp.sqrt(params['omega_r']) * cnst.H0_over_h / cnst.c_Mpc_over_s
593  # Having inferred correct omega_m and omega_r, compute correct omega_Lambda
594  params['omega_Lambda'] = params['h']**2 - params['omega_r'] - params['omega_m']
```

- **main.py:594 is the flat closure.** A curvature param enters here:
  `params['omega_k'] = jnp.array(params.get('omega_k', 0.))` among the defaults (~line 380), then
  `omega_Lambda = h² − omega_r − omega_m − omega_k`.
- Note the omega_r / omega_m inference loops sum **over species** at `lna=-23` and `lna=0`. If
  curvature is implemented as a pseudo-fluid (§2.3) with `rho_k ∝ a^-2`, its contamination of
  these loops is `O(a_early²) = e^{-46}` for `omega_r` and `0` for `omega_m` (`is_matter=False`)
  — numerically negligible but worth a guard/comment. Cleaner: exclude a curvature species
  by name from these loops, or use an explicit term (not a species).
- `params['om']` (591) feeds the adiabatic ICs (radiation-era `Ω_m H0/√Ω_r`); curvature is
  irrelevant there at `a ~ e^{-15}` — no change needed.
- Neff bookkeeping (443-459, 555-567) similarly loops species and buckets non-photon, non-neutrino
  species into `rho_extra` at `lna=-23`. A curvature pseudo-fluid would land in `rho_extra` with
  relative size `~(omega_k/omega_r)·a² ≈ 1e-20` — negligible, same caveat.

### 1.3 `expected_keys` guard (main.py:600-610)

```python
600  expected_keys = {
601      'h', 'H0', 'omega_cdm', 'omega_b', 'A_s', 'n_s', 'TCMB0',
602      'tau_reion', 'z_reion', 'Delta_z_reion', 'z_reion_He', 'Delta_z_reion_He', 'exp_reion',
603      'omega_Lambda', 'T_nu_massive', 'N_nu_massive', 'm_nu_massive',
604      'N_nu_massless', 'Neff', 'T_nu_massless', 'YHe',
605      'omega_m', 'R_b', 'omega_r', 'R_nu', 'om'
606  }
607  for key, value in param_in.items():
608      if key not in expected_keys:
609          params[key] = jnp.array(value)
```
Unknown user keys are wrapped in `jnp.array` (anti-recompilation). If `omega_k` becomes a
first-class parameter with an explicit `params.get` default, add it to this set (otherwise it
would be harmlessly double-wrapped). Use a **float** default (`0.` not `0`): int/bool leaves get
force-cast to float64 at the top of `run_cosmology_abbr` anyway (landmine #4), but a float default
avoids relying on that.

### 1.4 Orchestration and device boundaries (`run_cosmology_abbr`, main.py:187-235)

```python
207  def _to_float(v): ...               # int/bool → float64 (reverse-AD landmine #4)
212  params = jax.tree_util.tree_map(_to_float, params)
214  pre_BG = self.get_BG_pre_recomb(params)                       # @eqx.filter_jit (GPU)
216  cpu_dev = jax.devices('cpu')[0]
217  recomb_inputs_cpu = jax.device_put(pre_BG.recomb_inputs, cpu_dev)
218  params_cpu = jax.device_put(params, cpu_dev)
220  recomb_output = eqx.filter_jit(self.RecModel, backend='cpu')((recomb_inputs_cpu, params_cpu))
223  recomb_output = jax.device_put(recomb_output, jax.devices('gpu')[0])
233  recomb_output = jax.tree_util.tree_map(_to_float, recomb_output)   # landmine #6
235  return self._run_post_recomb(params, pre_BG, recomb_output)       # @eqx.filter_jit (GPU)
```

What crosses the GPU→CPU boundary: `pre_BG.recomb_inputs = RecombInputs(lna_grid, TCMB_arr, nH_arr, H_arr)`
(built in `background.py:94-99` by vmapping `TCMB`, `nH`, `H` over `RecModel.lna_axis_full`) plus the
full `params` dict. What crosses back: HyRex's `(xe, lna_xe, Tm, lna_Tm)` quadruple of
`array_with_padding` objects.
**Curvature implication: HyRex only ever sees `H_arr` — a corrected Friedmann equation propagates
into recombination automatically; HyRex itself is geometry-agnostic. No boundary change needed.**

Params flow into Background construction:
- `get_BG_pre_recomb` (237-263) → `BackgroundPreRecomb(params, self.species_list, self.RecModel, adjoint=...)`.
- `_run_post_recomb` (265-302) → `get_PTBG` (304-328) → `get_BG` (330-366), which `lax.cond`s on
  `specs["input_tau_reion"]` between `Background(pre_BG, recomb_output, params, ReionizationModelFromZ)` and
  `...FromTau`. Then `PT = self.PE.full_evolution((BG, params))`, `Cls = self.SS.get_Cl(PT, BG, params)`,
  `Pk = self.SS.Pk_lin(self.SS.k_axis_Pk_output, 0., PT, params)`.

So a new `omega_k` reaches everything simply by living in `params` (a pytree leaf — traced, no
recompilation). Anything that must change *array shapes* (e.g. a discrete-k grid for closed
universes, different Bessel tables) must instead happen at `Model.__init__`/`load_specs` time
(static), per the design rule "shape-changing logic → `__init__`".

### 1.5 Call graph (from `Model.__call__`)

`Model.__call__(params)` → `add_derived_parameters` (plain Python, LINX-on-CPU branch lives here) →
`run_cosmology_abbr` (plain Python orchestrator) → [`get_BG_pre_recomb` GPU jit: `BackgroundPreRecomb.__init__`
→ `_tabulate_conformal_time` (diffrax Kvaerno5, dense) → builds `RecombInputs`] → [CPU jit: `RecModel((recomb_inputs, params))`]
→ [`_run_post_recomb` GPU jit: `get_BG` (`Background.__init__` → reionization model → `_tabulate_optical_depth`
(diffrax, dense) → visibility-peak/transfer-start searches) → `PE.full_evolution` (vmap over k of
`evolution_one_k`: `get_starting_time` → `initial_conditions_one_k` → diffrax Kvaerno5 with
`get_derivatives` = Einstein constraints + per-species `y_prime`; then `make_output_table` → `PerturbationTable`)
→ `SS.get_Cl` (vmap `Cl_one_ell` over `lensing_ells_indices` → cubic-spline to dense ell grid →
optional `lensed_Cls` via `lensing_Cl` Limber + Wigner-d correlation method) → `SS.Pk_lin`] → `Output`.

---

## 2. `background.py`

### 2.1 Class split

- `BackgroundPreRecomb` (20-417): fields `species_list`, `lna_tau_tab = jnp.linspace(-33.0, 0.0, 10000)`
  (class-level constant, line 59), `tau_tab`, `tau0`, `recomb_inputs`, static `adjoint`.
  Methods: `rho_tot`, `P_tot`, `H`, `aH`, `aH_prime`, `d2adtau2_over_a`, `tau`, `nH`, `TCMB`, `R_ratio_lna`,
  `_tabulate_conformal_time`.
- `Background(BackgroundPreRecomb)` (420-936): adds `xe_tab/lna_xe_tab/Tm_tab/lna_Tm_tab`
  (`array_with_padding`, finite-padded per landmine #8/#9 at lines 544-550), `kappa_func`, `z_reion`,
  `tau_reion`, `lna_rec`, `rA_rec`, `lna_transfer_start`, `lna_visibility_stop`. Methods: `xe`, `Tm`,
  `tau_c`, `_tabulate_optical_depth`, `expmkappa`, `visibility`, `_tabulate_kappa_d`, `_tabulate_rs`,
  `z_d`, `rs_d`.

### 2.2 Friedmann assembly — exact code

```python
101  def rho_tot(self, lna, params):
119      rho_tot = 0.
120      for i in range(len(self.species_list)):
121          rho_tot += self.species_list[i].rho(lna, params)
122      return rho_tot

147  def H(self, lna, params):
166      return jnp.sqrt(8.*jnp.pi*cnst.G*self.rho_tot(lna, params)/3.)

168  def aH(self, lna, params):
187      return jnp.exp(lna)*self.H(lna, params) / cnst.c_Mpc_over_s

189  def aH_prime(self, lna, params):       # d(aH)/dlna, Mpc^-1
208      return -4.*jnp.pi*cnst.G*jnp.exp(lna)**2/3./self.aH(lna, params) \
             * (self.rho_tot(lna,params)+3.*self.P_tot(lna, params)) / cnst.c_Mpc_over_s**2

210  def d2adtau2_over_a(self, lna, params):
229      return self.aH(lna, params)**2 + self.aH(lna, params)*self.aH_prime(lna, params)
```

**No `-K/a²` term anywhere.** Curved Friedmann: `H² = (8πG/3)ρ_tot − K c²/a²`.

### 2.3 Curvature as a species vs explicit term — the key design question

The sum over `species_list[i].rho` is fully generic, so a curvature "species" slots in *exactly*
at the background level:

- Define `Curvature(BackgroundFluid)` with
  `rho = −omega_k · (3 H_100²/8πG) · a^{-2}` (sign: positive ρ_k for open, omega_k>0 ⇔ Ω_k>0
  means *adding* `+Ω_k H0²/a²` to H², i.e. `rho_k = +omega_k·(3H_100²/8πG)/a²` with the closure
  `omega_Lambda = h² − omega_r − omega_m − omega_k`; pick one sign convention and stick to it) and
  `P = −rho/3` (w = −1/3).
- Then `H` (166) is automatically the curved Friedmann equation, and — crucially —
  `aH_prime` (208) is automatically correct too, because for w = −1/3 the combination
  `ρ + 3P = 0`: differentiating `(aH)² = (8πG/3)a²ρ_fluids − Kc²` in lna, the constant `−Kc²`
  drops out, which is precisely what the `(ρ+3P)` form with a vanishing curvature contribution
  reproduces. `d2adtau2_over_a` (229) follows.
- `BackgroundFluid` (species.py:402-428) already provides trivial `y_ini`/`y_prime`
  (`jnp.array([])`, `num_equations = 0`) and zero `rho_delta`/`rho_plus_P_theta`/`rho_plus_P_sigma`
  — physically right: curvature has no fluid perturbations; the *perturbation* equations need
  explicit `K` factors instead (§4), which a species cannot provide.

Caveats for the species route (all enumerated in §1.2): the `add_derived_parameters` species loops
(omega_r at 577-586, Neff at 443-459/555-567, R_nu at 587) pick up `O(a²)≈1e-20` contamination —
negligible; `R_ratio_lna` (389-417) filters by name — unaffected; closed universes give ρ_k < 0,
fine since the total under the `sqrt` at line 166 stays positive for viable cosmologies.

**However, curvature cannot be *only* a species**: the perturbed Einstein equations
(perturbations.py), the radial functions and distances (spectrum.py), and the hierarchy
coefficients (species.py) need `K` explicitly. So the realistic design is: explicit
`params['omega_k']` consumed everywhere + optionally a thin background species (or, equally
clean, an explicit `−Kc²/a²` term inside `H`/`d2adtau2_over_a` and leave `rho_tot` as the fluid
sum — note `aH_prime` is *already* curvature-correct in the (ρ+3P) form, so explicit-term
implementers must NOT also modify line 208).

### 2.4 Conformal time (252-319)

`_tabulate_conformal_time`: integrates `dτ/dlna = 1/aH` (231-250) with Kvaerno5, `SaveAt(dense=True)`,
from `lna_cut = -16.1` to 0, stitched to the analytic radiation-era solution
```python
274  tau_approx = lambda lna: jnp.exp(lna) / (cnst.H0_over_h / cnst.c_Mpc_over_s) / jnp.sqrt(params["omega_r"])
```
for `lna < -16.1`, evaluated on `lna_tau_tab = linspace(-33, 0, 10000)`; non-finite entries fall back
to `tau_approx` (317). **Form-invariant under curvature** (τ is defined by the metric, K enters only
through `aH`); the early-time approx is radiation-only and stays valid (curvature ~ e^{-32} relative).
`tau(lna)` (321-347) is `fast_interp` on the uniform `lna_tau_tab` grid. `tau0 = tau(0.)` (89).

### 2.5 χ / comoving-distance usage

```python
559  self.rA_rec = self.tau0 - self.tau(self.lna_rec)   # "Comoving angular diameter distance at recombination."
```
This is the **flat identification r_A = χ = τ0 − τ**. In curved space `r_A = S_K(χ) =
sin(√K χ)/√K (closed) / sinh(√−K χ)/√−K (open)`. As of this checkout `rA_rec` is *stored but not
consumed* anywhere in `abcmb/` (grep: only definition + docstrings) — but the same flat
identification is load-bearing in `spectrum.py` (LOS argument line 763 and Limber lensing lines
412/426), see §5. A curved implementation should add an `S_K(chi)` helper on `Background`.

### 2.6 Sound horizon, optical depths, decoupling (682-936)

- `_tabulate_optical_depth` (682-720): `dκ/dlna = −1/(τ_c aH)`, Kvaerno5, t0=0 → t1=−10,
  `SaveAt(dense=True)`, max_steps=2048. Geometry-free given aH.
- `expmkappa` (722-742): `where(lna < −10, 0, exp(−κ(lna)))`.
- `visibility` (744-768): `g = expmkappa/τ_c`.
- `_tabulate_kappa_d` (824-859): `dκ_d/dlna = −1/(τ_c aH R)`, Tsit5, `SaveAt(ts=lna_tau_tab[::-1])`.
- `_tabulate_rs` (861-899): `drs/dlna = 1/(√(3(1+R)) aH)` with IC `rs0 = 1/√3/aH(lna_tau_tab[0])` —
  r_s is a *radial* comoving distance; **form-invariant in curved space**. `z_d`/`rs_d` (901-936)
  interpolate κ_d = 1.
- All four solves correct automatically once `aH` carries curvature.

### 2.7 lna grids / transfer-start logic (554-564)

```python
555  lna_vals = jnp.linspace(-8.0, -4.0, 1500)           # visibility-peak search window
557  self.lna_rec = lna_vals[jnp.argmax(vis_vals)]
558  self.lna_visibility_stop = lna_vals[jnp.argmin((vis_vals - 1.e-3)**2)]   # stored; not consumed elsewhere
562  lna_vals = jnp.linspace(-15.0, -6.0, 5000)
563  aH_tau_c_vals = vmap(self.aH,...)(lna_vals, params) * self.tau_c(lna_vals, params)
564  self.lna_transfer_start = lna_vals[jnp.argmin((aH_tau_c_vals-0.008)**2)]  # aH·τ_c ≈ 0.008
```
Geometry-free criteria (thermodynamic). `lna_transfer_start` sets the PE output grid
`linspace(lna_transfer_start, 0, 500)` (perturbations.py:100).

### 2.8 Where `omega_Lambda` is consumed

1. `species.py:471` — `DarkEnergy.rho = omega_Lambda · 3H_100²/8πG` (constant). Via `rho_tot` this is
   the only place Λ enters H(a).
2. `spectrum.py:377-380` — `Omega_L = omega_Lambda/h²` in the lensing `Om(z)` factor (flat, §5.6).
Both inherit the corrected closure automatically once main.py:594 subtracts `omega_k`.

### 2.9 Reionization models (939-1017)

`ReionizationModel.xe_reion` (CAMB tanh in `y=(1+z)^exp_reion`, + HeII tanh) and `tau_reion_fn`
(trapezoid of `Γ/aH` over `linspace(-5,0,2000)`): geometry enters only through `aH` — clean.
`ReionizationModelFromTau` root-finds z_reion with `optx.Newton` (1008-1017).

---

## 3. `species.py`

### 3.1 `Fluid` base interface (11-249)

```python
44  first_idx     : int = eqx.field(default=0, static=True)
45  num_equations : int = eqx.field(default=0, static=True)
46  name          : str = eqx.field(default="", static=True)
47  is_matter     : bool = eqx.field(default=False, static=True)
```
(Note: this checkout names the vector offset `first_idx`; the parent CLAUDE.md's "delta_idx /
diffrax_vector_idx" refers to the same role — `populate_species` passes a running
`diffrax_vector_idx` into each ctor as `first_idx`, model_specs.py:97-116.)
Abstract methods: `rho(lna, params)`, `P(lna, params)`, `w` (= P/ρ, concrete at 96-115),
`y_ini(k, tau_ini, params)`, `y_prime(k, lna, metric_h_prime, metric_eta_prime, y, args)`,
`rho_delta(lna, y, params)`, `rho_plus_P_theta`, `rho_plus_P_sigma`, `output_perturbations` (default `{}`).
`StandardFluid` (251-400) adds getters `get_delta/get_theta/get_sigma` (`y[first_idx + 0/1/2]`) and
generic `rho_delta = ρ·δ`, `rho_plus_P_theta = (ρ+P)θ`, `rho_plus_P_sigma = (ρ+P)σ`.
`BackgroundFluid` (402-428): `num_equations=0`, empty `y_ini`/`y_prime`, zero stress-energy
perturbations — the natural parent for a curvature pseudo-fluid.

**How `y_prime` receives metric quantities and k:** `PerturbationEvolver.get_derivatives`
(perturbations.py:201-248) computes `metric_h_prime`, `metric_eta_prime` from the Einstein
constraints, then calls every `species.y_prime(k, lna, metric_h_prime, metric_eta_prime, y, args)`
with `args = (BG, params, species_list, species_dict)`. All species derivatives are written in
d/dlna form: every conformal-time-derivative term is divided by `aH`. **How `y_ini` receives
ICs:** `initial_conditions_one_k` calls `p.y_ini(k, tau_ini, params)` with `tau_ini = BG.tau(lna_ini)`.

### 3.2 DarkEnergy (432-490)

`rho = params['omega_Lambda']·(3H_100²/8πG)` (471), `P = −ρ` (490). Background-only.

### 3.3 ColdDarkMatter (492-610)

`rho = omega_cdm·(3H_100²/8πG)/a³` (537); `P = 0`. `num_equations = 1`.
```python
580  delta = -(k*tau_ini)**2/4. * (1.-params["om"]*tau_ini/5.)        # y_ini
607  return jnp.array([-0.5*metric_h_prime])                          # y_prime: δ'_c = −h'/2
```

### 3.4 Baryon (1093-1291)

`rho = omega_b·(3H_100²/8πG)/a³` (1139); `P = 0`. `num_equations = 2`.
`cs2` (1163-1196, M&B eq. 68): `Tm/μ (5/3 − (2/3)(μ R)/(m_e aH τ_c)(Tg/Tm − 1))`.
ICs (1222-1243):
```python
1241  delta = -(k*tau_ini)**2/4. * (1.-params["om"]*tau_ini/5.)
1242  theta = - k**4 * tau_ini**3/36. * (1.-3.*(1.+5.*params['R_b']-params['R_nu'])/20./(1.-params['R_nu'])*params["om"]*tau_ini)
```
y_prime (1245-1285):
```python
1282  delta_prime = -theta/aH - metric_h_prime/2.
1283  theta_prime = -theta + cs2*k**2*delta/aH + R/tau_c/aH*(theta_g-theta)
```
Curvature: the `cs2 k²` pressure term is gauge/geometry standard; in curved space `k²` here is the
eigenvalue of the Laplacian — if perturbations are labeled by the curved eigenvalue this line is
form-invariant (it is the *metric* equations and the l≥2 couplings that pick up explicit K).

### 3.5 Photon (1293-1457) — temperature + polarization hierarchies (exact coefficients)

`rho = (π²/15) TCMB0⁴/a⁴/(c ħ)³` (1346); `P = ρ/3`. `num_F_ell_modes = l_max_g+1 = 13`,
`num_G_ell_modes = l_max_pol_g+1 = 11`, `num_equations = 24` (default).
ICs (1386-1388): same δ, θ as baryons (tight coupling); all higher F, all G start at 0.

y_prime (1390-1448), with `F = [δ, θ, σ, F₃, …]`, `G = [G₀…G_{Glmax}]`, primes = d/dlna:
```python
1432  delta_prime = -4./3./aH*theta - 2./3.*metric_h_prime
1433  theta_prime = k**2/aH*(delta/4.-sigma) + (theta_b-theta)/aH/tau_c
1434  sigma_prime = 4./15./aH*theta - 3./10.*k/aH*F[3] + 2./15.*metric_h_prime + 4./5.*metric_eta_prime \
                  - 9./10./aH/tau_c*sigma + (G[0]+G[2])/20./aH/tau_c
1435  F3_prime    = k/7./aH * (6.*sigma - 4.*F[4]) - F[3]/aH/tau_c
      # Temperature hierarchy, 4 <= L < Flmax:
1439  Fl_prime    = 1./(2.*L+1.)*k/aH * (L*F[L-1]-(L+1)*F[L+1]) - F[L]/aH/tau_c
1440  Flmax_prime = k/aH*F[Flmax-1] - (Flmax+1)/aH/tau*F[Flmax] - F[Flmax]/aH/tau_c
      # Polarization hierarchy, 0 <= L < Glmax:
1444  Gl_prime    = 1./(2.*L+1.)*k/aH * (L*G[L-1]-(L+1)*G[L+1]) - G[L]/aH/tau_c \
                  + (2.*sigma+G[0]+G[2])/2./aH/tau_c * jnp.concatenate((jnp.array([1., 0., 0.2]), jnp.zeros(Glmax-3)))
1447  Glmax_prime = k/aH*G[Glmax-1] - (Glmax+1)/aH/tau*G[Glmax] - G[Glmax]/aH/tau_c
```
(Note `Fnu_2 = 2σ` convention; σ' carries the metric source `2/15 h' + 4/5 η'`.)

**Curvature targets:** every `l→l±1` coupling `k·l/(2l+1)`, `k·(l+1)/(2l+1)` acquires curvature
factors. In the standard total-angular-momentum formulation (Hu, Seljak, White, Zaldarriaga 1998;
CLASS `perturbations.c` curved hierarchies), with `β_l ≡ 1 − (l²−1)K/k²` for temperature
(spin-0 source) and `β_l^{(pol)} = 1 − (l²−3)K/k²`-type factors for polarization (generally
`ₛκ_l = sqrt[(l²−s²)(1 − l²K/k²)]/l`-weighted couplings), the hierarchy becomes
`F'_l ∝ k/(2l+1)·[l·s_l F_{l−1} − (l+1)·s_{l+1} F_{l+1}]` with `s_l = sqrt(1 − (l²−1)K/k²)`
(temperature) — the exact convention should be matched to CLASS/CAMB during implementation.
Additionally `k` itself is replaced by the eigenvalue label (ν or q with `q² = k² − K` conventions),
and for closed universes the hierarchy must terminate at `l ≤ ν − 1` physically.
**Truncation (1440, 1447)** uses the flat free-streaming closure (M&B eq. 51)
`F'_{lmax} ≈ k F_{lmax−1} − (lmax+1)/τ · F_{lmax}` whose `(lmax+1)/(kτ)` comes from the flat
recurrence `j_{l+1}(x) = (2l+1)/x j_l − j_{l−1}`; the curved analog replaces `1/τ` with
`√|K| cot_K(√|K| τ)`-type functions (CLASS: `cotKgen`).

### 3.6 MasslessNeutrino (612-760)

`rho = N_nu_massless · 2·(7/8)(π²/30)(T_nu_massless·TCMB0)⁴/a⁴/(cħ)³` (654); `P = ρ/3`.
`num_equations = l_max_massless_nu + 1 = 18` (default).
ICs (677-705) — transcribed:
```python
698  delta = - (k*tau_ini)**2/3. * (1.-params["om"]*tau_ini/5.)
699  theta = - k*(k*tau_ini)**3/36./(4.*R_nu+15.) \
             * (4.*R_nu+11.+12.-3.*(8.*R_nu**2+50.*R_nu+275.)/20./(2.*R_nu+15.)*tau_ini*params["om"])
701  sigma = (k*tau_ini)**2/(45.+12.*R_nu) * 2. * (1.+(4.*R_nu-5.)/4./(2.*R_nu+15.)*tau_ini*params["om"])
```
y_prime (707-753):
```python
742  delta_prime = -4./3./aH*theta - 2./3.*metric_h_prime
743  theta_prime = k**2/aH*(delta/4.-sigma)
744  sigma_prime = 4./15./aH*theta - 3./10.*k/aH*F[3] + 2./15.*metric_h_prime + 4./5.*metric_eta_prime
745  F3_prime    = 1./7. * k/aH * (6.*sigma - 4.*F[4])
      # 4 <= L < lmax:
750  Fl_prime    = 1./(2.*L+1.)*k/aH * (L*F[L-1]-(L+1)*F[L+1])
751  Flmax_prime = k/aH*F[lmax-1] - (lmax+1)/aH/tau*F[lmax]
```
Same curvature factors and truncation issue as the photon temperature hierarchy (no τ_c terms).

### 3.7 MassiveNeutrino (762-1090)

3-point Gauss-Laguerre-style momentum quadrature: `q_3p = [0.913201, 3.37517, 7.79184]`,
`w_3p = [0.0687359, 3.31435, 2.29911]` for perturbations; 5-point (`q_5p/w_5p`, 793-794) for
background ρ and P (809-880, `∫ ∝ √(q²+x²)` and `∝ 1/√(q²+x²)` with `x = m/T(a)`).
`num_equations = 3·(l_max_massive_nu+1) = 54` (default). State per bin: `[Ψ₀, k·Ψ₁, Ψ₂, Ψ₃ …]`.
ICs (882-917): rescale the massless δ/4, θ/3, σ/2 by `q dlnf₀/dlnq = −q/(1+e^{−q})` per bin.
y_prime (919-977):
```python
962  Psi0_prime  = -q/epsilon/aH*Psi[1] + metric_h_prime/6. * dlnf0_dlnq
963  kPsi1_prime = q*k**2/3./epsilon/aH * (Psi[0] - 2.*Psi[2])
964  Psi2_prime  = q*k/5./epsilon/aH * (2.*Psi[1]/k - 3.*Psi[3]) - (metric_h_prime/15. + 2.*metric_eta_prime/5.) * dlnf0_dlnq
      # 3 <= L < lmax:
969  Psi_inter_prime = q*k/epsilon/aH/(2*L_inter+1) * (L_inter*Psi[L_inter-1] - (L_inter+1)*Psi[L_inter+1])
972  Psi_lmax_prime  = q*k/aH/epsilon*Psi[lmax-1] - (lmax+1)/aH/tau*Psi[lmax]
```
`epsilon = √(q²+x²)`. Same `l/(2l+1)`-type couplings → curvature factors; same flat truncation.
Stress-energy integrals (979-1075) use `w_3p` weights — geometry-free.

---

## 4. `perturbations.py`

### 4.1 `PerturbationEvolver` structure (23-117)

Fields: `species_list`, `species_dict`, `k_axis_perturbations`, `specs`, static `adjoint`.
`full_evolution` (75-117): `lna = jnp.linspace(BG.lna_transfer_start, 0., 500)` (100); on GPU
`vmap(self.evolution_one_k, in_axes=[0,None,None])(self.k_axis_perturbations, lna, args)` (110),
on CPU a `lax.scan` over k; result transposed to `(Ny, Nlna, Nk)` then `make_output_table`.

### 4.2 `get_starting_time` (119-159)

Searches `lna ∈ linspace(-20, -10, 10000)`; inverts `τ_c·aH = R_tc (0.0015)` and `k/aH = R_large (0.07)`,
takes the min, later clipped to ≤ −10 (279). Geometry-free criteria; `k/aH` is the flat-mode label
(would become the curved eigenvalue).

### 4.3 Adiabatic initial conditions (161-199) — transcribed

```python
190  tau_ini = BG.tau(lna_ini)
194  metric_eta_ini = (1.-k**2*tau_ini**2/12./(15.+4.*params['R_nu'])*(5.+4.*params['R_nu'] \
                     - (16.*params['R_nu']*params['R_nu']+280.*params['R_nu']+325)/10./(2.*params['R_nu']+15.)*tau_ini*om))
196  all_fluid_ini = jnp.concatenate([p.y_ini(k, tau_ini, params) for p in self.species_list])
197  y_ini = jnp.concatenate((jnp.array([metric_eta_ini]), all_fluid_ini))
```
State vector layout: `y = [metric_eta, CDM(1), Baryon(2), Photon(24), MasslessNu(18), (MassiveNu 54)]`
(via `diffrax_vector_idx` starting at 1, model_specs.py:97). These are the flat CLASS-style
superhorizon series in `(kτ)`; curved versions acquire `O(K/k²)` and `O(Kτ²)` corrections
(CLASS implements curved adiabatic ICs; for `|Ω_k| ≲ 0.1` the leading corrections are tiny at
`τ_ini`, but formal correctness needs the curved series).

### 4.4 Einstein equations (`get_derivatives`, 201-248) — where k² appears

```python
228  sum_rho_delta = 0.
229  sum_rho_plus_P_theta = 0.
231  for i in range(len(self.species_list)):
234      sum_rho_delta        += species.rho_delta(lna, y, params)
236      sum_rho_plus_P_theta += species.rho_plus_P_theta(lna, y, params)

238  metric_h_prime   = 2./aH**2 * (k**2*metric_eta + 4.*jnp.pi*cnst.G*a**2/cnst.c_Mpc_over_s**2 * sum_rho_delta)
239  metric_eta_prime = 4.*jnp.pi*cnst.G*a**2/aH/k**2 * sum_rho_plus_P_theta / cnst.c_Mpc_over_s**2
```
These are M&B (synchronous gauge) eqs. 21a/21b rearranged into d/dlna form. **Curved versions
(M&B eq. 23 / CLASS):**
- Energy constraint: `k²η − ½ aH h'_(τ) = −4πGa²δρ` becomes `(k² − 3K)η − ½ aH h'_(τ) = −4πGa²δρ`
  → line 238: `k²*metric_eta` → `(k²−3K)*metric_eta`.
- Momentum constraint: `k²η'_(τ) = 4πGa²(ρ+P)θ` becomes `(k²−3K)η'_(τ) = 4πGa²(ρ+P)θ`
  → line 239: `/k**2` → `/(k**2−3K)`.
The `metric_eta` evolution is *only* this constraint (η is `y[0]`, evolved via line 239 prepended
to `y_prime` at 243).

### 4.5 `make_output_table` (318-404): α and α′ expressions

Recomputed on the output grid (karr = k[None,:]):
```python
387  metric_h_prime     = 2./aH**2 * (karr**2*metric_eta + 4.*jnp.pi*cnst.G*a**2/cnst.c_Mpc_over_s**2 * sum_rho_delta)
388  metric_eta_prime   = 4.*jnp.pi*cnst.G*a**2/aH * sum_rho_plus_P_theta / cnst.c_Mpc_over_s**2 / karr**2
389  metric_alpha       = aH*(metric_h_prime + 6.*metric_eta_prime)/2./karr**2
390  metric_alpha_prime = metric_eta/aH - 2.*metric_alpha \
391                     - 12.*jnp.pi*cnst.G*a**2/aH * sum_rho_plus_P_sigma / cnst.c_Mpc_over_s**2 / karr**2
```
- `α = (ḣ + 6η̇)/2k²` (conformal-time form; here multiplied by aH because primes are d/dlna).
  Curved: the shear Einstein equation `k²(ḣ+6η̇)/2 ... = −8πGa²(ρ+P)σ` form becomes
  `(k² − 3K)`-weighted (CLASS: `α' = ... ` with `(1−3K/k²)` factors multiplying the σ source);
  specifically the M&B-style curved eq. is `(ḣ+6η̇)·k²/2 → ` source `(ρ+P)σ` gets a `(1−3K/k²)`
  factor. Lines 389-391 therefore all need explicit K factors.
- Note duplication: the same Einstein constraints live in *both* `get_derivatives` (238-239) and
  `make_output_table` (387-388) — both must be edited consistently. Also `theta_b_prime`
  recomputation at 365 (`cs2/aH*(karr**2*delta_b)`) repeats the baryon Euler equation.
- `delta_m` (385): matter-weighted sum, geometry-free.
- `PerturbationTable` (406-450): fields `k, lna, delta_m, theta_b_prime, metric_eta, metric_h_prime,
  metric_eta_prime, metric_alpha, metric_alpha_prime, species_perturbations` — all `(Nlna, Nk)` on the
  fixed 500-point lna grid and the Nk perturbation k-grid.

### 4.6 `evolution_one_k` (250-316)

Kvaerno5, `t0 = max(get_starting_time, ... min with −10)`, `t1 = 0`, `dt0 = 1e-2`,
PIDController with k-split tolerances (`k_split_PE = 0.01`: rtol 1e-5/1e-4, atol 1e-10/1e-6),
`SaveAt(ts=lna)`, `max_steps = specs["max_steps_PE"] = 2048`, `args=(k, BG, params)`.
No geometry here beyond what `get_derivatives` supplies.

---

## 5. `spectrum.py`

### 5.1 Bessel tables (lines 20-46) and format

```python
21  bessel_l_tab = jnp.array(np.loadtxt(file_dir+"/bessel_tab/l.txt"), dtype="int")
22  xphi0_tab = ...; phi0_tab = ...; xphi1_tab/phi1_tab; xphi2_tab/phi2_tab
```
Format (verified on disk): `l.txt` = **161 l-values**: 2,3,…,16 (step 1), then growing steps
(17→2 …) up to step 40, ending at 5000 (…4930, 4970, 5000). Each `xphiN.txt`/`phiN.txt` is
**5000 rows × 161 cols**: column i is a *uniform* grid of 5000 x-samples for l = bessel_l_tab[i],
spanning from where the function is ~1e-10 up to (per docstrings) the fifth local maximum;
`fast_interp` exploits the uniform spacing. Outside the table: 0 below, large-x asymptotics above.
**Curvature impact: the entire tabulation is a function of the single flat variable x = kχ. The
curved radial functions Φ_l^ν(χ) depend on (l, ν=k-label, χ) separately (plus sign of K), so this
table layout cannot represent them; either a 3-axis table, on-the-fly recursion (CLASS's
hyperspherical module), or a WKB/rescaling approximation is needed.**

### 5.2 Asymptotic branch + radial functions phi0/phi1/phi2 (49-109) — transcribed

```python
49  Q = lambda l, x : jnp.sqrt(x**2-l**2) - l*jnp.pi/2 + l * jnp.arcsin(l/x)
50  J = lambda l, x : jnp.sqrt(2/jnp.pi/jnp.sqrt(x**2-l**2)) * jnp.cos(Q(l, x) - jnp.pi/4)
51  j = lambda l, x : jnp.sqrt(jnp.pi/2/x) * J(l+1/2, x)        # WKB spherical Bessel
```
```python
53  def phi0(i, x):     # phi0 = j_l
62      x_safe = jnp.where(x >= xphi0_tab[-1, i], x, xphi0_tab[-1, i])    # reverse-AD landmine #3 pattern
63      return jnp.where(x < xphi0_tab[0, i], 0.,
66          jnp.where(x >= xphi0_tab[-1, i], j(l, x_safe),
69              tools.fast_interp(x, xphi0_tab[:, i].min(), xphi0_tab[:, i].max(), phi0_tab[:, i])))
```
```python
73  def phi1(i, x):     # phi1 = j_l' : asymptotic branch  l/x_safe*j(l, x_safe) - j(l+1, x_safe)      (line 87)
92  def phi2(i, x):     # phi2 = (3 j_l'' + j_l)/2 :
106     ((3*l*(l-1)-2*x_safe**2)*j(l, x_safe)+6*x_safe*j(l+1, x_safe))/2/x_safe**2
```
`Cl_one_ell` re-defines `phi0_local/phi1_local/phi2_local` (719-753) with pre-sliced columns
(`x0_min/x0_max/col_phi0_l` etc., 708-716) — identical math, **two copies of each radial function
to keep in sync** (module-level 53-109 and local 719-753; module-level `phi1/phi2` appear unused
by the production path but exist for parity). The `x_safe` pre-clip is reverse-AD landmine #3 —
any curved replacement must preserve the pattern (clip the argument *everywhere* in the unused
branch, not just inside the special function).

### 5.3 The LOS Cl integral (`Cl_one_ell`, 616-797)

Structure: per ell index `idx` into `bessel_l_tab`:
- `lna_axis = PT.lna[:-1]` (499 pts), `delta_lna` trapezoid weight (640-641); background vectors
  `tau0, tau(lna), g, g', aH, expmkappa, aH_dot` (645-651).
- Source-table k-interpolation: `CubicSpline(log10(PT.k), col)(log10(k_axis_transfer))` per lna row
  (664-678) for `delta_g, theta_b, theta_b_prime, sigma_g, Gg0, Gg2, eta, eta_prime, alpha, alpha_prime`.
- **Source functions (681-696) — transcribed:**
```python
681  sourceT0 = scale_sw * g * (delta_g/4. + aH*alpha_prime) \
           + scale_isw * ( g * (eta - aH*alpha_prime - 2.*aH*alpha)
                         + 2.*expmkappa * (aH*eta_prime - aH_dot*alpha - aH**2*alpha_prime) ) \
           + scale_dop * ( aH * (g*((theta_b_prime / k_axis**2) + alpha_prime)
                         + g_prime*((theta_b / k_axis**2) + alpha)) )
691  sourceT1 = scale_isw * expmkappa * ((aH*alpha_prime + 2.*aH*alpha - eta) * k_axis)
694  sourceT2 = scale_pol * g * (2*sigma_g + Gg0 + Gg2) / 8.
696  sourceE  = jnp.sqrt(6) * g * (2*sigma_g + Gg0 + Gg2) / 8.
```
  Curvature: `theta_b/k²` Doppler terms and the T1 `·k_axis` factor are tied to the flat radial
  decomposition (T0·φ0 + T1·φ1 + T2·φ2); the curved decomposition keeps the same structure but with
  `k → ν` and curvature-corrected derivative relations between the radial functions.
- **Rolling lna scan (755-783):**
```python
763  chi_l = (tau0 - tau_l) * k_axis          # <-- flat radial argument x = k·χ
764  phi0_l = phi0_local(chi_l); phi1_l = phi1_local(chi_l); phi2_l = phi2_local(chi_l)
767  eps_l  = phi0_l / chi_l**2 * ell_eps_factor     # ell_eps_factor = sqrt(3/8*(l+2)(l+1)l(l-1))  (line 717)
769  acc_T0 += w_l * sT0_l / aH_l * phi0_l   (similarly T1·phi1, T2·phi2, E·eps)
781  lax.scan(jax.checkpoint(scan_step), init, xs)   # checkpointed body = reverse-AD landmine #2
```
  **`chi_l = (tau0 − tau)·k` at line 763 is the single most geometry-laden line in spectrum.py**:
  in curved space the argument separates into (ν, χ) and `phiN(x)` → `Φ_l^ν(χ)` etc.; the E-mode
  `phi0/χ²k²` combination (`eps_l = phi0_l/chi_l²·…`) becomes the curved `ε_l^ν(χ)` with
  `S_K(χ)`-based denominators.
- **Final k-integral (789-796):**
```python
789  integrandTT = 4.*jnp.pi * params['A_s'] * (k_axis/self.k_pivot)**(params['n_s']-1.) * transferT**2 / k_axis
...
794  return (jnp.trapezoid(integrandTT, k_axis), ..., ...)
```
  This is `∫ dk/k P_dimensionless(k) T²`. In curved space the mode measure changes
  (open: `dν ν²/(ν²+|K|)`-type factors depending on convention; closed: a **discrete sum** over
  integer ν ≥ 3 — a shape change that must happen at `__init__`, not in the jit).

### 5.4 ell grid and spline reconstruction (`get_Cl`, 574-614; ctor 216-248)

```python
219  self.ells = jnp.arange(ellmin, ellmax+1)                       # default 2..2500 (output grid)
220  ell_idx_min = jnp.where(bessel_l_tab<=ellmin)[0][-1]
221  ell_idx_max = jnp.where(bessel_l_tab>=ellmax)[0][0]
222  self.ells_indices = jnp.arange(ell_idx_min, ell_idx_max+1)     # ~99 table l's for 2..2530
224  if self.lensing:  lensing_ellmax = ellmax+500                  # → table l's up to ≥3000 (~111 entries)
230      num_mu = lensing_ellmax + 70;  mu, w = tools.gauss_legendre_weights(num_mu)
```
`get_Cl` vmaps `Cl_one_ell` over `lensing_ells_indices` (594), then `CubicSpline(lensing_ells, raw)`
onto the dense integer ell grid (598-601), then `lax.cond(self.lensing, get_lensed_Cls, get_unlensed_Cls)`
(610-614). The sparse-l + spline trick assumes Cl smoothness — unchanged by curvature (though the
angular-scale shift from r_A would move features; the table's l ≤ 5000 ceiling is unchanged).

### 5.5 `primordial_spectrum` and `Pk_lin` (250-347)

```python
266  return params['A_s']*(k/self.k_pivot)**(params['n_s']-1.) * (2*jnp.pi**2/k**3)
302  return delta_m**2 * self.primordial_spectrum(k, params)        # Pk_lin
```
`Pk_lin` interpolates `PT.delta_m` in lna (jnp.interp) then in k (jnp.interp at 300). `Pk_cb`
(304-347) is the CDM+baryon analog weighted by `omega_b/omega_cdm/omega_m`. Curvature: the `1/k³`
measure and the meaning of P(k) at the largest scales change (`k³ → ν(ν²−K)`-type, convention-
dependent); δ_m transfer itself follows from the corrected perturbations.

### 5.6 Lensing (`lensing_power_spectrum` 349-384, `lensing_Cl` 386-435, `lensed_Cls` 437-572)

```python
376  Omega_m = params["omega_m"]/params["h"]**2
377  Omega_L = params["omega_Lambda"]/params["h"]**2
380  Om = (Omega_m * (1.+z)**3)/ ((Omega_m * (1.+z)**3) + Omega_L)   # flat; no Omega_k(1+z)^2 (nor radiation)
384  return 9./8./jnp.pi**2 * Om**2 * aH**4 * Pk / k                 # converts Pk → P_Psi via flat Poisson eq.
```
The Poisson conversion `Φ ∝ (3/2) Ω_m(a) (aH/k)² δ` is also flat: curved Poisson gives
`(k² − 3K) Φ = −4πGa²ρΔ`-type relations → `1/k` factors here become `k³/(k²−3K)²`-weighted.

**Limber integral (386-435) — transcribed:**
```python
411  coeff = 8.*jnp.pi**2/(ells+0.5)**3
412  chi = lambda lna : BG.tau0 - BG.tau(lna)
419  lna_axis = jnp.linspace(BG.lna_rec, 0., 500)
420  lna_floor = lna_axis[-2]
422  def integrand_func(lna):
423      lna_safe = jnp.where(lna < 0., lna, lna_floor)       # reverse-AD landmine #7 pattern
424      chi_safe = chi(lna_safe)
425      k = (ells+0.5)/chi_safe                              # Limber k = (l+1/2)/χ  → /S_K(χ)
426      window = (chi(BG.lna_rec) - chi_safe)/chi(BG.lna_rec)/chi_safe
427      res = ( chi_safe / BG.aH(lna_safe, params) * window**2
                * self.lensing_power_spectrum(k, lna_safe, PT, BG, params) )
432      return jnp.where(lna < 0., res, 0.)                  # boundary mask (landmine #7)
434  integrand = vmap(integrand_func)(lna_axis)
435  return coeff*jnp.trapezoid(integrand, lna_axis, axis=0)
```
Curved replacements: `k = (l+½)/S_K(χ)`; `window = S_K(χ*−χ)/(S_K(χ*) S_K(χ))`; the `chi_safe/aH`
volume factor follows from `dχ` with the appropriate `S_K²` already absorbed in the window² —
re-derive carefully against astro-ph/0601594's curved generalization. Preserve the
`lna_safe`/`where`-mask structure (landmine #7).

`lensed_Cls` (437-572): Wigner-d correlation-function method (`tools.d00/d1n/d2n/d3n/d4n`),
Gauss-Legendre quadrature in μ (`lensing_mus/lensing_ws`), X000/X220/X022/X121/X132/X242 kernels —
**all-sky harmonic machinery, geometry-independent given Clpp and the unlensed Cls.** No changes.

---

## 6. `model_specs.py`

### 6.1 `load_specs` defaults (7-82) — full relevant list

| key | default | key | default |
|-----|---------|-----|---------|
| `use_LCDM_species` | True | `k_step_sub` | 5e-2 |
| `input_tau_reion` | True | `k_step_super` | 2e-3 |
| `l_min` / `l_max` | 2 / 2500 | `k_step_transition` | 2e-1 |
| `lensing` | **False** | `k_step_super_reduction` | 1e-1 |
| `k_max` | 0.5 | `k_min_tau0` | 1e-1 |
| `bbn_type` | "" | `k_max_tau0_over_l_max` | 1.8 |
| `linx_reaction_net` | key_PRIMAT_2023 | `H0_fid` | 2.255560e-04 |
| `l_max_g` | **12** | `tau0_fid` | **1.418668e+04** |
| `l_max_pol_g` | **10** | `rs_rec_fid` | **1.446279e+02** |
| `l_max_massless_nu` | **17** | `k_transfer_linstep` | 4.5e-1 |
| `l_max_massive_nu` | **17** | `k_transfer_logstep` | 170. |
| `R_tc` / `R_large` | 0.0015 / 0.07 | `tau_rec_fid` | 281.040565 |
| `max_steps_PE` | 2048 | `k_pivot` | 0.05 |
| `k_split_PE` | 0.01 | rtol/atol small-k | 1e-5 / 1e-10 |
| `pcoeff/icoeff/dcoeff_PE` | 0.25/0.8/0. | rtol/atol large-k | 1e-4 / 1e-6 |
| `scale_sw/isw/dop/pol` | 1 | | |

Unknown spec keys are preserved (78-80) — an `omega_k`-related *spec* (e.g. a curved-mode-grid
switch) would pass through; but note specs are **static-ish** (carried as a dict pytree; branching
on them belongs in `__init__`).

### 6.2 `get_k_axis_perturbations` (120-175) — exact formula

```python
126  k_rec_fid  = 2.*jnp.pi/rs_rec_fid
128  k_min = specs["k_min_tau0"] / tau0_fid                         # ≈ 7.05e-6 Mpc^-1
129  k_max = specs["k_max_tau0_over_l_max"] / tau0_fid * specs["l_max"]   # ≈ 0.3172 for l_max=2500
134  while k < k_max:
135      step = (k_step_super + 0.5*(tanh((k-k_rec_fid)/k_rec_fid/k_step_transition)+1.)
                 * (k_step_sub-k_step_super)) * k_rec_fid
139      scale2 = H0_fid**2
141      step *= (k**2/scale2+1.)/(k**2/scale2+1./k_step_super_reduction)
143      k += step
147  specs["k_min"] = k_min;  specs["k_max_cmb"] = k
151  if specs["lensing"]: extend to k+0.3 with step 0.005
162  if k < specs["k_max"]: extend to k_max (0.5) with step 0.005
173  k_axis_Pk_output = ks[ks <= specs["k_max"]]
```
This is bit-identical to CLASS's perturbation k-sampler (per project memory). **Nk ≈ 548** for the
fiducial ΛCDM config (lensing=False, k_max=0.5; per CHANGELOG/CLAUDE.md benchmarks; buffer 2000).
Flatness: grid is built from *fiducial flat* `tau0_fid`, `rs_rec_fid`, `H0_fid` — these are
fixed reference scales, not cosmology-dependent, so mild curvature only mis-tunes density, not
correctness. **Closed universes are the real problem: k becomes discrete (ν = 3,4,5,…·√K), so the
continuous grid (and the trapezoid in Cl_one_ell) must be replaced/resampled — a shape change.**

### 6.3 `get_k_axis_transfer` (177-193) — exact formula

```python
180  k_period = 2*jnp.pi/(specs["tau0_fid"] - specs["tau_rec_fid"])      # ≈ 4.519e-4
185  while k < specs["k_max_cmb"]:
186      k = k + k_period * k_transfer_linstep * k / (k + k_transfer_linstep/k_transfer_logstep)
```
i.e. log-spaced (ratio ≈ 1+0.0768) at small k, linear (Δk ≈ 0.45·k_period ≈ 2.03e-4, sampling the
acoustic period) at large k. From k_min ≈ 7.05e-6 to k_max_cmb ≈ 0.317 this yields **≈ 1.6-1.7k
points** (estimate from the formula; buffer 8000; not executed). `k_period` is the flat fiducial
acoustic-oscillation period of the transfer functions `j_l(k(τ0−τrec))` — in curved space the
oscillation period in ν is analogous; fiducials again only affect sampling density.

### 6.4 `populate_species` (84-118)

LCDM tuple `(DarkEnergy, ColdDarkMatter, Baryon, Photon, MasslessNeutrino)` when
`use_LCDM_species`; each instantiated as `s(diffrax_vector_idx, specs)` with
`diffrax_vector_idx` starting at **1** (slot 0 = `metric_eta`) and advanced by
`instance.num_equations`; `user_species` appended after. A `Curvature` background species would be
passed via `user_species` (num_equations=0 → no state-vector impact) — or, if it becomes core,
appended to `lcdm_species` behind a spec flag (shape-neutral either way).

Default state vector (ΛCDM, no massive ν): `Ny = 1 + 1 + 2 + 24 + 18 = 46`
(`metric_eta`, CDM, Baryon, Photon F(13)+G(11), MasslessNu(18)); +54 with MassiveNeutrino.

---

## 7. `ABCMBTools.py` and `constants.py`

- `fast_interp` (268-308): uniform-grid linear interpolation (clips index into `[eps, n−1−eps]`).
  Used for: `tau_tab` (BG.tau), `xe_tab`/`Tm_tab` (BG.xe/Tm), and all Bessel-table lookups in
  `spectrum.py`. **Assumes uniform x-grids** — a curved radial-function table must either keep
  per-(l,ν) uniform χ-grids or switch interpolators.
- `bilinear_interp` (311-360): only used for the PArthENoPE YHe table (main.py:493).
- Wigner-d machinery (16-185) + `gauss_legendre_weights` (190-266): lensing correlation method —
  geometry-free.
- `constants.py`: `c = 2.998e10 cm/s`, `c_Mpc_over_s = 9.71561e-15`, `H0_over_h = 3.24078e-18 s^-1`,
  `hbar`, `G = 1.1898e-40 cm³ eV⁻¹ s⁻²` (G/c²), `kB`, masses, recombination energies, `thomson_xsec`.
  No curvature-related constants; a curvature radius / K in Mpc⁻² would be derived as
  `K = −omega_k·(H_100/c)² = −omega_k·(H0_over_h/c_Mpc_over_s)²` in code units — note
  `(H0_over_h/c_Mpc_over_s) = 3.3356e-4 Mpc⁻¹` is the conversion already used at main.py:591 and
  background.py:187.

---

## 8. Reverse-AD landmine patterns to respect when adding curvature

(From the repo CLAUDE.md — note the worktree itself has **no CLAUDE.md checked in**; the canonical
copy lives at `/pscratch/sd/c/carag/ABCMB/CLAUDE.md` and is untracked. Patterns verified present in
this checkout's code.)

1. **`jnp.where` dead-branch safety**: any new `where` whose unused branch can produce NaN/inf
   forward (sqrt of negative, /0, log 0) must pre-clip the argument (`x_safe` pattern,
   spectrum.py:62/81/100/720/732/744) — e.g. `sin(√K χ)/√K` vs `sinh(√−K χ)/√−K` branches for
   open/closed `S_K` are *exactly* this bug class if implemented as `jnp.where(K>0, sin..., sinh...)`
   with a `sqrt(K)` / `sqrt(−K)` in each branch. Prefer series-safe forms (e.g. `sinc`-based) or
   clipped arguments in both branches. Same for `1/(k²−3K)` near small k in closed models.
2. **No new int/bool pytree leaves** on the AD path: give `omega_k` a float default
   (`jnp.array(0.)`); any new static species metadata must be `eqx.field(static=True)`
   (cf. `Fluid.first_idx/num_equations/name/is_matter`, species.py:44-47).
3. **Shape logic in `__init__` only**: open vs closed *mode-grid* differences (discrete ν) and any
   new radial-function tables are construction-time; only K-values flow through the jit.
4. `lax.scan` body in `Cl_one_ell` is `jax.checkpoint`-wrapped (spectrum.py:781) — keep when
   editing `scan_step` (curved radial functions).
5. `lensing_Cl` boundary mask (landmine #7, spectrum.py:422-432) and `Background.__init__`
   finite-padding (landmine #8/#9, background.py:544-550) — preserve structure when touching
   either function.
6. New diffrax solves (if any) need `adjoint=self.adjoint()` plumbing and static adjoint fields.
7. Params int→float cast at run_cosmology_abbr top (main.py:207-212) — relies on float-meaningful
   params; don't encode "closed/open" as an int flag in `params` (use specs).

---

## 9. Summary of defaults requested

- **l_max per species**: photon temperature `l_max_g = 12`, photon polarization `l_max_pol_g = 10`,
  massless ν `l_max_massless_nu = 17`, massive ν `l_max_massive_nu = 17` (×3 momentum bins).
- **Nk**: perturbation grid ≈ 548 (formula-built, fiducial ΛCDM, lensing off; ≤2000 buffer);
  transfer grid ≈ 1.6-1.7k (formula estimate, ≤8000 buffer); `k_axis_Pk_output` = perturbation-grid
  points ≤ k_max=0.5.
- **Cl ell grid**: computed at the ~99 (unlensed, ellmax 2500) / ~111 (lensing, ellmax+500)
  `bessel_l_tab` entries spanning the request, cubic-splined onto `arange(l_min, l_max+1)`
  (spectrum.py:594-601); table tops out at l=5000.
- **Bessel tables**: 161 l's (2→5000), 5000 uniform x-samples per l, three function families
  (φ0=j_l, φ1=j_l′, φ2=(3j_l″+j_l)/2) + x-grids; flat-only by construction.

## 10. Suggested implementation order (minimal-blast-radius)

1. `params['omega_k']` default + closure fix (main.py:380ish, 594, expected_keys 600).
2. Background: explicit `−Kc²/a²` in `H` (background.py:166) **or** `Curvature(BackgroundFluid)`
   species (§2.3; `aH_prime` then needs no edit in either route — verify, since (ρ+3P)_k = 0).
   Add `S_K(χ)`/`cot_K` helpers on `BackgroundPreRecomb`.
3. Perturbations: `(k²−3K)` in the two constraint sites ×2 copies (perturbations.py:238-239,
   387-391) + curvature factors in the three hierarchies + truncations (species.py §3.5-3.7) +
   curved ICs (perturbations.py:194, species y_ini).
4. Spectrum: hyperspherical radial functions (replaces bessel_tab pipeline + line 763), measure
   factors (266, 789-791), Limber distances (412-432), `Om(z)` (380).
5. Closed-universe discrete-k grids (model_specs) last — open universes work with continuous ν.
