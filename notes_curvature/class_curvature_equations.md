# CLASS non-flat (curved FLRW) implementation — equations, code, and conventions

**Primary source:** `/pscratch/sd/c/carag/AxiCLASS` (CLASS `v3.3.0` per `include/common.h:46`). All curvature
code below was cross-checked line-by-line against `/pscratch/sd/c/carag/class_StepDR` (CLASS `v3.2.0`) and is
**identical base-CLASS code** in both (the Axi/StepDR modifications touch scalar-field/fluid sectors only, and
those are flagged where they appear). `class_EDE` is v2.6.3 (older) and was not used.

All `file:line` references below are into AxiCLASS unless noted. CLASS units convention (load-bearing):

- `pba->H0` is **H0/c in Mpc^-1** (i.e. `H0[km/s/Mpc] / 299792.458`).
- All densities are stored as `rho_class = (8 pi G / 3 c^2) rho_physical`, units Mpc^-2, so the Friedmann
  equation reads `H^2 = rho_tot - K/a^2` with **no 8piG/3 prefactor** (background.c:641-645).
- `a_today = 1` (so K carries no explicit a0 factors).
- `tau` = conformal time in Mpc. `pba->index_bg_H` is the **proper-time Hubble rate** `H = a'/a^2`
  (`'` = d/dtau), in Mpc^-1; `H_prime = dH/dtau`.

Gauge: everything in sections 4-6 is **synchronous gauge** (Ma & Bertschinger conventions, metric
`ds^2 = a^2[-dtau^2 + (delta_ij + h_ij)dx^i dx^j]`, scalar modes parametrized by `h` and `eta`).
The curvature generalization follows **Lesgourgues & Tram, arXiv:1305.3261** ("Fast and accurate CMB
computations in non-flat FLRW universes", CLASS IV paper) — CLASS cites it at perturbations.c:2784 and 9763-9764.

---

## 1. PARAMETER LAYER

### 1.1 Omega_k -> K, sgnK (input.c:3353-3363)

```c
  /** 6) Omega_0_k (effective fractional density of curvature) */
  /* Read */
  class_read_double("Omega_k",pba->Omega0_k);
  /* Complete set of parameters */
  pba->K = -pba->Omega0_k*pow(pba->H0,2);
  if (pba->K > 0.){
    pba->sgnK = 1;
  }
  else if (pba->K < 0.){
    pba->sgnK = -1;
  }
```

So with `pba->H0` in Mpc^-1:

- **K = -Omega_k * H0^2**, units Mpc^-2. (background.h:257 documents it as
  `K = -Omega0_k * a_today^2 * H0^2` with a_today = 1.)
- **Sign convention:** `Omega_k > 0` (open universe) => `K < 0`, `sgnK = -1`.
  `Omega_k < 0` (closed) => `K > 0`, `sgnK = +1`. Flat: `sgnK = 0`.
- Defaults (input.c:7663-7665): `pba->Omega0_k = 0.; pba->K = 0.; pba->sgnK = 0;`
- `pba->has_curvature` flag set in background_indices (background.c:1378-1379):
  ```c
  if (pba->sgnK != 0)
    pba->has_curvature = _TRUE_;
  ```
- The curvature radius is `1/sqrt(|K|)` Mpc (comoving). `enum spatial_curvature {flat,open,closed}` exists
  in background.h:50-52 but the workhorses are `K` and `sgnK`.

### 1.2 Dark-energy closure with curvature (input.c:4985-5006; default budget input.c:7764)

The budget equation subtracts Omega_k along with all species. Whichever DE component is left unspecified is
used to close the budget:

```c
  /* Step 2 */
  if (flag1 == _FALSE_) {
    /* Fill with Lambda */
    pba->Omega0_lambda= 1. - pba->Omega0_k - Omega_tot;
    ...
  }
  else if (flag2 == _FALSE_) {
    /* Fill up with fluid */
    pba->Omega0_fld = 1. - pba->Omega0_k - Omega_tot;
    ...
  }
  else if ((flag3 == _TRUE_) && (param3 < 0.)){
    /* Fill up with scalar field */
    pba->Omega0_scf = 1. - pba->Omega0_k - Omega_tot;
    ...
  }
```

(`Omega_tot` = sum of g, b, cdm, ur, ncdm, dcdmdr, idr, idm, and any explicitly-specified DE.) The default
initializer (input.c:7764) is the same formula. **Note Omega_k is NOT part of Omega_tot — it enters the
closure once, with a minus sign, exactly as in `sum_i Omega_i + Omega_k = 1`.**

### 1.3 Curvature in H(a) — background_functions (background.c:641-678)

```c
  /** - compute expansion rate H from Friedmann equation: this is the
      only place where the Friedmann equation is assumed. ... */
  pvecback[pba->index_bg_H] = sqrt(rho_tot-pba->K/a/a);

  /** - compute derivative of H with respect to conformal time */
  pvecback[pba->index_bg_H_prime] = - (3./2.) * (rho_tot + p_tot) * a + pba->K/a;
  ...
  /** - compute critical density */
  rho_crit = rho_tot-pba->K/a/a;

  class_test(rho_crit <= 0.,
             pba->error_message,
             "rho_crit = %e instead of strictly positive",rho_crit);

  /** - compute relativistic density to total density ratio */
  pvecback[pba->index_bg_Omega_r] = rho_r / rho_crit;
```

I.e. (CLASS density units, H in proper time, ' = d/dtau):

- `H^2 = rho_tot - K/a^2`
- `dH/dtau = -(3/2)(rho_tot + p_tot) a + K/a`
- `rho_crit(a) = rho_tot - K/a^2 = H^2`; `Omega_i(a)` ratios are taken w.r.t. this rho_crit, so curvature is
  excluded from the critical density (standard).

No other background equation carries an explicit K except the distances and rs below.

---

## 2. DISTANCES (background.c)

### 2.1 Comoving radius with sin/sinh branching (background.c:2444-2449)

Computed in a loop over the background table after integration (`conformal_age = tau_0`):

```c
    conformal_distance = pba->conformal_age - pba->tau_table[index_loga];
    pba->background_table[index_loga*pba->bg_size+pba->index_bg_conf_distance] = conformal_distance;

    if (pba->sgnK == 0) { comoving_radius = conformal_distance; }
    else if (pba->sgnK == 1) { comoving_radius = sin(sqrt(pba->K)*conformal_distance)/sqrt(pba->K); }
    else if (pba->sgnK == -1) { comoving_radius = sinh(sqrt(-pba->K)*conformal_distance)/sqrt(-pba->K); }
```

So with `chi = tau_0 - tau` (radial conformal distance, Mpc):

- flat: `r = chi`
- closed (K>0): `r = sin(sqrt(K) chi)/sqrt(K)`
- open (K<0): `r = sinh(sqrt(-K) chi)/sqrt(-K)`

### 2.2 Angular diameter & luminosity distance (background.c:2496-2497)

```c
    pba->background_table[index_loga*pba->bg_size+pba->index_bg_ang_distance] = comoving_radius/(1.+pba->z_table[index_loga]);
    pba->background_table[index_loga*pba->bg_size+pba->index_bg_lum_distance] = comoving_radius*(1.+pba->z_table[index_loga]);
```

`d_A = r/(1+z)`, `d_L = r(1+z)`. The "conformal distance" column (`index_bg_conf_distance`) remains the raw
`tau_0 - tau` regardless of curvature; only ang/lum distances go through the sin/sinh map. Output module and
`background_output_data` just read these columns.

### 2.3 Sound horizon rs (background.c:3213)

In `background_derivs` (CLASS 3.x integrates the background in `loga`, so this is d(rs)/d(loga); the `1/a/H`
factor is dtau/dloga):

```c
  dy[pba->index_bi_rs] = 1./a/H/sqrt(3.*(1.+3.*pvecback[pba->index_bg_rho_b]/4./pvecback[pba->index_bg_rho_g]))*sqrt(1.-pba->K*y[pba->index_bi_rs]*y[pba->index_bi_rs]); // TBC: curvature correction
```

I.e. `d(rs)/dtau = c_s * sqrt(1 - K rs^2)` with `c_s = 1/sqrt(3(1+R))`, `R = 3 rho_b/(4 rho_g)`. **The
curvature change is the `sqrt(1 - K*rs^2)` factor** (sound horizon measured along the curved radial geodesic:
d(r)/d(chi) for r = sin/sinh(sqrt|K| chi)/sqrt|K| is sqrt(1-K r^2), valid for both signs). CLASS itself flags
it `// TBC: curvature correction` — it is a tiny correction since rs << curvature radius.

### 2.4 angular_rescaling (thermodynamics.c:3808-3812; used widely in k/q sampling)

```c
  pth->rs_rec=pvecback[pba->index_bg_rs];
  pth->ds_rec=pth->rs_rec/(1.+pth->z_rec);
  pth->da_rec=pvecback[pba->index_bg_ang_distance];
  pth->ra_rec=pth->da_rec*(1.+pth->z_rec);
  pth->angular_rescaling=pth->ra_rec/(pba->conformal_age-pth->tau_rec);
```

`angular_rescaling = r(z_rec)/(tau_0 - tau_rec)` — ratio of curved comoving angular-diameter distance to the
flat conformal distance to recombination (=1 for K=0; header doc at thermodynamics.h:272). It rescales the
k-sampling (and l<->k mapping heuristics) so that the same angular resolution is achieved in curved space.

---

## 3. WAVENUMBER BOOKKEEPING: k, q, nu

### 3.1 Definitions

From transfer.c:46 (header comment) and `transfer_get_k_list` (transfer.c:1379-1418):

> `q2 = k2 + K(1+m)`, where m=0,1,2 for scalar, vector, tensor

```c
    if (_scalars_) { m=0.; }
    if (_vectors_) { m=1.; }
    if (_tensors_) { m=2.; }

    for (index_q=0; index_q < ptr->q_size; index_q++) {
      ptr->k[index_md][index_q] = sqrt(ptr->q[index_q]*ptr->q[index_q]-K*(m+1.));
    }
```

So for **scalars**:

- `q^2 = k^2 + K`  <=>  `k^2 = q^2 - K`
- `nu = q/sqrt(|K|)`. Diagnostic print (perturbations.c:965-966):
  ```c
  printf(" (for scalar modes, corresponds to nu=%e)",
         sqrt(ppt->k[index_md][index_k]*ppt->k[index_md][index_k]+pba->K)/sqrt(pba->sgnK*pba->K));
  ```
  i.e. `nu = sqrt(k^2 + K)/sqrt(|K|)` (the `sgnK*K = |K|`).
- `k` here is the eigenvalue label of the scalar Helmholtz equation such that the Laplacian eigenvalue is
  `-k^2`; `q` is the "wavevector modulus" appearing in the hyperspherical Bessel functions `Phi_l^nu(chi)`.
- For tensors the perturbations.c:6244-2256 doc block uses `k2 = q2 - 3K` (the comment at 6246 writes
  "q2 = k2 - 3K", a sign typo for `k2 = q2 - 3K`; the algebra two lines later,
  `k2(k2-K)/((k2+3K)(k2+2K)) = (q2-3K)(q2-4K)/(q2(q2-K))`, confirms `q^2 = k^2 + 3K`), consistent with
  `q^2 = k^2 + (1+m)K`, m=2.

### 3.2 Closed-universe discretization (integer nu >= 3)

**Where modes live:** in a closed universe the scalar spectrum is discrete: `nu = q/sqrt(K)` integer, `nu >= 3`
(nu=1,2 are pure gauge). CLASS handles this as follows (perturbations.c:2234-2242 comment):

```c
    /* if K>0, the transfer function will be calculated for discrete
       integer values of nu=3,4,5,... where nu=sqrt(k2+(1+m)K) and
       m=0,1,2 for scalars/vectors/tensors. However we are free to
       define in the perturbation module some arbitrary values of k:
       later on, the transfer module will interpolate at values of k
       corresponding exactly to integer values of nu. Hence, apart
       from the value of k_min and the step size in the vicinity of
       k_min, we define exactly the same sampling in the three cases
       K=0, K<0, K>0 */
```

**So the snapping to integer nu happens in the TRANSFER module's q-list, not in the perturbation module.**
The perturbation module integrates the hierarchy on a smooth k grid; the transfer module builds its q grid
with integer nu (below a threshold) and **interpolates the perturbation sources at k(q)**.

### 3.3 perturb_get_k_list curvature branches (perturbations.c:2143-2356)

First scalar k value:

```c
    /* first value */
    if (pba->sgnK == 0) {
      /* K<0 (flat)  : start close to zero */
      k_min=ppr->k_min_tau0/pba->conformal_age;
    }
    else if (pba->sgnK == -1) {
      /* K<0 (open)  : start close to sqrt(-K)
         (in transfer modules, for scalars, this will correspond to q close to zero;
         for vectors and tensors, this value is even smaller than the minimum necessary value) */
      k_min=sqrt(-pba->K+pow(ppr->k_min_tau0/pba->conformal_age/pth->angular_rescaling,2));
    }
    else if (pba->sgnK == 1) {
      /* K>0 (closed): start from q=sqrt(k2+(1+m)K) equal to 3sqrt(K), i.e. k=sqrt((8-m)K) */
      k_min = sqrt((8.-1.e-4)*pba->K);
    }
```

- Open: smallest k is `sqrt(-K)` (q -> 0 limit), nudged up by the flat k_min in quadrature.
- Closed: smallest scalar mode is nu=3, i.e. `q = 3 sqrt(K)`, i.e. `k^2 = 9K - K = 8K`; the `-1.e-4` puts
  k_min epsilon-below so interpolation brackets nu=3. (Vector/tensor sections at perturbations.c:2363-2376
  and 2496-2509 use `(7.-1e-4)K` and `(6.-1e-4)K` respectively — the `(8-m)K` pattern.)
- `k_max_cmb = k_max_tau0_over_l_max * l_scalar_max / conformal_age / angular_rescaling`
  (perturbations.c:2179-2180): the curvature enters only through `angular_rescaling`.

Step size near the curvature scale (perturbations.c:2280-2293):

```c
      /* ... There are two other characteristic scales that matter for
         the sampling: the Hubble scale today, k0=a0H0, and eventually
         curvature scale sqrt(|K|). We define "scale2" as the sum of the
         squared Hubble radius and squared curvature radius. We need to
         increase the sampling for k<sqrt(scale2) ... */

      scale2 = pow(pba->H0,2)+fabs(pba->K);

      step *= (k*k/scale2+1.)/(k*k/scale2+1./ppr->k_step_super_reduction);
```

### 3.4 transfer_get_q_list — q grid and closed-case integer snapping (transfer.c:1033-1282)

Endpoints:

```c
  if (sgnK == 0) {
    q_min = ppt->k_min;
    ...
    K=0;
  }
  else if (sgnK == -1) {
    q_min = sqrt(ppt->k_min*ppt->k_min+K);      /* scalars: q_min ~ 0+ */
    ...
    q_max = sqrt(k_max*k_max+K);
    if (ppt->has_vectors == _TRUE_)  q_max = MIN(q_max,sqrt(k_max*k_max+2.*K));
    if (ppt->has_tensors == _TRUE_)  q_max = MIN(q_max,sqrt(k_max*k_max+3.*K));
  }
  else if (sgnK == 1) {
    nu_min = 3;
    q_min = nu_min * sqrt(K);
    ...
  }
```

Closed-case snapping loop (transfer.c:1207-1234) — **integer nu enforced only while
`nu < ppr->hyper_flat_approximation_nu` (default 4000.0, precisions.h:438); above that the hyperspherical
Bessels are replaced by rescaled flat Bessels and snapping stops:**

```c
    else {
      if (nu < (int)ppr->hyper_flat_approximation_nu) {

        q = ptr->q[index_q-1]
          + q_period * ppr->q_linstep * ptr->q[index_q-1]
          / (ptr->q[index_q-1] + ppr->q_linstep/q_logstep_trapzd);

        nu_proposed = (int)(q/sqrt(K));
        if (nu_proposed <= nu+1)
          nu = nu+1;
        else
          nu = nu_proposed;

        q = nu*sqrt(K);
        last_step = q - ptr->q[index_q-1];
        last_index = index_q+1;
      }
      else {
        q_step = q_period * ppr->q_linstep * ptr->q[index_q-1] / (ptr->q[index_q-1] + ppr->q_linstep/q_logstep_spline);

        if (index_q-last_index < (int)ppr->q_numstep_transition)
          q = ptr->q[index_q-1] + (1-(double)(index_q-last_index)/ppr->q_numstep_transition) * last_step + (double)(index_q-last_index)/ppr->q_numstep_transition * q_step;
        else
          q = ptr->q[index_q-1] + q_step;
      }
    }
```

(Initialization at transfer.c:1166-1169: `ptr->q[0] = q_min; nu = 3;`. In the snapped regime every q is an
exact integer multiple of sqrt(K), with Delta-nu >= 1 — i.e. **no mode skipped near nu=3, every integer hit
until the sampling step exceeds sqrt(K)**.)

Flat-approximation switch index (transfer.c:1258-1278): first index with
`q > ppr->hyper_flat_approximation_nu * sqrt(sgnK*K)` is stored as `ptr->index_q_flat_approximation`.

### 3.5 Closed-case Cl sum correction (harmonic.c:920-921, 1328-1330)

The Cl "integral" over k is really a discrete sum in the closed case. CLASS evaluates it with
trapezoid/spline anyway (good approximation of the discrete sum) plus an exact end-point correction:

```c
  if (pba->sgnK == 1) {
    index_q_spline = ptr->index_q_flat_approximation;   /* trapezoid below this index */
  }
  ...
      if (pba->sgnK == 1) {
        clvalue += integrand[1+index_ct] * q_min/k_min*sqrt(pba->K)/2.;
      }
```

The primordial spectrum is always evaluated at `k = ptr->k[index_md][index_q] = sqrt(q^2-(1+m)K)`
(harmonic.c:926-930: `primordial_spectrum_at_k(ppm,index_md,linear,k,...)`), and the integration variable
is k with a dk/k-type measure — no extra curvature measure factor for scalars (the choice of writing
P(k) vs P(nu) absorbs it; see the long comment for tensors at perturbations.c:6226-6268).

---

## 4. SYNCHRONOUS-GAUGE EVOLUTION WITH CURVATURE

### 4.0 The s_l coefficients and s2_squared

Allocation + flat init, `perturbations_workspace_init` (perturbations.c:2784-2789):

```c
  /** - Allocate s_l[ ] array for freestreaming of multipoles (see arXiv:1305.3261) and initialize
      to 1.0, which is the K=0 value. */
  class_alloc(ppw->s_l, sizeof(double)*(ppw->max_l_max+1),ppt->error_message);
  for (l=0; l<=ppw->max_l_max; l++){
    ppw->s_l[l] = 1.0;
  }
```

Per-k update, `perturbations_solve` (perturbations.c:3061-3066):

```c
  /** - If non-zero curvature, update array of free-streaming coefficients ppw->s_l */
  if (pba->has_curvature == _TRUE_){
    for (l = 0; l<=ppw->max_l_max; l++){
      ppw->s_l[l] = sqrt(MAX(1.0-pba->K*(l*l-1.0)/k/k,0.));
    }
  }
```

**Exact formula:**

- `s_l = sqrt( 1 - K (l^2 - 1) / k^2 )`, clipped at 0 (the clip matters only in the closed case where
  `1 - K(l^2-1)/k^2` can go negative for l approaching nu; for such l the mode has no support and the
  coefficient is exactly zero in the exact theory: the hierarchy naturally truncates at l = nu-1).
- `s_0 = sqrt(1+K/k^2)`, `s_1 = 1` always, `s_2 = sqrt(1-3K/k^2)`.
- `s2_squared := 1 - 3K/k^2 = (s_l[2])^2`. Defined independently three times with the identical expression:
  perturbations_einstein (6870), perturbations_initial_conditions (5583), perturbations_derivs (9776), plus
  in the TCA function (11187) and PPF block (`s2sq = ppw->s_l[2]*ppw->s_l[2]`, 7683).
- **One single s_l array is shared by ALL sectors** — photon temperature, photon polarization, ur, ncdm, dr,
  idr all index the same `ppw->s_l`. There are no per-species curvature coefficients. (In 1305.3261 notation
  these are the scalar, m=0, coefficients `s_l = sqrt(1 - (l^2-1) K/k^2)`; the `(1+m)K` generalization for
  vectors/tensors enters via q, not via s_l.)

### 4.1 (a) Einstein constraints, perturbations_einstein (perturbations.c:6859-6990)

Preamble:

```c
  k2 = k*k;
  a = ppw->pvecback[pba->index_bg_a];
  a2 = a * a;
  a_prime_over_a = ppw->pvecback[pba->index_bg_H]*a;
  s2_squared = 1.-3.*pba->K/k2;
```

(`ppw->delta_rho`, `ppw->rho_plus_p_theta`, `ppw->rho_plus_p_shear`, `ppw->delta_p` are the totals
`sum_i rho_i delta_i`, `sum_i (rho_i+p_i) theta_i`, `sum_i (rho_i+p_i) sigma_i`, `sum_i delta p_i`, assembled
in `perturbations_total_stress_energy` with **no s_l factors** — e.g. perturbations.c:7337
`ppw->rho_plus_p_shear += 4./3.*rho_ur*shear_ur;`.)

Synchronous-gauge constraint equations (perturbations.c:6922-6959):

```c
    /* synchronous gauge */
    if (ppt->gauge == synchronous) {

      /* first equation involving total density fluctuation */
      ppw->pvecmetric[ppw->index_mt_h_prime] =
        ( k2 * s2_squared * y[ppw->pv->index_pt_eta] + 1.5 * a2 * ppw->delta_rho)/(0.5*a_prime_over_a);  /* h' */

      ...  /* (RSA re-evaluation hooks, see sec. 7) */

      /* second equation involving total velocity */
      ppw->pvecmetric[ppw->index_mt_eta_prime] = (1.5 * a2 * ppw->rho_plus_p_theta + 0.5 * pba->K * ppw->pvecmetric[ppw->index_mt_h_prime])/k2/s2_squared;  /* eta' */

      /* third equation involving total pressure */
      ppw->pvecmetric[ppw->index_mt_h_prime_prime] =
        - 2. * a_prime_over_a * ppw->pvecmetric[ppw->index_mt_h_prime]
        + 2. * k2 * s2_squared * y[ppw->pv->index_pt_eta]
        - 9. * a2 * ppw->delta_p;

      /* alpha = (h'+6eta')/2k^2 */
      ppw->pvecmetric[ppw->index_mt_alpha] = (ppw->pvecmetric[ppw->index_mt_h_prime] + 6.*ppw->pvecmetric[ppw->index_mt_eta_prime])/2./k2;
```

In equation form (Ma-Bertschinger generalized; `H_c = a'/a` conformal Hubble):

- **h' constraint:**  `h' = [ 2 k^2 (1-3K/k^2) eta + 3 a^2 delta_rho ] / H_c`
  — note `k^2 eta` -> `(k^2 - 3K) eta`.
- **eta' (momentum constraint):**  `eta' = [ (3/2) a^2 (rho+p)theta + (K/2) h' ] / [ k^2 (1-3K/k^2) ]`
  — i.e. `(k^2-3K) eta' = (3/2) a^2 sum(rho+p)theta + (K/2) h'`. **The +K h'/2 term is new in curved space.**
- **h'' (pressure):**  `h'' = -2 H_c h' + 2 (k^2-3K) eta - 9 a^2 delta_p`.
- **alpha definition unchanged:**  `alpha = (h' + 6 eta')/(2k^2)` (bare k^2, NOT k^2-3K).

Shear constraint / alpha' (perturbations.c:6985-6990) — **no explicit K** (and CLASS flags it `//TBC`, but it
is the standard form: curvature cancels in this combination):

```c
      /* fourth equation involving total shear */
      ppw->pvecmetric[ppw->index_mt_alpha_prime] =  //TBC
        - 2. * a_prime_over_a * ppw->pvecmetric[ppw->index_mt_alpha]
        + y[ppw->pv->index_pt_eta]
        - 4.5 * (a2/k2) * ppw->rho_plus_p_shear;
```

`alpha' = -2 H_c alpha + eta - (9/2)(a^2/k^2) sum (rho+p) sigma`.

TCA shear feedback into the totals (perturbations.c:6963-6974), needed because under TCA shear_g isn't
evolved:

```c
      if (ppw->approx[ppw->index_ap_tca] == (int)tca_on) {
        ...
          shear_g = 16./45./ppw->pvecthermo[pth->index_th_dkappa]*(y[ppw->pv->index_pt_theta_g]+k2*ppw->pvecmetric[ppw->index_mt_alpha]);
        ...
        ppw->rho_plus_p_shear += 4./3.*ppw->pvecback[pba->index_bg_rho_g]*shear_g;
      }
```

Gauge-invariant matter sources (perturbations.c:6997-7016) — no K factors:

```c
      if (ppt->has_source_delta_m == _TRUE_) {
        ppw->delta_m += 3. *ppw->pvecback[pba->index_bg_a]*ppw->pvecback[pba->index_bg_H] * ppw->theta_m/k2;
      }
      ...
      if (ppt->has_source_theta_m == _TRUE_) {
        if  (ppt->gauge == synchronous) {
          ppw->theta_m += ppw->pvecmetric[ppw->index_mt_alpha]*k2;
        }
      }
```

(For reference, the Newtonian-gauge comment at perturbations.c:6886-6895 records the curved Poisson constraint
`phi = -1.5 (a2/k2/k2/s2/s2) (k2 delta_rho + 3 H_c rho_plus_p_theta)`, "with s2_squared = sqrt(1-3K/k2) =
ppw->s_l[2]*ppw->s_l[2]" — note that comment's "sqrt" is sloppy: `s2_squared` IS `1-3K/k^2`. The evolved
Newtonian equations 6898-6901, `psi = phi - 4.5 (a2/k2) rho_plus_p_shear` and
`phi' = -H_c psi + 1.5 (a2/k2) rho_plus_p_theta`, carry no explicit K.)

### 4.2 (c-prelim) metric source shorthands in derivs (perturbations.c:9861-9868)

```c
    if (ppt->gauge == synchronous) {
      metric_continuity = pvecmetric[ppw->index_mt_h_prime]/2.;
      metric_euler = 0.;
      metric_shear = k2 * pvecmetric[ppw->index_mt_alpha];
      //metric_shear_prime = k2 * pvecmetric[ppw->index_mt_alpha_prime];
      metric_ufa_class = pvecmetric[ppw->index_mt_h_prime]/2.;
    }
```

So in synchronous gauge: continuity source = h'/2, Euler source = 0, shear source = k^2 alpha = (h'+6eta')/2.
All curvature dependence of the metric sources comes through h', eta', alpha computed above.

### 4.3 (d) Hierarchy truncation — cotKgen (perturbations.c:9763-9776)

```c
  /** - Compute 'generalised cotK function of argument sqrt(|K|)*tau, for closing hierarchy.
      (see equation 2.34 in arXiv:1305.3261): */
  if (pba->has_curvature == _FALSE_){
    cotKgen = 1.0/(k*tau);
  }
  else{
    sqrt_absK = sqrt(fabs(pba->K));
    if (pba->K < 0)
      cotKgen = sqrt_absK/k/tanh(sqrt_absK*tau);
    else
      cotKgen = sqrt_absK/k/tan(sqrt_absK*tau);
  }

  s2_squared = 1.-3.*pba->K/k2;
```

**Exact formula:**

- flat: `cotKgen = 1/(k tau)`
- open (K<0): `cotKgen = sqrt(|K|) coth( sqrt(|K|) tau ) / k`
- closed (K>0): `cotKgen = sqrt(K) cot( sqrt(K) tau ) / k`

(This is `cot_K(chi)/k` with `cot_K` the generalized cotangent of 1305.3261 eq. 2.34, evaluated at chi=tau.)
It is used in every species' l = l_max line (transcribed below): the flat MB truncation
`y'_lmax = k y_(lmax-1) - (lmax+1)/tau y_lmax` becomes
`y'_lmax = k [ s_lmax y_(lmax-1) - (lmax+1) cotKgen y_lmax ]`.

### 4.4 (c) Photon temperature hierarchy (perturbations.c:9973-10095)

l=0 (density; `metric_continuity = h'/2`):

```c
    if (ppw->approx[ppw->index_ap_rsa] == (int)rsa_off) {
      dy[pv->index_pt_delta_g] = -4./3.*(theta_g+metric_continuity);
    }
```

No-TCA branch (perturbations.c:10054-10095). `photon_scattering_rate = pvecthermo[pth->index_th_dkappa]`
(= kappa' = a n_e sigma_T; idm_g adds dmu_idm_g, line 9721):

```c
        /** - -----> define Pi = G_gamma0 + G_gamma2 + F_gamma2 */
        P0 = (y[pv->index_pt_pol0_g] + y[pv->index_pt_pol2_g] + 2.*s_l[2]*y[pv->index_pt_shear_g])/8.;

        /** - -----> photon temperature velocity */
        dy[pv->index_pt_theta_g] =
          k2*(delta_g/4.-s2_squared*y[pv->index_pt_shear_g])
          + metric_euler
          + pvecthermo[pth->index_th_dkappa]*(theta_b-theta_g);

        /** - -----> photon temperature shear */
        dy[pv->index_pt_shear_g] =
          0.5*(8./15.*(theta_g+metric_shear)
               -3./5.*k*s_l[3]/s_l[2]*y[pv->index_pt_l3_g]
               -photon_scattering_rate*(2.*y[pv->index_pt_shear_g]-4./5./s_l[2]*P0));

        /** - -----> photon temperature l=3 */
        l = 3;
        dy[pv->index_pt_l3_g] = k/(2.0*l+1.0)*
          (l*s_l[l]*2.*s_l[2]*y[pv->index_pt_shear_g]-(l+1.)*s_l[l+1]*y[pv->index_pt_l3_g+1])
          - photon_scattering_rate*y[pv->index_pt_l3_g];

        /** - -----> photon temperature l>3 */
        for (l = 4; l < pv->l_max_g; l++) {
          dy[pv->index_pt_delta_g+l] = k/(2.0*l+1.0)*
            (l*s_l[l]*y[pv->index_pt_delta_g+l-1]-(l+1)*s_l[l+1]*y[pv->index_pt_delta_g+l+1])
            - photon_scattering_rate*y[pv->index_pt_delta_g+l];
        }

        /** - -----> photon temperature lmax */
        l = pv->l_max_g; /* l=lmax */
        dy[pv->index_pt_delta_g+l] =
          k*(s_l[l]*y[pv->index_pt_delta_g+l-1]-(1.+l)*cotKgen*y[pv->index_pt_delta_g+l])
          - photon_scattering_rate*y[pv->index_pt_delta_g+l];
```

Notes on the variable convention (Ma-Bertschinger): the state stores `delta_g, theta_g, shear_g, F_3, F_4...`
where `shear_g = F_2/2`, so the l=2 and l=3 lines carry the conversion factors (`2 s_2 shear_g = s_2 F_2`).
In pure-F_l form the curved hierarchy is
`F_l' = k/(2l+1) [ l s_l F_(l-1) - (l+1) s_(l+1) F_(l+1) ] - kappa' F_l (l>=3)`; **the curvature change is
exactly the insertion of `s_l` on the lower neighbor and `s_(l+1)` on the upper neighbor**, plus
`k^2 shear_g -> (k^2-3K) shear_g = k^2 s2_squared shear_g` in theta_g', plus the `1/s_2` factors multiplying
the scattering Pi-source in shear' (and the s_2 in P0), per 1305.3261 eqs. 2.30-2.33.

If TCA on (perturbations.c:10138-10140), only theta_g evolves:

```c
        dy[pv->index_pt_theta_g] =
          -(dy[pv->index_pt_theta_b]+a_prime_over_a*theta_b-k2*delta_p_b_over_rho_b)/R
          +k2*(0.25*delta_g-s2_squared*ppw->tca_shear_g)+(1.+R)/R*metric_euler;
```

### 4.5 (c) Photon polarization hierarchy (perturbations.c:10097-10127)

```c
        /** - -----> photon polarization l=0 */
        dy[pv->index_pt_pol0_g] =
          -k*y[pv->index_pt_pol0_g+1]
          -photon_scattering_rate*(y[pv->index_pt_pol0_g]-4.*P0);

        /** - -----> photon polarization l=1 */
        dy[pv->index_pt_pol1_g] =
          k/3.*(y[pv->index_pt_pol1_g-1]-2.*s_l[2]*y[pv->index_pt_pol1_g+1])
          -photon_scattering_rate*y[pv->index_pt_pol1_g];

        /** - -----> photon polarization l=2 */
        dy[pv->index_pt_pol2_g] =
          k/5.*(2.*s_l[2]*y[pv->index_pt_pol2_g-1]-3.*s_l[3]*y[pv->index_pt_pol2_g+1])
          -photon_scattering_rate*(y[pv->index_pt_pol2_g]-4./5.*P0);

        /** - -----> photon polarization l>2 */
        for (l=3; l < pv->l_max_pol_g; l++)
          dy[pv->index_pt_pol0_g+l] = k/(2.*l+1)*
            (l*s_l[l]*y[pv->index_pt_pol0_g+l-1]-(l+1.)*s_l[l+1]*y[pv->index_pt_pol0_g+l+1])
            -photon_scattering_rate*y[pv->index_pt_pol0_g+l];

        /** - -----> photon polarization lmax_pol */
        l = pv->l_max_pol_g;
        dy[pv->index_pt_pol0_g+l] =
          k*(s_l[l]*y[pv->index_pt_pol0_g+l-1]-(l+1)*cotKgen*y[pv->index_pt_pol0_g+l])
          -photon_scattering_rate*y[pv->index_pt_pol0_g+l];
```

(So G_l: l=0 has no s factor on the upstream coupling; l=1 couples up through `2 s_2 G_2`... wait — careful:
CLASS's polarization vector stores `G_0, G_1, G_2, G_3, ...` directly; the `2 s_2` in the l=1 and l=2 lines
appear because the generic rule `l s_l G_(l-1) - (l+1) s_(l+1) G_(l+1)` at l=1 gives `(1*s_1*G_0 -
2*s_2*G_2)/3` with s_1=1, and at l=2 gives `(2 s_2 G_1 - 3 s_3 G_3)/5` — i.e. **the generic curved recursion
holds for ALL polarization l>=1**, and the special-casing is only about the scattering source 4/5 P0 at l=2
and (G_0 - 4 P0) at l=0. The same generic rule s_l/s_(l+1) used in the temperature hierarchy applies.)

P0 ("Pi/8"): `P0 = (G_0 + G_2 + 2 s_2 sigma_g)/8 = (G_0 + G_2 + s_2 F_2)/8` — curvature s_2 on the F_2 term
(perturbations.c:10057). The TCA fallback for sources uses `P = 5 s_2 tca_shear_g / 8` (perturbations.c:8100).

### 4.6 (c) Ultra-relativistic (massless neutrino) hierarchy (perturbations.c:10558-10643)

(`three_ceff2_ur = three_cvis2_ur = 1` in standard LCDM; the non-standard terms vanish then.)

```c
        /** - -----> ur density */
        dy[pv->index_pt_delta_ur] =
          -4./3.*(y[pv->index_pt_theta_ur] + metric_continuity)
          +(1.-ppt->three_ceff2_ur)*a_prime_over_a*(y[pv->index_pt_delta_ur] + 4.*a_prime_over_a*y[pv->index_pt_theta_ur]/k/k);

        /** - -----> ur velocity */
        dy[pv->index_pt_theta_ur] =
          k2*(ppt->three_ceff2_ur*y[pv->index_pt_delta_ur]/4.-s2_squared*y[pv->index_pt_shear_ur]) + metric_euler
          -(1.-ppt->three_ceff2_ur)*a_prime_over_a*y[pv->index_pt_theta_ur];

        if (ppw->approx[ppw->index_ap_ufa] == (int)ufa_off) {

          /** - -----> exact ur shear */
          dy[pv->index_pt_shear_ur] =
            0.5*(
                 8./15.*(y[pv->index_pt_theta_ur]+metric_shear)-3./5.*k*s_l[3]/s_l[2]*y[pv->index_pt_shear_ur+1]
                 -(1.-ppt->three_cvis2_ur)*(8./15.*(y[pv->index_pt_theta_ur]+metric_shear)));

          /** - -----> exact ur l=3 */
          l = 3;
          dy[pv->index_pt_l3_ur] = k/(2.*l+1.)*
            (l*2.*s_l[l]*s_l[2]*y[pv->index_pt_shear_ur]-(l+1.)*s_l[l+1]*y[pv->index_pt_l3_ur+1]);

          /** - -----> exact ur l>3 */
          for (l = 4; l < pv->l_max_ur; l++) {
            dy[pv->index_pt_delta_ur+l] = k/(2.*l+1)*
              (l*s_l[l]*y[pv->index_pt_delta_ur+l-1]-(l+1.)*s_l[l+1]*y[pv->index_pt_delta_ur+l+1]);
          }

          /** - -----> exact ur lmax_ur */
          l = pv->l_max_ur;
          dy[pv->index_pt_delta_ur+l] =
            k*(s_l[l]*y[pv->index_pt_delta_ur+l-1]-(1.+l)*cotKgen*y[pv->index_pt_delta_ur+l]);
        }
```

Identical structure to the photon temperature hierarchy with kappa' = 0. Note the l=3 line:
`l * 2*s_l[3]*s_l[2] * shear_ur` = `3 s_3 (s_2 F_2)`... i.e. again the generic `l s_l F_(l-1)` with
F_2 = 2 shear and an extra s_2 from the 1305.3261 form (the CLASS IV convention attaches an s_2 to the
F_2 <-> F_3 coupling in the shear'/l3' pair: shear' has `-3/5 k (s_3/s_2) F_3/2` and l3' has
`+3/7 k s_3 (2 s_2 shear)`; the s_2's cancel against a redefinition `sigma = F_2/(2 s_2)`-style bookkeeping —
**port these lines verbatim rather than re-deriving**).

### 4.7 (c) ncdm (massive neutrino) hierarchy (perturbations.c:10728-10777)

Exact Boltzmann hierarchy on the momentum grid (Psi_l(q) per momentum bin; `idx` walks the state vector):

```c
        for (n_ncdm=0; n_ncdm<pv->N_ncdm; n_ncdm++) {
          for (index_q=0; index_q < pv->q_size_ncdm[n_ncdm]; index_q++) {

            /** - -----> define intermediate quantities */
            dlnf0_dlnq = pba->dlnf0_dlnq_ncdm[n_ncdm][index_q];
            q = pba->q_ncdm[n_ncdm][index_q];
            epsilon = sqrt(q*q+a2*pba->M_ncdm[n_ncdm]*pba->M_ncdm[n_ncdm]);
            qk_div_epsilon = k*q/epsilon;

            /** - -----> ncdm density for given momentum bin */
            dy[idx] = -qk_div_epsilon*y[idx+1]+metric_continuity*dlnf0_dlnq/3.;

            /** - -----> ncdm velocity for given momentum bin */
            dy[idx+1] = qk_div_epsilon/3.0*(y[idx] - 2*s_l[2]*y[idx+2])
              -epsilon*metric_euler/(3*q*k)*dlnf0_dlnq;

            /** - -----> ncdm shear for given momentum bin */
            dy[idx+2] = qk_div_epsilon/5.0*(2*s_l[2]*y[idx+1]-3.*s_l[3]*y[idx+3])
              -s_l[2]*metric_shear*2./15.*dlnf0_dlnq;

            /** - -----> ncdm l>3 for given momentum bin */
            for (l=3; l<pv->l_max_ncdm[n_ncdm]; l++){
              dy[idx+l] = qk_div_epsilon/(2.*l+1.0)*(l*s_l[l]*y[idx+(l-1)]-(l+1.)*s_l[l+1]*y[idx+(l+1)]);
            }

            /** - -----> ncdm lmax for given momentum bin (truncation as in Ma and Bertschinger)
                but with curvature taken into account a la arXiv:1305.3261 */
            dy[idx+l] = qk_div_epsilon*y[idx+l-1]-(1.+l)*k*cotKgen*y[idx+l];

            /** - -----> jump to next momentum bin or species */
            idx += (pv->l_max_ncdm[n_ncdm]+1);
          }
        }
```

Key curvature points for ncdm:

- Free-streaming rate is `qk/epsilon` (unchanged), but the **inter-multipole couplings get the SAME s_l
  factors as massless species**: `Psi_1' = (qk/eps)/3 (Psi_0 - 2 s_2 Psi_2) - ...`,
  `Psi_2' = (qk/eps)/5 (2 s_2 Psi_1 - 3 s_3 Psi_3) - (2/15) s_2 (h'+6eta')/2 * dlnf0/dlnq / k^2 * k^2`
  (note the **extra s_2 on the metric shear source** in Psi_2', absent in the photon/ur shear equation
  written in sigma variables), and generic `l s_l / (l+1) s_(l+1)` for l>=3.
- l_max truncation: `Psi'_lmax = (qk/eps) Psi_(lmax-1) - (l+1) k cotKgen Psi_lmax`. **Note: NO s_lmax factor
  on the upstream term here** (unlike photon/ur lmax lines which have `k*s_l[l]*y[l-1]`), and the cotKgen
  term uses `k*cotKgen` with the free-streaming rate on the first term being qk/eps. This is the MB
  truncation a la 1305.3261 as CLASS chose to implement it for massive species — transcribe as-is.
- The ncdm fluid approximation (ncdmfa, perturbations.c:10654-10725) keeps an `s_l[2]` on its shear source:
  `dy[idx+2] = ... + 8/3 cvis2/(1+w) s_l[2] (y[idx+1]+metric_shear)`; CLASS marks the whole ncdmfa block
  `//TBC: curvature`.

### 4.8 (c) Decaying-radiation (dr) hierarchy — same pattern (perturbations.c:10264-10289)

Included for completeness since ABCMB may host similar species; F_l-form with f_dr weighting:

```c
      dy[pv->index_pt_F0_dr+1] = k/3.*y[pv->index_pt_F0_dr]-2./3.*k*y[pv->index_pt_F0_dr+2]*s2_squared +
        4*metric_euler/(3.*k)*f_dr + fprime_dr/k*y[pv->index_pt_theta_dcdm];

      dy[pv->index_pt_F0_dr+2] = 8./15.*(3./4.*k*y[pv->index_pt_F0_dr+1]+metric_shear*f_dr) -3./5.*k*s_l[3]/s_l[2]*y[pv->index_pt_F0_dr+3];

      l = 3;
      dy[pv->index_pt_F0_dr+3] = k/(2.*l+1.)*
        (l*s_l[l]*s_l[2]*y[pv->index_pt_F0_dr+2]-(l+1.)*s_l[l+1]*y[pv->index_pt_F0_dr+4]);

      for (l = 4; l < pv->l_max_dr; l++) {
        dy[pv->index_pt_F0_dr+l] = k/(2.*l+1)*
          (l*s_l[l]*y[pv->index_pt_F0_dr+l-1]-(l+1.)*s_l[l+1]*y[pv->index_pt_F0_dr+l+1]);
      }

      l = pv->l_max_dr;
      dy[pv->index_pt_F0_dr+l] =
        k*(s_l[l]*y[pv->index_pt_F0_dr+l-1]-(1.+l)*cotKgen*y[pv->index_pt_F0_dr+l]);
```

Note the F_1' line shows the **pure F-form of the curved dipole equation**:
`F_1' = k/3 F_0 - (2/3) k s2_squared F_2 + ...` — i.e. in F_l variables the only l=1 curvature factor is
`s_2^2 = 1-3K/k^2` multiplying F_2. (idr free-streaming hierarchy at 10184-10216 is structurally identical.)

### 4.9 (e) Baryons and CDM — no bare K

Baryon continuity & Euler, TCA off (perturbations.c:10006-10019):

```c
    dy[pv->index_pt_delta_b] = -(theta_b+metric_continuity);

    if (ppw->approx[ppw->index_ap_tca] == (int)tca_off) {
      dy[pv->index_pt_theta_b] =
        - a_prime_over_a*theta_b
        + metric_euler
        + k2*delta_p_b_over_rho_b
        + R*pvecthermo[pth->index_th_dkappa]*(theta_g-theta_b);
    }
```

Baryon Euler with TCA on (perturbations.c:10034-10038) — curvature enters only via `s2_squared` on the photon
TCA shear:

```c
      dy[pv->index_pt_theta_b] =
        (-a_prime_over_a*theta_b
         +k2*(delta_p_b_over_rho_b+R*(delta_g/4.-s2_squared*ppw->tca_shear_g))
         +R*ppw->tca_slip)/(1.+R)
        +metric_euler;
```

CDM (perturbations.c:10168-10170), synchronous gauge:

```c
      if (ppt->gauge == synchronous) {
        dy[pv->index_pt_delta_cdm] = -metric_continuity; /* cdm density */
      }
```

**No explicit K anywhere in baryon/CDM equations** — curvature reaches them only through `metric_continuity
= h'/2` (which contains K via the Einstein constraints), through `s2_squared * shear_g` in the photon-drag
coupling, and (TCA slip, see sec. 7) `s2_squared` factors inside the slip formula.

### 4.10 (f) Metric variable evolution (perturbations.c:10784-10788)

```c
    if (ppt->gauge == synchronous) {
      dy[pv->index_pt_eta] = pvecmetric[ppw->index_mt_eta_prime];
    }
```

`eta` is the only evolved metric DOF in synchronous gauge; `h` itself is never integrated (only h', h'' as
constraints; when h is needed as output CLASS uses `_set_source_(h) = -2 delta_cdm`, perturbations.c:8314).

### 4.11 Vectors/tensors (for the record, not for porting)

- Tensor GW equation (perturbations.c:7059): `gw'' = -2 H_c gw' - (k^2 + 2K) gw + gw_source`
  — the `(1+m)K`-type shift appears as `k^2+2K` here.
- Vector source factor `sqrt(1 - 2K/k^2)` (perturbations.c:7799, 10803 `ssqrt3 = sqrt(1.-2.*pba->K/k2)`).
- Tensor primordial normalization in curved space (perturbations.c:6277-6288):
  `gw *= sqrt(k2*(k2-K)/(k2+3K)/(k2+2K))`, and in the open case an extra `sqrt(tanh(pi/2 * sqrt(k2+3K)/sqrt(-K)))`,
  with the long derivation comment at 6226-6268 (q^2 = k^2+3K measure manipulation).

---

## 5. INITIAL CONDITIONS (perturbations_initial_conditions, perturbations.c:5414-6132)

### 5.1 Setup / expansion variables (perturbations.c:5543-5583)

```c
    /* f_nu = Omega_nu(t_i) / Omega_r(t_i) */
    fracnu = rho_nu/rho_r;
    /* f_g = Omega_g(t_i) / Omega_r(t_i) */
    fracg = ppw->pvecback[pba->index_bg_rho_g]/rho_r;
    /* f_b = Omega_b(t_i) / Omega_m(t_i) */
    fracb = ppw->pvecback[pba->index_bg_rho_b]/rho_m;
    ...
    /* Omega_m(t_i) / Omega_r(t_i) */
    rho_m_over_rho_r = rho_m/rho_r;

    /* omega = Omega_m(t_i) a(t_i) H(t_i) / sqrt(Omega_r(t_i)) ...
       a = [H(t_0)^2 Omega_m(t_0) a(t_0)^3 / 4] x [tau^2 + 4 tau / omega]  */
    om = a*rho_m/sqrt(rho_r);

    /* (k tau)^2, (k tau)^3 */
    ktau_two=k*k*tau*tau;
    ktau_three=k*tau*ktau_two;

    /* curvature-dependent factors */
    s2_squared = 1.-3.*pba->K/k/k;
```

So the expansion is a double series in `(k tau)` and `(om tau)`; **the only curvature-dependent factor is
`s2_squared = 1-3K/k^2`** (no cotKgen, no s_l beyond s_2 — ICs are set at ktau<<1 where higher corrections
are negligible).

### 5.2 Adiabatic ICs (perturbations.c:5591-5786) — every K-dependent expression

Rationale comment (perturbations.c:5593-5605):

```c
      /* The following formulas are valid at leading order in
         (k*tau) and (om*tau), and order zero in tight-coupling. ...

         In the non-flat case the relation R=eta is still valid
         outside the horizon for adiabatic IC. Hence eta is still
         set to ppr->curvature_ini at leading order.  Factors s2
         appear through the solution of Einstein equations and
         equations of motion. */
```

(`ppr->curvature_ini = 1.0` by default, precisions.h:316.)

Photons & baryons & cdm:

```c
      /* photon density */
      ppw->pv->y[ppw->pv->index_pt_delta_g] = - ktau_two/3. * (1.-om*tau/5.)
        * ppr->curvature_ini * s2_squared;

      /* photon velocity */
      ppw->pv->y[ppw->pv->index_pt_theta_g] = - k*ktau_three/36. * (1.-3.*(1.+5.*fracb-fracnu)/20./(1.-fracnu)*om*tau)
        * ppr->curvature_ini * s2_squared;

      /* tighly-coupled baryons */
      ppw->pv->y[ppw->pv->index_pt_delta_b] = 3./4.*ppw->pv->y[ppw->pv->index_pt_delta_g]; /* baryon density */
      ppw->pv->y[ppw->pv->index_pt_theta_b] = ppw->pv->y[ppw->pv->index_pt_theta_g]; /* baryon velocity */

      if (pba->has_cdm == _TRUE_) {
        ppw->pv->y[ppw->pv->index_pt_delta_cdm] = 3./4.*ppw->pv->y[ppw->pv->index_pt_delta_g]; /* cdm density */
        /* cdm velocity vanishes in the synchronous gauge */
      }
```

In equations (eta_ini = curvature_ini = 1 normalization):

- `delta_g = -(1/3)(k tau)^2 (1 - om tau/5) * s_2^2`
- `theta_g = -(1/36) k (k tau)^3 [1 - 3(1+5 f_b - f_nu)/(20(1-f_nu)) om tau] * s_2^2`
- `delta_b = delta_cdm = (3/4) delta_g`, `theta_b = theta_g`, `theta_cdm = 0` (gauge).

Fluid adiabatic IC (vanilla CLASS form; class_StepDR:5643-5645; AxiCLASS carries the same line at 5664 plus
axion-specific variants around it):

```c
          ppw->pv->y[ppw->pv->index_pt_delta_fld] = - ktau_two/4.*(1.+w_fld)*(4.-3.*pba->cs2_fld)/(4.-6.*w_fld+3.*pba->cs2_fld) * ppr->curvature_ini * s2_squared; /* from 1004.5509 */ //TBC: curvature

          ppw->pv->y[ppw->pv->index_pt_theta_fld] = - k*ktau_three/4.*pba->cs2_fld/(4.-6.*w_fld+3.*pba->cs2_fld) * ppr->curvature_ini * s2_squared; /* from 1004.5509 */ //TBC:curvature
```

Relativistic relics (ur / early ncdm / dr) — perturbations.c:5764-5780:

```c
      if ((pba->has_ur == _TRUE_) || (pba->has_ncdm == _TRUE_) || (pba->has_dr == _TRUE_) || (pba->has_idr == _TRUE_)) {

        delta_ur = ppw->pv->y[ppw->pv->index_pt_delta_g]; /* density of ultra-relativistic neutrinos/relics */

        /* velocity of ultra-relativistic neutrinos/relics */ //TBC
        theta_ur = - k*ktau_three/36./(4.*fracnu+15.) * (4.*fracnu+11.+12.*s2_squared-3.*(8.*fracnu*fracnu+50.*fracnu+275.)/20./(2.*fracnu+15.)*tau*om) * ppr->curvature_ini * s2_squared;

        shear_ur = ktau_two/(45.+12.*fracnu) * (3.*s2_squared-1.) * (1.+(4.*fracnu-5.)/4./(2.*fracnu+15.)*tau*om) * ppr->curvature_ini;//TBC /s2_squared; /* shear of ultra-relativistic neutrinos/relics */  //TBC:0

        l3_ur = ktau_three*2./7./(12.*fracnu+45.)* ppr->curvature_ini;//TBC

        if (pba->has_dr == _TRUE_) delta_dr = delta_ur;
      }
```

Equations:

- `delta_ur = delta_g` (adiabaticity)
- `theta_ur = -(k^4 tau^3 / 36) * [4 f_nu + 11 + 12 s_2^2 - 3(8 f_nu^2 + 50 f_nu + 275)/(20(2 f_nu+15)) tau om]
   / (4 f_nu + 15) * s_2^2`
  — **note the `12 s2_squared` replacing the flat-space "12"** (flat limit: 4f_nu+23) **and the overall
  s_2^2.**
- `shear_ur = (k tau)^2 / (45 + 12 f_nu) * (3 s_2^2 - 1)/2 * ... ` — wait, transcribed exactly:
  `shear_ur = ktau_two/(45+12 fracnu) * (3 s2_squared - 1) * (1 + (4 fracnu - 5)/(4(2 fracnu+15)) tau om)`.
  **The flat limit of `(3 s_2^2 - 1)` is 2**, recovering the familiar
  `shear_ur = 2 (k tau)^2 / (3(45+12 f_nu)) * 3/2...` i.e. `(2/3) ktau^2/(45/3...)` — bottom line: in flat
  space this reduces to `2 ktau^2/(45+12 f_nu) (1 + ...)`, the MB value `(4/15)(k^2 tau^2)/(15+4 f_nu)`.
  **No s_2^2 overall factor on shear_ur** (a commented-out `/s2_squared` shows the devs deliberated).
- `l3_ur = (2/7) (k tau)^3 / (12 f_nu + 45)` — **no curvature factor at this order**.

Synchronous eta IC (perturbations.c:5782-5785; two rejected variants left in comments — useful provenance):

```c
      /* synchronous metric perturbation eta */
      //eta = ppr->curvature_ini * (1.-ktau_two/12./(15.+4.*fracnu)*(5.+4.*fracnu - (16.*fracnu*fracnu+280.*fracnu+325)/10./(2.*fracnu+15.)*tau*om)) /  s2_squared;
      //eta = ppr->curvature_ini * s2_squared * (1.-ktau_two/12./(15.+4.*fracnu)*(15.*s2_squared-10.+4.*s2_squared*fracnu - (16.*fracnu*fracnu+280.*fracnu+325)/10./(2.*fracnu+15.)*tau*om));
      eta = ppr->curvature_ini * (1.-ktau_two/12./(15.+4.*fracnu)*(5.+4.*s2_squared*fracnu - (16.*fracnu*fracnu+280.*fracnu+325)/10./(2.*fracnu+15.)*tau*om));
```

`eta = 1 - (k tau)^2/(12(15+4 f_nu)) * [5 + 4 s_2^2 f_nu - (16 f_nu^2 + 280 f_nu + 325)/(10(2 f_nu+15)) tau om]`
— **leading order eta = curvature_ini exactly (R=eta superhorizon relation survives curvature); the only
K-dependence is `4 f_nu -> 4 s_2^2 f_nu` in the (k tau)^2 correction.**

### 5.3 Mapping onto ur / ncdm / dr state vectors (perturbations.c:6068-6131)

```c
    if (pba->has_ur == _TRUE_) {
      ppw->pv->y[ppw->pv->index_pt_delta_ur] = delta_ur;
      ppw->pv->y[ppw->pv->index_pt_theta_ur] = theta_ur;
      ppw->pv->y[ppw->pv->index_pt_shear_ur] = shear_ur;
      ppw->pv->y[ppw->pv->index_pt_l3_ur] = l3_ur;
    }
    ...
    if (pba->has_ncdm == _TRUE_) {
      idx = ppw->pv->index_pt_psi0_ncdm1;
      for (n_ncdm=0; n_ncdm < pba->N_ncdm; n_ncdm++){
        for (index_q=0; index_q < ppw->pv->q_size_ncdm[n_ncdm]; index_q++) {

          q = pba->q_ncdm[n_ncdm][index_q];
          epsilon = sqrt(q*q+a*a*pba->M_ncdm[n_ncdm]*pba->M_ncdm[n_ncdm]);

          ppw->pv->y[idx] = -0.25 * delta_ur * pba->dlnf0_dlnq_ncdm[n_ncdm][index_q];
          ppw->pv->y[idx+1] =  -epsilon/3./q/k*theta_ur* pba->dlnf0_dlnq_ncdm[n_ncdm][index_q];
          ppw->pv->y[idx+2] = -0.5 * shear_ur * pba->dlnf0_dlnq_ncdm[n_ncdm][index_q];
          ppw->pv->y[idx+3] = -0.25 * l3_ur * pba->dlnf0_dlnq_ncdm[n_ncdm][index_q];

          idx += (ppw->pv->l_max_ncdm[n_ncdm]+1);
        }
      }
    }
```

The ncdm Psi_l ICs inherit all curvature factors through `delta_ur/theta_ur/shear_ur/l3_ur` — no additional
K factors in the q-space mapping.

### 5.4 Gauge-transform alpha (only if target gauge is Newtonian; perturbations.c:5942-5998)

The curved expression for alpha at IC time, used in the synchronous->Newtonian transformation:

```c
      alpha = (eta + 3./2.*a_prime_over_a*a_prime_over_a/k/k/s2_squared*(delta_tot + 3.*a_prime_over_a/k/k*velocity_tot))/a_prime_over_a;
```

`alpha = [eta + (3/2) H_c^2/(k^2 s_2^2) (delta_tot + 3 H_c velocity_tot / k^2)] / H_c` — the `1/s_2^2` is the
curvature change (it comes from inverting the curved h'-constraint). Not needed for a synchronous-gauge port,
but it documents where 1/s2 factors appear when solving the constraints for metric potentials.

Isocurvature ICs (cdi/bi/nid/niv, perturbations.c:5800-5930) carry **no s2_squared factors at all** — CLASS
has only validated/curvature-corrected the adiabatic mode (isocurvature in curved space is not supported;
the formulas are flat-space BMT99).

---

## 6. SOURCE FUNCTIONS (perturbations_sources, perturbations.c:~7990-8330)

### 6.1 Polarization weight P (perturbations.c:8090-8104)

```c
    if (ppw->approx[ppw->index_ap_rsa] == (int)rsa_on) {
      delta_g = ppw->rsa_delta_g;
      P = 0.;
    }
    else {
      delta_g = y[ppw->pv->index_pt_delta_g];
      if (ppw->approx[ppw->index_ap_tca] == (int)tca_on)
        P = 5.* ppw->s_l[2] * ppw->tca_shear_g/8.; /* (2.5+0.5+2)shear_g/8 */
      else
        P = (y[ppw->pv->index_pt_pol0_g] + y[ppw->pv->index_pt_pol2_g] + 2.* ppw->s_l[2] *y[ppw->pv->index_pt_shear_g])/8.;
    }
```

`P = Pi/8 = (G_0 + G_2 + s_2 F_2)/8` — the **s_l[2] multiplies shear_g (=F_2/2)** both in the exact and TCA
forms. This is the same Pi/8 as `P0` in the derivs.

### 6.2 Temperature T0/T1/T2 — synchronous gauge

"Simplest form" reference (kept as a comment, perturbations.c:8156-8163) — this is the cleanest statement of
the curvature content:

```c
      /* synchronous gauge: simplest form, not efficient numerically */
      /*
        if (ppt->gauge == synchronous) {
        _set_source_(ppt->index_tp_t0) = - pvecthermo[pth->index_th_exp_m_kappa] * pvecmetric[ppw->index_mt_h_prime] / 6. + pvecthermo[pth->index_th_g] / 4. * delta_g;
        _set_source_(ppt->index_tp_t1) = pvecthermo[pth->index_th_g] * y[ppw->pv->index_pt_theta_b] / k;
        _set_source_(ppt->index_tp_t2) = pvecthermo[pth->index_th_exp_m_kappa] * k*k* 2./3. * ppw->s_l[2] * pvecmetric[ppw->index_mt_alpha] + pvecthermo[pth->index_th_g] * P;
        }
      */
```

**T2_simple = e^-kappa * (2/3) k^2 s_2 alpha + g P** — the ISW-quadrupole piece carries an explicit `s_l[2]`,
and P carries its internal s_2 (sec 6.1). T0_simple and T1_simple have no explicit curvature factor.

Production ("efficient", integrated-by-parts) form actually used (perturbations.c:8167-8215, non-idm branch;
`g` = visibility, `exp_m_kappa` = e^-kappa, `a_prime_over_a_prime` = d(aH_c... i.e. (a'/a)' precomputed):

```c
      if (ppt->gauge == synchronous) {

        theta_b += pvecmetric[ppw->index_mt_alpha] *k*k;            // absorb alpha in here to make the formulas readable
        theta_b_prime += pvecmetric[ppw->index_mt_alpha_prime] *k*k;
        ...
          _set_source_(ppt->index_tp_t0) =
            ppt->switch_sw * g * (delta_g/4. + pvecmetric[ppw->index_mt_alpha_prime])
            + switch_isw * ( g * (y[ppw->pv->index_pt_eta]
                                  - pvecmetric[ppw->index_mt_alpha_prime]
                                  - 2 * a_prime_over_a * pvecmetric[ppw->index_mt_alpha])
                             + exp_m_kappa * 2. * (pvecmetric[ppw->index_mt_eta_prime]
                                                   - a_prime_over_a_prime * pvecmetric[ppw->index_mt_alpha]
                                                   - a_prime_over_a * pvecmetric[ppw->index_mt_alpha_prime]))
            + ppt->switch_dop /k/k * ( g * theta_b_prime + g_prime * theta_b );

          _set_source_(ppt->index_tp_t1) =
            switch_isw * exp_m_kappa * k * (pvecmetric[ppw->index_mt_alpha_prime]
                                            + 2. * a_prime_over_a * pvecmetric[ppw->index_mt_alpha]
                                            - y[ppw->pv->index_pt_eta]);

        _set_source_(ppt->index_tp_t2) =
          ppt->switch_pol * g * P;
      }
```

**Curvature content of the efficient form:** no bare K appears; all curvature flows in through (i) `eta'`,
`alpha`, `alpha'`, `h'` (Einstein constraints, sec 4.1), and (ii) the `s_l[2]` inside `P`. The IBP
manipulation that converts T2_simple into `T2 = g P` + extra T0/T1 pieces is curvature-consistent because the
transfer module's curved radial functions are constructed to match this decomposition (1305.3261 sec. 3;
the radial functions for T0/T1/T2 in curved space are built from hyperspherical Bessel functions
`Phi_l^nu(chi)` and their derivatives — see 6.4).

### 6.3 Polarization source (perturbations.c:8219-8229)

```c
    /* scalar polarization */
    if (ppt->has_source_p == _TRUE_) {
      /* all gauges. Note that the correct formula for the E source
         should have a minus sign, as shown in Hu & White. We put a
         plus sign to comply with the 'historical convention'
         established in CMBFAST and CAMB. */
      _set_source_(ppt->index_tp_p) = sqrt(6.) * g * P;
    }
```

`S_P = sqrt(6) g P` — curvature only via the s_2 inside P. (The curvature-dependent
`sqrt((l+2)!/(l-2)!)`-analogue factors for E-modes in curved space live in the transfer module's radial
function, not here.)

### 6.4 What is handed to the transfer module, and its curved radial machinery (summary)

The perturbation module hands `S_T0(k,tau), S_T1(k,tau), S_T2(k,tau), S_P(k,tau)` on its smooth k grid; the
transfer module interpolates them at `k(q) = sqrt(q^2 - (1+m)K)` and convolves with **curved radial
functions**. Key transcriptions:

`transfer_radial_coordinates` (transfer.c:2169-2206) — argument and generalized trig factors:

```c
  switch (ptw->sgnK){
  case 1:  /* closed */
    sqrt_absK = sqrt(ptw->K);
    for (index_tau=0; index_tau < ptw->tau_size; index_tau++) {
      ptw->chi[index_tau] = sqrt_absK*ptw->tau0_minus_tau[index_tau];
      ptw->cscKgen[index_tau] = sqrt_absK/ptr->k[index_md][index_q]/sin(ptw->chi[index_tau]);
      ptw->cotKgen[index_tau] = ptw->cscKgen[index_tau]*cos(ptw->chi[index_tau]);
    }
    break;
  case 0:  /* flat */
    ...
      ptw->chi[index_tau] = ptr->k[index_md][index_q] * ptw->tau0_minus_tau[index_tau];
      ptw->cscKgen[index_tau] = 1.0/ptw->chi[index_tau];
      ptw->cotKgen[index_tau] = 1.0/ptw->chi[index_tau];
    ...
  case -1: /* open */
    sqrt_absK = sqrt(-ptw->K);
    ...
      ptw->chi[index_tau] = sqrt_absK*ptw->tau0_minus_tau[index_tau];
      ptw->cscKgen[index_tau] = sqrt_absK/ptr->k[index_md][index_q]/sinh(ptw->chi[index_tau]);
      ptw->cotKgen[index_tau] = ptw->cscKgen[index_tau]*cosh(ptw->chi[index_tau]);
    ...
  }
```

The hyperspherical Bessel functions `Phi_l^nu(chi)` themselves are computed in
`source/hyperspherical.c` + `include/hyperspherical.h` (HIS = hyperspherical interpolation structure,
initialized in transfer.c around lines 3460-3480, with turning point
`x_turning_point = asin/asinh(sqrt(l(l+1))/nu)`); above `nu > hyper_flat_approximation_nu` a rescaled flat
Bessel `j_l(...)` is used instead (transfer.c:2044-2074). Porting the transfer-side machinery is a separate
job from the perturbation-side equations above.

CMB lensing source curvature rescaling (transfer.c:2406-2438):

```c
            switch (pba->sgnK){
            case 1:
              rescaling = sqrt(pba->K)
                *sin((tau_rec-tau)*sqrt(pba->K))
                /sin((tau0-tau)*sqrt(pba->K))
                /sin((tau0-tau_rec)*sqrt(pba->K));
              break;
            case 0:
              rescaling = (tau_rec-tau)/(tau0-tau)/(tau0-tau_rec);
              break;
            case -1:
              rescaling = sqrt(-pba->K)
                *sinh((tau_rec-tau)*sqrt(-pba->K))
                /sinh((tau0-tau)*sqrt(-pba->K))
                /sinh((tau0-tau_rec)*sqrt(-pba->K));
              break;
            }
            // Note: until 2.4.3 there was a bug here: the curvature effects had been omitted.
```

I.e. the flat lensing kernel `(tau_rec-tau)/[(tau0-tau)(tau0-tau_rec)]` becomes
`sin_K(chi_rec - chi)/[sin_K(chi) sin_K(chi_rec)] * 1/sqrt? ` — exactly
`f_K'(...)`-style ratio with `sin_K(x) = sin/sinh(sqrt|K| x)/sqrt|K|`.

Other sources for completeness (no K factors, synchronous): `phi = eta - H_c alpha` (8268),
`phi' = eta' - (a'/a)' alpha - H_c alpha'` (8279-8281), `phi+psi = eta + alpha'` (8292-8293),
`psi = H_c alpha + alpha'` (8305-8306), `h_prime = pvecmetric[h']` (8317), N-body gauge `H_T_Nb_prime`
(8237-8239).

---

## 7. GAUGE AND APPROXIMATION SCHEMES (one-liners, not for porting)

- **Gauge:** all of the above is synchronous gauge (`ppt->gauge == synchronous`, the CLASS default).
  Newtonian-gauge curved equations also exist (psi/phi forms quoted in sec 4.1) but were not the focus here.
- **TCA (tight coupling):** curvature-aware. `perturbations_tca_slip_and_shear` defines
  `s2_squared = 1.-3.*pba->K/k2` (perturbations.c:11187) and the slip carries it, e.g.
  `slip = (1.-2*a_prime_over_a*F)*slip + F*k2*s2_squared*(2.*a_prime_over_a*shear_g+shear_g_prime) ...`
  (11442; second-order variant 11460); post-TCA reinitialization uses s_l:
  `l3_g = 6/7 k/kappa' s_l[3] shear_g`, `pol1_g = k/kappa' (5-2 s_l[2])/6 shear_g`,
  `pol3_g = k/kappa' 3 s_l[3]/14 shear_g` (perturbations.c:4614-4618).
- **RSA (radiation streaming):** the rsa delta/theta closed-form expressions
  (`perturbations_rsa_delta_and_theta`, perturbations.c:~11480-11570) contain **no explicit K factors** — the
  scheme is the flat-space asymptotics applied as-is (curvature enters only implicitly via the metric inputs).
- **UFA (ur fluid approx):** flat-space closure; the shear equations at perturbations.c:10617-10641 have no K
  and CLASS marks the block `//TBC: curvature?` (10615).
- **ncdm fluid approx (ncdmfa):** partially curvature-dressed (single `s_l[2]` on its shear source,
  perturbations.c:10704/10711/10718) and marked `//TBC: curvature` (10653).

---

## Porting checklist (condensed)

1. Background: add `-K/a^2` to H^2, `+K/a` to H' (conformal); K = -Omega_k H0^2 in (1/Mpc)^2; close budget
   with `Omega_L = 1 - Omega_k - sum Omega_i`; distances via `r = sin_K(chi)`; rs gains `sqrt(1-K rs^2)`.
2. Per-k constants: `s_l = sqrt(max(0, 1 - K(l^2-1)/k^2))` for l = 0..l_max+1; `s2sq = 1-3K/k^2`.
3. Per-(k,tau): `cotKgen = sqrt|K| cot_K(sqrt|K| tau)/k` (tan/tanh/1-over-x).
4. Einstein (sync): `h' = [2(k^2-3K)eta + 3a^2 drho]/H_c`; `eta' = [1.5 a^2 (r+p)th + K h'/2]/(k^2-3K)`;
   `h'' = -2H_c h' + 2(k^2-3K)eta - 9a^2 dp`; `alpha = (h'+6eta')/2k^2` (bare k^2);
   `alpha' = -2H_c alpha + eta - 4.5(a^2/k^2)(r+p)sigma` (unchanged).
5. Hierarchies: insert s_l per the transcriptions (sec 4.4-4.7); `k^2 sigma -> k^2 s2sq sigma` in every
   dipole equation; Pi = G0+G2+ s_2 F_2; truncation `... - (l+1) k cotKgen y_lmax`.
6. ICs: multiply delta_g/theta_g (and fld) by s2sq; theta_ur `(4f+11+12 s2sq)` and overall s2sq; shear_ur
   `(3 s2sq - 1)` no overall factor; eta `5+4 s2sq f_nu`; l3_ur unchanged.
7. Sources: T2/P pick up s_2 via Pi (and explicit `s_2` in the non-IBP T2); the efficient T0/T1/T2 forms are
   structurally unchanged. Lensing kernel and LOS radial functions need sin_K geometry + hyperspherical
   Bessels (transfer side; separate work).
8. Closed universe only: discrete `nu = sqrt(k^2+K)/sqrt(K)` integer >= 3; smooth PE k-grid +
   integer-nu resampling at the transfer step (CLASS pattern), with k_min = sqrt(8K)(1-eps).
