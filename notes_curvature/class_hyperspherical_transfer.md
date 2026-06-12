# CLASS hyperspherical Bessel functions and curved-space transfer machinery

**Source examined:** AxiCLASS checkout (`/pscratch/sd/c/carag/AxiCLASS`, CLASS v3.3.0 — the newest of the three local CLASS copies; the hyperspherical/transfer machinery is stock CLASS, untouched by the axion mods. class_StepDR is v3.2.0 with identical hyperspherical code; class_EDE is v2.6.3).

Key files:
- `tools/hyperspherical.c` (1738 lines, author Thomas Tram, 2013) + `include/hyperspherical.h`
- `source/transfer.c` (5358 lines) + `include/transfer.h`
- `source/harmonic.c`, `source/lensing.c`, `source/primordial.c`, `source/perturbations.c`
- `include/precisions.h` (precision defaults)

**Method papers:** The curved-universe method is Lesgourgues & Tram 2014, *"Fast and accurate CMB computations in non-flat FLRW universes"*, *arXiv:1305.3261* — cited repeatedly in `perturbations.c` (e.g. lines 2784, 6333, 9535, 9764, 10768, 10834: "s_l array for freestreaming of multipoles (see arXiv:1305.3261)", "Eq 2.35 of 1305.3261", "Eq. B.23 in 1305.3261"). The hyperspherical Bessel algorithm itself is Tram's companion paper (arXiv:1311.0839, "Computation of hyperspherical Bessel functions"); the module header (`hyperspherical.c:1-6`) just credits "Thomas Tram, 11.01.2013". The Limber-2 formula cites 0809.5112; lensing module cites Challinor & Lewis astro-ph/0502425.

**Conventions used below:** `K` = curvature sign (+1 closed, 0 flat, −1 open) inside hyperspherical.c, while `pba->K` (in Mpc⁻²) is the dimensionful curvature and `sgnK` its sign in transfer.c. `beta` ≡ `nu` = q/√|K| is the dimensionless wavenumber. `x` = √|K|·χ is the dimensionless radial coordinate. Definitions of the generalized trig functions:

| K | sinK(x) | cotK(x) | turning point x_tp(l,ν) |
|---|---------|---------|--------------------------|
| 0 | x | 1/x | √(l(l+1))/ν |
| +1 | sin x | cot x | asin(√(l(l+1))/ν) |
| −1 | sinh x | coth x | asinh(√(l(l+1))/ν) |

---

## 1. hyperspherical.c — algorithms

### 1.1 Definition and normalization

Φ_l^ν(x) solves the radial equation (implicit in the derivative formulas at `hyperspherical.c:357`):

```
Φ'' = −2 cotK(x) Φ' + [ l(l+1)/sinK²(x) − ν² + K ] Φ
```

with normalization fixed by the **l=0 solution** (the Miller-algorithm anchor, `hyperspherical.c:497`):

```c
phi0 = sin(beta*x)/(beta*sinK);      // Φ_0^ν(x) = sin(νx)/(ν·sinK(x))
```

**Flat limit:** for K=0, sinK=x so Φ_0 = sin(νx)/(νx) = j_0(νx), and the whole family reduces to Φ_l^ν(x) = j_l(νx). CLASS exploits this: the flat HIS is built once with `K=0, beta=1`, storing literally j_l(x).

The l-dependent recurrence coefficients ("sqrtK") are set in `hyperspherical_HIS_create` (`hyperspherical.c:88-130`):

```c
case 0:   sqrtK[l] = beta;                  // flat
case 1:   sqrtK[l] = sqrt(beta2 - l*l);     // closed: sqrt(nu^2 - l^2)
case -1:  sqrtK[l] = sqrt(beta2 + l*l);     // open:   sqrt(nu^2 + l^2)
```

i.e. sqrtK[l] = √(ν² − sgnK·l²). Note that for K=+1 this requires **l ≤ ν−1** (else sqrt of negative); closed-universe modes have integer ν ≥ 3 and only l < ν exist.

### 1.2 The HyperInterpStruct (HIS) — what is tabulated

`include/hyperspherical.h:18-32`:

```c
typedef struct HypersphericalInterpolationStructure{
  int K;                  // Sign of the curvature, (0,-1,1)
  double beta;            // nu
  double delta_x;         // x-spacing (uniform grid)
  int trig_order;         // order of interpolation formula for SinK/CosK
  int l_size;  int *l;    // vector of l values stored (the sparse CLASS l-list)
  double *chi_at_phimin;  // per-l x_min below which Phi is negligible (≈0)
  int x_size;  double *x; // uniform x grid
  double *sinK; double *cotK;   // tabulated sin_K(x), cot_K(x)
  double *phi;            // size nl*nx, layout phi[index_l*nx + index_x]
  double *dphi;           // same layout, dPhi/dx
} HyperInterpStruct;
```

So per (ν, l): **Φ and dΦ/dx on a uniform x grid**, plus a per-l cutoff `chi_at_phimin`. Higher derivatives needed by the Hermite interpolation are reconstructed *on the fly from the ODE* (no extra storage — see §1.7).

**x-sampling formula** (`hyperspherical_HIS_create`, `hyperspherical.c:34-39`):

```c
beta2 = beta*beta;
lambda = 2*_PI_/beta;                       // one oscillation wavelength in x
nx = (int) ((xmax-xmin)*sampling/lambda);   // 'sampling' points per wavelength
nx = MAX(nx,2);
deltax = (xmax-xmin)/(nx-1.0);
```

`sampling` = `hyper_sampling_flat` (8.0) for the flat table, `hyper_sampling_curved_low_nu` (7.0) or `hyper_sampling_curved_high_nu` (3.0) for curved tables (switch at ν = `hyper_nu_sampling_step` = 1000). The grid covers `[hyper_x_min, xmax]`; closed case caps `xmax ≤ π/2 − hyper_x_min` (symmetry, §1.8).

**Trig interpolation order** (`hyperspherical.c:58-64`, Taylor-remainder test against `_TRIG_PRECISSION_ = 1e-7`):

```c
if (0.5*deltax*deltax < 1e-7)        trig_order = 1;
else if (pow(deltax,4)/24.0 < 1e-7)  trig_order = 3;
else                                 trig_order = 5;
```

### 1.3 Regime selection inside HIS_create

For each grid point x_j (`hyperspherical.c:75-201`):

- Compute `xfwd` = turning point of the **highest l < l_WKB** in the l-list: `xfwd = asinK(sqrt(l_rec_max*(l_rec_max+1.))/beta)` (lines 90/103/116, by K). Points **x < xfwd** (at least one mode still evanescent): **backward recurrence** (Miller + CF1). Points **x ≥ xfwd** (all modes oscillatory): **forward recurrence** (stable there), done in chunks of `_HYPER_CHUNK_ = 16` x-values for vectorization.
- `l_WKB` is passed as `ptr->l[l_size_max-1]+1` from transfer.c — i.e. in current CLASS **WKB is never used to fill the table** (all l use recurrences); `hyperspherical_WKB` survives only for the x_min root-finding (§1.9) and the open-case lmax search. (`l_recurrence_max` = highest l in lvec with l < l_WKB.)
- Closed-universe guard (lines 141-144, 171-176): if `(int)(beta+0.2) == lmax+1` (i.e. ν = lmax+1, the highest l exactly at the cutoff l=ν−1... the l=ν mode doesn't exist), set `PhiL[lmax+1] = 0` and decrement lmax — Φ_{l=ν}^ν ≡ 0.

After PhiL is filled at a given x, Φ and dΦ are stored for the *sparse* l-list only (lines 156-160):

```c
pHIS->phi[k*nx+j]  = PhiL[l];
pHIS->dphi[k*nx+j] = l*pHIS->cotK[j]*PhiL[l] - sqrtK[l+1]*PhiL[l+1];
```

**Derivative relation:** dΦ_l/dx = l·cotK(x)·Φ_l − √(ν²−sgnK(l+1)²)·Φ_{l+1}.

### 1.4 Forward recurrence (transcribed; `hyperspherical.c:440-456`)

```c
PhiL[0] = 1.0/beta*sin(beta*x)/sinK;
PhiL[1] = PhiL[0]*(cotK - beta/tan(beta*x))*one_over_sqrtK[1];
for (l=2; l<=lmax; l++){
  PhiL[l] = ((2*l-1)*cotK*PhiL[l-1] - PhiL[l-2]*sqrtK[l-1]) * one_over_sqrtK[l];
}
```

i.e. the three-term recurrence **√(ν²−sgnK·l²)·Φ_l = (2l−1)·cotK(x)·Φ_{l−1} − √(ν²−sgnK·(l−1)²)·Φ_{l−2}**, seeded from the exact Φ_0 and Φ_1 = Φ_0·(cotK(x) − ν·cot(νx))/√(ν²−sgnK). (Chunked version `hyperspherical_forwards_recurrence_chunk`, lines 458-482, identical math over 16 x at a time.)

### 1.5 Backward recurrence — Miller algorithm with CF1 seed (`hyperspherical.c:485-571`)

```c
phi0 = sin(beta*x)/(beta*sinK);
if (K==1){
  if (beta > 1.5*lmax)  funcreturn = get_CF1(K,lmax,beta,cotK,&phipr1,&isign);
  if (funcreturn == _FAILURE_)
    CF1_from_Gegenbauer(lmax,(int)(beta+0.2),sinK,cotK,&phipr1);
  phi1 = 1.0;
}
else{
  get_CF1(K,lmax,beta,cotK,&phipr1,&isign);
  phi1 = isign;  phipr1 *= phi1;
}
PhiL[lmax] = phi1;
phi_plus_1_times_sqrtK = lmax*cotK*phi1 - phipr1;   // = sqrtK[lmax+1]*Phi_{lmax+1}
for (l=lmax; l>=1; l--){            // (blocked by _HYPER_BLOCK_=8 in real code)
  phi_minus_1 = ((2*l+1)*cotK*phi - phi_plus_1_times_sqrtK) * one_over_sqrtK[l];
  phi_plus_1_times_sqrtK = phi*sqrtK[l];
  phi = phi_minus_1;
  PhiL[l-1] = phi;
  if (fabs(phi) > _HYPER_OVERFLOW_ /*1e200*/){ rescale phi, phi_plus_1, and PhiL[l..lmax] by 1e-200; }
}
scaling = phi0/phi;                  // normalize so PhiL[0] == exact Phi_0
for (k=0; k<=lmax; k++) PhiL[k] *= scaling;
```

Downward recurrence: **√(ν²−sgnK·l²)·Φ_{l−1} = (2l+1)·cotK(x)·Φ_l − √(ν²−sgnK(l+1)²)·Φ_{l+1}**, started from an arbitrary-amplitude top value whose *logarithmic derivative* is fixed by the continued fraction; final normalization against the closed-form Φ_0. Overflow rescaling every `_HYPER_BLOCK_=8` steps.

### 1.6 CF1 — continued fraction for Φ'_{lmax}/Φ_{lmax} (`get_CF1`, lines 645-684)

Modified-Lentz evaluation of the continued fraction for dΦ/dx / Φ at l=lmax:

```c
int get_CF1(int K,int l,double beta, double cotK, double *CF, int *isign){
  int maxiter = 1000000;  double tiny = 1e-100;  double reltol = DBL_EPSILON;
  if (K==1) maxiter = (int)(beta-l-10);   // closed case: CF terminates near j=nu-l
  bj = l*cotK;            // b_0
  fj = bj;  Cj = bj;  Dj = 0.0;  *isign = 1;
  for(j=1; j<=maxiter; j++){
    sqrttmp = sqrt(beta2 - K*(l+j+1)*(l+j+1));
    aj = -sqrt(beta2 - K*(l+j)*(l+j)) / sqrttmp;
    if (j==1)  aj = sqrt(beta2 - K*(l+1)*(l+1)) * aj;
    bj = (2*(l+j)+1)/sqrttmp * cotK;
    Dj = bj + aj*Dj;   if (Dj==0.0) Dj = tiny;
    Cj = bj + aj/Cj;   if (Cj==0.0) Cj = tiny;
    Dj = 1.0/Dj;
    Delj = Cj*Dj;  fj = fj*Delj;
    if (Dj<0) *isign *= -1;             // tracks sign of Phi_lmax (Miller seed sign)
    if (fabs(Delj-1.0) < reltol){ *CF = fj; return _SUCCESS_; }
  }
  return _FAILURE_;
}
```

So Φ'_l/Φ_l = l·cotK + a_1/(b_1 + a_2/(b_2 + …)) with a_j, b_j built from the √(ν²−K(l+j)²) coefficients; `isign` accumulates (−1)^{#negative D_j}, giving the sign of Φ_lmax so the Miller seed has the right sign before normalization (only used for K=0,−1; for K=+1 the seed sign is fixed via Gegenbauer).

**Closed-universe fallback** — `CF1_from_Gegenbauer` (lines 686-738): when ν ≤ 1.5·lmax the CF converges slowly/fails (maxiter = ν−l−10 can even be ≤0), so Φ'_l/Φ_l is computed from the exact Gegenbauer-polynomial representation: with n = ν−l−1, α = l+1, X = cos x:

```c
// upward recurrence for Gegenbauer C_n^alpha(X):
G_k = (2*(k+alpha-1)*X*G_{k-1} - (k+2*alpha-2)*G_{k-2}) / k;   // k=4..n  (n<=3 explicit)
dG = (-n*X*G + (n+2*alpha-1)*G_{n-1}) / (1-X*X);
*CF = l*cotK - sinK*dG/G;    // since Phi ~ sin^l(x) C_{nu-l-1}^{l+1}(cos x)
```

(with the same 1e200 overflow rescaling). This encodes the closed-form **Φ_l^ν(x) ∝ sin^l(x)·C^{(l+1)}_{ν−l−1}(cos x)** for integer ν.

### 1.7 Hermite interpolation and ODE-derived derivatives

Evaluation at arbitrary x uses Hermite interpolation on the uniform grid, in two flavors:
- `hyperspherical_Hermite_interpolation_vector` (`hyperspherical.c:252-438`): order-6 Hermite (quintic polynomial per interval, matching y,y',y'' at both ends), with the higher derivatives generated from the ODE at the bracketing nodes (line 357 ff.):

```c
d2y = -2*dy*cotK + y*(l(l+1)/sinK² - beta² + K);
d3y = -2*cotK*d2y - 2*y*l(l+1)*cotK/sinK² + dy*(K - beta² + (2+l(l+1))/sinK²);
d4y = -2*cotK*d3y + d2y*(K - beta² + (4+l(l+1))/sinK²)
      + dy*(-4*(1+l(l+1))*cotK/sinK²) + y*(2*l(l+1)/sinK²*(2*cotK²+1/sinK²));
```

  Out-of-range x → 0 (lines 333-342). For K=+1, x is first folded into [0, π/2] by `ClosedModY` with the appropriate signs (§1.8). The quintic coefficients (a1..a5 from ym,dym,d2ym / yp,dyp,d2yp) are at lines 401-421.
- The production path in transfer.c instead uses the preprocessor-generated `hyperspherical_Hermite{3,4,6}_interpolation_vector_{Phi,dPhi,d2Phi,...}` family (`hyperspherical.c:1432-1737` including `tools/hermite{3,4,6}_interpolation_csource.h`), same math at orders 3/4/6 with monotonically-increasing-x optimization. Transfer uses **HERMITE6 for genuinely-curved tables** (sampling as low as 3 pts/wavelength) and **HERMITE4 for the flat table** (8 pts/wavelength) — `transfer.c:4072-4097`.

### 1.8 Closed-universe symmetries — `ClosedModY` (`hyperspherical.c:996-1022`)

Tables only cover x ∈ [0, π/2]; everything else is reflection:

```c
while (*y > 2π) *y -= 2π;
if (*y > π){ *y = 2π − *y;          // Phi parity in l
  if (l odd)  *phisign = −*phisign;  else *dphisign = −*dphisign; }
if (*y > π/2){ *y = π − *y;          // parity in (beta−l−1)
  if ((beta−l)%2==0) *phisign = −*phisign;  else *dphisign = −*dphisign; }
```

i.e. Φ_l^ν(2π−x) = (−1)^l Φ_l^ν(x) and Φ_l^ν(π−x) = (−1)^{ν−l−1} Φ_l^ν(x); derivatives pick up the complementary signs. Combined with integer ν and the l ≤ ν−1 cutoff this is everything the closed case needs.

### 1.9 WKB approximation — `hyperspherical_WKB` (`hyperspherical.c:793-853`)

Uniform (Langer/Airy) WKB used today only for x_min root-finding and the open-case l_max search. With e = 1/√(l(l+1)), α = βe, CscK = 1/sinK(y), w = α/CscK, turning point ytp = asinK(1/α):

```c
// K = -1 (open):
if (y > ytp){   // classically allowed (oscillatory)
  S = alpha*log((sqrt(w2-1.0)+sqrt(w2+alpha2))/sqrt(1.0+alpha2))
      + atan(1.0/alpha*sqrt((w2+alpha2)/(w2-1.0))) - 0.5*_PI_;
  airy_sign = -1;
}else{          // evanescent
  t = sqrt(1.0-w2)/sqrt(1.0+w2/alpha2);
  S = atanh(t) - alpha*atan(t/alpha);
  airy_sign = 1;
}
// K = +1 (closed), after ClosedModY folding:
if (y > ytp){
  t = sqrt(1-w2/alpha2)/sqrt(w2-1.0);
  S = atan(t) + alpha*atan(1.0/(t*alpha)) - 0.5*_PI_;
  airy_sign = -1;
}else{
  S = atanh(sqrt(1.0-w2)/sqrt(1.0-w2/alpha2))
      - alpha*log((sqrt(alpha2-w2)+sqrt(1.0-w2))/sqrt(alpha2-1.0));
  airy_sign = 1;
}
argu = 3.0*S/(2.0*e);
Q = CscK*CscK - alpha2;
C = 0.5*sqrt(alpha)/beta;
Ai = airy_cheb_approx(airy_sign*pow(argu,2.0/3.0));
*Phi = phisign*2.0*sqrt(_PI_)*C*pow(argu,1.0/6.0)*pow(fabs(Q),-0.25)*Ai*CscK;
```

(`hyperspherical_WKB_vec`, lines 740-790, is the flat-case K=0 vectorized variant with sinK_vec = x.) `airy_cheb_approx` (857-978) is a 4-region Chebyshev fit of Ai(z) (z≤−7 asymptotic-oscillatory `coef1`; −7<z≤0 `coef2` power series; 0<z<7 `coef3` exp-damped; z≥7 `coef4` asymptotic), each with hard-coded coefficient tables and a Clenshaw evaluator `cheb()`.

### 1.10 xmin / turning-point logic

Two routines find, per (l,ν), the x below which |Φ| < phiminabs (so the LOS integral can skip it):

- **`hyperspherical_get_xmin_from_approx`** (lines 1392-1423) — the production default (called at HIS_create:208 for every l in the table). Pure closed-form estimate via the flat-Bessel evanescent asymptotic + geometry correction:

```c
l_plus_half = l+0.5;
lhs = 1.0/l_plus_half*log(2*phiminabs*l_plus_half);
alpha = -2.0*lhs/5.0*(1.0+2.0*cosh(1.0/3.0*acosh(1.0+375.0/(16.0*lhs*lhs)))); // Chebyshev cubic root
x = l_plus_half/cosh(alpha)/nu;
if (K==-1){ x *= asinh(l/nu)/(l/nu);  x *= ((nu+0.4567)/(nu+1.24)-2.209e-3); } // + small-nu fudge
else if (K==1){ x *= asin(l/nu)/(l/nu); }
*xmin = x;
```

- **`hyperspherical_get_xmin_from_Airy`** (lines 1079-1164): root-find |Φ_WKB(x)| = phiminabs with Ridder's method (`fzero_ridder`, lines 1173-1253), stepping away from the turning point `xtp = asinK(sqrt(l(l+1))/beta)` in units of λ=2π/(β+5). Used by `transfer_get_lmax` for the open-case per-ν l_max refinement (transfer.c:4634).
- `hyperspherical_get_xmin` (lines 1025-1077): brute-force scan + Hermite refinement on the table itself (commented out at the call site).

`get_value_at_small_phi` (981-994) is the same closed-form inversion exposed standalone. `HypersphericalExplicit` (1255-1390) has exact closed forms `Φ = (γ·β·cos(xβ) + δ·sin(xβ))·CscK/√(N_K)` with hand-expanded polynomial γ, δ for l ≤ 9 and `N_K = β²·Π_{n=1..l}(β²−K n²)` — present but not called on the production path.

---

## 2. transfer.c usage

### 2.1 Top-level flow (`transfer_init`, transfer.c:116-406)

1. `q_period = 2π/(τ0−τ_rec) * angular_rescaling` (line 209), where `angular_rescaling = r_a(rec)/(τ0−τ_rec)` = sinK(√|K|(τ0−τrec))/(√|K|(τ0−τrec)) is set in `thermodynamics.c:3812` (`pth->angular_rescaling = pth->ra_rec/(pba->conformal_age-pth->tau_rec)`; =1 flat, >1 open, <1 closed).
2. `transfer_indices` → `transfer_get_l_list` (l grid, log step `pow(l_logstep, angular_rescaling)`, linear step `l_linstep*angular_rescaling` — lines 880-894: curvature stretches/shrinks the l sampling) → `transfer_get_q_list` (§2.2) → `transfer_get_k_list` (k=√(q²−(m+1)K), §2.3).
3. **One flat HIS ("BIS") for the whole run** (lines 256-275): `hyperspherical_HIS_create(0, 1., l_size_max, ptr->l, hyper_x_min, xmax, hyper_sampling_flat, l[lmax]+1, hyper_phi_min_abs, &BIS, ...)` with `xmax = q_max*tau0`, inflated for K<0 when the flat rescaling approximation is in play (lines 259-261: `xmax *= sqrt(|K|)/q_max*(l_max+1)/asinh((l_max+1)*sqrt(|K|)/q_max)*1.01` — because the rescaled argument overshoots).
4. Parallel loop over index_q: `transfer_update_HIS` (per-q curved HIS if needed, §2.4) then `transfer_compute_for_each_q`.

### 2.2 q grid — `transfer_get_q_list` (transfer.c:1033-1282)

Endpoints (lines 1056-1094):

```c
if (sgnK == 0){ q_min = ppt->k_min;  q_max = max over modes of k[k_size_cl-1]; K=0; }
else if (sgnK == -1){                        // open
  q_min = sqrt(ppt->k_min*ppt->k_min + K);   // note K<0 here: q = sqrt(k^2 - |K|)... 
                                             // (K stored negative; q² = k² + K means q < k)
  q_max = sqrt(k_max*k_max + K);
  if (has_vectors) q_max = MIN(q_max, sqrt(k_max*k_max + 2.*K));
  if (has_tensors) q_max = MIN(q_max, sqrt(k_max*k_max + 3.*K));
}
else if (sgnK == 1){ nu_min = 3;  q_min = nu_min*sqrt(K);  q_max = k_max; }   // closed
```

(Careful with sign conventions: in this routine the open branch is written with the *dimensionful* K<0 substituted, so `q² = k² + K` matches `k² = q² − K(m+1)` with m=0 in `transfer_get_k_list`. **Scalars: q² = k² + K; vectors q² = k²+2K; tensors q² = k²+3K**, with K = −|K| for open and +|K| for closed.)

Step-size law, flat/open (lines 1187-1193) — log step morphing into linear step:

```c
q = q[i-1] + q_period * q_linstep * q[i-1] / (q[i-1] + q_linstep/q_logstep_spline);
```

with `q_logstep_spline = ppr->q_logstep_spline / pow(angular_rescaling, ppr->q_logstep_open)` (line 1098) — i.e. **open models densify the low-q log sampling by (r_a/Δτ)^6** by default.

Closed case (lines 1207-1233): below `nu < hyper_flat_approximation_nu` (4000) it uses the same morphing formula but with the (much smaller) `q_logstep_trapzd` (=20) and then **snaps q to integer ν**:

```c
nu_proposed = (int)(q/sqrt(K));
if (nu_proposed <= nu+1) nu = nu+1;  else nu = nu_proposed;
q = nu*sqrt(K);
last_step = q - q[i-1];  last_index = index_q+1;
```

(ν always advances by ≥1 — never skips back; q-list is exactly {ν√K} until the step formula wants Δν>1, then it jumps in integer multiples.) Above the flat-approximation threshold the step transitions smoothly over `q_numstep_transition` (=250) points to the standard spline step (lines 1227-1232), no more integer snapping.

Finally (lines 1258-1278) `index_q_flat_approximation` = first index with q > `hyper_flat_approximation_nu`·√|K|. For all q above it, **no curved HIS is built at all** — flat Bessels are rescaled instead (§2.6).

`transfer_get_q_limber_list` (1296-1366): same endpoints, pure log grid `q[i] = q_logstep_limber * q[i-1]` (1.025), used only for the new full-Limber lensing-potential scheme.

### 2.3 k(q) and consistency — `transfer_get_k_list` (1379-1476)

```c
ptr->k[index_md][index_q] = sqrt(q*q − K*(m+1.));   // m = 0/1/2 for S/V/T
```

Sources from the perturbation module are interpolated **at this k** (spline in k, `transfer_interpolate_sources`, 2226+).

### 2.4 Per-q curved HIS — `transfer_update_HIS` (transfer.c:4558-4674)

Called once per q **only when `sgnK != 0 && index_q < index_q_flat_approximation`**; frees/rebuilds the workspace HIS (one HIS per q, per thread — this is the dominant curved-case cost):

```c
xmin = ppr->hyper_x_min;
sqrt_absK = sqrt(sgnK*K);
xmax = sqrt_absK*tau0;
nu = q/sqrt_absK;
if (sgnK == 1){
  xmax = MIN(xmax, _PI_/2.0 - hyper_x_min);     // only need [0, pi/2]
  int_nu = (int)(nu+0.2);  nu = (double)int_nu; // assert integer nu
}
sampling = (nu > hyper_nu_sampling_step) ? hyper_sampling_curved_high_nu
                                         : hyper_sampling_curved_low_nu;
// l_size_max: closed → drop all l >= nu;
// open → bisection with get_xmin_from_approx then get_xmin_from_Airy to find
//        the largest l whose x_nonzero < xmax (transfer_get_lmax, 4676-4807)
hyperspherical_HIS_create(sgnK, nu, l_size_max, ptr->l, xmin, xmax, sampling,
                          ptr->l[l_size_max-1]+1, hyper_phi_min_abs, &ptw->HIS, ...);
```

`transfer_get_lmax` (4676-4807): geometric hunt + binary search over the l-list for the boundary l where x_nonzero(l,ν) crosses xmax — for open universes high l never "turns on" inside the horizon at small ν, so the table is trimmed. (Modes with `index_l >= ptw->HIS.l_size` are then explicitly zeroed: `transfer.c:2048-2050`; closed modes with l ≥ ν likewise: lines 2044-2046.)

### 2.5 χ sampling and the LOS convolution

The time sampling is **the perturbation module's source sampling** (`ppt->tau_sampling`), not a Bessel-driven grid: `transfer_source_tau_size` (1672-1843) gives `tau_size = ppt->tau_size` for T0/T1/T2/E (line 1701); for lcmb early times (τ<τ_rec) are dropped (1708-1716). `transfer_radial_coordinates` (2169-2206) then converts:

```c
case 1:  chi = sqrt(K)*(tau0-tau);    cscKgen = sqrt(K)/k/sin(chi);   cotKgen = cscKgen*cos(chi);
case 0:  chi = k*(tau0-tau);          cscKgen = 1/chi;                cotKgen = 1/chi;
case -1: chi = sqrt(-K)*(tau0-tau);   cscKgen = sqrt(-K)/k/sinh(chi); cotKgen = cscKgen*cosh(chi);
```

Note the *flat* convention bakes k into χ (χ=k(τ0−τ), ν=1) while curved uses χ=√|K|(τ0−τ), ν=q/√|K| — both give Bessel argument "νχ" consistently. Also note `cscKgen` carries an extra 1/k normalization in the curved cases (= √|K|/(k·sinK)) — this is the curved generalization of 1/(k(τ0−τ)) that appears in the flat radial functions.

`transfer_integrate` (3425-3585): trapezoidal convolution `Δ_l(q) = ∫ dτ S(k,τ)·f_l(χ(τ))` (`array_trapezoidal_convolution`), truncated below `tau0_minus_tau_min_bessel`:

```c
if (sgnK==0)  tau0_minus_tau_min_bessel = pBIS->chi_at_phimin[index_l]/k;
else if (index_q < index_q_flat_approximation)
              tau0_minus_tau_min_bessel = HIS.chi_at_phimin[index_l]/sqrt(|K|);
else {        // flat rescaling regime: rescale the flat cutoff by x_tp ratio
  tau0_minus_tau_min_bessel = pBIS->chi_at_phimin[index_l]/sqrt(|K|);
  x_turning_point = asinK(sqrt(l(l+1))/q*sqrt(|K|));        // asin/asinh by sgnK
  tau0_minus_tau_min_bessel *= x_turning_point/sqrt(l(l+1));
}
```

with an exact triangle correction at the Bessel cutoff (lines 3576-3580).

### 2.6 `transfer_radial_function` (transfer.c:4019-4293) — THE curved radial functions

Setup (4053-4097):

```c
K = ptw->K;  k2 = k*k;
sqrt_absK_over_k = (sgnK==0) ? 1.0 : sqrt(sgnK*K)/k;     // converts dPhi/dx → dPhi/d(k(tau0-tau))
absK_over_k2 = sqrt_absK_over_k^2;
if (sgnK == 0)                 { pHIS = pBIS; rescale_argument=1; rescale_amplitude=1; HERMITE4; }
else if (index_q < index_q_flat_approximation)
                               { pHIS = &HIS; rescale_argument=1; rescale_amplitude=1; HERMITE6; }
else {                                                    // FLAT RESCALING APPROXIMATION
  nu = q/sqrt(|K|);
  chi_tp = asinK(sqrt(l(l+1))/nu);
  rescale_argument  = sqrt(l*(l+1.))/chi_tp;              // maps x → x*rescale so turning points coincide
  rescale_amplitude = pow(1.-K*l*(l+1.)/q/q, -1./12.);    // WKB amplitude ratio at turning point
  pHIS = pBIS; HERMITE4;
}
```

Flat-rescaling correction function (4124-4151), an empirical 2nd-order expansion around the turning point, clipped by the exact WKB amplitude ratio x/sinK(x):

```c
chireverse[j] = chi[x_size-1-j]*rescale_argument;        // chi increasing for interpolator
// closed:
rescale_function[j] = MIN( rescale_amplitude*(1 + 0.34*atan(l/nu)*(chi-chi_tp)
                                                + 2.00*pow(atan(l/nu)*(chi-chi_tp),2)),
                           chi/sin(chi) );
// open:
rescale_function[j] = MAX( rescale_amplitude*(1 - 0.38*atan(l/nu)*(chi-chi_tp)
                                                + 0.40*pow(atan(l/nu)*(chi-chi_tp),2)),
                           chi/sinh(chi) );
```

So for ν > 4000 (default), **Φ_l^ν(χ) ≈ rescale_function(χ) · j_l(ν·χ·√(l(l+1))/(ν·χ_tp))** — flat Bessel with stretched argument and slowly-varying amplitude correction. This is the Lesgourgues–Tram flat approximation.

Radial functions (transcribed, 4168-4284; `rescale_argument` factors accompany each dΦ/d2Φ to express derivatives w.r.t. the unrescaled argument; in the exact branches rescale_*=1):

```c
case SCALAR_TEMPERATURE_0:                                  // f = Phi
  radial[..] = Phi[j]*rescale_function[j];

case SCALAR_TEMPERATURE_1:                                  // f = (sqrt|K|/k) dPhi/dx  ( = j_l' flat)
  radial[..] = sqrt_absK_over_k*dPhi[j]*rescale_argument*rescale_function[j];

case SCALAR_TEMPERATURE_2:                                  // f = [3 (|K|/k²) d²Phi + Phi] / (2 s2)
  s2 = sqrt(1.0-3.0*K/k2);  factor = 1.0/(2.0*s2);
  radial[..] = factor*(3*absK_over_k2*d2Phi[j]*resc_arg²+Phi[j])*rescale_function[j];

case SCALAR_POLARISATION_E:                                 // f = sqrt(3(l+2)!/(8(l-2)!)) /s2 * cscKgen² * Phi
  s2 = sqrt(1.0-3.0*K/k2);
  factor = sqrt(3.0/8.0*(l+2.0)*(l+1.0)*l*(l-1.0))/s2;
  radial[..] = factor*cscKgen[..]²*Phi[j]*rescale_function[j];

case VECTOR_TEMPERATURE_1:
  s0 = sqrt(1.0+K/k2);  factor = sqrt(0.5*l*(l+1))/s0;
  radial[..] = factor*cscKgen[..]*Phi[j]*...;

case VECTOR_TEMPERATURE_2:
  s0 = sqrt(1.0+K/k2);  ssqrt3 = sqrt(1.0-2.0*K/k2);
  factor = sqrt(1.5*l*(l+1))/s0/ssqrt3;
  radial[..] = factor*cscKgen[..]*(sqrt_absK_over_k*dPhi[j]*resc_arg - cotKgen[j]*Phi[j])*...;

case VECTOR_POLARISATION_E:
  factor = 0.5*sqrt((l-1.0)*(l+2.0))/s0/ssqrt3;
  radial[..] = factor*cscKgen[..]*(cotKgen[j]*Phi[j] + sqrt_absK_over_k*dPhi[j]*resc_arg)*...;

case VECTOR_POLARISATION_B:
  si = sqrt(1.0+2.0*K/k2);
  factor = 0.5*sqrt((l-1.0)*(l+2.0))*si/s0/ssqrt3;
  radial[..] = factor*cscKgen[..]*Phi[j]*...;

case TENSOR_TEMPERATURE_2:
  ssqrt2 = sqrt(1.0-1.0*K/k2);  si = sqrt(1.0+2.0*K/k2);
  factor = sqrt(3.0/8.0*(l+2.0)*(l+1.0)*l*(l-1.0))/si/ssqrt2;
  radial[..] = factor*cscKgen[..]²*Phi[j]*...;

case TENSOR_POLARISATION_E:
  factor = 0.25/si/ssqrt2;
  radial[..] = factor*( absK_over_k2*d2Phi[j]*resc_arg²
                        + 4.0*cotKgen[..]*sqrt_absK_over_k*dPhi[j]*resc_arg
                        - (1.0+4*K/k2-2.0*cotKgen[..]²)*Phi[j] )*...;

case TENSOR_POLARISATION_B:
  ssqrt2i = sqrt(1.0+3.0*K/k2);
  factor = 0.5*ssqrt2i/ssqrt2/si;
  radial[..] = factor*(sqrt_absK_over_k*dPhi[j]*resc_arg + 2.0*cotKgen[..]*Phi[j])*...;

case NC_RSD:
  radial[..] = absK_over_k2*d2Phi[j]*resc_arg²*...;   // bug fixed in 2.4.3: factor absK_over_k2 was missing
```

The **curvature s-factors** are `s_n = sqrt(1 − (n²−1)K/k²)`-type combinations: `s2 = √(1−3K/k²)` (the same `s2_squared` of the perturbation ICs), `s0 = √(1+K/k²)`, `√(1−2K/k²)`, `√(1±…)` etc. The E-polarization "√((ν²−4)…)" factors of the literature appear here through cscKgen² (= |K|/(k²sinK²) = (ν²/ (ν²) ) …) together with 1/s2; CLASS's normalization choice puts √(1−3K/k²) = √((q²−4K)/(q²−K)) into the radial function (denominator) and the matching factor into the ICs (§3).

**Which radial type for which source** (`transfer_select_radial_function`, 4295-4389): scalar t0→T0, t1→T1, t2→T2, e→POL_E; **lcmb, density, lensing all use SCALAR_TEMPERATURE_0** (plain Φ — generic case, line 4303-4304); nc_rsd→NC_RSD; nc_d1, nc_g5→T1.

Temperature assembly: Δ_l^T = Δ_l^{t0} + Δ_l^{t1} + Δ_l^{t2} (harmonic.c:962).

### 2.7 Flat-limit handling

CLASS uses the hyperspherical machinery for **every K≠0 run, but only below ν = hyper_flat_approximation_nu = 4000**; above that it uses the rescaled flat j_l (§2.6). There is no |K| threshold under which the whole run silently becomes flat — the switch is per-q in ν. (For very small |Omega_k|, *all* q have ν > 4000, so effectively the entire run uses rescaled flat Bessels — see §6.)

### 2.8 Limber in curved space — `transfer_limber` (3606-3759)

Only SCALAR_TEMPERATURE_0 (and the flat-only T1/NC_RSD variants) implement Limber. Curved version of the peak position and prefactor:

```c
if (sgnK == 0)        tau0_minus_tau_limber = (l+0.5)/q;
else if (sgnK == 1) { x_limber = asin(sqrt(l*(l+1.))/q*sqrt(K));   tau0_minus_tau_limber = x_limber/sqrt(K); }
else                { x_limber = asinh((l+0.5)/q*sqrt(-K));        tau0_minus_tau_limber = x_limber/sqrt(-K); }
...
IPhiFlat = sqrt(_PI_/(2.*l))*(1.-0.25/l+1./32./(l*l));   // ∫ j_l peak weight, Stirling-corrected
*trsf = IPhiFlat*S;
if (sgnK == 0)  *trsf /= (l+0.5);
else            *trsf *= pow(1.-K*l*l/q/q, -1./4.) / (tau0_minus_tau_limber*q);
```

i.e. the curved Limber kernel = flat one evaluated at the curved turning point with WKB amplitude factor (1−Kl²/q²)^{−1/4}. Source interpolation interpolates S·(τ0−τ) parabolically (3763-3828; regular at τ→τ0 for lensing). Comment at 3360: "in principle the Limber condition should be adapted to account for curvature effects — TBC". `transfer_limber2` (3850-3912) is flat-only (`(l+0.5)/k`, "to be updated to include curvature effects").

Limber switching (3336-3401): lcmb uses Limber for l > `l_switch_limber` (=10) — **so almost the entire CMB-lensing-potential transfer is curved-Limber**, plus the optional full-Limber scheme on the separate log q grid.

---

## 3. Primordial spectrum and Cl assembly (harmonic.c)

### 3.1 Where P is evaluated — at k, not q or ν

`harmonic_compute_cl` (harmonic.c:854-1345), the integrand loop (924-933):

```c
for (index_q=0; index_q < ptr->q_size; index_q++) {
  k = ptr->k[index_md][index_q];                       // k = sqrt(q² − (m+1)K)
  cl_integrand[index_q*ncol+0] = k;                    // integration variable is k!
  class_call(primordial_spectrum_at_k(ppm,index_md,linear,k,primordial_pk), ...);
  ...
  factor = 4. * _PI_ / k;
  cl_integrand[...+index_ct_tt] = primordial_pk * T1 * T2 * factor;   // etc. for ee/te/bb/pp/tp/ep/dd/...
```

So **C_l = ∫ (dk/k) · 4π · 𝒫(k) · Δ1_l(q(k)) · Δ2_l(q(k))**, with 𝒫(k) the *same pure power law as flat* (`primordial.c` contains no curvature anywhere — checked; `calP = A_s (k/k_pivot)^(n_s-1)`).

### 3.2 The curvature-factor bookkeeping (verbatim comment, harmonic.c:1032-1071)

```
   C_l = int [4 pi dk/k calP(k) Delta1_l(q) Delta2_l(q)]
   where ... q=sqrt(k2+K) (scalars) or sqrt(k2+2K) (vectors) or sqrt(k2+3K) (tensors)

   In the literature, people often rewrite the integral in terms of q ...
   dk/k = kdk/k2 = qdq/k2 = dq/q * (q/k)^2 = dq/q * [q2/(q2-K)] = q2dq * 1/[q(q2-K)]
   This factor 1/[q(q2-K)] is commonly absorbed in the definition of calP. ...
   Sometimes in the literature, the factor (k2-3K)=(q2-4K) present
   in the initial conditions of scalar transfer functions (if
   normalized to curvature R=1) is also absorbed in the definition
   of the power spectrum. Then the curvature power spectrum reads
   calP = (q2-4K)/[q(q2-K)] * (k/k)^ns

   In CLASS we prefer to define calP = (k/k)^ns like in the flat
   case, to have the factor (q2-4K) in the initial conditions,
   and the factor 1/[q(q2-K)] doesn't need to be there since we
   integrate over dk/k.

   For tensors ... dk/k = ... = q2dq * 1/[q(q2-3K)]
   But for tensors there are extra curvature-related correction factors ...
```

**Where (q²−4K) = (k²−3K) actually lives:** scalar adiabatic initial conditions, `perturbations.c:5583-5613`:

```c
s2_squared = 1.-3.*pba->K/k/k;
ppw->pv->y[index_pt_delta_g] = ... * ppr->curvature_ini * s2_squared;   // etc. for all species
```

and it cancels against the 1/s2 = 1/√(1−3K/k²) factors in the T2/E radial functions (and in the Einstein equations: `perturbations.c:6870,6890,6927`: `k2*s2_squared*eta ...`). The free-streaming curvature factors in the Boltzmann hierarchy are `ppw->s_l[l] = sqrt(MAX(1.0-pba->K*(l*l-1.0)/k/k,0.))` (`perturbations.c:3064`), per arXiv:1305.3261. Tensor normalization comment (`perturbations.c:6246-6252`): Σ<h h> = ∫dk/k (q²−3K)(q²−4K)/(q²(q²−K)) 𝒫_h(k).

### 3.3 Integration measure: spline-in-k with closed-case trapezoid + discrete-sum correction

- Integration is over the **k column** with `array_spline` + `array_integrate_all_trapzd_or_spline` (harmonic.c:1295-1316). For sgnK=+1, indices below `index_q_flat_approximation` are forced trapezoidal (`index_q_spline = ptr->index_q_flat_approximation`, lines 906-922) because integer-ν snapping makes dq jumpy and splines inaccurate.
- **Closed-case discrete-sum correction** (lines 1318-1330): the C_l is really a sum over integer ν; the trapezoid underweights the first point, so:

```c
if (pba->sgnK == 1)  clvalue += integrand[1+index_ct] * q_min/k_min*sqrt(pba->K)/2.;
```

(the missing half-weight of the ν=3 term, with dq=√K and the dk/dq=q/k Jacobian).

### 3.4 Matter P(k) in curved space

`fourier.c` (and harmonic Pk output) contains **no sgnK/K anywhere** — P(k) is computed exactly as in flat space from the δ_m source at wavenumber k (δ_m defined gauge-invariantly per 1307.1459). Curvature only enters through the background/perturbation evolution. The perturbation k-grid itself starts at `k_min` tied to curvature (largest of a few cutoffs; for closed, k discreteness handled in perturbations input — not transcribed here as it does not affect the transfer port).

---

## 4. Lensing

### 4.1 lensing.c — no explicit curvature

`source/lensing.c` (Challinor & Lewis astro-ph/0502425 full-sky correlation-function method) contains **zero references to pba->K or sgnK** (grep confirms). It consumes Cl_pp and the unlensed Cls on the sphere; all curvature is already inside those Cls. Nothing to port beyond what ABCMB's lensing already does — *provided* Cl_pp itself is computed with the curved kernel below.

### 4.2 CMB lensing potential source — curved Weyl kernel (`transfer_sources`, transfer.c:2396-2445)

```c
/* lensing source = - W(tau) (phi+psi) Heaviside(tau-tau_rec)
   W = (tau-tau_rec)/(tau_0-tau)/(tau_0-tau_rec)   [flat form]            */
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
sources[..] = interpolated_sources[..] * rescaling * lcmb_rescale * pow(k/lcmb_pivot, lcmb_tilt);
```

i.e. W_K(τ) = √|K|·sinK(χ_rec−χ)/(sinK(χ)·sinK(χ_rec)) — the angular-diameter-distance ratio kernel, sign convention via sin(τ_rec−τ) (negative for τ>τ_rec ⇒ overall minus). The corresponding radial function is plain Φ (SCALAR_TEMPERATURE_0).

### 4.3 Galaxy lensing / number-count lensing windows (`transfer_precompute_selection`, transfer.c:5147-5229)

Same structure with a selection-function integral:

```c
sinKgen_source        = sinK( (tau0−tau_source)·√|K| ) / √|K|;
sinKgen_source_to_lens= sinK( ((tau0−tau)−(tau0−tau_source))·√|K| ) / √|K|;
cscKgen_lens          = √|K| / sinK( √|K|·(tau0−tau) );
rescaling += sinKgen_source_to_lens * cscKgen_lens / sinKgen_source
             * selection[..] * w_trapz[..];               // lensing potential (tt_lensing)
// nc_lens: same × −(2−5s_bias)/2 ;  nc_g4: (2−5s_bias)·cotKgen_source
```

i.e. the standard sinK(χ_s−χ)/(sinK(χ)·sinK(χ_s)) lensing efficiency.

---

## 5. Cost notes (as visible in the code)

- The curved path rebuilds a full HIS **per q below ν=4000** (`transfer_update_HIS` frees and re-creates each time; one per OpenMP workspace). Each HIS costs O(nx · lmax) recurrence work, nx ≈ xmax·ν·sampling/(2π).
- Parallelization is over q (`transfer_init:301`), with HIS_create itself parallelized over x-chunks only in the flat case (`class_setup_parallel_optional(K == 0)`, hyperspherical.c:135).
- `q_linstep=0.45` (in units of 2π/r_a(rec)) controls the asymptotic q density; the **closed case forces Δν ≥ 1 steps** which at small |K| is *coarser* than the flat sampling wants — fine, since the physical modes are genuinely discrete.
- The CMB temperature/polarization convolutions reuse ppt->tau_sampling; no extra χ grid is invented for curvature.

---

## 6. Practical scales for |Ω_k| ≤ 0.1 (Planck-like, h≈0.67, lmax≈2500, q_max≈0.18 Mpc⁻¹, τ0≈14000 Mpc)

K = −Ω_k H0² ⇒ √|K| = √|Ω_k| · H0 ≈ √|Ω_k| · 2.2×10⁻⁴ Mpc⁻¹.

| |Ω_k| | √|K| (Mpc⁻¹) | x_max = √|K|·τ0 | ν_max = q_max/√|K| | ν at flat-approx switch (q=4000√|K|) | fraction of q-range needing true hyperspherical |
|------|------|------|------|------|------|
| 0.01 | 2.2e-5 | ≈0.31 | ≈8000 | q ≈ 0.09 Mpc⁻¹ | low-q half (l ≲ 1200 equivalent) |
| 0.05 | 5.0e-5 | ≈0.70 | ≈3600 | q ≈ 0.20 > q_max | essentially **all** q |
| 0.1  | 7.0e-5 | ≈0.99 | ≈2600 | q ≈ 0.28 > q_max | **all** q |

- ν range: from ν_min ≈ q_min/√|K| (a few, for q_min ~ 1e-4 Mpc⁻¹... e.g. ν_min ≈ 5 at Ω_k=0.01; closed case literally starts at ν=3) up to ν_max above.
- x range: [1e-5, √|K|τ0] ≤ ~1.0 for |Ω_k| ≤ 0.1 — so **sinK(x) ≈ x to within ~17% at most, and the tables never approach x=π/2 issues for realistic open/closed Planck-like models** (closed cap at π/2 only matters for very closed universes).
- Number of x points per HIS at ν=2600, x_max≈1.0, sampling=3 (high-ν): nx ≈ 1.0·2600·3/(2π) ≈ 1240. At ν~500, sampling=7: nx ≈ 560·(x_max). Cheap per table; the cost is the *number of tables* (= number of q below the switch, hundreds to ~2000).
- Defaults recap (`include/precisions.h:427-501`): `hyper_x_min=1e-5`, `hyper_sampling_flat=8.0` (">7.5"), `hyper_sampling_curved_low_nu=7.0`, `hyper_sampling_curved_high_nu=3.0`, `hyper_nu_sampling_step=1000`, `hyper_phi_min_abs=1e-10`, `hyper_x_tol=1e-4`, `hyper_flat_approximation_nu=4000`; q grid: `q_linstep=0.45`, `q_logstep_spline=170`, `q_logstep_open=6.0`, `q_logstep_trapzd=20`, `q_numstep_transition=250`, `q_logstep_limber=1.025`; `l_linstep=40`, `l_logstep=1.12`, `l_switch_limber=10`; `_TRIG_PRECISSION_=1e-7`, `_HYPER_OVERFLOW_=1e200`, `_HYPER_BLOCK_=8`, `_HYPER_CHUNK_=16` (hyperspherical.h:9-16).

---

## 7. JUDGEMENT — minimal faithful subset for ABCMB at 0.1% for |Ω_k| ≤ 0.1

(Clearly labeled as judgement, not transcription.)

1. **WKB is NOT needed for function values.** CLASS itself never fills tables with WKB anymore (l_WKB is passed as lmax+1); everything is the two three-term recurrences + CF1 seed + Φ_0 normalization. WKB/Airy survives only for (a) finding x_min cutoffs (an optimization — the closed-form `hyperspherical_get_xmin_from_approx` is the default anyway and is ~10 lines) and (b) trimming l_max per ν in open models (also an optimization; you can simply compute all l and let small values multiply through). **Port: forward recurrence, backward recurrence with CF1 (modified Lentz), Φ_0 anchor, overflow rescaling. Skip: WKB, Airy-Chebyshev, Ridder, HypersphericalExplicit, Gegenbauer CF.**

2. **Closed-universe integer-ν snapping: needed in principle for K>0, but cheap.** For Ω_k ~ −0.01 (closed), √K ≈ 2.2e-5 ⇒ Δq = √K ≈ 2e-5 Mpc⁻¹, far finer than CLASS's own continuum sampling at low q — so the integer grid is *denser* than needed and a continuum treatment with the (q²−4K) factors intact is an excellent approximation except for the very first modes (ν=3..~20, affecting only l≲20 cosmic-variance-limited scales). CLASS itself approximates the discrete sum by an integral (harmonic.c:1318-1330) with a one-point correction. **Judgement: for |Ω_k| ≤ 0.1 closed you can start with a continuum q grid + the q_min = 3√K cutoff + the half-weight first-point correction, and snap to integers only if low-l TT residuals vs CLASS exceed target.** The l ≤ ν−1 cutoff must be respected either way (zero the transfer there).

3. **The flat-rescaling approximation is the big win and is essentially mandatory to match CLASS.** For |Ω_k| ≤ 0.01, *all or most* of the q-range uses rescaled flat j_l (ν > 4000): argument stretch √(l(l+1))/χ_tp, amplitude (1−Kl(l+1)/q²)^{−1/12}, and the quadratic-in-(χ−χ_tp) correction with the 0.34/2.00 (closed) and 0.38/0.40 (open) coefficients, clipped by x/sinK(x). Since ABCMB already has flat j_l tabulated + asymptotics, **a curvature port could plausibly use ONLY this rescaling for |Ω_k| ≲ 0.01** — that reproduces what CLASS actually computes there. For |Ω_k| up to 0.1 the true hyperspherical recurrences are needed below ν=4000 (which becomes the whole range); matching CLASS at 0.1% then requires the recurrence machinery of item 1. An alternative worth testing: lower the effective "flat approximation" threshold and check against CLASS — CLASS's own threshold (ν=4000) was tuned for ~0.1% Cl accuracy per 1305.3261.

4. **χ sampling can reuse ABCMB's existing lna grid.** CLASS evaluates the radial functions on its source time grid (ppt->tau_sampling), exactly analogous to ABCMB's fixed 500-point lna grid; Φ tables are interpolated to those χ values (Hermite there, jnp.interp/cubic in ABCMB). What changes is only: χ = √|K|(τ0−τ) instead of k(τ0−τ); kernel = Φ_l^ν(χ) and its first two derivatives instead of j_l and derivatives; the s2-factors in T2/E; cscKgen²=|K|/(k²sinK²) replacing 1/(k(τ0−τ))² in E; the (1−3K/k²) factor in ICs; the q vs k bookkeeping (P(k) at k=√(q²−K), transfer at q); the sinK lensing kernel (trivial); the curved Limber prefactor for Cl_pp ((1−Kl²/q²)^{−1/4} at the curved turning point) if ABCMB Limbers its lensing potential.

5. **JAX-specific note:** the per-q HIS rebuild is hostile to vmap (per-q x-grid sizes differ). Two same-answer options: (a) fix one x grid (size set by ν_max and the high-ν sampling) for all q — wasteful at low ν but static-shape; (b) evaluate Φ_l^ν directly at the χ source points by recurrence (no table at all): backward recurrence over l is a `lax.scan` of length lmax with (Nq, Nχ)-shaped carries — fully batched, no interpolation error, at the cost of computing all l instead of the sparse l-list (ABCMB computes all ell up to lmax via spline reconstruction over ~99 sampled ells anyway, so only the sampled ells need extraction). CF1 is a while-loop (or fixed ~few-hundred-iteration scan; convergence is fast in the oscillatory region and the closed case terminates after ν−l steps).

6. **Sanity anchors for a port:** Φ_0^ν(x)=sin(νx)/(ν sinK x); flat limit j_l(νχ); the dΦ relation dΦ_l = l·cotK·Φ_l − √(ν²−sgnK(l+1)²)Φ_{l+1}; the ODE for d²Φ (used in T2/E and Hermite); closed symmetries Φ(π−x)=(−1)^{ν−l−1}Φ(x).
