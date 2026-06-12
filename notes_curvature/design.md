# Curved-geometry (Omega_k != 0) design for ABCMB

Decisions condensed from the three research reports in this directory
(`abcmb_flat_audit.md`, `class_curvature_equations.md`, `class_hyperspherical_transfer.md`).
Goal: match CLASS Cls for |Omega_k| <= ~0.1 (open and closed), with maximum parsimony
w.r.t. existing ABCMB structure. Sign convention (CLASS): `K = -Omega_k * (H0/c)^2` [Mpc^-2],
so Omega_k > 0 = open = K < 0.

## Parameters (main.py)
- `params['omega_k']` (= Omega_k h^2, float default 0.) among defaults; added to `expected_keys`.
- `params['K'] = -omega_k * (cnst.H0_over_h/cnst.c_Mpc_over_s)**2` derived once (Mpc^-2).
- Closure: `omega_Lambda = h^2 - omega_r - omega_m - omega_k`.

## Background — curvature as a species
- New `Curvature(BackgroundFluid)` in species.py: `rho = omega_k * (3 H100^2/8 pi G) / a^2`,
  `P = -rho/3`, `num_equations = 0`, `is_matter = False`. Added to the LCDM tuple in
  `populate_species`. Then `H`, `aH_prime`, `d2adtau2_over_a` are automatically the curved
  Friedmann forms (rho+3P = 0 for w=-1/3 — the -K/a^2 constant drops out of (aH)^2-prime exactly).
  Contamination of the omega_r / Neff inference loops is O(1e-20) (audit §1.2) — accepted.
- `_tabulate_rs`: ODE gains factor `sqrt(1 - K rs^2)` (CLASS background.c:3213).
- `rA_rec = S_K(tau0 - tau_rec)`.

## Geometry helpers (ABCMBTools.py) — AD-safe, smooth through K=0
All built from two entire functions of `u = K*chi^2` (series for |u| < u0, clipped trig/hyper
branches outside; correct d/dK at K=0, no 0-gradient dead zones):
- `f(u)`: S_K(chi) = chi*f(u); f = sin(sqrt(u))/sqrt(u) (u>0) / sinh(sqrt(-u))/sqrt(-u) (u<0);
  series 1 - u/6 + u^2/120 - u^3/5040.
- `g(u)`: cot_K(chi) = g(u)/chi; g = sqrt(u)cot(sqrt(u)) / sqrt(-u)coth(sqrt(-u));
  series 1 - u/3 - u^2/45 - 2u^3/945.
Validity domain: sqrt|K|*tau0 < pi/2 (covers |Omega_k| <= ~0.1 Planck-like; documented).

## Perturbations — always-on, value-level (exactly flat-reducing at K=0)
- `s_l(k, K) = sqrt(max(1 - K(l^2-1)/k^2, 0))` (safe-sqrt + where; clips genuinely for closed
  l >= nu — physical hierarchy termination). Shared by photon T/pol, massless nu, massive nu.
- Einstein constraints (BOTH copies — get_derivatives and make_output_table):
  `k^2*eta -> (k^2-3K)*eta` in h'; `eta' = [4 pi G a^2 sum rpt / c^2 / aH + K*h'_lna/2] / (k^2-3K)`.
  `alpha = (h'+6 eta')/2k^2` and `alpha'` unchanged (CLASS forms carry no explicit K).
- Hierarchies (port CLASS verbatim, perturbations.c:9973-10777):
  theta': `k^2(delta/4 - s2^2 sigma)`; photon sigma': F3 coupling *(s3/s2), pol source /s2,
  -9/10 kappa' sigma unchanged; F3': `k/7(6 s3 s2 sigma - 4 s4 F4)`; generic l: `l s_l / (l+1) s_{l+1}`;
  lmax: `k s_lmax F_{lmax-1} - (lmax+1) cot_K(tau)/aH F_lmax`. Pol generic + P0 = (G0+G2+2 s2 sigma)/8.
  Massive nu: s2 on Psi1<->Psi2 couplings and on the Psi2 metric source; lmax per CLASS (no s on
  upstream term).
- Adiabatic ICs: s2^2 = 1-3K/k^2 factors per CLASS perturbations.c:5591-5786 (delta_g/theta_g/
  delta_b/theta_b/delta_c * s2^2; theta_nu `12->12 s2^2` + overall s2^2; sigma_nu `2->(3 s2^2-1)`;
  eta_ini `5+4R -> 5+4 s2^2 R`). This is where the (q^2-4K)=(k^2-3K) normalization lives —
  CLASS keeps P(k) a pure power law (harmonic.c:1032-1071), so NO measure changes in the Cl
  k-integral or Pk_lin.

## Spectrum — static switch `specs['curvature']` (default False = byte-identical flat path)
Curved path replaces the per-ell flat-Bessel-table LOS with ONE exact hyperspherical
recurrence shared by all ells (loop inversion):
- q^2 = k^2 + K per transfer-grid k (q_safe + where-mask where q^2 <= 0 [open] or
  nu < 3 i.e. k^2 < 8K [closed]; masks are value-level, K traced, static grid unchanged).
- Dimensionful recurrence in chi = tau0 - tau (smooth through K=0, no sign branches):
  `sqrt(q^2 - K l^2) Phi_l = (2l-1) cot_K(chi) Phi_{l-1} - sqrt(q^2 - K(l-1)^2) Phi_{l-2}`,
  seeds `Phi_0 = sin(q chi)/(q S_K(chi))`, `Phi_1 = Phi_0 (cot_K(chi) - q cot(q chi))/sqrt(q^2-K)`.
  dPhi/dchi = sqrt(q^2 - K l^2) Phi_{l-1} - (l+1) cot_K(chi) Phi_l; d2Phi from the ODE.
  Forward recurrence only (exact seeds): oscillatory region is stable; evanescent-region garbage
  is clipped (carry clamp) and masked at emission by CLASS's closed-form x_min(l, nu) estimate
  (hyperspherical_get_xmin_from_approx — same |Phi|<1e-10 cut CLASS applies to its LOS integral).
  Justification: report §7 — CLASS itself never needs evanescent values below 1e-10.
- Structure: outer lax.scan over the ~99-111 table ells, inner scan over intermediate l
  (jax.checkpoint on inner body for reverse-AD); carry (Phi_{l-1}, Phi_l) of shape (Nlna, Nk);
  sources S_T0/T1/T2/E built ONCE (Nlna, Nk) (shared with flat path via small refactor);
  contraction over lna and the dk/k integral inside the emission step -> per-ell (ClTT, ClTE, ClEE).
  Memory ~tens of MB; no new big tensors.
- Radial functions (CLASS transfer.c:4019-4293, dimensionful): T0 = Phi; T1 = dPhi/dchi / k;
  T2 = [3 d2Phi/dchi^2 / k^2 + Phi]/(2 s2); E = ell_eps_factor/s2 * Phi/(k S_K(chi))^2.
  Flat limits = phi0, phi1, phi2, eps exactly.
- Source change (both paths): polarization Pi = (2 s2 sigma_g + G0 + G2)/8 (s2 on transfer k grid).
- Closed universes: continuum-q approximation initially (no integer-nu snapping; report §7.2:
  integer grid is denser than continuum sampling for |Omega_k| ~ 0.01; CLASS approximates the
  discrete sum by an integral anyway). l <= nu-1 enforced by the s_l/sqrtK masks. Revisit only
  if low-l TT vs CLASS misses target.

## Lensing
- `lensing_power_spectrum`: Om(z) -> omega_m/(a^3 (H/H100)^2 h^-2...) exact from BG (drops the
  flat no-radiation approximation); Poisson `1/k -> k^3/(k^2-3K)^2`.
- `lensing_Cl` Limber: k = (l+1/2)/S_K(chi); window = S_K(chi*-chi)/(S_K(chi*) S_K(chi));
  derive volume factor carefully (flat code is the K->0 anchor); preserve landmine-#7 masks.
- Wigner-d lensed-Cl machinery: untouched (geometry-free).

## Out of scope (documented, deliberate)
- Integer-nu discrete sum for closed (continuum approx, see above).
- CLASS's flat-rescaling approximation, WKB/Airy, Hermite tables (we are exact instead).
- TCA/RSA/UFA curvature variants (ABCMB has no approximation schemes).
- Vectors/tensors.
- Low-q grid densification for very open models (CLASS q_logstep_open): static-grid knob,
  add later if low-l accuracy vs CLASS demands it.

## Validation plan (accuracy first)
1. Unit: recurrence vs scipy j_l at K=0; vs mpmath high-precision forward recurrence and the
   closed-universe Gegenbauer closed form for K != 0.
2. omega_k=0 regressions: curvature=False byte-identical to main; curvature=True vs flat path
   (differences bounded by flat-table interpolation error).
3. CLASS (classy) comparison at Omega_k = +/-0.01, +/-0.05: TT/TE/EE/PP + P(k), thresholds
   matching pytests/accuracy_test.py.
4. GPU timing: flat-path no-regression + curved-path cost (short salloc, released promptly).

## Implementation status (2026-06-12)
- All sections above implemented on branch worktree-curvature (3 commits:
  notes, background+perturbations, spectrum+lensing).
- Validation scripts: test_hyper_unit.py (CPU; helpers + recurrence vs
  mpmath/scipy), smoke_trace.py (CPU; full pipeline trace at l_max=250 for
  flat/curved x open/closed), test_flat_vs_curved_path.py (GPU; K=0 path
  consistency + timings), test_curved_vs_class.py <Omega_k> (classy
  comparison; target 2e-3, ell=2 exempt).
- Flat-path regression guarantee: with curvature=False and omega_k=0 every
  new factor is exactly 1.0 or +0.0 in the same evaluation order (the
  truncation uses a multiplicative g factor for this reason); the spectrum
  source block was refactored (built once in get_Cl instead of per-ell) which
  is mathematically identical.
