"""
ABCMB (curvature=True) vs CLASS for non-flat cosmologies.

Usage:  python notes_curvature/test_curved_vs_class.py <Omega_k> [nolensing]
        e.g.  python test_curved_vs_class.py 0.01
              python test_curved_vs_class.py -0.05

Target: <= 2e-3 relative on TT/EE (ell=2 exempt per known small error there).
Mirrors pytests/accuracy_test.py conventions. classy must be CLASS >= 3
(curvature supported). Runtime: CLASS ~1-3 min (curved is slow in CLASS) +
ABCMB compile+run.
"""
import os, sys, time
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, file_dir + "/..")

Omega_k = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01
lensing = "nolensing" not in sys.argv[2:]
flat_path = "flat" in sys.argv[2:]   # control: use the table path (requires Omega_k = 0)
hiprec = "hiprec" in sys.argv[2:]    # crank CLASS precision (discriminates whose error)

from classy import Class
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
from abcmb.main import Model

h = 0.6762
ellmin, ellmax = 2, 2500
omega_k = Omega_k * h**2

params = {
    'h': h, 'omega_cdm': 0.1193, 'omega_b': 0.0225,
    'A_s': 2.12424e-9, 'n_s': 0.9709, 'Neff': 3.044, 'YHe': 0.245,
    'TCMB0': 2.34865418e-4, 'N_nu_massive': 0,
    "tau_reion": 0.0544, "Delta_z_reion": 0.5, "z_reion_He": 3.5,
    "Delta_z_reion_He": 0.5, "exp_reion": 1.5,
    'omega_k': omega_k,
}

print(f"=== Omega_k = {Omega_k:+.4f} (omega_k = {omega_k:+.6f}), lensing={lensing}, path={'flat' if flat_path else 'curved'} ===")

model = Model(
    l_max=ellmax,
    lensing=lensing,
    curvature=not flat_path,
    omega_k_ref=omega_k,
    l_max_g=12, l_max_pol_g=10,
)
full_params = model.add_derived_parameters(params)

CLASS_params = {
    "output": "mPk, tCl, pCl, lCl" if lensing else "mPk, tCl, pCl",
    "l_max_scalars": ellmax,
    "P_k_max_1/Mpc": 0.5,
    "lensing": "yes" if lensing else "no",
    "accurate_lensing": 1,
    "H0": h*100,
    "Omega_k": Omega_k,
    "omega_b": float(full_params["omega_b"]),
    "omega_cdm": float(full_params["omega_cdm"]),
    "A_s": float(full_params["A_s"]),
    "n_s": float(full_params["n_s"]),
    "N_ur": float(full_params["Neff"]),
    "YHe": float(full_params["YHe"]),
    "N_ncdm": 0,
    "reio_parametrization": "reio_camb",
    "tau_reio": params["tau_reion"],
    "reionization_width": params["Delta_z_reion"],
    "helium_fullreio_redshift": params["z_reion_He"],
    "helium_fullreio_width": params["Delta_z_reion_He"],
    "reionization_exponent": params["exp_reion"],
    "l_max_g": 12,
    "l_max_pol_g": 10,
    "l_max_ur": 17,
}

if hiprec:
    CLASS_params.update({
        "q_linstep": 0.11,
        "l_linstep": 10,
        "l_logstep": 1.026,
        "hyper_sampling_flat": 12.,
        "hyper_sampling_curved_low_nu": 14.,
        "hyper_sampling_curved_high_nu": 6.,
        "hyper_phi_min_abs": 1.e-12,
        # (hyper_flat_approximation_nu left at default: raising it breaks
        # CLASS's closed-universe harmonic spline indexing)
        "tol_perturbations_integration": 1.e-6,
        "perturbations_sampling_stepsize": 0.05,
    })
    print("CLASS high-precision settings ON")

t0 = time.time()
CLASS_Model = Class()
CLASS_Model.set(CLASS_params)
CLASS_Model.compute()
cl = CLASS_Model.lensed_cl(ellmax) if lensing else CLASS_Model.raw_cl(ellmax)
print(f"CLASS done in {time.time()-t0:.0f} s")
cltt = cl["tt"][ellmin:]
clee = cl["ee"][ellmin:]
clte = cl["te"][ellmin:]

t0 = time.time()
output = model(params)
jax.block_until_ready(output.ClTT)
print(f"ABCMB done in {time.time()-t0:.0f} s (incl. compile)")

ells = np.asarray(output.l)
ABC = {"TT": np.asarray(output.ClTT), "EE": np.asarray(output.ClEE),
       "TE": np.asarray(output.ClTE)}
CLA = {"TT": cltt, "EE": clee, "TE": clte}

worst = 0.
for nm in ("TT", "EE", "TE"):
    scale = np.abs(CLA[nm]) if nm != "TE" else np.sqrt(cltt*clee)
    rel = np.abs(ABC[nm] - CLA[nm]) / scale
    for lo, hi in ((2, 2), (3, 29), (30, 800), (801, 2500)):
        band = (ells >= lo) & (ells <= hi)
        mx = rel[band].max()
        am = ells[band][rel[band].argmax()]
        print(f"  {nm} rel diff, ell {lo:4d}-{hi:4d}: max {mx:.3e} (at l={am})")
        if lo > 2:
            worst = max(worst, mx)

# Clpp (Limber) vs CLASS, where lensing is on
if lensing and "pp" in cl:
    import jax.numpy as jnp
    clpp_abc = np.asarray(model.SS.lensing_Cl(jnp.asarray(ells, dtype=jnp.float64),
                                              output.PT, output.BG, full_params))
    clpp_cla = cl["pp"][ellmin:]
    relpp = np.abs(clpp_abc - clpp_cla)/np.abs(clpp_cla)
    for lo, hi in ((2, 29), (30, 800), (801, 2500)):
        band = (ells >= lo) & (ells <= hi)
        print(f"  PP rel diff, ell {lo:4d}-{hi:4d}: max {relpp[band].max():.3e} (at l={ells[band][relpp[band].argmax()]})")

# P(k) — compare only at k physical for this geometry (k > sqrt(|K|) open)
ABC_Pk = np.asarray(output.Pk)
ABC_k = np.asarray(output.k)
H100 = 3.33564095e-4
sqrtabsK = np.sqrt(abs(omega_k))*H100
kmin_phys = 1.2*sqrtabsK if Omega_k > 0 else (np.sqrt(8.*abs(omega_k))*H100 if Omega_k < 0 else 0.)
sel = ABC_k > max(kmin_phys, ABC_k.min())
CLA_Pk = np.vectorize(CLASS_Model.pk)(ABC_k[sel], 0.)
err_pk = np.abs(CLA_Pk - ABC_Pk[sel])/np.abs(CLA_Pk)
print(f"  Pk rel diff (k > {max(kmin_phys, ABC_k.min()):.2e}): max {err_pk.max():.3e}")

print(f"WORST (TT/EE/TE, ell>=3): {worst:.3e}  ({'PASS' if worst <= 2e-3 else 'ABOVE 2e-3'})")
