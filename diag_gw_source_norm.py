"""Leading hypothesis for the bulk of the low-l tensor-BB excess: a small
(~0.1%) bias in the GW-SOURCE density normalization.

The GW source is  S = -sqrt6 * 4 a^2 * rho_unit * (rho_g*(..) + rho_u*(..)),
rho_unit = 8 pi G / 3 / c_Mpc^2. A constant ~0.1% bias in rho_unit*rho is
amplified for low-k (source-dominated, super-horizon) modes and washes out at
high k where -k^2 h dominates -> biases gwdot -> Pi -> BB largest at low l,
decaying to 0 at high l. Matches the observed shape, EE~BB, TT-less, h-exact.

CLASS GW source uses a^2 * (.)rho_g * (..)  with (.)rho == 8 pi G/3 rho (its
background, units Mpc^-2). So compare:
    ABCMB  rho_unit*rho_g(lna)   vs   CLASS  (.)rho_g(z)
    ABCMB  rho_unit*rho_u(lna)   vs   CLASS  (.)rho_ur(z)
and the ratio rho_u/rho_g (sets the neutrino damping). A ~0.1% offset here is
the smoking gun.

Background only (no PE). GPU build + one CLASS background run. ~3 min warm.
"""
import os
os.environ["JAX_PLATFORM_NAME"] = "gpu"
import sys
file_dir = "/pscratch/sd/c/carag/ABCMB-bmodes"
sys.path.insert(0, file_dir); sys.path.insert(0, file_dir + "/pytests")
import jax
from jax import vmap
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
from classy import Class
from abcmb.main import Model
from abcmb import constants as cnst
from accuracy_test_bb import PARAMS, ELLMAX

model = Model(l_max=ELLMAX, lensing=False, tensors=True, l_max_g=12, l_max_pol_g=10)
out = model(PARAMS)
BG, params = out.BG, out.params
TPE = model.TPE
ig = model.species_dict["Photon"]
photon = model.species_list[ig]
rho_unit = 8. * np.pi * cnst.G / 3. / cnst.c_Mpc_over_s**2
print(f"ABCMB done; rho_unit={rho_unit:.6e}", flush=True)

z = np.logspace(2, 7, 400)          # 1e2 .. 1e7, radiation+matter era
lna = jnp.asarray(-np.log(1.0 + z))
rho_g_ab = rho_unit * np.asarray(vmap(photon.rho, in_axes=[0, None])(lna, params))
rho_u_ab = rho_unit * np.asarray(vmap(TPE.rho_relativistic, in_axes=[0, None])(lna, params))

# Match ABCMB's TCMB (TCMB0 is in eV in PARAMS); convert eV->K.
TCMB_K = float(params["TCMB0"]) / 8.617333262e-5   # eV -> K
M = Class()
M.set({
    "output": "tCl", "modes": "t", "r": PARAMS["r"], "n_t": -0.0127075,
    "l_max_tensors": 500, "lensing": "no", "T_cmb": TCMB_K,
    "H0": PARAMS["h"] * 100, "omega_b": PARAMS["omega_b"],
    "omega_cdm": PARAMS["omega_cdm"], "A_s": PARAMS["A_s"],
    "N_ur": PARAMS["Neff"], "YHe": PARAMS["YHe"], "N_ncdm": 0,
})
M.compute()
bg = M.get_background()
print(f"CLASS done; TCMB_K={TCMB_K:.6f}", flush=True)
print("CLASS background keys:", [k for k in bg if "rho" in k.lower()], flush=True)
zc = np.asarray(bg["z"]); o = np.argsort(zc)
def bgi(key):
    return np.interp(z, zc[o], np.asarray(bg[key])[o])
rho_g_cl = bgi("(.)rho_g")
rho_u_cl = bgi("(.)rho_ur")
M.struct_cleanup(); M.empty()

print("\n--- GW-source density coefficients: ABCMB vs CLASS ---")
print("    z     | rho_unit*rho_g rel | rho_unit*rho_u rel | (rho_u/rho_g) rel")
for zi in [1e2, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6, 1e7]:
    j = int(np.argmin(np.abs(z - zi)))
    rg = rho_g_ab[j] / rho_g_cl[j] - 1
    ru = rho_u_ab[j] / rho_u_cl[j] - 1
    rr = (rho_u_ab[j] / rho_g_ab[j]) / (rho_u_cl[j] / rho_g_cl[j]) - 1
    print(f" {z[j]:.3e} |   {rg:+.4e}     |   {ru:+.4e}     |   {rr:+.4e}")

print("\n[summary] over z in [1e2, 1e7]:")
print(f"  rho_g coeff:  mean rel = {np.mean(rho_g_ab/rho_g_cl-1):+.4e}, "
      f"max|rel| = {np.max(np.abs(rho_g_ab/rho_g_cl-1)):.4e}")
print(f"  rho_u coeff:  mean rel = {np.mean(rho_u_ab/rho_u_cl-1):+.4e}, "
      f"max|rel| = {np.max(np.abs(rho_u_ab/rho_u_cl-1)):.4e}")
print(f"  rho_u/rho_g:  mean rel = {np.mean((rho_u_ab/rho_g_ab)/(rho_u_cl/rho_g_cl)-1):+.4e}")
print("DONE", flush=True)
