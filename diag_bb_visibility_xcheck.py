"""Is the polarization-source bias in the VISIBILITY g, or in the tensor
quadrupole Pi?

Established: low-l BB excess is real (CLASS converged), in source_E = sqrt6*g*Pi
(EE~BB, TT less), all grids converged, all tensor equations/ICs/radials match
CLASS verbatim. With identical equations+ICs, Pi can only differ through the
BACKGROUND fed to the source. source_E contains the VISIBILITY g (from HyRex
recomb). A g bias enters EE, BB, and TT's g*Pi term but NOT TT's -hdot*e^-kappa
term -> exactly the EE~BB, TT-less pattern. A ~0.2% g difference vs CLASS near
last scattering would be hidden under the scalar test's 1% bar.

Compare ABCMB's background visibility g(z), exp(-kappa), kappa'=1/tau_c, tau(z)
to CLASS thermodynamics (default + hp precision) near recombination. If g
matches -> bias is in Pi (tensor hierarchy). If g differs -> found it (a
recomb/visibility issue, shared with scalars but masked).

One model build (background only needed) + CLASS thermo. GPU, ~3 min warm.
"""
import os
os.environ["JAX_PLATFORM_NAME"] = "gpu"
import sys
file_dir = "/pscratch/sd/c/carag/ABCMB-bmodes"
sys.path.insert(0, file_dir)
sys.path.insert(0, file_dir + "/pytests")
import jax
from jax import vmap
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
from classy import Class
from abcmb.main import Model
from accuracy_test_bb import PARAMS, ELLMAX, class_tensor_hp_reference

# ---- ABCMB background ----
model = Model(l_max=ELLMAX, lensing=False, tensors=True,
              l_max_g=12, l_max_pol_g=10)
output = model(PARAMS)
BG, params = output.BG, output.params
print("ABCMB BG done", flush=True)

z = np.concatenate([np.linspace(200, 700, 60),
                    np.linspace(700, 1500, 240),   # dense across last scattering
                    np.linspace(1500, 3000, 60)])
lna = jnp.asarray(-np.log(1.0 + z))
g_ab = np.asarray(vmap(BG.visibility, in_axes=[0, None])(lna, params))
emk_ab = np.asarray(vmap(BG.expmkappa)(lna))
tauc_ab = np.asarray(vmap(BG.tau_c, in_axes=[0, None])(lna, params))
kp_ab = 1.0 / tauc_ab                      # kappa' = 1/tau_c
tau_ab = np.asarray(vmap(BG.tau)(lna))
tau0_ab = float(BG.tau0)

# ---- CLASS thermodynamics (default + hp) ----
def class_thermo(extra):
    M = Class()
    cfg = {
        "output": "tCl,pCl", "modes": "t", "r": PARAMS["r"], "n_t": -0.0127075,
        "l_max_tensors": 500, "lensing": "no",
        "H0": PARAMS["h"] * 100, "omega_b": PARAMS["omega_b"],
        "omega_cdm": PARAMS["omega_cdm"], "A_s": PARAMS["A_s"],
        "N_ur": PARAMS["Neff"], "YHe": PARAMS["YHe"], "N_ncdm": 0,
        "reio_parametrization": "reio_camb", "tau_reio": PARAMS["tau_reion"],
        "reionization_width": PARAMS["Delta_z_reion"],
        "helium_fullreio_redshift": PARAMS["z_reion_He"],
        "helium_fullreio_width": PARAMS["Delta_z_reion_He"],
        "reionization_exponent": PARAMS["exp_reion"],
    }
    cfg.update(extra)
    M.set(cfg)
    M.compute()
    th = M.get_thermodynamics()
    tau0 = M.get_current_derived_parameters(['conformal_age'])['conformal_age']
    return th, tau0, M

HP = {"k_step_sub": 0.005, "k_step_super": 0.0002, "q_linstep": 0.05,
      "perturbations_sampling_stepsize": 0.02,
      "tol_perturbations_integration": 1.e-7,
      "tight_coupling_trigger_tau_c_over_tau_h": 0.0015,
      "tight_coupling_trigger_tau_c_over_tau_k": 0.001,
      "start_small_k_at_tau_c_over_tau_h": 0.00015,
      "radiation_streaming_trigger_tau_over_tau_k": 1.e4}

th, tau0_cl, M = class_thermo(HP)
print("CLASS thermo keys:", list(th.keys()), flush=True)
print(f"tau0: ABCMB={tau0_ab:.4f}  CLASS={tau0_cl:.4f}  "
      f"rel={tau0_ab/tau0_cl-1:+.3e}", flush=True)

zc = np.asarray(th["z"])
order = np.argsort(zc)
zc = zc[order]
def cl_interp(key):
    return np.interp(z, zc, np.asarray(th[key])[order])

# find the CLASS visibility key
gkey = next(k for k in th if k.strip().lower().startswith("g "))
emkey = next(k for k in th if "exp(-kappa)" in k)
g_cl = cl_interp(gkey)
emk_cl = cl_interp(emkey)

# normalization check: integral of g dtau (should be ~1 both)
tauc = cl_interp("conf. time [Mpc]")
def trap_int(g_, tau_):
    o = np.argsort(tau_); return np.trapezoid(g_[o], tau_[o])
print(f"\n[norm] int g dtau:  ABCMB={trap_int(g_ab, tau_ab):.5f}  "
      f"CLASS={trap_int(g_cl, tauc):.5f}", flush=True)

# peak location and value
ip_a, ip_c = g_ab.argmax(), g_cl.argmax()
print(f"[peak] ABCMB z*={z[ip_a]:.2f} g={g_ab[ip_a]:.5e} | "
      f"CLASS z*={z[ip_c]:.2f} g={g_cl[ip_c]:.5e}")

print("\n--- visibility g(z) and exp(-kappa): ABCMB vs CLASS-hp ---")
print("   z   |   g_ABCMB     g_CLASS    g rel  |  emk rel | tau rel")
for zi in [400, 700, 900, 1000, 1050, 1080, 1100, 1150, 1200, 1300, 1500, 2000]:
    j = int(np.argmin(np.abs(z - zi)))
    grel = g_ab[j] / g_cl[j] - 1 if g_cl[j] != 0 else float("nan")
    erel = emk_ab[j] / emk_cl[j] - 1
    trel = tau_ab[j] / cl_interp("conf. time [Mpc]")[j] - 1
    print(f" {z[j]:6.1f}| {g_ab[j]: .4e}  {g_cl[j]: .4e}  {grel:+.2e} | "
          f"{erel:+.2e} | {trel:+.2e}")

# integrated visibility-weighted check: does g differ in a way that biases
# polarization? print max |g rel| over the last-scattering window
win = (z > 900) & (z < 1300)
print(f"\n[summary] over last-scattering window 900<z<1300:")
print(f"  max |g_ABCMB/g_CLASS - 1| = {np.nanmax(np.abs(g_ab[win]/g_cl[win]-1)):.3e}")
print(f"  mean (g_ABCMB/g_CLASS - 1) = {np.nanmean(g_ab[win]/g_cl[win]-1):+.3e}")
M.struct_cleanup(); M.empty()
print("DONE", flush=True)
