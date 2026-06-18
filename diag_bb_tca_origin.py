"""Confirm the TCA-origin hypothesis for the residual ~0.1% Pi (-> +0.4% low-l
BB) difference.

CLASS uses the tight-coupling approximation (TCA) before recomb: it reports the
photon quadrupole shear_g, pol2, l4_g as ZERO (closed-form) and DROPS the
photon term from the GW source until TCA ends. ABCMB integrates the full stiff
hierarchy with those small-but-nonzero quadrupoles throughout. Hypothesis: the
+0.1% Pi at recomb is established PRE-recomb, during tight coupling, from this
treatment difference.

Compare Pi(z) and shear_g(z) ABCMB vs CLASS across the TCA era (z~8000) into
recomb (z~1100), for 4 k. Detect CLASS's TCA-exit (first z, scanning from high
z, where CLASS shear_g departs from 0). If the Pi ratio is ALREADY ~+0.1% at /
just after TCA-exit and persists to recomb -> origin is pre-recomb/TCA. If Pi
matches at TCA-exit and diverges only into recomb -> not TCA.

GPU: one model build + 4 ABCMB single-k solves + one CLASS k_output run. ~4 min.
"""
import os
os.environ["JAX_PLATFORM_NAME"] = "gpu"
import sys
file_dir = "/pscratch/sd/c/carag/ABCMB-bmodes"
sys.path.insert(0, file_dir); sys.path.insert(0, file_dir + "/pytests")
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
from classy import Class
from abcmb.main import Model
from accuracy_test_bb import PARAMS, ELLMAX

SQRT6 = np.sqrt(6.)
model = Model(l_max=ELLMAX, lensing=False, tensors=True, l_max_g=12, l_max_pol_g=10)
out = model(PARAMS)
BG, params = out.BG, out.params
TPE = model.TPE
tau0 = float(BG.tau0)
NF = model.specs["l_max_g_ten"] + 1
print(f"ABCMB BG done; tau0={tau0:.2f}", flush=True)

ks_grid = np.asarray(TPE.k_axis_tensor)
targets = {10: 10 / tau0, 20: 20 / tau0, 40: 40 / tau0, 80: 80 / tau0}
kpick = {L: float(ks_grid[np.argmin(np.abs(ks_grid - kt))]) for L, kt in targets.items()}
print("k picks:", {L: f"{k:.4e}" for L, k in kpick.items()}, flush=True)

def pi_from(dg, sh, F4, G0, G2, G4):
    return -1. / SQRT6 * (dg / 10. + 2. / 7. * sh + 3. / 70. * F4
                          - 3. / 5. * G0 + 6. / 7. * G2 - 3. / 70. * G4)

lna_ab = jnp.linspace(float(BG.lna_transfer_start), 0., 6000)
z_ab = np.exp(-np.asarray(lna_ab)) - 1.0
abcmb = {}
for L, k in kpick.items():
    y = np.asarray(TPE.evolution_one_k(k, lna_ab, (BG, params)))
    P2 = pi_from(y[:, 0], y[:, 2], y[:, 4], y[:, NF], y[:, NF + 2], y[:, NF + 4])
    abcmb[L] = dict(z=z_ab, P2=P2, shear=y[:, 2], gwdot=y[:, -1], gw=y[:, -2])
    print(f"  ABCMB k(l={L}) done", flush=True)

kvals = ",".join(f"{kpick[L]:.8e}" for L in sorted(kpick))
M = Class()
M.set({
    "output": "tCl,pCl", "modes": "t", "r": PARAMS["r"], "n_t": -0.0127075,
    "l_max_tensors": 500, "lensing": "no", "k_output_values": kvals,
    "H0": PARAMS["h"] * 100, "omega_b": PARAMS["omega_b"],
    "omega_cdm": PARAMS["omega_cdm"], "A_s": PARAMS["A_s"],
    "N_ur": PARAMS["Neff"], "YHe": PARAMS["YHe"], "N_ncdm": 0,
    "reio_parametrization": "reio_camb", "tau_reio": PARAMS["tau_reion"],
    "reionization_width": PARAMS["Delta_z_reion"],
    "helium_fullreio_redshift": PARAMS["z_reion_He"],
    "helium_fullreio_width": PARAMS["Delta_z_reion_He"],
    "reionization_exponent": PARAMS["exp_reion"],
    "k_step_sub": 0.005, "k_step_super": 0.0002, "q_linstep": 0.05,
    "perturbations_sampling_stepsize": 0.02, "tol_perturbations_integration": 1.e-7,
    "tight_coupling_trigger_tau_c_over_tau_h": 0.0015,
    "tight_coupling_trigger_tau_c_over_tau_k": 0.001,
    "start_small_k_at_tau_c_over_tau_h": 0.00015,
    "radiation_streaming_trigger_tau_over_tau_k": 1.e4,
})
M.compute()
pt = M.get_perturbations()["tensor"]
cls = {}
for i, L in enumerate(sorted(kpick)):
    d = pt[i]
    z = 1.0 / np.asarray(d["a"]) - 1.0
    P2 = pi_from(np.asarray(d["delta_g"]), np.asarray(d["shear_g"]),
                 np.asarray(d["l4_g"]), np.asarray(d["pol0_g"]),
                 np.asarray(d["pol2_g"]), np.asarray(d["pol4_g"]))
    cls[L] = dict(z=z, P2=P2, shear=np.asarray(d["shear_g"]),
                  gwdot=np.asarray(d["Hdot (gwdot)"]))
M.struct_cleanup(); M.empty()

def interp_z(rec, key, zq):
    z = rec["z"]; o = np.argsort(z)
    return np.interp(zq, z[o], np.asarray(rec[key])[o])

for L in sorted(kpick):
    # detect CLASS TCA-exit: highest z where |shear_g| first becomes nonzero
    zc = cls[L]["z"]; shc = cls[L]["shear"]
    o = np.argsort(zc)[::-1]                  # high z -> low z
    nz = np.where(np.abs(shc[o]) > 0)[0]
    z_tca_exit = zc[o][nz[0]] if len(nz) else float("nan")
    print(f"\n===== l~{L}, k={kpick[L]:.4e}  (CLASS TCA-exit at z~{z_tca_exit:.0f}) =====")
    print("    z   |  Pi ratio  | gwdot ratio | shear_g: ABCMB / CLASS(0=TCA)")
    for zq in [4000, 3000, 2000, 1600, 1400, 1300, 1200, 1100, 1050, 1000, 950]:
        pir = interp_z(abcmb[L], "P2", zq) / interp_z(cls[L], "P2", zq)
        gdr = interp_z(abcmb[L], "gwdot", zq) / interp_z(cls[L], "gwdot", zq)
        sh_a = interp_z(abcmb[L], "shear", zq)
        sh_c = interp_z(cls[L], "shear", zq)
        tag = "  <-- TCA (CLASS shear=0)" if abs(sh_c) == 0 or zq > z_tca_exit else ""
        print(f" {zq:5d} |  {pir:+.5f} |  {gdr:+.5f}  | {sh_a:+.4e} / {sh_c:+.4e}{tag}")
print("DONE", flush=True)
