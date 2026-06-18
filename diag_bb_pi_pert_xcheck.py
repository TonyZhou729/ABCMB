"""Perturbation-level Pi(k,tau) comparison: ABCMB vs CLASS, to localize the
k-dependent tensor polarization-quadrupole bias (epoch + which moment).

Established: BB excess is real (CLASS converged), in source_E = sqrt6*g*Pi, all
grids converged, all equations/ICs/radials/source-formula match CLASS verbatim.
g is k-independent but the bias DECAYS with l~k and vanishes by l~490 -> the
bias is in the k-dependent quadrupole Pi, largest for low-k (super-horizon-at-
recomb) modes.

CLASS exposes tensor moments (delta_g, shear_g, l4_g, pol0_g, pol2_g, pol4_g,
gw, gwdot) at chosen k with the SAME gw=1/sqrt6 IC as ABCMB, so Pi can be
compared ABSOLUTELY. Compare, for a few low k:
  - gw(z), gwdot(z): is the GW amplitude h matched?  (TT-clean said ~yes)
  - Pi = P2(z): the source. Ratio ABCMB/CLASS through last scattering.
  - shear_g, pol0_g, pol2_g: which moment carries the bias.

One model build + per-k ABCMB single-mode solves + one CLASS run with
k_output_values. GPU, ~4 min warm.
"""
import os
os.environ["JAX_PLATFORM_NAME"] = "gpu"
import sys
file_dir = "/pscratch/sd/c/carag/ABCMB-bmodes"
sys.path.insert(0, file_dir)
sys.path.insert(0, file_dir + "/pytests")
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
from classy import Class
from abcmb.main import Model
from accuracy_test_bb import PARAMS, ELLMAX

SQRT6 = np.sqrt(6.)

model = Model(l_max=ELLMAX, lensing=False, tensors=True,
              l_max_g=12, l_max_pol_g=10)
output = model(PARAMS)
BG, params = output.BG, output.params
TPE = model.TPE
tau0 = float(BG.tau0)
print(f"ABCMB BG done; tau0={tau0:.2f}", flush=True)

# target ells -> k; pick nearest ABCMB tensor-grid k
ks_grid = np.asarray(TPE.k_axis_tensor)
targets = {10: 10/tau0, 20: 20/tau0, 40: 40/tau0, 80: 80/tau0}
kpick = {L: float(ks_grid[np.argmin(np.abs(ks_grid - kt))])
         for L, kt in targets.items()}
print("k picks (Mpc^-1):", {L: f"{k:.5e}" for L, k in kpick.items()}, flush=True)

NF = model.specs["l_max_g_ten"] + 1   # 6
def p2_from_y(y):  # y: (Nlna, Ny)
    dg, sh, F4 = y[:, 0], y[:, 2], y[:, 4]
    G0, G2, G4 = y[:, NF], y[:, NF + 2], y[:, NF + 4]
    P2 = -1. / SQRT6 * (dg / 10. + 2. / 7. * sh + 3. / 70. * F4
                        - 3. / 5. * G0 + 6. / 7. * G2 - 3. / 70. * G4)
    return P2, sh, G0, G2

lna_ab = jnp.linspace(float(BG.lna_transfer_start), 0., 4000)
z_ab = np.exp(-np.asarray(lna_ab)) - 1.0
abcmb = {}
for L, k in kpick.items():
    y = np.asarray(TPE.evolution_one_k(k, lna_ab, (BG, params)))
    P2, sh, G0, G2 = p2_from_y(y)
    abcmb[L] = dict(z=z_ab, P2=P2, gw=y[:, -2], gwdot=y[:, -1],
                    shear_g=sh, pol0=G0, pol2=G2)
    print(f"  ABCMB k(l={L}) done", flush=True)

# ---- CLASS with k_output_values ----
kvals = ",".join(f"{kpick[L]:.8e}" for L in sorted(kpick))
M = Class()
M.set({
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
    "k_output_values": kvals,
    "k_step_sub": 0.005, "k_step_super": 0.0002, "q_linstep": 0.05,
    "perturbations_sampling_stepsize": 0.02,
    "tol_perturbations_integration": 1.e-7,
    "tight_coupling_trigger_tau_c_over_tau_h": 0.0015,
    "tight_coupling_trigger_tau_c_over_tau_k": 0.001,
    "start_small_k_at_tau_c_over_tau_h": 0.00015,
    "radiation_streaming_trigger_tau_over_tau_k": 1.e4,
})
M.compute()
pt = M.get_perturbations()["tensor"]
print("CLASS tensor pert keys:", list(pt[0].keys()), flush=True)

def key(d, *subs):
    for k in d:
        kl = k.lower()
        if all(s in kl for s in subs):
            return k
    raise KeyError(subs)

cls = {}
for i, L in enumerate(sorted(kpick)):
    d = pt[i]
    a = np.asarray(d[key(d, "a")]) if "a" in d else np.asarray(d["a"])
    a = np.asarray(d["a"])
    z = 1.0 / a - 1.0
    dg = np.asarray(d["delta_g"]); sh = np.asarray(d["shear_g"])
    F4 = np.asarray(d[key(d, "l4_g")]); G0 = np.asarray(d["pol0_g"])
    G2 = np.asarray(d["pol2_g"]); G4 = np.asarray(d[key(d, "pol4_g")])
    P2 = -1. / SQRT6 * (dg / 10. + 2. / 7. * sh + 3. / 70. * F4
                        - 3. / 5. * G0 + 6. / 7. * G2 - 3. / 70. * G4)
    cls[L] = dict(z=z, P2=P2, gw=np.asarray(d[key(d, "h (gw)")]),
                  gwdot=np.asarray(d[key(d, "hdot")]), shear_g=sh,
                  pol0=G0, pol2=G2)
M.struct_cleanup(); M.empty()

ZQ = [950, 1000, 1050, 1080, 1100, 1150, 1200, 1300]
def interp_z(rec, k, zq):
    z = rec["z"]; o = np.argsort(z)
    return np.interp(zq, z[o], np.asarray(rec[k])[o])

for L in sorted(kpick):
    print(f"\n===== l~{L}, k={kpick[L]:.4e} Mpc^-1 "
          f"(ABCMB/CLASS ratios) =====")
    print("   z   |   P2 ratio |  gw ratio | gwdot rat | shear_g r | pol2 ratio")
    for zq in ZQ:
        r = {}
        for q in ["P2", "gw", "gwdot", "shear_g", "pol2"]:
            a = interp_z(abcmb[L], q, zq)
            c = interp_z(cls[L], q, zq)
            r[q] = a / c if c != 0 else float("nan")
        print(f" {zq:5d} |  {r['P2']:+.5f} | {r['gw']:+.5f} | {r['gwdot']:+.5f} "
              f"| {r['shear_g']:+.5f} | {r['pol2']:+.5f}")
print("DONE", flush=True)
