"""Three-way low-l raw-BB: ABCMB vs default-CLASS vs hp-CLASS.

The low-l nodes show ABCMB ~0.3-0.4% high vs the hp (cranked-precision)
CLASS reference. This decides the fork: is ABCMB high, or is the cranked
hp reference corrupting low l? If default and hp CLASS AGREE at low l, the
difference is a robust ABCMB-vs-CLASS effect; if they DIVERGE, the test's
hp reference is the problem. Both CLASS runs use n_t=scc and CLASS's default
tensor_method (= massless_approximation, matching ABCMB). GPU.
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
from classy import Class
from abcmb.main import Model
from abcmb.spectrum import bessel_l_tab
from accuracy_test_bb import PARAMS, R_TENSOR, ELLMIN, ELLMAX, \
    class_tensor_hp_reference

LMAX_T = 500

model = Model(l_max=ELLMAX, lensing=False, tensors=True, l_max_g=12, l_max_pol_g=10)
abcmb_bb = np.asarray(model(PARAMS).ClBB)
print("ABCMB done", flush=True)

# CLASS tensor-only at DEFAULT precision (no cranking, no l_linstep).
n_t_scc = -R_TENSOR / 8. * (2. - R_TENSOR / 8. - PARAMS["n_s"])
Md = Class()
Md.set({
    "output": "tCl, pCl", "modes": "t", "r": R_TENSOR, "n_t": n_t_scc,
    "l_max_tensors": LMAX_T, "lensing": "no",
    "H0": PARAMS["h"] * 100, "omega_b": PARAMS["omega_b"],
    "omega_cdm": PARAMS["omega_cdm"], "A_s": PARAMS["A_s"],
    "N_ur": PARAMS["Neff"], "YHe": PARAMS["YHe"], "N_ncdm": 0,
    "reio_parametrization": "reio_camb", "tau_reio": PARAMS["tau_reion"],
    "reionization_width": PARAMS["Delta_z_reion"],
    "helium_fullreio_redshift": PARAMS["z_reion_He"],
    "helium_fullreio_width": PARAMS["Delta_z_reion_He"],
    "reionization_exponent": PARAMS["exp_reion"],
})
Md.compute()
bb_def = Md.raw_cl(LMAX_T)["bb"]
print("CLASS default done", flush=True)

bb_hp = class_tensor_hp_reference()["bb"]
print("CLASS hp done", flush=True)

nodes = np.asarray(bessel_l_tab)
nodes = nodes[(nodes >= ELLMIN) & (nodes <= 490)]

print("\n  l   |   ABCMB    | CLASS-def  |  CLASS-hp  | A/def-1   | "
      "A/hp-1    | def/hp-1")
for L in nodes[(nodes <= 100) | np.isin(nodes, [152, 237, 331, 450, 490])]:
    a, d, h = abcmb_bb[L - ELLMIN], bb_def[L], bb_hp[L]
    print(f" {L:4d} | {a:.4e} | {d:.4e} | {h:.4e} | "
          f"{a/d-1:+.3e} | {a/h-1:+.3e} | {d/h-1:+.3e}")

lo = nodes[nodes <= 100]
defhp = np.array([abs(bb_def[L]/bb_hp[L]-1) for L in lo])
adef = np.array([abs(abcmb_bb[L-ELLMIN]/bb_def[L]-1) for L in lo])
print(f"\nmax |def/hp - 1| over l<=100 nodes: {defhp.max():.3e} "
      f"at l={lo[defhp.argmax()]}")
print(f"max |ABCMB/def - 1| over l<=100 nodes: {adef.max():.3e} "
      f"at l={lo[adef.argmax()]}")
print("\nIf def/hp ~ 0 everywhere -> CLASS precisions agree -> the low-l "
      "excess is\na robust ABCMB-vs-CLASS effect, not an hp-reference "
      "artifact. DONE")
