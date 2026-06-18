"""One-off diagnostic: BB error profile vs l for the raw (unlensed) config."""
from classy import Class
import os
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
import sys
file_dir = os.path.dirname(__file__)
sys.path.insert(0, file_dir)
import jax
jax.config.update("jax_enable_x64", True)
from abcmb.main import Model
import numpy as np

sys.path.insert(0, file_dir + '/pytests')
from accuracy_test_bb import PARAMS, R_TENSOR, ELLMIN, ELLMAX

model = Model(l_max=ELLMAX, lensing=False, tensors=True, l_max_g=12, l_max_pol_g=10)
output = model(PARAMS)

CLASS_Model = Class()
CLASS_Model.set({
    "output": "tCl, pCl",
    "modes": "s,t",
    "r": R_TENSOR,
    "l_max_scalars": ELLMAX,
    "l_max_tensors": model.specs["l_tensor_max"],
    "lensing": "no",
    "H0": PARAMS["h"] * 100,
    "omega_b": PARAMS["omega_b"],
    "omega_cdm": PARAMS["omega_cdm"],
    "A_s": PARAMS["A_s"],
    "n_s": PARAMS["n_s"],
    "N_ur": PARAMS["Neff"],
    "YHe": PARAMS["YHe"],
    "N_ncdm": 0,
    "reio_parametrization": "reio_camb",
    "tau_reio": PARAMS["tau_reion"],
    "reionization_width": PARAMS["Delta_z_reion"],
    "helium_fullreio_redshift": PARAMS["z_reion_He"],
    "helium_fullreio_width": PARAMS["Delta_z_reion_He"],
    "reionization_exponent": PARAMS["exp_reion"],
    "l_max_g": 12,
    "l_max_pol_g": 10,
    "l_max_ur": 17,
})
CLASS_Model.compute()
cl = CLASS_Model.raw_cl(ELLMAX)

ells = np.arange(ELLMIN, ELLMAX + 1)
ours = np.asarray(output.ClBB)
theirs = np.asarray(cl["bb"][ELLMIN:])

np.savez(file_dir + "/diag_bb_profile.npz", ells=ells, abcmb_bb=ours,
         class_bb=theirs, abcmb_tt=np.asarray(output.ClTT),
         class_tt=np.asarray(cl["tt"][ELLMIN:]),
         abcmb_ee=np.asarray(output.ClEE),
         class_ee=np.asarray(cl["ee"][ELLMIN:]),
         abcmb_te=np.asarray(output.ClTE),
         class_te=np.asarray(cl["te"][ELLMIN:]))

mask = theirs != 0.
err = np.where(mask, np.abs(ours - theirs) / np.abs(np.where(mask, theirs, 1.)), 0.)

print("l, BB_abcmb, BB_class, rel_err")
for l in [2, 3, 5, 10, 20, 50, 90, 100, 150, 200, 250, 300, 350, 380, 400,
          410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510]:
    i = l - ELLMIN
    print(f"{l:5d}  {ours[i]: .5e}  {theirs[i]: .5e}  {err[i]:.4f}")

for lo, hi in [(2, 50), (50, 100), (100, 200), (200, 300), (300, 400),
               (400, 450), (450, 500)]:
    sel = (ells >= lo) & (ells < hi)
    print(f"max rel err in [{lo},{hi}): {err[sel].max():.4f} at l={ells[sel][err[sel].argmax()]}")

# TE sign/profile check (tensor TE has the CLASS historical sign convention)
te_o = np.asarray(output.ClTE)
te_c = np.asarray(cl["te"][ELLMIN:])
sel = ells <= 450
print("tensor-region TE max abs diff / max abs TE:",
      np.abs(te_o - te_c)[sel].max() / np.abs(te_c[sel]).max())
print("DONE")
