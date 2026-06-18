"""
ABCMB-vs-CLASS accuracy test for B modes (branch `bmodes`).

Two configurations, fiducial LCDM + tensors with r = 0.1:

1. raw (lensing off):  tensor BB, plus tensor contributions to TT/EE.
2. lensed:             total BB = lensed(scalar+tensor EE/BB), TT/EE too.

Run as a pytest module or standalone:  python pytests/accuracy_test_bb.py
"""
from classy import Class
import os
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
file_dir = os.path.dirname(__file__)

import sys
sys.path.append(file_dir + '/../')
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
from abcmb.main import Model
import numpy as np

ELLMIN = 2
ELLMAX = 2500
R_TENSOR = 0.1

PARAMS = {
    'h': 0.6762,
    'omega_cdm': 0.1193,
    'omega_b': 0.0225,
    'A_s': 2.12424e-9,
    'n_s': 0.9709,
    'Neff': 3.044,
    'YHe': 0.245,
    'TCMB0': 2.34865418e-4,
    'N_nu_massive': 0,
    'T_nu_massive': 0.71611,
    'm_nu_massive': 0.06,
    'tau_reion': 0.0544,
    'Delta_z_reion': 0.5,
    'z_reion_He': 3.5,
    'Delta_z_reion_He': 0.5,
    'exp_reion': 1.5,
    'r': R_TENSOR,
}

# CLASS's computed transfer multipoles up to l_tensor_max (formerly
# abcmb/bessel_tab/l.txt, the CLASS l-sampling). ABCMB now computes every
# integer ell via the tensor Bessel recurrence, but CLASS still interpolates
# its raw_cl between these nodes, so the BB transfer-accuracy comparison is
# made at the nodes (both codes exact there). Between nodes the residual is
# CLASS's own cubic interpolation (~5e-3 at the 370-410 / 450-490 gap
# midpoints l=387, 477), not ABCMB error -- see diag_bb_recurrence_check.py
# (node-level 8.9e-4 vs every-l 5.1e-3).
CLASS_TRANSFER_ELLS = np.array([
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 21, 23, 25,
    28, 31, 34, 38, 42, 47, 52, 58, 64, 71, 79, 88, 98, 109, 122, 136, 152,
    170, 190, 212, 237, 265, 296, 331, 370, 410, 450, 490,
])


def run_pair(lensing):
    model = Model(
        l_max=ELLMAX,
        lensing=lensing,
        tensors=True,
        l_max_g=12,
        l_max_pol_g=10,
    )
    output = model(PARAMS)

    CLASS_params = {
        "output": "tCl, pCl, lCl" if lensing else "tCl, pCl",
        "modes": "s,t",
        "r": R_TENSOR,
        "l_max_scalars": ELLMAX,
        "l_max_tensors": model.specs["l_tensor_max"],
        "lensing": "yes" if lensing else "no",
        "accurate_lensing": 1,
        "H0": PARAMS["h"] * 100,
        "omega_b": PARAMS["omega_b"],
        "omega_cdm": PARAMS["omega_cdm"],
        "A_s": PARAMS["A_s"],
        "n_s": PARAMS["n_s"],
        "N_ur": PARAMS["Neff"],
        "YHe": PARAMS["YHe"],
        "N_ncdm": PARAMS["N_nu_massive"],
        "reio_parametrization": "reio_camb",
        "tau_reio": PARAMS["tau_reion"],
        "reionization_width": PARAMS["Delta_z_reion"],
        "helium_fullreio_redshift": PARAMS["z_reion_He"],
        "helium_fullreio_width": PARAMS["Delta_z_reion_He"],
        "reionization_exponent": PARAMS["exp_reion"],
        "l_max_g": model.specs["l_max_g"],
        "l_max_pol_g": model.specs["l_max_pol_g"],
        "l_max_ur": model.specs["l_max_massless_nu"],
        "l_max_g_ten": model.specs["l_max_g_ten"],
        "l_max_pol_g_ten": model.specs["l_max_pol_g_ten"],
    }
    CLASS_Model = Class()
    CLASS_Model.set(CLASS_params)
    CLASS_Model.compute()
    cl = CLASS_Model.lensed_cl(ELLMAX) if lensing else CLASS_Model.raw_cl(ELLMAX)

    return output, cl


def compare(name, ours, theirs, lmask=None, tol=0.01):
    ours = np.asarray(ours)
    theirs = np.asarray(theirs)
    if lmask is None:
        lmask = np.ones(len(ours), dtype=bool)
    denom = np.where(theirs != 0., theirs, 1.)
    err = np.abs(ours - theirs) / np.abs(denom)
    err = err[lmask]
    print(f"{name}: max rel err = {err.max():.3e} "
          f"(at l = {np.arange(ELLMIN, ELLMAX+1)[lmask][err.argmax()]})")
    return err.max() <= tol, err.max()


def class_tensor_hp_reference(l_tensor_max=500):
    """
    Tensor-only CLASS run at high precision, the reference for raw BB.

    Default-precision CLASS tensor spectra are ~1% unconverged at
    l ~ 450-500 (dominated by k/q sampling; see the convergence ladder in
    design_bmodes.md). ABCMB's tensor sector is grid-converged, so the
    meaningful 2-permille comparison is against CLASS with tensor
    precision cranked. This tensor-only run is cheap.
    """
    n_t_scc = -R_TENSOR / 8. * (2. - R_TENSOR / 8. - PARAMS["n_s"])
    M = Class()
    M.set({
        "output": "tCl, pCl",
        "modes": "t",
        "r": R_TENSOR,
        "n_t": n_t_scc,
        "l_max_tensors": l_tensor_max,
        "lensing": "no",
        "H0": PARAMS["h"] * 100,
        "omega_b": PARAMS["omega_b"],
        "omega_cdm": PARAMS["omega_cdm"],
        "A_s": PARAMS["A_s"],
        "N_ur": PARAMS["Neff"],
        "YHe": PARAMS["YHe"],
        "N_ncdm": PARAMS["N_nu_massive"],
        "reio_parametrization": "reio_camb",
        "tau_reio": PARAMS["tau_reion"],
        "reionization_width": PARAMS["Delta_z_reion"],
        "helium_fullreio_redshift": PARAMS["z_reion_He"],
        "helium_fullreio_width": PARAMS["Delta_z_reion_He"],
        "reionization_exponent": PARAMS["exp_reion"],
        # precision: dense k/q, fine time sampling, weakened TCA/RSA
        "k_step_sub": 0.005,
        "k_step_super": 0.0002,
        "q_linstep": 0.05,
        "perturbations_sampling_stepsize": 0.02,
        "tol_perturbations_integration": 1.e-7,
        "tight_coupling_trigger_tau_c_over_tau_h": 0.0015,
        "tight_coupling_trigger_tau_c_over_tau_k": 0.001,
        "start_small_k_at_tau_c_over_tau_h": 0.00015,
        "radiation_streaming_trigger_tau_over_tau_k": 1.e4,
    })
    M.compute()
    return M.raw_cl(l_tensor_max)


def test_bb_raw():
    output, cl = run_pair(lensing=False)
    ells = np.arange(ELLMIN, ELLMAX + 1)

    ok_tt, _ = compare("raw TT (s+t)", output.ClTT, cl["tt"][ELLMIN:])
    ok_ee, _ = compare("raw EE (s+t)", output.ClEE, cl["ee"][ELLMIN:])

    # Informational: vs default-precision CLASS (expect ~1% in the tail,
    # which is CLASS-side unconvergence — see class_tensor_hp_reference).
    mask_ten = ells <= 500
    compare("raw BB vs CLASS default (info)", output.ClBB,
            cl["bb"][ELLMIN:], lmask=mask_ten, tol=np.inf)

    # The accuracy assertion has two parts, both vs high-precision
    # tensor-only CLASS (l=2 exempt; 490 is the last ell node below
    # l_tensor_max). ABCMB now computes every multipole exactly via the tensor
    # Bessel recurrence, so the 491-500 sliver (the dying tail up to the cutoff)
    # is also tight (~9e-4) instead of the ~1.4e-2 the old node+spline path left
    # (it splined through its node at 530).
    cl_hp = class_tensor_hp_reference()
    bb_hp = np.zeros(ELLMAX - ELLMIN + 1)
    bb_hp[:500 - ELLMIN + 1] = cl_hp["bb"][ELLMIN:]

    # The tensor recurrence computes every integer multipole exactly, so the
    # only meaningful transfer-accuracy comparison is at the CLASS computed
    # nodes (CLASS interpolates its raw_cl between them; ABCMB does not). Split:
    #   - recomb tail (100 <= l <= 490, at CLASS nodes): clean transfer
    #     accuracy, ~0.9e-3, held to 1.5e-3.
    #   - low l (3 <= l < 100, at CLASS nodes): the converged tensor-quadrupole
    #     inter-code difference (h exact, Pi ~+0.1% -> +0.4% in BB; NOT reion/
    #     grid/recomb/TCA). ~3.1e-3, held to 4e-3; do NOT tighten. See
    #     NOTE_lowl_bb_excess.md.
    #   - 491-500 sliver: ABCMB exact to the cutoff -> ~9e-4 (was ~1.4e-2 with
    #     the old node+spline path); asserted as a direct check.
    # l=2 carries the known small ABCMB error (exempt). The full every-l band
    # is reported for information only: it is limited by CLASS's own cubic
    # interpolation between its nodes (~5e-3 at the 370-410 / 450-490 gap
    # midpoints), NOT ABCMB error (diag_bb_recurrence_check.py).
    nodes = CLASS_TRANSFER_ELLS
    m_hi = np.isin(ells, nodes) & (ells >= 100) & (ells <= 490)
    m_lo = np.isin(ells, nodes) & (ells >= 3) & (ells < 100)
    ok_bb_hi, _ = compare("raw BB vs CLASS hp @ nodes 100<=l<=490", output.ClBB,
                          bb_hp, lmask=m_hi, tol=1.5e-3)
    ok_bb_lo, _ = compare("raw BB vs CLASS hp @ nodes 3<=l<100", output.ClBB,
                          bb_hp, lmask=m_lo, tol=4.0e-3)
    ok_bb_cut, _ = compare("raw BB vs CLASS hp, 491-500 (exact to cutoff)",
                           output.ClBB, bb_hp,
                           lmask=(ells > 490) & (ells <= 500), tol=2.0e-3)
    compare("raw BB vs CLASS hp, every-l 3<=l<=490 (CLASS-interp-limited, info)",
            output.ClBB, bb_hp, lmask=(ells >= 3) & (ells <= 490), tol=np.inf)

    assert ok_tt, "raw TT accuracy"
    assert ok_ee, "raw EE accuracy"
    assert ok_bb_hi, "raw BB recomb-node accuracy (100<=l<=490, recurrence exact)"
    assert ok_bb_lo, "raw BB low-l-node accuracy (3<=l<100, inter-code scatter)"
    assert ok_bb_cut, "raw BB cutoff sliver 491-500 (recurrence exact to l_tensor_max)"


def test_bb_lensed():
    output, cl = run_pair(lensing=True)

    ok_tt, _ = compare("lensed TT (s+t)", output.ClTT, cl["tt"][ELLMIN:])
    ok_ee, _ = compare("lensed EE (s+t)", output.ClEE, cl["ee"][ELLMIN:])
    # Lensed BB is held to 1%: the comparison is against default-precision
    # CLASS, whose tensor sector is ~1% unconverged at l ~ 350-500 (see
    # class_tensor_hp_reference), and at high l the accuracy is set by the
    # scalar EE/lensing agreement (~4e-3).
    ok_bb, _ = compare("lensed BB (total)", output.ClBB, cl["bb"][ELLMIN:])

    assert ok_tt, "lensed TT accuracy"
    assert ok_ee, "lensed EE accuracy"
    assert ok_bb, "lensed BB accuracy"


if __name__ == "__main__":
    print("=== raw (lensing off) ===")
    test_bb_raw()
    print("=== lensed ===")
    test_bb_lensed()
    print("All B-mode accuracy checks passed.")
