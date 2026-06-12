"""
Localize the curved-path blowup at K=0 (smoke showed ClTT ~ 1e13 vs 1e-12).

CPU, l_max=250. Compares, at table ells, the flat Cl_one_ell against the
curved _Cl_all_ells_curved output, and recomputes single-l emissions directly
(python loop, production formulas) with diagnostics on the radial functions.
"""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, file_dir + "/..")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from abcmb.main import Model
from abcmb import ABCMBTools as tools
import abcmb.spectrum as spectrum

base = {
    'h': 0.6762, 'omega_cdm': 0.1193, 'omega_b': 0.0225,
    'A_s': 2.12424e-9, 'n_s': 0.9709, 'Neff': 3.044, 'YHe': 0.245,
    'TCMB0': 2.34865418e-4, 'N_nu_massive': 0,
    "tau_reion": 0.0544, "Delta_z_reion": 0.5, "z_reion_He": 3.5,
    "Delta_z_reion_He": 0.5, "exp_reion": 1.5,
}

m = Model(l_max=250, lensing=False, curvature=True)
out = m(base)
params = m.add_derived_parameters(base)
PT, BG = out.PT, out.BG
SS = m.SS

sources = SS._transfer_sources(PT, BG, params)
(sT0, sT1, sT2, sE), aH_1d, tau, weights, tau0 = sources

# 1. flat vs curved per table ell
tt_c, te_c, ee_c = SS._Cl_all_ells_curved(sources, params)
tt_c = np.asarray(tt_c)
print("table-ell comparison (flat Cl_one_ell vs curved emission):")
ltab = np.asarray(spectrum.bessel_l_tab)
for idx in np.asarray(SS.lensing_ells_indices)[:12]:
    l = int(ltab[idx])
    f = float(SS.Cl_one_ell(int(idx), sources, params)[0])
    c = float(tt_c[l-2])
    print(f"  l={l:4d}: flat {f: .6e}  curved {c: .6e}  ratio {c/f: .3e}")

# 2. direct single-l recomputation with diagnostics
k_axis = np.asarray(SS.k_axis_transfer)
K = float(params['K'])
chi = np.asarray(tau0 - tau)[:, None]
q2 = k_axis**2 + K
q = np.sqrt(np.clip(q2, 1e-30, None))
s2 = np.sqrt(np.clip(1.-3.*K/k_axis**2, 1e-30, None))
sinK = np.asarray(tools.sin_K(jnp.array(chi), K))
cotK = np.asarray(tools.cot_K(jnp.array(chi), K))
uK = K*chi**2
qchi = q*chi
Phi0 = np.asarray(jnp.sinc(jnp.array(qchi)/jnp.pi)/tools._curv_f(jnp.array(uK)))
Phi1 = Phi0*np.asarray(tools._curv_g_diff(jnp.array(uK), jnp.array(qchi)**2))/(chi*k_axis)
s2d = np.sqrt(np.clip(q2-4.*K, 1e-30, None))
Phi2 = np.clip((3.*cotK*Phi1 - k_axis*Phi0)/s2d, -1e10, 1e10)

wa = (np.asarray(weights)/np.asarray(aH_1d))[:, None]
SW0, SW1, SW2, SWE = np.asarray(sT0)*wa, np.asarray(sT1)*wa, np.asarray(sT2)*wa, np.asarray(sE)*wa
dk = np.diff(k_axis)
wk = np.concatenate(([dk[0]/2.], (dk[1:]+dk[:-1])/2., [dk[-1]/2.]))
prim = wk*4.*np.pi*float(params['A_s'])*(k_axis/0.05)**(float(params['n_s'])-1.)/k_axis

xmin_all = np.asarray(SS.curv_xmin)
x_eff = q*sinK

Phi_lm1, Phi_l = Phi1, Phi2
for l in range(2, 251):
    lf = float(l)
    sld = np.sqrt(np.clip(q2-K*lf**2, 1e-30, None))
    if l in (5, 10, 30, 100, 250):
        dPhi = sld*Phi_lm1 - (lf+1.)*cotK*Phi_l
        d2Phi = -2.*cotK*dPhi + (lf*(lf+1.)/sinK**2 - q2 + K)*Phi_l
        mask = x_eff >= xmin_all[l-2]
        r0 = np.where(mask, Phi_l, 0.)
        r1 = np.where(mask, dPhi, 0.)/k_axis
        r2 = np.where(mask, 3.*d2Phi/k_axis**2 + Phi_l, 0.)/(2.*s2)
        epsf = np.sqrt(3./8.*(lf+2.)*(lf+1.)*lf*(lf-1.))
        rE = epsf/s2*np.where(mask, Phi_l/(k_axis*sinK)**2, 0.)
        T = (SW0*r0 + SW1*r1 + SW2*r2).sum(axis=0)
        E = (SWE*rE).sum(axis=0)
        cl = float((prim*T*T).sum())
        for nm, r in (("r0", r0), ("r1", r1), ("r2", r2), ("rE", rE)):
            am = np.unravel_index(np.abs(r).argmax(), r.shape)
            print(f"  l={l:4d} {nm}: max|{nm}| {np.abs(r).max():.3e} at (lna_i={am[0]}, k={k_axis[am[1]]:.3e}, x_eff={x_eff[am]:.3e}, xmin={xmin_all[l-2]:.3e})")
        am = np.abs(T).argmax()
        print(f"  l={l:4d}: ClTT direct {cl: .3e}; max|T| {np.abs(T).max():.3e} at k={k_axis[am]:.3e}")
    # advance
    slp_arg = q2 - K*(lf+1.)**2
    slpd = np.sqrt(np.clip(slp_arg, 1e-30, None))
    Phi_next = np.where(slp_arg > 0., np.clip(((2.*lf+1.)*cotK*Phi_l - sld*Phi_lm1)/slpd, -1e10, 1e10), 0.)
    Phi_lm1, Phi_l = Phi_l, Phi_next

print("done")
