from classy import Class

import sys
sys.path.append('../')
# sys.path.append('../ABCMB')
# print(sys.path)

import os
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
# print(os.getcwd())
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)
from ABCMB.main import Model
import ABCMB.spectrum as spectrum
from scipy.interpolate import interp1d
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import pytest
import numpy as np
np.seterr(all='raise') 

def test_accuracy_checker(h = 0.6762):
    try:
        # ABCMB:
        params = {
            'h': h,
            'omega_cdm': 0.1193,
            'omega_b': 0.0225,
            'A_s': 2.12424e-9,
            'n_s': 0.9709,
            'Neff': 3.044,
            'YHe': 0.245,
            'TCMB0': 2.34865418e-4,
            'T_nu': (4. / 11.)**(1. / 3.) * 2.34865418e-4,
            'N_ncdm': 0,
            'T_ncdm': 0.71611 * 2.34865418e-4,
        }

        model = Model(ellmin=2, ellmax=2000, lensing=False, has_MasslessNeutrinos=True) # ZZ: model now takes ellmin, ellmax for Cls, and want_lensing
        k = jnp.logspace(-3., 0., 200, base=10)
        k_np = np.logspace(-3., 0., 200, base=10)
        #PT, BG = model.get_PTBG(params)
        #SS = spectrum.SpectrumSolver(switch_sw=1., switch_isw=1., switch_dop=1., switch_pol=1.)
        #SS = model.SS

        #idxs = jnp.arange(18, 80) # Only compute at tabulated l positions. Future: Adjust l_max # CG: 80 for l = 2000
        #ABC_Cl = SS.get_Cl(idxs, PT, BG)[0] # [0] is TT
        #ABC_ell = spectrum.bessel_l_tab[idxs]
        ABC_Cls = model.run_cosmology(params)
        ABC_Cl = ABC_Cls[0] # [0] is TT
        ABC_ell = model.SS.ells # SpectrumSolver now automatically computes ells between specified ellmin and ellmax

        print(ABC_Cl)


        # CLASS:
        CLASS_params = {
            'h': params["h"],
            'omega_cdm': params['omega_cdm'],
            'omega_b': params['omega_b'],
            'A_s': params['A_s'],
            'n_s': params['n_s'],
            'N_ur' : params['Neff'],
            'YHe': params['YHe'],
            'N_ncdm': 0,
            'output':'mPk, tCl, pCl, lCl',
            'lensing':'no',
            'P_k_max_h/Mpc':1.0
        }

        CLASS_Model = Class()
        CLASS_Model.set(CLASS_params)

        CLASS_Model.compute()
        cl = CLASS_Model.raw_cl(2000)
        cltt=cl["tt"][2:]
        ell = cl["ell"][2:]
        print(cltt)


        ABC_interp = interp1d(np.asarray(ABC_ell),np.asarray(ABC_ell * (ABC_ell + 1)/2* ABC_Cl),'cubic')
        CLASS_interp = interp1d(ell,ell*(ell+1)/2 * cltt,'cubic')

        diff = []
        for ell in [100,300,1000]:
            diff = np.append(diff,np.abs((ABC_interp(ell) - CLASS_interp(ell))/CLASS_interp(ell)))

        assert max(diff) <= 0.21, f"Accuracy check failed: {diff}"
    
    except Exception as e:
        pytest.fail(f"accuracy_checks raised an exception: {e}")

#print(test_accuracy_checker())