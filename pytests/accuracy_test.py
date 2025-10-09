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
    ellmin = 2
    ellmax = 2500
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

        model = Model(ellmin=ellmin, ellmax=ellmax, lensing=False) # ZZ: model now takes ellmin, ellmax for Cls, and want_lensing
        ABC_Cls = model.run_cosmology(params)
        ABC_tt = ABC_Cls[0] 
        ABC_te = ABC_Cls[1] 
        ABC_ee = ABC_Cls[2] 
        ABC_ell = model.SS.ells # SpectrumSolver now automatically computes ells between specified ellmin and ellmax


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
            'output':'mPk, tCl, pCl',
            'lensing':'no',
            'P_k_max_h/Mpc':1.0
        }

        CLASS_Model = Class()
        CLASS_Model.set(CLASS_params)

        CLASS_Model.compute()
        cl = CLASS_Model.raw_cl(ellmax)
        cltt=cl["tt"][ellmin:]
        ell = cl["ell"][ellmin:]

        # Compare all ells
        err_tt = abs(cltt-ABC_tt)/cltt
        print(err_tt.max())

        assert max(err_tt) <= 0.21, f"Accuracy check failed: {err_tt}"
    
    except Exception as e:
        pytest.fail(f"accuracy_checks raised an exception: {e}")

#print(test_accuracy_checker())