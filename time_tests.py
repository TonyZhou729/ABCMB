from classy import Class

import sys
sys.path.append('../')
# sys.path.append('../JaxCMB')
# print(sys.path)

import os
# os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
# print(os.getcwd())

import sys

# assert "jax" not in sys.modules, "jax already imported: you must restart your runtime"
# os.environ['XLA_FLAGS'] = "--xla_force_host_platform_device_count=8"

import jax
print(jax.devices())
# import jax
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

import time

h = 0.6762

for i in range(2):
    start=time.time()
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

    model = Model(ellmin=2, ellmax=2500, lensing=False)

    ABC_Cl = model.run_cosmology(params)[0]
    ABC_ell = model.SS.ells

    print(ABC_Cl)
    print(time.time()-start)
