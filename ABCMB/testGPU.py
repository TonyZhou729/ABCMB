import numpy as np 
import sys
import time
import jax

import os
os.environ['XLA_FLAGS']='--xla_cpu_enable_xprof_traceme'

sys.path.append('..')

from JaxCMB.main import Model
# from jax.profiler import ProfileOptions

# opts = ProfileOptions()
# opts.host_tracer_level   = 0   # drop all host/thread events
# opts.python_tracer_level = 1   # drop Python-level call events
# opts.device_tracer_level = 0  # only record high-level kernel-launch events



params = {
    'h': 0.6762,
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
    'm_ncdm': 0.06,
}

# cosmo = Model(has_MasslessNeutrinos=True)
cosmo = Model() # faster for debugging
pt, bg = jax.block_until_ready(cosmo.get_PTBG(params))
# res = jax.block_until_ready(cosmo.run_cosmology(params))

for i in range(5):
    start = time.time()
    pt,bg = jax.block_until_ready(cosmo.get_PTBG(params))
    # res = jax.block_until_ready(cosmo.run_cosmology(params))
    print(time.time() - start)


# jax.profiler.start_trace("/home/cg3566/profile-data")#,profiler_options=opts)
# pt, bg = jax.block_until_ready(cosmo.get_PTBG())
# pt, bg = jax.block_until_ready(cosmo.get_PTBG())
# jax.profiler.stop_trace()

# jax.profiler.start_trace("/home/cg3566/profile-data2")
# pt, bg = jax.block_until_ready(jit_get_PTBG())
# jax.profiler.stop_trace()


