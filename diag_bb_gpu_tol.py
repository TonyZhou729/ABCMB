"""GPU tolerance/cost study for the tensor solver.

For each (rtol_ten, atol_ten): warm forward time and tensor BB at shared
ell nodes (lensing off, so ClBB at a node = the raw computed node value).
Reference values from the CPU tight-tolerance run (diag_bb_tol.py).
"""
import sys, time
file_dir = '/pscratch/sd/c/carag/ABCMB-bmodes'
sys.path.insert(0, file_dir)
import jax
jax.config.update("jax_enable_x64", True)
print(jax.devices())
from abcmb.main import Model
sys.path.insert(0, file_dir + '/pytests')
from accuracy_test_bb import PARAMS

REF = {237: 1.988653e-20, 296: 1.073287e-20, 450: 2.158397e-21,
       490: 1.256664e-21}
NODES = [237, 296, 450, 490]

configs = {
    "off":              None,
    "T1_1e-5_1e-9":     (1.e-5, 1.e-9),
    "T2_1e-6_1e-10":    (1.e-6, 1.e-10),
    "T3_1e-6_1e-11":    (1.e-6, 1.e-11),
}

for name, tol in configs.items():
    kw = dict(l_max=2500, lensing=False, l_max_g=12, l_max_pol_g=10)
    if tol is None:
        kw["tensors"] = False
    else:
        kw["tensors"] = True
        kw["rtol_ten"], kw["atol_ten"] = tol
    model = Model(**kw)
    p = dict(PARAMS)
    times = []
    for i in range(3):
        t0 = time.time()
        out = model(p)
        out.ClBB.block_until_ready()
        times.append(time.time() - t0)
    line = f"{name:16s} warm={min(times[1:]):.2f}s"
    if tol is not None:
        for l in NODES:
            v = float(out.ClBB[l - 2])
            line += f"  l{l}:{v/REF[l]-1.:+.2e}"
    print(line)
print("DONE")
