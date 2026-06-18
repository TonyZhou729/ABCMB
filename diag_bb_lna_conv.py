"""Is the low-l BB excess a TIME-grid (Nlna_ten) under-resolution of the
recombination visibility peak?

Established: the excess is real on ABCMB's side (CLASS converged), lives in the
polarization quadrupole source g*Pi (EE~BB, TT less), and every hierarchy
equation / source / radial / background matches CLASS verbatim. The ONLY grid
axis never tested is the lna (time) grid. The recomb visibility g has width
dlna~0.15; the source/trapezoid grid (Nlna_ten=500 over [lna_transfer_start,0],
dlna~0.03) samples that peak with only ~5 pts. A coarse trapezoid over the
g*Pi peak biases the polarization integral, and would be HIDDEN in scalar EE
(held to 1% in the test). Both the source-table resolution AND the time
trapezoid scale with Nlna_ten.

Re-sample the SAME adaptive PE solution at finer Nlna (reuse model.TPE/TSS, no
rebuild) and re-integrate. If low-l BB drops toward CLASS as Nlna grows ->
found it.

Cost: one base build + 4 PE re-samples (SaveAt grid changes -> recompile each).
GPU, a few minutes warm.
"""
import os
os.environ["JAX_PLATFORM_NAME"] = "gpu"
import sys
file_dir = "/pscratch/sd/c/carag/ABCMB-bmodes"
sys.path.insert(0, file_dir)
sys.path.insert(0, file_dir + "/pytests")
import jax
from jax import vmap
jax.config.update("jax_enable_x64", True)
import numpy as np
import jax.numpy as jnp
from abcmb.main import Model
from abcmb.spectrum import bessel_l_tab
from accuracy_test_bb import PARAMS, ELLMIN, ELLMAX, class_tensor_hp_reference

cl_hp = class_tensor_hp_reference()
bb_hp = cl_hp["bb"]
print("CLASS hp done", flush=True)

model = Model(l_max=ELLMAX, lensing=False, tensors=True,
              l_max_g=12, l_max_pol_g=10)
output = model(PARAMS)
BG, params = output.BG, output.params
print("baseline ABCMB done", flush=True)

TPE, TSS = model.TPE, model.TSS
args = (BG, params)
node_ells = np.asarray(bessel_l_tab)[np.asarray(TSS.tensor_ells_indices)]


def build_TPT(Nlna):
    lna = jnp.linspace(BG.lna_transfer_start, 0., Nlna)
    res = vmap(TPE.evolution_one_k, in_axes=[0, None, None])(
        TPE.k_axis_tensor, lna, args)
    res = res.transpose(2, 1, 0)   # (Ny, Nlna, Nk)
    return TPE.make_output_table(lna, res, args)


def bb_nodes(TPT):
    out = vmap(TSS.Cl_one_ell, in_axes=(0, None, None, None))(
        TSS.tensor_ells_indices, TPT, BG, params)
    return np.asarray(jax.block_until_ready(out[3]))


NLNAS = (500, 1000, 2000, 4000)
bb = {}
for N in NLNAS:
    bb[N] = bb_nodes(build_TPT(N))
    print(f"  Nlna={N} done", flush=True)

print("\n--- BB low-l nodes: rel err vs CLASS-hp at increasing Nlna_ten ---")
print("  l   | " + "".join(f" err N{N:<5d}" for N in NLNAS) + " |   hp Cl")
for L in node_ells[node_ells <= 130]:
    c = bb_hp[L]
    i = np.where(node_ells == L)[0][0]
    cells = "".join(f" {(bb[N][i]-c)/c:+.3e}" for N in NLNAS)
    print(f" {L:4d} |{cells} | {c:.3e}")

lo = node_ells[(node_ells >= 3) & (node_ells < 100)]
def maxerr(arr):
    es = np.array([abs(arr[np.where(node_ells == L)[0][0]] - bb_hp[L]) / bb_hp[L]
                   for L in lo])
    return float(es.max()), int(lo[es.argmax()])
print("\n[summary] max |rel err| over low-l nodes (3<=l<100):")
for N in NLNAS:
    print(f"  Nlna={N}: {maxerr(bb[N])}")
print("DONE", flush=True)
