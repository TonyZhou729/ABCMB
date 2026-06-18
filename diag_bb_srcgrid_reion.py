"""Two decisive tests for the low-l raw-BB ~0.4% excess.

Part (A) [diag_bb_kgrid_conv.py] already ruled OUT the transfer-trapezoid grid:
refining TSS.k_axis_transfer x1->x2->x4 left the l=10 excess rock-stable at
+4.2e-3. So the transfer quadrature is converged. Two axes remain:

  (B) SOURCE / perturbation k-grid (TPT.k): 38 pts below k(l=10), dk/k~0.063,
      ~10x coarser than CLASS-hp (k_step_super 2e-4). Refining the transfer
      trapezoid integrates the spline through COARSE source points accurately
      but cannot recover undersampled source structure -- must add real source
      points by re-running the PE on a denser tensor k grid. Densify only the
      TENSOR grid (k_max~0.064, cheap), avoiding the 2000-pt preallocation
      overflow in get_k_axis_perturbations.

  (C) REIONIZATION source. BB uses ONLY source_E = sqrt(6) g Pi. Decompose
      ABCMB BB into recomb-only by masking the source past recomb
      (lna > lna_cut). bb_full - bb_recomb = total reion contribution
      (incl. cross term). If reion contributes only a tiny fraction of
      BB(l~10), a reion mismatch cannot produce the +0.4% excess.

Cost: 3 tensor PE solves (source x1,x2,x4) + cheap re-integrations. GPU,
a few minutes warm; the source-grid compiles dominate.
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
import jax.numpy as jnp
import equinox as eqx
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

node_ells = np.asarray(bessel_l_tab)[np.asarray(model.TSS.tensor_ells_indices)]
LOWMASK = node_ells <= 130


def subdivide(k, m):
    if m == 1:
        return jnp.asarray(k)
    k = np.asarray(k)
    segs = [np.linspace(k[i], k[i + 1], m, endpoint=False)
            for i in range(len(k) - 1)]
    return jnp.asarray(np.concatenate(segs + [k[-1:]]))


def raw_bb_nodes(TPT_in, k_transfer):
    TSS2 = eqx.tree_at(lambda s: s.k_axis_transfer, model.TSS, k_transfer)
    res = jax.vmap(TSS2.Cl_one_ell, in_axes=(0, None, None, None))(
        TSS2.tensor_ells_indices, TPT_in, BG, params)
    return np.asarray(jax.block_until_ready(res[3]))


def relerr_row(L, vals_by_tag, tags):
    c = bb_hp[L]
    cells = "".join(f" {(vals_by_tag[t][np.where(node_ells == L)[0][0]]-c)/c:+.3e} "
                    for t in tags)
    return f" {L:4d} |{cells}| {c:.3e}"


# ======================================================================
# (B) SOURCE-grid refinement: re-run the tensor PE on a denser k grid.
print("\n" + "=" * 74)
print("(B) SOURCE/perturbation k-grid refinement (re-run PE, denser tensor k)")
print("=" * 74, flush=True)
k_src0 = model.TPE.k_axis_tensor
bb_B = {}
for m in (1, 2, 4):
    k_src_m = subdivide(k_src0, m)
    TPE_m = eqx.tree_at(lambda e: e.k_axis_tensor, model.TPE, k_src_m)
    TPT_m = jax.block_until_ready(TPE_m.full_evolution((BG, params)))
    # integrate on the dense source grid itself (no interpolation loss,
    # trapezoid already shown converged at this density in part A)
    bb_B[m] = raw_bb_nodes(TPT_m, k_src_m)
    print(f"  source x{m} done (N_k = {len(k_src_m)})", flush=True)
    if m == 1:
        TPT_base = TPT_m  # reuse for part (C)

print("\n--- (B) source refinement: low-l nodes (rel err vs CLASS-hp) ---")
print("  l   |  err x1     err x2     err x4    |   hp Cl")
for L in node_ells[LOWMASK]:
    print(relerr_row(L, bb_B, (1, 2, 4)))

lo = node_ells[(node_ells >= 3) & (node_ells < 100)]
def maxerr(arr):
    es = np.array([abs(arr[np.where(node_ells == L)[0][0]] - bb_hp[L]) / bb_hp[L]
                   for L in lo])
    return float(es.max()), int(lo[es.argmax()])
print("\n[summary] max |rel err| over low-l nodes (3<=l<100):")
for m in (1, 2, 4):
    print(f"  source x{m}: {maxerr(bb_B[m])}")

# ======================================================================
# (C) REIONIZATION decomposition (mask source past recomb), baseline grid.
print("\n" + "=" * 74)
print("(C) REIONIZATION decomposition: full vs recomb-only ABCMB BB")
print("=" * 74, flush=True)
LNA_CUT = -4.0   # z~53: between recomb (lna~-7) and reion (lna~-2.2)
lna = TPT_base.lna
mask = (lna < LNA_CUT)[:, None]
print(f"  lna_cut = {LNA_CUT} (keeps {int(mask.sum())}/{mask.shape[0]} "
      f"lna pts = recomb side)", flush=True)
TPT_recomb = eqx.tree_at(lambda t: t.source_E, TPT_base,
                         TPT_base.source_E * mask)
bb_full = raw_bb_nodes(TPT_base, model.TSS.k_axis_transfer)
bb_recomb = raw_bb_nodes(TPT_recomb, model.TSS.k_axis_transfer)

print("\n--- (C) reion contribution to ABCMB BB, low-l nodes ---")
print("  l   | reion frac (full-recomb)/full | excess vs hp | reion>>excess?")
for L in node_ells[LOWMASK]:
    i = np.where(node_ells == L)[0][0]
    frac = (bb_full[i] - bb_recomb[i]) / bb_full[i]
    exc = (bb_full[i] - bb_hp[L]) / bb_hp[L]
    flag = "yes" if abs(frac) > 5 * abs(exc) else ""
    print(f" {L:4d} | {frac:+.4e}                  | {exc:+.3e}   | {flag}")
print("DONE", flush=True)
