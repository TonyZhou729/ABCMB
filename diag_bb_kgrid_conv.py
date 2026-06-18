"""Is the low-l raw-BB ~0.4% excess a low-k QUADRATURE convergence gap?

The hp CLASS reference samples low k ~10x denser than ABCMB (k_step_super
0.0002 vs ABCMB's 2e-3; q_linstep 0.05). Low l <-> low k. If ABCMB's BB is
undersampled at low k, ABCMB-vs-hp shows exactly a smooth excess largest at
low l decaying to ~0 by l~490 -- the observed shape.

Two convergence axes, tested separately:
  (A) TRANSFER trapezoid grid  (TSS.k_axis_transfer) -- cheap: re-integrate
      the already-computed source table on a refined k grid, NO PE re-solve.
  (B) SOURCE / perturbation grid (TPT.k) -- needs one PE rebuild with a
      denser k_axis_perturbations / k_axis_transfer.

If BB at low-l nodes moves toward CLASS-hp under (A) and/or (B), the excess is
grid convergence, not a physics bug. If neither moves it, it is structural
(reion source / GW source) -> go to the reion test.

Cost: ~1 PE solve for baseline + (A) is cheap; (B) is a 2nd model build (one
more compile + PE solve). GPU, a few minutes total warm.
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

# ----------------------------------------------------------------------
# CLASS high-precision tensor-only reference (CPU, cheap)
cl_hp = class_tensor_hp_reference()
bb_hp = cl_hp["bb"]          # indexed by ell, up to 500
print("CLASS hp done", flush=True)

# ----------------------------------------------------------------------
# Baseline ABCMB model
model = Model(l_max=ELLMAX, lensing=False, tensors=True,
              l_max_g=12, l_max_pol_g=10)
output = model(PARAMS)
BG, params = output.BG, output.params
print("baseline ABCMB done", flush=True)

# Heavy step: the tensor source table (one PE solve)
TPT = model.TPE.full_evolution((BG, params))
TPT = jax.block_until_ready(TPT)
print("TPT done", flush=True)

node_idx = np.asarray(model.TSS.tensor_ells_indices)
node_ells = np.asarray(bessel_l_tab)[node_idx]


def raw_bb_nodes(k_transfer, TPT_in=TPT, TSS=model.TSS):
    """Raw (un-splined) BB at the tensor spline nodes for a given transfer grid."""
    TSS2 = eqx.tree_at(lambda s: s.k_axis_transfer, TSS, k_transfer)
    res = jax.vmap(TSS2.Cl_one_ell, in_axes=(0, None, None, None))(
        TSS2.tensor_ells_indices, TPT_in, BG, params)
    return np.asarray(jax.block_until_ready(res[3]))


def subdivide(k, m):
    """Refine each interval of k into m sub-intervals (linear). Strict trapezoid
    refinement -> converges to the exact integral of the interpolated integrand."""
    if m == 1:
        return jnp.asarray(k)
    k = np.asarray(k)
    segs = [np.linspace(k[i], k[i + 1], m, endpoint=False)
            for i in range(len(k) - 1)]
    return jnp.asarray(np.concatenate(segs + [k[-1:]]))


# ----------------------------------------------------------------------
# Grid-density context
k_t0 = np.asarray(model.TSS.k_axis_transfer)
k_src = np.asarray(TPT.k)
tau0 = float(BG.tau0)
k_l10 = 10.0 / tau0      # k that dominates l~10 BB
print(f"\n[context] tau0 = {tau0:.1f} Mpc;  k(l=10) ~ {k_l10:.3e} Mpc^-1")
print(f"[context] transfer grid: N={len(k_t0)}, "
      f"k in [{k_t0[0]:.3e}, {k_t0[-1]:.3e}]")
print(f"[context] source/pert grid: N={len(k_src)}, "
      f"k in [{k_src[0]:.3e}, {k_src[-1]:.3e}]")
below = lambda g: int((np.asarray(g) <= k_l10).sum())
print(f"[context] transfer pts below k(l=10): {below(k_t0)};  "
      f"source pts below k(l=10): {below(k_src)}")
# local dk/k at k(l=10)
def dkk_at(g, kq):
    g = np.asarray(g); i = np.searchsorted(g, kq)
    i = min(max(i, 1), len(g) - 1)
    return (g[i] - g[i - 1]) / g[i]
print(f"[context] transfer dk/k at k(l=10): {dkk_at(k_t0, k_l10):.4f};  "
      f"source dk/k at k(l=10): {dkk_at(k_src, k_l10):.4f}")

# ======================================================================
# (A) TRANSFER-grid refinement (cheap, no PE re-solve)
print("\n" + "=" * 78)
print("(A) TRANSFER trapezoid refinement (same source table)")
print("=" * 78, flush=True)
bb_A = {}
for m in (1, 2, 4):
    bb_A[m] = raw_bb_nodes(subdivide(k_t0, m))
    print(f"  transfer x{m} done (N={len(subdivide(k_t0, m))})", flush=True)


def show(tag, bb_dict_or_arr, ms):
    print(f"\n--- {tag}: low-l nodes (rel err vs CLASS-hp) ---")
    hdr = "  l   |" + "".join(f"  err x{m}   " for m in ms) + " |   hp Cl"
    print(hdr)
    for L in node_ells[node_ells <= 130]:
        c = bb_hp[L]
        cells = ""
        for m in ms:
            a = bb_dict_or_arr[m][np.where(node_ells == L)[0][0]]
            e = (a - c) / c
            cells += f" {e:+.3e} "
        print(f" {L:4d} |{cells}| {c:.3e}")


show("(A) transfer refinement", bb_A, (1, 2, 4))

# ======================================================================
# (B) SOURCE / perturbation-grid refinement (one PE rebuild, ~4x denser low-k)
print("\n" + "=" * 78)
print("(B) SOURCE/perturbation-grid refinement (denser k_axis_perturbations)")
print("=" * 78, flush=True)
DENSE_SPECS = dict(
    k_step_super=5.e-4,     # 2e-3 -> 5e-4  (4x finer super-horizon)
    k_step_sub=1.25e-2,     # 5e-2 -> 1.25e-2
    k_transfer_linstep=1.1e-1,  # 4.5e-1 -> 1.1e-1
)
model_d = Model(l_max=ELLMAX, lensing=False, tensors=True,
                l_max_g=12, l_max_pol_g=10, **DENSE_SPECS)
print(f"  dense model built; tensor source N_k = "
      f"{len(model_d.TPE.k_axis_tensor)}, "
      f"transfer N_k = {len(model_d.TSS.k_axis_transfer)}", flush=True)
out_d = model_d(PARAMS)
BG_d, params_d = out_d.BG, out_d.params
TPT_d = jax.block_until_ready(model_d.TPE.full_evolution((BG_d, params_d)))
print("  dense TPT done", flush=True)

# raw nodes from the dense model (use its own TSS + source table)
res_d = jax.vmap(model_d.TSS.Cl_one_ell, in_axes=(0, None, None, None))(
    model_d.TSS.tensor_ells_indices, TPT_d, BG_d, params_d)
bb_dense = np.asarray(jax.block_until_ready(res_d[3]))
node_idx_d = np.asarray(model_d.TSS.tensor_ells_indices)
node_ells_d = np.asarray(bessel_l_tab)[node_idx_d]

print("\n--- (B) baseline vs dense-grid model vs CLASS-hp, low-l nodes ---")
print("  l   |  err base  |  err dense |   hp Cl")
for L in node_ells[node_ells <= 130]:
    c = bb_hp[L]
    a_b = bb_A[1][np.where(node_ells == L)[0][0]]
    a_d = bb_dense[np.where(node_ells_d == L)[0][0]]
    print(f" {L:4d} | {(a_b-c)/c:+.3e} | {(a_d-c)/c:+.3e} | {c:.3e}")

# summary maxima over low-l nodes
lo = node_ells[(node_ells >= 3) & (node_ells < 100)]
def maxerr(bb_arr, ne):
    es = np.array([abs(bb_arr[np.where(ne == L)[0][0]] - bb_hp[L]) / bb_hp[L]
                   for L in lo])
    return es.max(), lo[es.argmax()]
print("\n[summary] max |rel err| over low-l nodes (3<=l<100):")
print(f"  baseline transfer x1 : {maxerr(bb_A[1], node_ells)}")
print(f"  transfer x4          : {maxerr(bb_A[4], node_ells)}")
print(f"  dense pert+transfer  : {maxerr(bb_dense, node_ells_d)}")
print("DONE", flush=True)
