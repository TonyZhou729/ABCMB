"""Confirm the restructured raw-BB assertion (node 2.5e-3 + band 1%) passes.
Runs only test_bb_raw (lensed pair unchanged). GPU."""
import os
os.environ["JAX_PLATFORM_NAME"] = "gpu"
import sys
sys.path.insert(0, "/pscratch/sd/c/carag/ABCMB-bmodes/pytests")
from accuracy_test_bb import test_bb_raw

test_bb_raw()
print("\nraw-BB check PASSED")
