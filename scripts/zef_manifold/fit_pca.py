#!/usr/bin/env python
"""Step 3: fit THE canonical PCA basis on the curvature dataset (single source of truth for all
downstream analyses -- every analysis loads pca_basis.npz, nobody refits).

PCA via numpy SVD on mean-centered kappa*BL over VALID frames only.
Deterministic sign convention: each component is flipped so its largest-|loading| station is
positive (so re-runs and independent verifiers agree on signs).

Output: demo_out/zef_manifold/pca_basis.npz
  mean (K,), components (K,K) [row i = PC i], explained_variance_ratio (K,),
  singular_values (K,), coeffs (T,K) [NaN rows for invalid frames], n90/n95/n99 (ints)
"""
import numpy as np

from os.path import expandvars as _expandvars

def _P(s):
    """Expand the ${FISH_ROOT}/${ZEF_ROOT}/... placeholders used in this release."""
    import os
    os.environ.setdefault("FISH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return _expandvars(s)


OUT = _P("${FISH_ROOT}/demo_out/zef_manifold")

d = np.load(f"{OUT}/curvature_dataset.npz")
kap, valid = d["kappa_bl"], d["valid"]
X = kap[valid]
mu = X.mean(0)
Xc = X - mu
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
evr = S ** 2 / np.sum(S ** 2)

# deterministic signs
flip = np.sign(Vt[np.arange(Vt.shape[0]), np.abs(Vt).argmax(1)])
flip[flip == 0] = 1.0
Vt = Vt * flip[:, None]

T, K = kap.shape
coeffs = np.full((T, K), np.nan, np.float32)
coeffs[valid] = ((kap[valid] - mu) @ Vt.T).astype(np.float32)

cum = np.cumsum(evr)
n90, n95, n99 = (int(np.searchsorted(cum, q) + 1) for q in (0.90, 0.95, 0.99))
np.savez(f"{OUT}/pca_basis.npz", mean=mu.astype(np.float32), components=Vt.astype(np.float32),
         explained_variance_ratio=evr.astype(np.float32), singular_values=S.astype(np.float32),
         coeffs=coeffs, n90=n90, n95=n95, n99=n99)

print(f"[pca] {X.shape[0]} valid frames, K={K}")
print("  EVR per PC:", " ".join(f"{v:.4f}" for v in evr[:10]), "...")
print("  cumulative:", " ".join(f"{v:.4f}" for v in cum[:10]), "...")
print(f"  components for 90%: {n90}   95%: {n95}   99%: {n99}")
print(f"  wrote {OUT}/pca_basis.npz")
