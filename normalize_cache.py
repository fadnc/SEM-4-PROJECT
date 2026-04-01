# quick_fix_normalize_cache.py
import numpy as np
from sklearn.preprocessing import StandardScaler

print("Loading cached X...")
X = np.load('output/feature_cache_X.npy')
print(f"X shape: {X.shape}, dtype: {X.dtype}")

n_seqs, seq_len, n_feat = X.shape
X_2d = X.reshape(-1, n_feat)

print("Fitting StandardScaler...")
scaler = StandardScaler()
X_2d = scaler.fit_transform(X_2d).astype(np.float32)
np.nan_to_num(X_2d, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

X_norm = X_2d.reshape(n_seqs, seq_len, n_feat)
np.save('output/feature_cache_X.npy', X_norm)
print("Saved normalized X. Re-run pipeline normally.")