"""
verify_export.py  –  multi-class OvO SVM, 5 classes / 10 pairs
────────────────────────────────────────────────────────────────
sklearn's decision_function() for multi-class returns VOTE COUNTS
per class (shape = n_classes), NOT the 10 raw pair scores.

This script:
  1. Computes all 10 raw OvO pair scores correctly
  2. Converts them to per-class votes  →  matches sklearn's decision output
  3. Computes predict_proba via Platt + Wu et al. coupling
  4. Prints a clear PASS/FAIL for each stage
"""

import joblib
import numpy as np
from scipy.stats import skew as scipy_skew
import pandas as pd

model  = joblib.load("svm_retrain_model.pkl")
scaler = joblib.load("svm_retrain_scaler.pkl")

# ── Sample window ─────────────────────────────────────────────────────────────
np.random.seed(42)
data = np.random.randn(75, 4) * 0.5
df   = pd.DataFrame(data, columns=["ay", "az", "gx", "gz"])

# ── Feature extraction ────────────────────────────────────────────────────────
def extract_features(df):
    feats = []
    for col in ["ay", "az", "gx", "gz"]:
        s    = df[col].values
        mean = np.mean(s)
        mn   = np.min(s)
        mx   = np.max(s)
        rms  = np.sqrt(np.mean(s**2))
        m2   = np.mean((s - mean)**2)
        m3   = np.mean((s - mean)**3)
        std  = np.sqrt(m2)
        sk   = 0.0 if std < 1e-9 else m3 / std**3
        energy = np.sum(s**2) / len(s)
        zcr    = np.sum(np.diff(np.sign(s)) != 0) / len(s)
        feats.extend([mn, mx, mx - mn, rms, sk, energy, zcr])
    return np.array(feats)

feat        = extract_features(df)
feat_scaled = scaler.transform([feat])[0]

# ── sklearn reference ─────────────────────────────────────────────────────────
pred_sk = model.predict([feat_scaled])[0]
prob_sk = model.predict_proba([feat_scaled])[0]
dec_sk  = model.decision_function([feat_scaled])[0]   # shape (5,) = vote counts

print("=== sklearn reference ===")
print(f"  classes  : {model.classes_}")
print(f"  decision : {dec_sk}   shape={dec_sk.shape}")
print(f"  predict  : {pred_sk}")
print(f"  proba    : {prob_sk}")

# ── Model parameters ──────────────────────────────────────────────────────────
sv        = model.support_vectors_     # (361, 28)
dc        = model.dual_coef_           # (4, 361)
b         = model.intercept_           # (10,)
g         = model._gamma               # 0.1
pA        = model.probA_               # (10,)
pB        = model.probB_               # (10,)
n_support = model.n_support_           # [61, 41, 101, 104, 54]
n_classes = len(model.classes_)        # 5
n_pairs   = n_classes * (n_classes - 1) // 2   # 10

sv_start  = np.concatenate([[0], np.cumsum(n_support[:-1])])

# ── RBF kernel vector ─────────────────────────────────────────────────────────
K = np.exp(-g * np.sum((sv - feat_scaled)**2, axis=1))   # (361,)

# ── Compute all 10 raw OvO pair scores ───────────────────────────────────────
pair_scores = np.zeros(n_pairs)
pair_idx    = 0
pair_classes = []   # which (i, j) each pair corresponds to

for i in range(n_classes):
    for j in range(i + 1, n_classes):
        si = slice(sv_start[i], sv_start[i] + n_support[i])
        sj = slice(sv_start[j], sv_start[j] + n_support[j])
        score = (np.dot(dc[j-1, si], K[si])
               + np.dot(dc[i,   sj], K[sj])
               + b[pair_idx])
        pair_scores[pair_idx] = score
        pair_classes.append((i, j))
        pair_idx += 1

print(f"\n=== 10 raw OvO pair scores ===")
for p, ((i, j), s) in enumerate(zip(pair_classes, pair_scores)):
    print(f"  pair {p:2d}  ({model.classes_[i]:8s} vs {model.classes_[j]:8s}):  {s:+.6f}")

# ── Convert pair scores → per-class vote counts (matches sklearn decision_function) ──
# For pair (i,j): if score > 0  →  vote for class i,  else  vote for class j
votes = np.zeros(n_classes)
for p, (i, j) in enumerate(pair_classes):
    if pair_scores[p] > 0:
        votes[i] += 1
    else:
        votes[j] += 1

print(f"\n=== per-class votes (matches sklearn decision_function) ===")
for i in range(n_classes):
    print(f"  {model.classes_[i]:8s}: {votes[i]}")

print(f"\n  votes    : {votes}")
print(f"  dec_sk   : {dec_sk}")
print(f"  match    : {np.allclose(votes, dec_sk, atol=1e-5)}")

# ── Platt + Wu et al. coupling ────────────────────────────────────────────────
def platt_prob(f, A, B):
    fApB = f * A + B
    return (np.exp(-fApB) / (1 + np.exp(-fApB))) if fApB >= 0 else (1 / (1 + np.exp(fApB)))

def pairwise_to_proba(pair_scores, pA, pB, n_classes, pair_classes):
    r = np.zeros((n_classes, n_classes))
    for p, (i, j) in enumerate(pair_classes):
        rij      = platt_prob(pair_scores[p], pA[p], pB[p])
        r[i, j]  = rij
        r[j, i]  = 1.0 - rij

    p = np.ones(n_classes) / n_classes
    for _ in range(100):
        Q   = np.array([sum(r[j, i] * p[j] for j in range(n_classes) if j != i)
                        for i in range(n_classes)])
        pQp  = np.dot(p, Q)
        diff = p * (Q - pQp)
        p   -= diff
        if np.max(np.abs(diff)) < 1e-5:
            break
    return p

prob_manual = pairwise_to_proba(pair_scores, pA, pB, n_classes, pair_classes)
pred_manual = model.classes_[np.argmax(prob_manual)]

print(f"\n=== predict_proba ===")
print(f"  manual : {prob_manual}")
print(f"  sklearn: {prob_sk}")
print(f"  match  : {np.allclose(prob_manual, prob_sk, atol=1e-4)}")
print(f"\n=== prediction ===")
print(f"  manual : {pred_manual}")
print(f"  sklearn: {pred_sk}")
print(f"  match  : {pred_manual == pred_sk}")