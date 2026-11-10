import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, brier_score_loss

BASE = Path(r'E:\ML') / 'Data'

# ---------- data (validation + train labels only) ----------
X_val = pd.read_csv(BASE / 'X_val.csv').to_numpy(dtype=float)
y_val = pd.read_csv(BASE / 'y_val.csv')['readmit_30d'].to_numpy()
y_train = pd.read_csv(BASE / 'y_train.csv')['readmit_30d'].to_numpy()
Xb_val = np.hstack((np.ones((X_val.shape[0], 1)), X_val))

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))

# ---------- metrics from scratch ----------
def brier(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return np.mean((p - y) ** 2)

def auc(y, s):
    y = np.asarray(y)
    ranks = rankdata(s)                      # average ranks for ties
    n_pos = int(np.sum(y == 1))
    n_neg = len(y) - n_pos
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

# ---------- toy checks (known answers) ----------
print(auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8])))   # 0.75
print(auc(np.array([0, 1, 0, 1]), np.array([0.5, 0.5, 0.5, 0.5])))    # 0.5 (all ties)
print(brier(np.array([0, 1, 1, 0]), np.array([0.1, 0.9, 0.8, 0.3])))  # 0.0375

# ---------- models on validation ----------
W_nt = np.load(BASE / 'W_nt.npy')
W_r4 = np.load(BASE / 'W_ridge_1e-4.npy')

p_nt = sigmoid(Xb_val @ W_nt)
p_r4 = sigmoid(Xb_val @ W_r4)
p_base = np.full(len(y_val), y_train.mean())   # constant baseline: training base rate

rows = []
for name, p in [('baseline (train base rate)', p_base),
                ('Newton (no penalty)', p_nt),
                ('Ridge lam=1e-4', p_r4)]:
    a, b = auc(y_val, p), brier(y_val, p)
    a_sk, b_sk = roc_auc_score(y_val, p), brier_score_loss(y_val, p)
    print(f"{name:28s} AUC = {a:.6f} (sklearn diff {abs(a - a_sk):.1e})   "
          f"Brier = {b:.6f} (sklearn diff {abs(b - b_sk):.1e})")
    assert abs(a - a_sk) < 1e-10 and abs(b - b_sk) < 1e-12
    rows.append({'model': name, 'val_auc': a, 'val_brier': b})

print("base rate (train):", y_train.mean(), "  base rate (val):", y_val.mean())
pv = y_val.mean()
print("expected baseline Brier ~ p(1-p) =", pv*(1-pv) + (y_train.mean()-pv)**2)

pd.DataFrame(rows).to_csv(BASE / 'metrics_val.csv', index=False)

# ---------- confusion-matrix metrics at a threshold ----------
def confusion_metrics(y, p, t):
    pred = p >= t
    pos = (y == 1)
    tp = np.sum(pred & pos);  fp = np.sum(pred & ~pos)
    fn = np.sum(~pred & pos); tn = np.sum(~pred & ~pos)
    return {
        'threshold': t, 'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
        'flagged_rate': pred.mean(),
        'sensitivity': tp / (tp + fn),
        'specificity': tn / (tn + fp),
        'precision': tp / (tp + fp) if (tp + fp) > 0 else np.nan,
    }

thr = [0.5, 0.3, 0.2, 0.15, y_val.mean(), 0.10, 0.08]
cm = pd.DataFrame([confusion_metrics(y_val, p_nt, t) for t in thr])
print(cm.to_string(float_format=lambda v: f"{v:.4f}"))
cm.to_csv(BASE / 'confusion_val_newton.csv', index=False)

# ---------- paired bootstrap on validation rows ----------
rng = np.random.default_rng(0)
B = 1000
n = len(y_val)
boot = {'auc_nt': [], 'brier_nt': [], 'brier_base': [], 'auc_diff_nt_minus_r4': []}
for _ in range(B):
    idx = rng.integers(0, n, n)
    yb = y_val[idx]
    boot['auc_nt'].append(auc(yb, p_nt[idx]))
    boot['brier_nt'].append(brier(yb, p_nt[idx]))
    boot['brier_base'].append(brier(yb, p_base[idx]))
    boot['auc_diff_nt_minus_r4'].append(auc(yb, p_nt[idx]) - auc(yb, p_r4[idx]))

def ci(a):
    return np.percentile(a, [2.5, 97.5])

print("AUC Newton            ", auc(y_val, p_nt), ci(boot['auc_nt']))
print("Brier Newton          ", brier(y_val, p_nt), ci(boot['brier_nt']))
print("Brier gain vs baseline", 
      np.mean(np.array(boot['brier_base']) - np.array(boot['brier_nt'])),
      ci(np.array(boot['brier_base']) - np.array(boot['brier_nt'])))
print("AUC diff Newton-Ridge ", np.mean(boot['auc_diff_nt_minus_r4']), ci(boot['auc_diff_nt_minus_r4']))


from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import log_loss as sk_log_loss

X_train = pd.read_csv(BASE / 'X_train.csv').to_numpy(dtype=float)

# fixed settings chosen in advance; any later tuning must use validation only
models = {
    'Random forest': RandomForestClassifier(n_estimators=300, min_samples_leaf=50,
                                            n_jobs=-1, random_state=0),
    'Gradient boosting': HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                                        max_leaf_nodes=15, early_stopping=False,
                                                        random_state=0),
}

probs = {'Baseline': p_base, 'Newton': p_nt, 'Ridge 1e-4': p_r4}
for name, m in models.items():
    m.fit(X_train, y_train)
    probs[name] = m.predict_proba(X_val)[:, 1]

rows = []
for name, p in probs.items():
    rows.append({'model': name, 'val_logloss': sk_log_loss(y_val, p),
                 'val_auc': auc(y_val, p), 'val_brier': brier(y_val, p)})
res = pd.DataFrame(rows)
print(res.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
res.to_csv(BASE / 'metrics_val_all.csv', index=False)

# paired bootstrap: each tree model minus Newton
rng = np.random.default_rng(1)
for name in ['Random forest', 'Gradient boosting']:
    d_auc, d_ll = [], []
    for _ in range(500):
        idx = rng.integers(0, len(y_val), len(y_val))
        yb = y_val[idx]
        d_auc.append(auc(yb, probs[name][idx]) - auc(yb, p_nt[idx]))
        d_ll.append(sk_log_loss(yb, probs[name][idx]) - sk_log_loss(yb, p_nt[idx]))
    print(name, 'minus Newton  AUC', np.mean(d_auc), np.percentile(d_auc, [2.5, 97.5]),
          ' logloss', np.mean(d_ll), np.percentile(d_ll, [2.5, 97.5]))

pd.DataFrame({'y': y_val, **probs}).to_csv(BASE / 'val_probs.csv', index=False)
