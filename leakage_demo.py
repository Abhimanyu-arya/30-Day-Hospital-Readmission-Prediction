import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from model import add_intercept, fit_newton, predict_proba, log_loss, auc, brier

BASE = Path(r'E:\ML') / 'Data'
ps = pd.read_csv(BASE / 'patient_splits.csv')
print(ps['split'].value_counts())
assert ps['patient_nbr'].is_unique        # one row per patient: each patient is in exactly one split

Xtr = pd.read_csv(BASE / 'X_train.csv').to_numpy(float)
Xva = pd.read_csv(BASE / 'X_val.csv').to_numpy(float)
ytr = pd.read_csv(BASE / 'y_train.csv')['readmit_30d'].to_numpy()
yva = pd.read_csv(BASE / 'y_val.csv')['readmit_30d'].to_numpy()
n_tr, N = len(ytr), len(ytr) + len(yva)
Xall, yall = np.vstack([Xtr, Xva]), np.concatenate([ytr, yva])    # train + val only: the test set is not used here

GB = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, early_stopping=False, random_state=0)
GB_FLEX = dict(max_iter=300, learning_rate=0.1, max_leaf_nodes=63, early_stopping=False, random_state=0)  # demo only

def evaluate(Xa, ya, Xb, yb, tag):
    rows = []
    W = fit_newton(add_intercept(Xa), ya)
    fits = {'Newton': lambda X: predict_proba(W, add_intercept(X))}
    for name, params in [('GB (frozen settings)', GB), ('GB flexible (demo only)', GB_FLEX)]:
        m = HistGradientBoostingClassifier(**params).fit(Xa, ya)
        fits[name] = lambda X, m=m: m.predict_proba(X)[:, 1]
    for name, f in fits.items():
        pa, pb = f(Xa), f(Xb)
        rows.append({'split': tag, 'model': name, 'train_auc': auc(ya, pa),
                     'eval_logloss': log_loss(yb, pb), 'eval_auc': auc(yb, pb), 'eval_brier': brier(yb, pb)})
    return rows

rows = evaluate(Xtr, ytr, Xva, yva, 'patient-level (actual)')
for seed in range(5):
    perm = np.random.default_rng(seed).permutation(N)
    tr, va = perm[:n_tr], perm[n_tr:]
    rows += evaluate(Xall[tr], yall[tr], Xall[va], yall[va], f'random encounters, seed {seed}')

res = pd.DataFrame(rows)
res.to_csv(BASE / 'leakage_demo_raw.csv', index=False)
res['kind'] = np.where(res['split'].str.startswith('patient'), 'patient-level (actual)', 'random encounters (mean of 5)')
summary = res.groupby(['model', 'kind'])[['train_auc', 'eval_logloss', 'eval_auc', 'eval_brier']].agg(['mean', 'std'])
print(summary.to_string(float_format=lambda v: f"{v:.4f}"))
summary.to_csv(BASE / 'leakage_demo.csv')