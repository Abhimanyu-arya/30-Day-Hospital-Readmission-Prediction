import numpy as np
import pandas as pd
from pathlib import Path
from model import add_intercept, fit_newton, predict_proba, log_loss, auc, brier

BASE = Path(r'E:\ML') / 'Data'
Xtr = pd.read_csv(BASE / 'X_train.csv'); Xva = pd.read_csv(BASE / 'X_val.csv')
ytr = pd.read_csv(BASE / 'y_train.csv')['readmit_30d'].to_numpy()
yva = pd.read_csv(BASE / 'y_val.csv')['readmit_30d'].to_numpy()
ftr = pd.read_csv(BASE / 'fair_train.csv'); fva = pd.read_csv(BASE / 'fair_val.csv')
assert len(ftr) == len(Xtr) and len(fva) == len(Xva)

payer = [c for c in Xtr.columns if c.startswith('payer_code_')]
spec = [c for c in Xtr.columns if c.startswith('medical_specialty_grouped_')]
assert len(payer) == 10 and len(spec) == 12
DROPS = {
    f'drop all payer ({len(payer)} cols)': payer,
    f'drop all specialty ({len(spec)} cols)': spec,
    'drop payer_code_Missing only': ['payer_code_Missing'],
    'drop specialty Missing only': ['medical_specialty_grouped_Missing'],
}
VISITS = ['number_outpatient', 'number_emergency', 'number_inpatient']
race_cols = [c for c in ftr.columns if c.startswith('race_')]

def run(A_tr, A_va):
    W = fit_newton(add_intercept(A_tr.to_numpy(float)), ytr)
    return predict_proba(W, add_intercept(A_va.to_numpy(float)))

preds, nfeat = {}, {}
base = 'baseline (all features)'
preds[base] = run(Xtr, Xva); nfeat[base] = Xtr.shape[1]
p0 = preds[base]
assert abs(log_loss(yva, p0) - 0.339326) < 2e-6 and abs(auc(yva, p0) - 0.669633) < 2e-6, \
    "baseline does not reproduce the earlier Newton validation numbers"

# 1-4: drops
for name, cols in DROPS.items():
    preds[name] = run(Xtr.drop(columns=cols), Xva.drop(columns=cols))
    nfeat[name] = Xtr.shape[1] - len(cols)

# 5: log1p on visit counts, recovering raw counts from the standardized columns
def recover_counts(z_tr, z_va):
    u = np.unique(z_tr)
    gap = np.diff(u).min()                         # one raw count in standardized units
    r_tr, r_va = (z_tr - u[0]) / gap, (z_va - u[0]) / gap   # assumes smallest count is 0
    assert np.abs(r_tr - np.round(r_tr)).max() < 1e-6 and np.abs(r_va - np.round(r_va)).max() < 1e-6
    return np.round(r_tr), np.round(r_va)

A_tr, A_va = Xtr.copy(), Xva.copy()
for c in VISITS:
    r_tr, r_va = recover_counts(Xtr[c].to_numpy(), Xva[c].to_numpy())
    print(f"{c}: recovered raw counts min={r_tr.min():.0f} max={r_tr.max():.0f} mean={r_tr.mean():.3f}")
    a, b = np.log1p(r_tr), np.log1p(r_va)
    mu, sd = a.mean(), a.std()                     # TRAIN statistics only
    A_tr[c], A_va[c] = (a - mu) / sd, (b - mu) / sd
name = 'log1p visit counts'
preds[name] = run(A_tr, A_va); nfeat[name] = Xtr.shape[1]

# 6: add race dummies back
name = f'add race ({len(race_cols)} dummies)'
preds[name] = run(pd.concat([Xtr, ftr[race_cols]], axis=1), pd.concat([Xva, fva[race_cols]], axis=1))
nfeat[name] = Xtr.shape[1] + len(race_cols)

# paired bootstrap against the baseline (same resamples for every ablation)
rng = np.random.default_rng(3)
n, B = len(yva), 500
idxs = [rng.integers(0, n, n) for _ in range(B)]
rows = []
for name, p in preds.items():
    r = {'ablation': name, 'n_features': nfeat[name],
         'val_logloss': log_loss(yva, p), 'val_auc': auc(yva, p), 'val_brier': brier(yva, p)}
    if name != base:
        dll = np.array([log_loss(yva[i], p[i]) - log_loss(yva[i], p0[i]) for i in idxs])
        dau = np.array([auc(yva[i], p[i]) - auc(yva[i], p0[i]) for i in idxs])
        r.update({'dlogloss': dll.mean(), 'dll_lo': np.percentile(dll, 2.5), 'dll_hi': np.percentile(dll, 97.5),
                  'dauc': dau.mean(), 'dauc_lo': np.percentile(dau, 2.5), 'dauc_hi': np.percentile(dau, 97.5)})
    rows.append(r)
res = pd.DataFrame(rows)
print(res.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
res.to_csv(BASE / 'ablations_val.csv', index=False)