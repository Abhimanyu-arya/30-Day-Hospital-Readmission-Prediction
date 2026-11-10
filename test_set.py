import json, time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from model import add_intercept, fit_newton, predict_proba, log_loss, auc, brier

BASE = Path(r'E:\ML') / 'Data'
FORCE = False                     # leave False: the test set is scored once
if (BASE / 'test_probs.csv').exists() and not FORCE:
    raise SystemExit("test_probs.csv already exists: the test set has been used. Not running again.")

RATIOS = [2, 3, 5, 10]            # frozen headline range; threshold = 1/(1+r)
GB_PARAMS = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                 early_stopping=False, random_state=0)

# ---------- train + validation only: refit and verify ----------
Xtr = pd.read_csv(BASE / 'X_train.csv').to_numpy(float)
Xva = pd.read_csv(BASE / 'X_val.csv').to_numpy(float)
ytr = pd.read_csv(BASE / 'y_train.csv')['readmit_30d'].to_numpy()
yva = pd.read_csv(BASE / 'y_val.csv')['readmit_30d'].to_numpy()

W_nt = fit_newton(add_intercept(Xtr), ytr)
gb = HistGradientBoostingClassifier(**GB_PARAMS).fit(Xtr, ytr)

ll_nt = log_loss(yva, predict_proba(W_nt, add_intercept(Xva)))
ll_gb = log_loss(yva, gb.predict_proba(Xva)[:, 1])
print(f"validation log loss: Newton {ll_nt:.6f} (expect 0.339326), GB {ll_gb:.6f} (expect about 0.336527)")
assert abs(ll_nt - 0.339326) < 2e-6, "Newton does not reproduce the validation result: stop"
if abs(ll_gb - 0.336527) > 2e-5:
    print("WARNING: gradient boosting differs from the earlier validation run (library version?). "
          "Do not continue until you understand why.")
    raise SystemExit

frozen = {'models': {'Newton': 'no penalty, fit on train only',
                     'Gradient boosting': GB_PARAMS},
          'threshold_rule': 't = 1/(1+r)', 'ratios': RATIOS,
          'written_before_test_data_loaded': time.strftime('%Y-%m-%d %H:%M:%S')}
(BASE / 'frozen_config.json').write_text(json.dumps(frozen, indent=2))

# ---------- the single test-set evaluation ----------
Xte = pd.read_csv(BASE / 'X_test.csv').to_numpy(float)
yte = pd.read_csv(BASE / 'y_test.csv')['readmit_30d'].to_numpy()
n = len(yte)
P = {'Baseline': np.full(n, ytr.mean()),
     'Newton': predict_proba(W_nt, add_intercept(Xte)),
     'Gradient boosting': gb.predict_proba(Xte)[:, 1]}
pd.DataFrame({'y': yte, **P}).to_csv(BASE / 'test_probs.csv', index=False)
print("test rows:", n, " test base rate:", yte.mean(), " train base rate:", ytr.mean())

rng = np.random.default_rng(10)
B = 1000
idxs = [rng.integers(0, n, n) for _ in range(B)]
ci = lambda a: np.percentile(a, [2.5, 97.5])

# metrics with bootstrap intervals
rows = []
for m, p in P.items():
    bl = [log_loss(yte[i], p[i]) for i in idxs]
    bb = [brier(yte[i], p[i]) for i in idxs]
    ba = [auc(yte[i], p[i]) for i in idxs]
    rows.append({'model': m, 'logloss': log_loss(yte, p), 'll_lo': ci(bl)[0], 'll_hi': ci(bl)[1],
                 'auc': auc(yte, p), 'auc_lo': ci(ba)[0], 'auc_hi': ci(ba)[1],
                 'brier': brier(yte, p), 'br_lo': ci(bb)[0], 'br_hi': ci(bb)[1]})
met = pd.DataFrame(rows)
print(met.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
met.to_csv(BASE / 'test_metrics.csv', index=False)

d_ll = [log_loss(yte[i], P['Gradient boosting'][i]) - log_loss(yte[i], P['Newton'][i]) for i in idxs]
d_au = [auc(yte[i], P['Gradient boosting'][i]) - auc(yte[i], P['Newton'][i]) for i in idxs]
print("GB minus Newton  logloss", np.mean(d_ll), ci(d_ll), "  AUC", np.mean(d_au), ci(d_au))

# calibration slope (1 = perfect)
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
for m in ['Newton', 'Gradient boosting']:
    lr = LogisticRegression(C=1e10, max_iter=1000).fit(logit(P[m]).reshape(-1, 1), yte)
    print(f"{m} calibration slope on test: {lr.coef_[0, 0]:.3f}")

# cost at the frozen thresholds
def counts(y, p, t):
    f = p >= t
    return (int(np.sum(f & (y == 1))), int(np.sum(f & (y == 0))),
            int(np.sum(~f & (y == 1))), int(np.sum(~f & (y == 0))))
def cost_at(y, p, r):
    tp, fp, fn, tn = counts(y, p, 1 / (1 + r)); return (r * fn + fp) / len(y)
def trivial(y, r):
    return min(r * y.mean(), 1 - y.mean())

crow = []
for r in RATIOS:
    for m in ['Newton', 'Gradient boosting']:
        p = P[m]
        tp, fp, fn, tn = counts(yte, p, 1 / (1 + r))
        sav = [trivial(yte[i], r) - cost_at(yte[i], p[i], r) for i in idxs]
        crow.append({'r': r, 'model': m, 'threshold': 1 / (1 + r),
                     'cost': cost_at(yte, p, r), 'cost_best_trivial': trivial(yte, r),
                     'saving': trivial(yte, r) - cost_at(yte, p, r), 'sav_lo': ci(sav)[0], 'sav_hi': ci(sav)[1],
                     'flagged_per_1000': 1000 * (tp + fp) / n, 'caught_per_1000': 1000 * tp / n,
                     'missed_per_1000': 1000 * fn / n, 'unnecessary_per_1000': 1000 * fp / n})
    diff = [cost_at(yte[i], P['Newton'][i], r) - cost_at(yte[i], P['Gradient boosting'][i], r) for i in idxs]
    print(f"r={r}: GB cheaper than Newton by {np.mean(diff):.4f} {ci(diff)}")
cost = pd.DataFrame(crow)
print(cost.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
cost.to_csv(BASE / 'test_cost.csv', index=False)