import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(r'E:\ML') / 'Data'
d = pd.read_csv(BASE / 'val_probs.csv')
y = d['y'].to_numpy()
n = len(y)
prev = y.mean()

models = ['Newton', 'Gradient boosting']
ratios = [1, 2, 3, 5, 10, 20]            # cost of a missed readmission / cost of an unnecessary intervention
grid = np.round(np.arange(0.01, 0.60, 0.005), 3)

def counts(y, p, t):
    flag = p >= t
    tp = int(np.sum(flag & (y == 1))); fp = int(np.sum(flag & (y == 0)))
    fn = int(np.sum(~flag & (y == 1))); tn = int(np.sum(~flag & (y == 0)))
    return tp, fp, fn, tn

def cost(y, p, t, r):
    tp, fp, fn, tn = counts(y, p, t)
    return (r * fn + fp) / len(y)        # in units of one unnecessary-intervention cost

# threshold table, saved for later plots and the website slider (cost for any r = (r*FN + FP)/n)
tab = []
for m in models:
    p = d[m].to_numpy()
    for t in grid:
        tp, fp, fn, tn = counts(y, p, t)
        tab.append({'model': m, 'threshold': t, 'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn})
pd.DataFrame(tab).to_csv(BASE / 'threshold_table_val.csv', index=False)

rows = []
for m in models:
    p = d[m].to_numpy()
    for r in ratios:
        t_th = 1 / (1 + r)
        c_th = cost(y, p, t_th, r)
        costs = np.array([cost(y, p, t, r) for t in grid])
        t_emp, c_emp = grid[costs.argmin()], costs.min()
        flag_none = r * prev              # nobody flagged: pay for every readmission
        flag_all = 1 - prev               # everybody flagged: pay for every unnecessary intervention
        trivial = min(flag_none, flag_all)
        tp, fp, fn, tn = counts(y, p, t_th)
        rows.append({'model': m, 'r': r, 't_theory': t_th, 'cost_theory': c_th,
                     't_empirical': t_emp, 'cost_empirical': c_emp,
                     'cost_flag_none': flag_none, 'cost_flag_all': flag_all,
                     'saving_vs_best_trivial': trivial - c_th,
                     'flagged_rate': (tp + fp) / n, 'sensitivity': tp / (tp + fn), 'precision': tp / (tp + fp) if tp + fp else np.nan})
res = pd.DataFrame(rows)
print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
res.to_csv(BASE / 'cost_summary_val.csv', index=False)
print("flag-all beats flag-none when r <", (1 - prev) / prev)

rng = np.random.default_rng(2)
B = 1000
P = {m: d[m].to_numpy() for m in models}

def cost_at(y, p, r):
    flag = p >= 1 / (1 + r)
    fn = np.sum(~flag & (y == 1)); fp = np.sum(flag & (y == 0))
    return (r * fn + fp) / len(y)

def cost_trivial(y, r):
    return min(r * y.mean(), 1 - y.mean())

for r in [2, 3, 5, 10]:
    sav_nt, sav_gb, diff = [], [], []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        c_nt = cost_at(yb, P['Newton'][idx], r)
        c_gb = cost_at(yb, P['Gradient boosting'][idx], r)
        triv = cost_trivial(yb, r)
        sav_nt.append(triv - c_nt); sav_gb.append(triv - c_gb); diff.append(c_nt - c_gb)
    f = lambda a: f"{np.mean(a):.4f} [{np.percentile(a, 2.5):.4f}, {np.percentile(a, 97.5):.4f}]"
    print(f"r={r:2d}  Newton saving {f(sav_nt)}   GB saving {f(sav_gb)}   GB cheaper by {f(diff)}")

for m in models:
    for r in [2, 3, 5, 10]:
        tp, fp, fn, tn = counts(y, d[m].to_numpy(), 1 / (1 + r))
        print(f"{m:18s} r={r:2d}  caught={tp}  missed={fn}  unnecessary={fp}  correct_no_flag={tn}")