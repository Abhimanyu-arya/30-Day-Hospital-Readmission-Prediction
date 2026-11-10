"""Builds site/results.json for the static website. Aggregates only: no patient-level rows.

Reads the saved outputs of the earlier scripts (val_probs.csv and test_probs.csv are row-level
files and stay on your machine; only counts, rates and curves go into the JSON).
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_curve

BASE = Path(r'E:\ML') / 'Data'
OUT = BASE / 'site'
OUT.mkdir(exist_ok=True)

MODELS = ['Newton', 'Gradient boosting']
R_LIST = [1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]      # slider stops
GRID = np.round(np.arange(0.02, 0.7001, 0.001), 3)               # thresholds for the free slider
B = 300                                                          # bootstrap resamples for the intervals
CHECK_README_NUMBERS = True                                      # compare against numbers already in the README


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return round(float(o), 6) if np.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def records(path):
    return pd.read_csv(path).to_dict('records')


def counts_at(y, p, t):
    flag = p >= t
    tp = int(np.sum(flag & (y == 1))); fp = int(np.sum(flag & (y == 0)))
    fn = int(np.sum(~flag & (y == 1))); tn = int(np.sum(~flag & (y == 0)))
    return tp, fp, fn, tn


def cost_at(y, p, r):
    tp, fp, fn, tn = counts_at(y, p, 1 / (1 + r))
    return (r * fn + fp) / len(y)


def trivial(y, r):
    return min(r * y.mean(), 1 - y.mean())


def grid_counts(y, p):
    """Counts of positives and negatives with score >= t, for every threshold t on GRID."""
    ps, ns = np.sort(p[y == 1]), np.sort(p[y == 0])
    tp = len(ps) - np.searchsorted(ps, GRID, side='left')
    fp = len(ns) - np.searchsorted(ns, GRID, side='left')
    return tp.tolist(), fp.tolist()


def calib(y, p, bins=10):
    out = []
    for c in np.array_split(np.argsort(p, kind='stable'), bins):
        obs = y[c].mean()
        out.append({'pred': p[c].mean(), 'obs': obs, 'n': len(c), 'half': 1.96 * np.sqrt(obs * (1 - obs) / len(c))})
    return out


def roc(y, p, points=150):
    fpr, tpr, _ = roc_curve(y, p)
    idx = np.unique(np.linspace(0, len(fpr) - 1, points).astype(int))
    return {'fpr': fpr[idx], 'tpr': tpr[idx]}


def cost_block(y, P, seed):
    """Cost, saving over the best trivial policy and per-1,000 counts at t = 1/(1+r), with bootstrap intervals."""
    rng = np.random.default_rng(seed)
    n = len(y)
    sav = {(m, r): [] for m in MODELS for r in R_LIST}
    dif = {r: [] for r in R_LIST}
    for _ in range(B):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        pb = {m: P[m][idx] for m in MODELS}
        for r in R_LIST:
            tr = trivial(yb, r)
            c = {m: cost_at(yb, pb[m], r) for m in MODELS}
            for m in MODELS:
                sav[(m, r)].append(tr - c[m])
            dif[r].append(c['Newton'] - c['Gradient boosting'])
    out = {m: [] for m in MODELS}
    for m in MODELS:
        for r in R_LIST:
            t = 1 / (1 + r)
            tp, fp, fn, tn = counts_at(y, P[m], t)
            cost = (r * fn + fp) / n
            tv = trivial(y, r)
            out[m].append({'r': r, 'threshold': t, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                           'cost': cost, 'trivial': tv, 'saving': tv - cost,
                           'sav_lo': np.percentile(sav[(m, r)], 2.5), 'sav_hi': np.percentile(sav[(m, r)], 97.5)})
    out['gb_cheaper_than_newton'] = [{'r': r, 'diff': float(np.mean(dif[r])),
                                      'lo': np.percentile(dif[r], 2.5), 'hi': np.percentile(dif[r], 97.5)} for r in R_LIST]
    return out


# ---------------- load ----------------
splits = {'validation': pd.read_csv(BASE / 'val_probs.csv'),
          'test': pd.read_csv(BASE / 'test_probs.csv')}
y_train = pd.read_csv(BASE / 'y_train.csv')['readmit_30d'].to_numpy()

result = {'meta': {'models': MODELS, 'r_list': R_LIST, 'grid_start': float(GRID[0]), 'grid_step': 0.001,
                   'grid_len': len(GRID), 'bootstrap_resamples': B,
                   'train_base_rate': float(y_train.mean()),
                   'note': 'Aggregates only. Counts at threshold t are TP=pos_ge[t], FP=neg_ge[t], FN=P-TP, TN=N-FP.'},
          'splits': {}}
fz = BASE / 'frozen_config.json'
if fz.exists():
    result['meta']['frozen_config'] = json.loads(fz.read_text())

for si, (name, d) in enumerate(splits.items()):
    y = d['y'].to_numpy().astype(int)
    P = {m: d[m].to_numpy() for m in d.columns if m != 'y'}
    S = {'n': len(y), 'positives': int(y.sum()), 'prevalence': float(y.mean()), 'grid': {}, 'calibration': {}, 'roc': {}}
    for m in MODELS:
        tp, fp = grid_counts(y, P[m])
        S['grid'][m] = {'pos_ge': tp, 'neg_ge': fp}
    for m in [c for c in P if c != 'Baseline']:
        S['calibration'][m] = calib(y, P[m])
        S['roc'][m] = roc(y, P[m])
    S['cost'] = cost_block(y, P, seed=100 + si)
    result['splits'][name] = S

    # consistency checks: grid counts agree with direct counts
    for m in MODELS:
        for t in [0.1, 0.167, 0.25]:
            k = int(round((t - GRID[0]) / 0.001))
            direct = counts_at(y, P[m], round(GRID[k], 3))
            assert (S['grid'][m]['pos_ge'][k], S['grid'][m]['neg_ge'][k]) == (direct[0], direct[1]), (name, m, t)

# ---------------- tables copied from earlier outputs ----------------
result['metrics'] = {'validation': records(BASE / 'metrics_val_all.csv'), 'test': records(BASE / 'test_metrics.csv')}
result['fairness_test'] = records(BASE / 'fairness_test.csv')
result['ablations_val'] = records(BASE / 'ablations_val.csv')
result['coefficients'] = records(BASE / 'coef_se_cluster.csv')

conv = {'newton': pd.read_csv(BASE / 'loss_history_newton.csv').to_dict('list')}
gd_path = BASE / 'loss_history_gd.csv'
if gd_path.exists():
    gd = pd.read_csv(gd_path)
    idx = np.unique(np.round(np.geomspace(1, len(gd), 200)).astype(int)) - 1
    conv['gd'] = gd.iloc[idx].to_dict('list')
result['convergence'] = conv

# ---------------- write ----------------
path = OUT / 'results.json'
path.write_text(json.dumps(clean(result), separators=(',', ':')))
print(f"wrote {path}  ({path.stat().st_size / 1024:.0f} KB)")

# ---------------- checks against numbers already reported ----------------
for name in result['splits']:
    row = {r['r']: r for r in result['splits'][name]['cost']['Newton']}[5]
    n = result['splits'][name]['n']
    print(f"{name:10s} r=5 Newton: saving {row['saving']:.4f} [{row['sav_lo']:.4f}, {row['sav_hi']:.4f}]  per 1000: "
          f"flagged {1000 * (row['tp'] + row['fp']) / n:.1f}, caught {1000 * row['tp'] / n:.1f}, "
          f"missed {1000 * row['fn'] / n:.1f}, unnecessary {1000 * row['fp'] / n:.1f}")
if CHECK_README_NUMBERS:
    t5 = {r['r']: r for r in result['splits']['test']['cost']['Newton']}[5]
    v5 = {r['r']: r for r in result['splits']['validation']['cost']['Newton']}[5]
    g5 = {r['r']: r for r in result['splits']['test']['cost']['Gradient boosting']}[5]
    assert abs(t5['saving'] - 0.0669) < 5e-4, "test r=5 Newton saving differs from the README (0.0669)"
    assert abs(g5['saving'] - 0.0742) < 5e-4, "test r=5 GB saving differs from the README (0.0742)"
    assert abs(v5['saving'] - 0.0772) < 5e-4, "validation r=5 Newton saving differs from the earlier output (0.0772)"
    print("headline numbers match the README")