import numpy as np
import pandas as pd
from pathlib import Path
from model import add_intercept, predict_proba

BASE = Path(r'E:\ML') / 'Data'
Xtr = pd.read_csv(BASE / 'X_train.csv')
Xva = pd.read_csv(BASE / 'X_val.csv')
yva = pd.read_csv(BASE / 'y_val.csv')['readmit_30d'].to_numpy()
fva = pd.read_csv(BASE / 'fair_val.csv')
W = np.load(BASE / 'W_nt.npy')
p = predict_proba(W, add_intercept(Xva.to_numpy(float)))

vp = pd.read_csv(BASE / 'val_probs.csv')           # row alignment check with the earlier outputs
assert len(vp) == len(Xva) == len(fva) and np.allclose(vp['Newton'], p) and (vp['y'].to_numpy() == yva).all()

T = 1 / 6                                          # r = 5, the worked example
flag = p >= T
TP, FN = flag & (yva == 1), ~flag & (yva == 1)
FP, TN = flag & (yva == 0), ~flag & (yva == 0)
print(f"validation, threshold {T:.3f}: TP={TP.sum()} FN={FN.sum()} FP={FP.sum()} TN={TN.sum()}")

# raw visit counts, recovered from the standardized columns (same trick as the ablation)
VISITS = ['number_outpatient', 'number_emergency', 'number_inpatient']
def raw_counts(col):
    u = np.unique(Xtr[col]); gap = np.diff(u).min()
    r = (Xva[col].to_numpy() - u[0]) / gap
    assert np.abs(r - np.round(r)).max() < 1e-6
    return np.round(r).astype(int)
raw = {c: raw_counts(c) for c in VISITS}

dis = [c for c in Xva.columns if c.startswith('discharge_group_')]
home = (Xva[dis].sum(axis=1) == 0).to_numpy()      # all discharge dummies 0 = reference group (check which one in DataSplit.py)
age = fva['age_numeric'].to_numpy()

# ---- A. where do the readmissions sit? ----
dec = pd.qcut(pd.Series(p).rank(method='first'), 10, labels=False)   # 0 = lowest risk
A = pd.DataFrame({'decile': dec, 'p': p, 'y': yva}).groupby('decile').agg(
    n=('y', 'size'), mean_pred=('p', 'mean'), observed=('y', 'mean'), readmissions=('y', 'sum'))
A = A.iloc[::-1]                                    # highest risk first
A['cum_share_of_readmissions'] = A['readmissions'].cumsum() / yva.sum()
print("\nA. risk deciles (highest risk first)")
print(A.to_string(float_format=lambda v: f"{v:.3f}"))

# ---- B. profile of the four error groups ----
def prof(m):
    return {'n': int(m.sum()), 'mean_pred': p[m].mean(), 'age': age[m].mean(),
            'prior_inpatient': raw['number_inpatient'][m].mean(),
            'share_no_prior_inpatient': (raw['number_inpatient'][m] == 0).mean(),
            'prior_emergency': raw['number_emergency'][m].mean(),
            'prior_outpatient': raw['number_outpatient'][m].mean(),
            'share_home_discharge': home[m].mean(),
            'share_rehab': Xva['discharge_group_Rehab'].to_numpy()[m].mean(),
            'diagnoses_z': Xva['number_diagnoses'].to_numpy()[m].mean(),
            'time_in_hospital_z': Xva['time_in_hospital'].to_numpy()[m].mean()}
B = pd.DataFrame({'TP (caught)': prof(TP), 'FN (missed)': prof(FN),
                  'FP (false alarm)': prof(FP), 'TN (correct)': prof(TN)})
print("\nB. group profiles (columns ending _z are in standard deviations)")
print(B.to_string(float_format=lambda v: f"{v:.3f}"))
B.to_csv(BASE / 'error_groups_val.csv')

# ---- C. does prior history explain who gets missed? ----
def slice_stats(label, m):
    pos = yva[m].sum()
    return {'slice': label, 'n': int(m.sum()), 'readmissions': int(pos), 'prevalence': yva[m].mean(),
            'share_of_all_readmissions': pos / yva.sum(), 'flagged_rate': flag[m].mean(),
            'sensitivity': (flag & (yva == 1) & m).sum() / pos if pos else np.nan,
            'precision': (flag & (yva == 1) & m).sum() / flag[m].sum() if flag[m].sum() else np.nan}
ni = raw['number_inpatient']
C = pd.DataFrame([slice_stats('no prior inpatient visit', ni == 0),
                  slice_stats('1+ prior inpatient visits', ni >= 1),
                  slice_stats('home discharge (reference)', home),
                  slice_stats('other discharge', ~home),
                  slice_stats('no prior inpatient AND home discharge', (ni == 0) & home)])
print("\nC. sensitivity by history slice")
print(C.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

# ---- D. a few individual cases: top contributions (coef x standardized value) ----
Xv = Xva.to_numpy(float)
def case(i, tag):
    c = Xv[i] * W[1:]
    top = np.argsort(-np.abs(c))[:5]
    items = ', '.join(f"{Xva.columns[j]} ({c[j]:+.2f})" for j in top)
    print(f"[{tag}] row {i}: p={p[i]:.3f} y={yva[i]} age={age[i]:.0f} prior_inpatient={raw['number_inpatient'][i]} "
          f"prior_emergency={raw['number_emergency'][i]} home_discharge={bool(home[i])}\n     top log-odds contributions: {items}")
fn_idx = np.where(FN)[0]; fp_idx = np.where(FP)[0]
print("\nD. cases (log-odds contributions; the intercept is excluded)")
for i in fn_idx[np.argsort(p[fn_idx])][:3]:   case(i, 'missed, lowest score')
for i in fp_idx[np.argsort(-p[fp_idx])][:3]:  case(i, 'false alarm, highest score')
for i in fp_idx[np.argsort(-p[fp_idx])][len(fp_idx)//2:len(fp_idx)//2+2]: case(i, 'false alarm, typical')

print("\nE. calibration by prior inpatient visits")
for lo, hi in [(0, 0), (1, 1), (2, 2), (3, 4), (5, 7), (8, 100)]:
    m = (ni >= lo) & (ni <= hi)
    print(f"prior inpatient {lo}-{hi}: n={m.sum()}, readmissions={yva[m].sum()}, "
          f"mean_pred={p[m].mean():.3f}, observed={yva[m].mean():.3f}")

import pandas as pd
d = pd.read_csv(r'E:\ML\Data\patient_splits.csv')
print(d.shape); print(d.head()); print(d.nunique())