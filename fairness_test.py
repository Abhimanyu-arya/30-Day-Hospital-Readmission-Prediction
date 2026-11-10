import numpy as np
import pandas as pd
from pathlib import Path
from model import auc

BASE = Path(r'E:\ML') / 'Data'
T = pd.read_csv(BASE / 'test_probs.csv')
F = pd.read_csv(BASE / 'fair_test.csv')
assert len(T) == len(F) == 19835
y = T['y'].to_numpy()

print(F['gender'].value_counts(dropna=False))
print(F['age_numeric'].describe())
assert F['age_numeric'].min() >= 0, "age_numeric looks standardized: send me this output"

# race: rows with all dummies 0 are the reference group (confirm which one in DataSplit.py)
race = pd.Series('Reference', index=F.index)
for c in [c for c in F.columns if c.startswith('race_')]:
    race[F[c] == 1] = c.replace('race_', '')
age = pd.cut(F['age_numeric'], [0, 50, 60, 70, 80, 200], right=False,
             labels=['<50', '50-59', '60-69', '70-79', '80+']).astype(str)
attrs = {'race': race.astype(str), 'gender': F['gender'].astype(str), 'age': age}

def wilson(k, n, z=1.96):
    if n == 0: return np.nan, np.nan
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h

rows = []
for attr, labels in attrs.items():
    groups = [('All', np.ones(len(y), bool))] if attr == 'race' else []
    groups += [(g, (labels == g).to_numpy()) for g in sorted(labels.unique())]
    for g, m in groups:
        yy = y[m]
        for model in ['Newton', 'Gradient boosting']:
            p = T[model].to_numpy()[m]
            for r in [3, 5, 10]:
                flag = p >= 1 / (1 + r)
                tp = int((flag & (yy == 1)).sum()); fn = int((~flag & (yy == 1)).sum())
                fp = int((flag & (yy == 0)).sum()); tn = int((~flag & (yy == 0)).sum())
                s_lo, s_hi = wilson(tp, tp + fn); f_lo, f_hi = wilson(fp, fp + tn)
                rows.append({'attribute': attr, 'group': g, 'model': model, 'r': r,
                             'n': len(yy), 'positives': int(yy.sum()), 'prevalence': yy.mean(),
                             'mean_pred': p.mean(),
                             'auc': auc(yy, p) if 0 < yy.sum() < len(yy) else np.nan,
                             'flagged_rate': flag.mean(),
                             'sensitivity': tp / (tp + fn) if tp + fn else np.nan, 'sens_lo': s_lo, 'sens_hi': s_hi,
                             'fpr': fp / (fp + tn) if fp + tn else np.nan, 'fpr_lo': f_lo, 'fpr_hi': f_hi,
                             'precision': tp / (tp + fp) if tp + fp else np.nan,
                             'cost': (r * fn + fp) / len(yy)})
res = pd.DataFrame(rows)
res.to_csv(BASE / 'fairness_test.csv', index=False)
show = ['attribute', 'group', 'n', 'positives', 'prevalence', 'mean_pred', 'auc', 'flagged_rate',
        'sensitivity', 'sens_lo', 'sens_hi', 'fpr', 'precision']
for model in ['Newton', 'Gradient boosting']:
    print(f"\n=== {model}, r=5 (threshold {1/6:.3f}) ===")
    print(res[(res.model == model) & (res.r == 5)][show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))