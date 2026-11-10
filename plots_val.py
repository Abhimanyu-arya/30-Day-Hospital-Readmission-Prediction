import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_curve

BASE = Path(r'E:\ML') / 'Data'
FIG = BASE / 'figures'
FIG.mkdir(exist_ok=True)

d = pd.read_csv(BASE / 'val_probs.csv')
y = d['y'].to_numpy()
models = ['Newton', 'Ridge 1e-4', 'Random forest', 'Gradient boosting']

# ---- 1. ROC curves ----
plt.figure(figsize=(5, 5))
for m in models:
    fpr, tpr, _ = roc_curve(y, d[m])
    plt.plot(fpr, tpr, label=m)
plt.plot([0, 1], [0, 1], 'k--', label='chance')
plt.xlabel('False positive rate'); plt.ylabel('True positive rate')
plt.title('ROC, validation'); plt.legend(); plt.tight_layout()
plt.savefig(FIG / 'roc_val.png', dpi=200); plt.close()

# ---- 2. Calibration (10 equal-count bins, with 95% intervals) ----
def calib(y, p, bins=10):
    out = []
    for c in np.array_split(np.argsort(p), bins):
        obs = y[c].mean()
        out.append([p[c].mean(), obs, 1.96 * np.sqrt(obs * (1 - obs) / len(c))])
    return np.array(out)

plt.figure(figsize=(5.5, 5.5))
for m in ['Newton', 'Random forest', 'Gradient boosting']:
    c = calib(y, d[m].to_numpy())
    plt.errorbar(c[:, 0], c[:, 1], yerr=c[:, 2], fmt='o-', capsize=2, label=m)
lim = 0.35
plt.plot([0, lim], [0, lim], 'k--', label='perfect')
plt.xlim(0, lim); plt.ylim(0, lim)
plt.xlabel('Mean predicted probability'); plt.ylabel('Observed readmission rate')
plt.title('Calibration, validation (95% intervals)'); plt.legend(); plt.tight_layout()
plt.savefig(FIG / 'calibration_val.png', dpi=200); plt.close()

# ---- calibration slope: regress y on logit(p). slope 1 = perfect, <1 overconfident, >1 underconfident ----
from sklearn.linear_model import LogisticRegression
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

for m in ['Newton', 'Random forest', 'Gradient boosting']:
    z = logit(d[m].to_numpy()).reshape(-1, 1)
    lr = LogisticRegression(C=1e10, max_iter=1000).fit(z, y)
    print(f"{m:18s} slope = {lr.coef_[0, 0]:.3f}   mean predicted = {d[m].mean():.4f}   observed = {y.mean():.4f}")

# ---- 3. Convergence: Newton vs gradient descent ----
nt = pd.read_csv(BASE / 'loss_history_newton.csv')
plt.figure(figsize=(6, 4))
plt.loglog(nt['iteration'] + 1, nt['grad_norm'], 'o-', label='Newton')
gd_path = BASE / 'loss_history_gd.csv'
if gd_path.exists():
    gd = pd.read_csv(gd_path)
    plt.loglog(gd['iteration'] + 1, gd['grad_norm'], label='Gradient descent')
plt.xlabel('Updates + 1'); plt.ylabel('Gradient norm')
plt.title('Convergence'); plt.legend(); plt.tight_layout()
plt.savefig(FIG / 'convergence.png', dpi=200); plt.close()
print('saved to', FIG)
