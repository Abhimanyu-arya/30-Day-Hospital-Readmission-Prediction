import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm
from model import add_intercept, hessian, gradient, sigmoid

BASE = Path(r'E:\ML') / 'Data'
df = pd.read_csv(BASE / 'Final_Data.csv')
ps = pd.read_csv(BASE / 'patient_splits.csv')
Xdf = pd.read_csv(BASE / 'X_train.csv')
y = pd.read_csv(BASE / 'y_train.csv')['readmit_30d'].to_numpy()

# rebuild the patient id of every training row (same boolean mask logic as DataSplit.py)
mask = df['patient_nbr'].isin(set(ps.loc[ps['split'] == 'train', 'patient_nbr'])).to_numpy()
assert mask.sum() == len(Xdf) == len(y)
assert (df.loc[mask, 'readmit_30d'].to_numpy() == y).all()                 # labels line up row by row
raw = df.loc[mask, 'number_inpatient']
assert np.allclose((raw - raw.mean()) / raw.std(ddof=0), Xdf['number_inpatient'])   # features line up too
pid = df.loc[mask, 'patient_nbr'].to_numpy()
codes = np.unique(pid, return_inverse=True)[1]
G = codes.max() + 1

Xb = add_intercept(Xdf.to_numpy(float))
N, K = Xb.shape
names = ['intercept'] + list(Xdf.columns)
W = np.load(BASE / 'W_nt.npy')
assert np.linalg.norm(gradient(W, Xb, y)) < 1e-8

p = sigmoid(Xb @ W)
Ainv = np.linalg.inv(hessian(W, Xb)) / N               # inverse of the total information
S = Xb * (p - y)[:, None]                              # per-row score contributions
Gs = np.zeros((G, K)); np.add.at(Gs, codes, S)         # summed within each patient
cov = Ainv @ (Gs.T @ Gs) @ Ainv                        # sandwich
cov *= G / (G - 1) * (N - 1) / (N - K)                 # small-sample correction

se_naive = np.sqrt(np.diag(Ainv))
se_cl = np.sqrt(np.diag(cov))
z = W / se_cl
pv = 2 * norm.sf(np.abs(z))
tab = pd.DataFrame({'feature': names, 'coef': W, 'se_naive': se_naive, 'se_cluster': se_cl,
                    'ratio': se_cl / se_naive, 'z_cluster': z, 'p_cluster': pv,
                    'or': np.exp(W), 'or_lo': np.exp(W - 1.96 * se_cl), 'or_hi': np.exp(W + 1.96 * se_cl)})
tab.to_csv(BASE / 'coef_se_cluster.csv', index=False)

print(f"{N} encounters from {G} patients ({N / G:.2f} per patient; {np.mean(np.bincount(codes) > 1):.1%} of patients have 2+)")
print("SE ratio (cluster / naive): median %.3f, max %.3f (%s)" % (
    tab.ratio.median(), tab.ratio.max(), tab.loc[tab.ratio.idxmax(), 'feature']))
print(f"p < 0.05: naive {(2 * norm.sf(np.abs(W / se_naive)) < 0.05).sum()}, cluster {(pv < 0.05).sum()}; "
      f"Bonferroni: cluster {(pv < 0.05 / K).sum()}")
print(tab.reindex(tab.z_cluster.abs().sort_values(ascending=False).index).head(12)
      [['feature', 'coef', 'se_naive', 'se_cluster', 'z_cluster', 'or_lo', 'or_hi']]
      .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

try:
    import statsmodels.api as sm
    fit = sm.Logit(y, Xb).fit(method='newton', disp=0, maxiter=50, cov_type='cluster',
                              cov_kwds={'groups': codes, 'use_correction': False})
    cov0 = Ainv @ (Gs.T @ Gs) @ Ainv                   # uncorrected version, for the cross-check
    print("max |SE - statsmodels cluster SE|:", np.max(np.abs(np.sqrt(np.diag(cov0)) - fit.bse)))
except ImportError:
    print("statsmodels not installed: skipping the cross-check")