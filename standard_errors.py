import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm
from model import add_intercept, hessian, gradient

BASE = Path(r'E:\ML') / 'Data'
Xdf = pd.read_csv(BASE / 'X_train.csv')
y = pd.read_csv(BASE / 'y_train.csv')['readmit_30d'].to_numpy()
Xb = add_intercept(Xdf.to_numpy(float))
n = Xb.shape[0]
names = ['intercept'] + list(Xdf.columns)

W = np.load(BASE / 'W_nt.npy')
assert np.linalg.norm(gradient(W, Xb, y)) < 1e-8, "W_nt is not the converged MLE"

H = hessian(W, Xb)                    # average Hessian (divided by n)
cov = np.linalg.inv(H) / n            # covariance = inverse of the total information n*H
se = np.sqrt(np.diag(cov))
z = W / se
p = 2 * norm.sf(np.abs(z))
tab = pd.DataFrame({'feature': names, 'coef': W, 'se': se, 'z': z, 'p': p,
                    'odds_ratio': np.exp(W),
                    'or_lo': np.exp(W - 1.96 * se), 'or_hi': np.exp(W + 1.96 * se)})

# how far did Ridge (lam = 1e-4) move each coefficient, in standard errors?
W_r = np.load(BASE / 'W_ridge_1e-4.npy')
tab['ridge_1e-4'] = W_r
tab['shift_in_se'] = (W_r - W) / se
tab.to_csv(BASE / 'coef_se.csv', index=False)

# ---- checks ----
print("condition number of the Hessian at the MLE:", np.linalg.cond(H))
print("condition number of the design matrix (singular value ratio):", np.linalg.cond(Xb))
try:
    import statsmodels.api as sm
    fit = sm.Logit(y, Xb).fit(method='newton', disp=0, maxiter=50)
    print("max |SE - statsmodels SE|:", np.max(np.abs(se - fit.bse)))
except ImportError:
    print("statsmodels not installed (pip install statsmodels) -> skipping the cross-check")

# ---- results ----
pd.set_option('display.width', 200)
fmt = lambda v: f"{v:.4f}"
print(f"\n{(p < 0.05).sum()} of {len(p)} coefficients have p < 0.05; "
      f"{(p < 0.05 / len(p)).sum()} survive a Bonferroni correction (p < {0.05/len(p):.1e})")
print("\nlargest |z|:")
print(tab.reindex(tab['z'].abs().sort_values(ascending=False).index).head(15)
      [['feature', 'coef', 'se', 'z', 'odds_ratio', 'or_lo', 'or_hi']].to_string(index=False, float_format=fmt))
rare = ['rosiglitazone_Down', 'pioglitazone_Up', 'glipizide_Down']
print("\nrare columns, and how much Ridge moved them:")
print(tab[tab.feature.isin(rare)][['feature', 'coef', 'se', 'z', 'ridge_1e-4', 'shift_in_se']].to_string(index=False, float_format=fmt))
print("\nlargest Ridge shifts (in SE):")
print(tab.reindex(tab['shift_in_se'].abs().sort_values(ascending=False).index).head(8)
      [['feature', 'coef', 'se', 'ridge_1e-4', 'shift_in_se']].to_string(index=False, float_format=fmt))