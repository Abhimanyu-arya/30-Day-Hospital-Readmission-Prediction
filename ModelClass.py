import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import log_loss as sk_log_loss
from sklearn.linear_model import LogisticRegression

# Set True once to run the slow gradient descent fit and save W_gd.npy / loss_history_gd.csv,
# then set it back to False. When False, W_gd.npy is loaded if it exists.
RUN_GD = False 

BASE = Path(r'E:\ML') / 'Data'
X_train_df = pd.read_csv(BASE / 'X_train.csv')
X_test_df = pd.read_csv(BASE / 'X_test.csv')
X_val_df = pd.read_csv(BASE / 'X_val.csv')

X_train = X_train_df.to_numpy(dtype = float)
X_test = X_test_df.to_numpy(dtype = float)
X_val = X_val_df.to_numpy(dtype = float)

y_train = pd.read_csv(BASE / 'y_train.csv')['readmit_30d'].to_numpy()
y_test = pd.read_csv(BASE / 'y_test.csv')['readmit_30d'].to_numpy()
y_val = pd.read_csv(BASE / 'y_val.csv')['readmit_30d'].to_numpy()

fair_train = pd.read_csv(BASE / 'fair_train.csv')
fair_test = pd.read_csv(BASE / 'fair_test.csv')
fair_val = pd.read_csv(BASE / 'fair_val.csv')

feature_names = X_train_df.columns.to_list()
names = ['intercept'] + feature_names

# Race is excluded from X on purpose (it is only kept in the fair_* files for the fairness audit)
assert not any(c.startswith('race_') for c in feature_names)
assert set(np.unique(y_train)) <= {0, 1}


#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)

#print(X_train_df.describe())



def add_intercept(X):
    Xb = np.hstack((np.ones((X.shape[0], 1)), X))
    return Xb

Xb_train = add_intercept(X_train)
Xb_val = add_intercept(X_val)
Xb_test = add_intercept(X_test)
# expect (69396, 100) (10089, 100) (19835, 100) for the 99-feature version
print(Xb_train.shape)
print(Xb_val.shape)
print(Xb_test.shape)
assert Xb_train.shape == (len(y_train), X_train.shape[1] + 1)
assert Xb_val.shape == (len(y_val), X_val.shape[1] + 1)
assert Xb_test.shape == (len(y_test), X_test.shape[1] + 1)
assert Xb_train.shape[1] == Xb_val.shape[1] == Xb_test.shape[1]

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))

z = np.array([-1000, -40, -1, 0, 1, 40, 1000])
print(sigmoid(z))
print(sigmoid(0), sigmoid(2) + sigmoid(-2))

def log_loss(y_true, y_pred, eps = 1e-15):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred = np.clip(y_pred, eps, 1-eps)
    loss = -np.mean(y_true * np.log(y_pred) + (1-y_true)*np.log(1-y_pred))
    return loss

#y = np.array([1, 0])
#print(log_loss(y, np.array([0.5, 0.5])))   # should be ln 2 = 0.6931
#print(log_loss(np.array([0]), np.array([0.9])))  # should be -ln(0.1) = 2.3026

rng = np.random.default_rng(0)
#y = rng.integers(0, 2, 1000)
#p = rng.uniform(0.01, 0.99, 1000)
#print(log_loss(y, p), sk_log_loss(y, p))




def gradient(W, Xb, y):
    p = sigmoid(Xb @ W)
    n = Xb.shape[0]
    grad = Xb.T@(p-y)/n
    return grad

def grad_check(W, Xb, y, idx, h=1e-5):
    e = np.zeros_like(W)
    e[idx] = h
    loss_plus = log_loss(y, sigmoid(Xb @ (W + e)))
    loss_minus = log_loss(y, sigmoid(Xb @ (W-e)))
    return ((loss_plus - loss_minus)/(2*h))

rng = np.random.default_rng(0)
W = rng.normal(0, 0.1, Xb_train.shape[1])
g = gradient(W, Xb_train, y_train)
for i in [0, 1, 5, 50, 99]:
    print(i, g[i], grad_check(W, Xb_train, y_train, i))


# ---------- 2. Gradient descent (full batch) ----------
# History convention (same for every fit function): index k = state after k updates,
# so index 0 is the starting point (loss ln 2 for W = 0).
def fit_gd(Xb, y, Xb_val, y_val, lr=0.1, max_iter=5000, tol=1e-6, print_every=100):
    W = np.zeros(Xb.shape[1])
    g = gradient(W, Xb, y)
    train_hist = [log_loss(y, sigmoid(Xb@W))]
    val_hist = [log_loss(y_val, sigmoid(Xb_val@W))]
    gnorm_hist = [np.linalg.norm(g)]
    print(f"iteration = 0, train_loss = {train_hist[0]:.6f}, val_loss = {val_hist[0]:.6f}, norm_g = {gnorm_hist[0]:.3e}")
    for k in range(1, max_iter+1):
        if gnorm_hist[-1] < tol:
            break
        W = W - lr*g
        g = gradient(W, Xb, y)
        train_hist.append(log_loss(y, sigmoid(Xb@W)))
        val_hist.append(log_loss(y_val, sigmoid(Xb_val@W)))
        gnorm_hist.append(np.linalg.norm(g))
        if k % print_every == 0:
            print(f"iteration = {k}, train_loss = {train_hist[-1]:.6f}, val_loss = {val_hist[-1]:.6f}, norm_g = {gnorm_hist[-1]:.3e}")
    return W, train_hist, val_hist, gnorm_hist

W_gd = None
if RUN_GD:
    W_gd, tr_h, va_h, gn_h = fit_gd(Xb_train, y_train, Xb_val, y_val, lr=0.1, max_iter=50000)
    np.save(BASE / 'W_gd.npy', W_gd)
    pd.DataFrame({
        'iteration': np.arange(len(tr_h)),
        'train_loss': tr_h,
        'val_loss': va_h,
        'grad_norm': gn_h,
    }).to_csv(BASE / 'loss_history_gd.csv', index = False)
elif (BASE / 'W_gd.npy').exists():
    W_gd = np.load(BASE / 'W_gd.npy')




# ---------- 3. Hessian + Newton's method ----------
def hessian(W, Xb):
    p = sigmoid(Xb@W)
    s = p*(1-p)
    n = Xb.shape[0]
    H = (Xb.T * s) @ Xb
    return H/n 

def fit_newton(Xb, y, Xb_val, y_val, max_iter=25, tol=1e-10):
    W = np.zeros(Xb.shape[1])
    g = gradient(W, Xb, y)
    train_hist = [log_loss(y, sigmoid(Xb@W), 1e-15)]
    val_hist = [log_loss(y_val, sigmoid(Xb_val@W), 1e-15)]
    gnorm_hist = [np.linalg.norm(g)]
    print("initial train loss")
    print(f"{train_hist[0]:.6f}")
    print(f"0, train_loss = {train_hist[0]:.6f}, val_loss = {val_hist[0]:.6f}, g_norm = {gnorm_hist[0]:.3e}")
    for k in range(1, max_iter+1):
        if gnorm_hist[-1] < tol:
            break
        H = hessian(W, Xb)
        step = np.linalg.solve(H, g)
        W = W - step
        g = gradient(W, Xb, y)
        train_hist.append(log_loss(y, sigmoid(Xb@W), 1e-15))
        val_hist.append(log_loss(y_val, sigmoid(Xb_val@W), 1e-15))
        gnorm_hist.append(np.linalg.norm(g))
        print(f"{k}, train_loss = {train_hist[-1]:.6f}, val_loss = {val_hist[-1]:.6f}, g_norm = {gnorm_hist[-1]:.3e}")
    return W, train_hist, val_hist, gnorm_hist

W = np.zeros(Xb_train.shape[1])
e = np.zeros_like(W)
e[3] = 1e-5
fd = (gradient(W+e, Xb_train, y_train) - gradient(W-e, Xb_train, y_train))/2e-5
print(np.max(np.abs(fd-hessian(W, Xb_train)[:, 3])))
H = hessian(W, Xb_train)
print(H.shape)
print(np.allclose(H, H.T))

W_nt, tr_nt, va_nt, gn_nt = fit_newton(Xb_train, y_train, Xb_val, y_val)
np.save(BASE / 'W_nt.npy', W_nt)

pd.DataFrame({
    'iteration': np.arange(len(tr_nt)),
    'train_loss': tr_nt,
    'val_loss': va_nt,
    'grad_norm': gn_nt,
}).to_csv(BASE / 'loss_history_newton.csv', index = False)


coef_df = pd.DataFrame({'feature': names, 'coef': W_nt})
if W_gd is not None:
    coef_df['coef_gd'] = W_gd
coef_df.to_csv(BASE / 'coefficients.csv', index = False)

print(np.max(np.abs(W_gd - W_nt)))
loss_gd = log_loss(y_train, sigmoid(Xb_train @ W_gd))
loss_nt = log_loss(y_train, sigmoid(Xb_train @ W_nt))
print(loss_gd, loss_nt, loss_gd - loss_nt)
#d = np.abs(W_gd - W_nt)
#order = np.argsort(d)[::-1][:8]
#for j in order:
    #print(names[j], W_gd[j], W_nt[j], d[j])

clf = LogisticRegression(penalty = None, solver = 'newton-cg', tol = 1e-12, max_iter = 1000).fit(X_train, y_train)
print(clf.intercept_[0], W_nt[0])
print(np.max(np.abs(clf.coef_[0] - W_nt[1:])))



def ridge_gradient(W, Xb, y, lam):
    g = gradient(W, Xb, y)
    pen = lam*W
    pen[0] = 0.0
    return g + pen

def ridge_hessian(W, Xb, lam):
    H = hessian(W, Xb)
    P = lam*np.eye(Xb.shape[1])
    P[0, 0] = 0.0
    return H + P

def fit_newton_ridge(Xb, y, Xb_val, y_val, lam, max_iter = 25, tol = 1e-10):
    W = np.zeros(Xb.shape[1])
    g = ridge_gradient(W, Xb, y, lam)
    train_hist = [log_loss(y, sigmoid(Xb@W), 1e-15)]
    val_hist = [log_loss(y_val, sigmoid(Xb_val@W), 1e-15)]
    gnorm_hist = [np.linalg.norm(g)]
    #print("initial train loss")
    #print(f"{train_hist[0]:.6f}")
    for k in range(1, max_iter+1):
        if gnorm_hist[-1] < tol:
            break
        H = ridge_hessian(W, Xb, lam)
        step = np.linalg.solve(H, g)
        W = W - step
        g = ridge_gradient(W, Xb, y, lam)
        train_hist.append(log_loss(y, sigmoid(Xb@W), 1e-15))
        val_hist.append(log_loss(y_val, sigmoid(Xb_val@W), 1e-15))
        gnorm_hist.append(np.linalg.norm(g))
        #print(f"{k}, train_loss = {train_hist[-1]:.6f}, val_loss = {val_hist[-1]:.6f}, g_norm = {gnorm_hist[-1]:.3e}")
    return W, train_hist, val_hist, gnorm_hist

W_r_1e3, _, _, _ = fit_newton_ridge(Xb_train, y_train, Xb_val, y_val, lam=1e-3, max_iter = 25, tol = 1e-10)
#print(np.max(np.abs(W_r_1e3 - W_nt)))
#print(W_r_1e3 is W_nt)
#print(np.linalg.norm(W_r_1e3[1:]), np.linalg.norm(W_nt[1:]))
#print(W_r_1e3[:3], W_nt[:3])

j = np.argmax(np.abs(W_r_1e3 - W_nt))
#print(names[j], W_r_1e3[j], W_nt[j])

def J(W, lam):
    return log_loss(y_train, sigmoid(Xb_train @ W)) + 0.5*lam*np.sum(W[1:]**2)

rng = np.random.default_rng(0)
W0 = rng.normal(0, 0.1, Xb_train.shape[1])
for lam_t in [0.0, 1e-3]:
    g = ridge_gradient(W0, Xb_train, y_train, lam_t)
    for i in [0, 1, 50]:
        e = np.zeros_like(W0); e[i] = 1e-5
        #print(lam_t, i, g[i], (J(W0+e, lam_t) - J(W0-e, lam_t)) / 2e-5)

n = Xb_train.shape[0]
clf = LogisticRegression(penalty='l2', C=1/(1e-3*n), solver='newton-cg', tol=1e-12, max_iter=1000).fit(X_train, y_train)
print(clf.intercept_[0] - W_r_1e3[0], np.max(np.abs(clf.coef_[0] - W_r_1e3[1:])))


lams = [0, 1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3]
rare = ['rosiglitazone_Down', 'pioglitazone_Up', 'glipizide_Down']
rare_idx = [names.index(r) for r in rare]

weights = {}
rows = []
for lam in lams:
    W_l, _, _, _ = fit_newton_ridge(Xb_train, y_train, Xb_val, y_val, lam=lam)
    weights[lam] = W_l
    row = {
        'lam': lam,
        'train_loss': log_loss(y_train, sigmoid(Xb_train @ W_l)),
        'val_loss': log_loss(y_val, sigmoid(Xb_val @ W_l)),
        'norm_W': np.linalg.norm(W_l[1:]),
    }
    for r, j in zip(rare, rare_idx):
        row[r] = W_l[j]
    rows.append(row)

sweep = pd.DataFrame(rows)
sweep.to_csv(BASE / 'ridge_sweep.csv', index=False)
print(sweep.to_string(float_format=lambda v: f"{v:.6f}"))

best = float(sweep.loc[sweep['val_loss'].idxmin(), 'lam'])
print("best lam by validation loss:", best)
np.save(BASE / 'W_ridge_best.npy', weights[best])
np.save(BASE / 'W_ridge_1e-4.npy', weights[1e-4])

# ---------- 4. Prediction helpers ----------
def predict_proba(W, Xb):
    return sigmoid(Xb @ W)

def predict(W, Xb, threshold=0.5):
    return (predict_proba(W, Xb) >= threshold).astype(int)

# sanity: mean predicted probability should be close to the base rate (about 0.116 on val)
print(predict_proba(W_nt, Xb_val).mean(), y_val.mean())
print(predict_proba(weights[1e-4], Xb_val).mean(), y_val.mean())