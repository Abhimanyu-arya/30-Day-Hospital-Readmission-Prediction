import numpy as np
from scipy.stats import rankdata

def add_intercept(X):
    return np.hstack((np.ones((X.shape[0], 1)), X))

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))

def log_loss(y_true, y_pred, eps=1e-15):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def gradient(W, Xb, y):
    return Xb.T @ (sigmoid(Xb @ W) - y) / Xb.shape[0]

def hessian(W, Xb):
    p = sigmoid(Xb @ W)
    return (Xb.T * (p * (1 - p))) @ Xb / Xb.shape[0]

def fit_newton(Xb, y, lam=0.0, max_iter=25, tol=1e-10):
    """Newton's method; lam=0 is plain logistic regression. The intercept is never penalized."""
    n, d = Xb.shape
    pen = np.full(d, lam); pen[0] = 0.0
    W = np.zeros(d)
    for _ in range(max_iter):
        g = gradient(W, Xb, y) + pen * W
        if np.linalg.norm(g) < tol:
            break
        H = hessian(W, Xb) + np.diag(pen)
        W = W - np.linalg.solve(H, g)
    return W

def predict_proba(W, Xb):
    return sigmoid(Xb @ W)

def brier(y, p):
    return np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2)

def auc(y, s):
    y = np.asarray(y)
    ranks = rankdata(s)
    n_pos = int(np.sum(y == 1)); n_neg = len(y) - n_pos
    return (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)