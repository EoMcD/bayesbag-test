import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from scipy.stats import norm as sp_norm

os.makedirs("figs", exist_ok=True)

# ----------------------------
# Load saved fits + data
# ----------------------------
with open("out/microcredit.pkl", "rb") as f:
    payload_std = pickle.load(f)
standard_fit = payload_std["mg"]
data = payload_std["data"]

data["g"] = data["g"].astype("category")
data["t"] = data["t"].astype("category")

print("here")

with open("out/microcredit_bayesbag.pkl", "rb") as f:
    payload_bag = pickle.load(f)
bayesbag_fits = payload_bag["bayesbag"]

K = standard_fit["theta"]["k"].nunique()
sites = [f"Site {i+1}" for i in range(K)]

print("here")

# ----------------------------
# Helpers: stack draws, HDI, predictive
# ----------------------------
def stack_theta_draws(fit):
    """
    fit: dict with fit["theta"] DataFrame (j, k, mu_k, tau_k, sigma_k)
    Returns (mu, tau, sig) arrays of shape (S, K).
    """
    th = fit["theta"]
    mu = th.pivot(index="j", columns="k", values="mu_k").sort_index(axis=1).values
    ta = th.pivot(index="j", columns="k", values="tau_k").sort_index(axis=1).values
    sg = th.pivot(index="j", columns="k", values="sigma_k").sort_index(axis=1).values
    return mu, ta, sg

def hdi_min_width(draws_2d, prob=0.90):
    """Minimum-width HDI for each column; draws_2d shape (S, K)."""
    S, K = draws_2d.shape
    k = max(1, int(np.ceil(prob * S)))
    out = np.empty((K, 2), dtype=float)
    for g in range(K):
        s = np.sort(draws_2d[:, g])
        if k >= S:
            out[g] = [s[0], s[-1]]
            continue
        w = s[k-1:] - s[:S-k+1]
        j = int(np.argmin(w))
        out[g] = [s[j], s[j+k-1]]
    return out

def avg_log_pred_density_from_theta(fit, df):
    """
    Compute average log predictive density per observation under the posterior mixture.
    df must contain columns g (categorical), t (categorical), y (float).
    """
    th = fit["theta"]
    mu = th.pivot(index="j", columns="k", values="mu_k").sort_index(axis=1).values  # (S,K)
    ta = th.pivot(index="j", columns="k", values="tau_k").sort_index(axis=1).values
    sg = th.pivot(index="j", columns="k", values="sigma_k").sort_index(axis=1).values
    S, K = mu.shape

    y = df["y"].to_numpy()
    gix = df["g"].cat.codes.to_numpy()       
    tit = df["t"].cat.codes.to_numpy()        

    total = 0.0
    for i in range(len(y)):
        k = gix[i]
        loc = mu[:, k] + ta[:, k] * tit[i]
        scale = sg[:, k]
        lp = sp_norm.logpdf(y[i], loc=loc, scale=scale)   # (S,)
        total += logsumexp(lp) - np.log(S)
    return total / len(y)

# ----------------------------
# Build Standard and Bagged posteriors (comparable)
# ----------------------------
muS, tauS, sigS = stack_theta_draws(standard_fit)

muB, tauB, sigB = [], [], []
for fit in bayesbag_fits:
    m, t, s = stack_theta_draws(fit)
    muB.append(m); tauB.append(t); sigB.append(s)
muB = np.vstack(muB)   # (S_total, K)
tauB = np.vstack(tauB)
sigB = np.vstack(sigB)

# ----------------------------
# Summaries per site
# ----------------------------
tau_std_mean = tauS.mean(axis=0)
tau_std_sd   = tauS.std(axis=0, ddof=1)
tau_std_hdi  = hdi_min_width(tauS, 0.90)

tau_bag_mean = tauB.mean(axis=0)
tau_bag_sd   = tauB.std(axis=0, ddof=1)
tau_bag_hdi  = hdi_min_width(tauB, 0.90)

# ----------------------------
# Predictive accuracy (ALPD)
# ----------------------------
alpd_std = avg_log_pred_density_from_theta(standard_fit, data)
theta_bag_df = []
offset = 0
for bf in bayesbag_fits:
    th = bf["theta"].copy()
    th["j"] = th["j"] + offset
    offset = th["j"].max()
    theta_bag_df.append(th)
bag_fit = {"theta": pd.concat(theta_bag_df, ignore_index=True)}
alpd_bag = avg_log_pred_density_from_theta(bag_fit, data)

print(f"\nAvg log predictive density — standard={alpd_std:.4f}  bayesbag={alpd_bag:.4f}  Δ={alpd_bag - alpd_std:+.4f}")

# ----------------------------
# Plots: τ posterior intervals per site
# ----------------------------
x = np.arange(1, K + 1)

plt.figure(figsize=(8, 5))
yerr_std = np.vstack([tau_std_mean - tau_std_hdi[:, 0], tau_std_hdi[:, 1] - tau_std_mean])
yerr_bag = np.vstack([tau_bag_mean - tau_bag_hdi[:, 0], tau_bag_hdi[:, 1] - tau_bag_mean])
plt.errorbar(x - 0.05, tau_std_mean, yerr=yerr_std, fmt='o', capsize=3, label='Standard')
plt.errorbar(x + 0.05, tau_bag_mean, yerr=yerr_bag, fmt='s', capsize=3, label='BayesBag')
plt.xlabel("Site"); plt.ylabel("Treatment effect τ")
plt.title("Posterior 90% intervals by site")
plt.xticks(x, sites, rotation=45)
plt.grid(True, alpha=0.2)
plt.legend()
plt.tight_layout()
plt.savefig("figs/microcredit_tau_intervals.pdf", bbox_inches="tight")
plt.close()

# ----------------------------
# (Optional) per-site posterior SD comparison
# ----------------------------
plt.figure(figsize=(8, 5))
plt.plot(x, tau_std_sd, 'o-', label="Std posterior SD")
plt.plot(x, tau_bag_sd, 's--', label="BayesBag posterior SD")
plt.xlabel("Site"); plt.ylabel("Posterior SD of τ")
plt.title("Posterior spread of τ: Standard vs BayesBag")
plt.xticks(x, sites, rotation=45)
plt.grid(True, alpha=0.2)
plt.legend()
plt.tight_layout()
plt.savefig("figs/microcredit_tau_sd.pdf", bbox_inches="tight")
plt.close()

# ----------------------------
# Mismatch index (coherent version using posterior SDs)
# ----------------------------
var_std = float(np.sum(tau_std_sd ** 2))
var_bag = float(np.sum(tau_bag_sd ** 2))
mismatch_index = 1 - ((2 * var_std) / var_bag) if var_bag > var_std else np.nan
print(f"Mismatch index = {mismatch_index:.4f}")

# ----------------------------
# Relative squared error (keep for continuity; note: heuristic)
# ----------------------------
eps = 1e-8
rse = (tau_bag_mean - tau_std_mean) ** 2 / (tau_bag_sd ** 2 + eps)
print(f"Relative squared error (mean) = {np.mean(rse):.4f}")

print("\nSaved figures:")
print("  figs/microcredit_tau_intervals.pdf")
print("  figs/microcredit_tau_sd.pdf")
