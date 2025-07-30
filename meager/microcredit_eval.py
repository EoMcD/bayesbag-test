import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("figs", exist_ok=True)

with open("out/microcredit.pkl", "rb") as f:
    standard_fit = pickle.load(f)["mg"]

with open("out/microcredit_bayesbag.pkl", "rb") as f:
    bayesbag_fits = pickle.load(f)["bayesbag"]

K = standard_fit["theta"]["k"].nunique()
sites = [f"Site {i+1}" for i in range(K)]

theta_standard = standard_fit["theta"]
theta_standard_mean = theta_standard.groupby("k")["tau_k"].mean().values
theta_standard_std = theta_standard.groupby("k")["tau_k"].std().values

tau_k_bagged = []

for fit in bayesbag_fits:
    theta_df = fit["theta"]
    tau_k_means = theta_df.groupby("k")["tau_k"].mean()
    tau_k_bagged.append(tau_k_means)

tau_k_bagged = np.vstack(tau_k_bagged)
theta_bagged_mean = tau_k_bagged.mean(axis=0)
theta_bagged_std = tau_k_bagged.std(axis=0)

# ---- Plot posterior means ----
plt.figure(figsize=(7, 5))
plt.plot(theta_standard_mean, 'o-', label="Standard Posterior Mean", color='blue')
plt.plot(theta_bagged_mean, 's--', label="BayesBag Posterior Mean", color='red')
plt.xlabel("Site Index")
plt.ylabel("Estimated Treatment Effect (τ)")
plt.title("Standard vs BayesBag Posterior Means")
plt.xticks(np.arange(K), labels=sites, rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figs/microcredit_means.pdf", bbox_inches="tight")
plt.close()

# ---- Plot posterior std deviations ----
plt.figure(figsize=(7, 5))
plt.plot(theta_standard_std, 'o-', label="Standard Posterior Std", color='blue')
plt.plot(theta_bagged_std, 's--', label="BayesBag Posterior Std", color='red')
plt.xlabel("Site Index")
plt.ylabel("Posterior Std Dev")
plt.title("Posterior Spread: Standard vs BayesBag")
plt.xticks(np.arange(K), labels=sites, rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figs/microcredit_std.pdf", bbox_inches="tight")
plt.close()

# ---- Error bar plot ----
x = np.arange(1, K + 1)
plt.figure(figsize=(7, 5))
plt.errorbar(x, theta_standard_mean, yerr=0, fmt='o', label='Standard')
plt.errorbar(x, theta_bagged_mean, yerr=theta_bagged_std, fmt='o', label='BayesBag')
plt.legend()
plt.xlabel("Site")
plt.ylabel("Estimated Effect")
plt.title("Standard vs BayesBag Inference")
plt.xticks(x, sites, rotation=45)
plt.tight_layout()
plt.savefig("figs/microcredit_comparison.pdf", bbox_inches="tight")
plt.close()

# ---- Mismatch Index ----
var_std = np.sum(theta_standard_std ** 2)
var_bag = np.sum(theta_bagged_std ** 2)
mismatch_index = 1 - ((2 * var_std) / var_bag) if var_bag > var_std else np.nan
print(f"MISMATCH INDEX = {mismatch_index:.4f}")

# ---- Relative Squared Error ----
eps = 1e-8
rse = (theta_bagged_mean - theta_standard_mean) ** 2 / (theta_bagged_std ** 2 + eps)
rse_mean = np.mean(rse)
print(f"RELATIVE SQUARED ERROR = {rse_mean:.4f}")
