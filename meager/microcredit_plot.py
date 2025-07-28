import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load model fit
with open("out/microcredit.pkl", "rb") as f:
    fit = pickle.load(f)["mg"]

theta_df = fit["theta"]
lambda_df = fit["lambda"]

# Load original data (required for empirical means & PPC)
from pyreadr import read_r
result = read_r("microcredit-profit-independent.Rdata")
data = result["data"]
data = data.rename(columns={"site": "g", "treatment": "t", "profit": "y"})
data["g"] = data["g"].astype("category")
data["t"] = data["t"].astype("category")

# -- PLOTS --
# --- Posterior means per site ---
posterior_mu = theta_df.groupby("k")["mu_k"].mean().reset_index()
posterior_mu.columns = ["site", "posterior_mu"]

# --- Empirical means per site ---
empirical_mu = data.groupby("g")["y"].mean().reset_index()
empirical_mu["site"] = empirical_mu["g"].cat.codes + 1
empirical_mu = empirical_mu[["site", "y"]].rename(columns={"y": "empirical_mu"})

# --- Merge and plot ---
mu_compare = pd.merge(posterior_mu, empirical_mu, on="site")

plt.figure(figsize=(8, 6))
plt.scatter(mu_compare["empirical_mu"], mu_compare["posterior_mu"])
plt.plot([mu_compare.min().min(), mu_compare.max().max()],
         [mu_compare.min().min(), mu_compare.max().max()], 'r--')
plt.xlabel("Empirical Site Mean")
plt.ylabel("Posterior Mean (mu_k)")
plt.title("Posterior vs Empirical Means per Site")
plt.grid(True)
plt.tight_layout()
plt.savefig("figs/posterior_vs_empirical.pdf")

# PPC
ppc = theta_df.copy()
n_samples = ppc["j"].nunique()
n_sites = ppc["k"].nunique()

# One sample per site per draw
ppc["y_rep"] = np.random.normal(loc=ppc["mu_k"], scale=ppc["sigma_k"])

# Merge site names back
ppc = ppc.merge(data[["g"]].drop_duplicates().reset_index(drop=True), left_on="k", right_index=True)

# Plot PPC
g = sns.FacetGrid(ppc, col="g", col_wrap=3, height=3.5, sharex=False, sharey=False)
g.map(sns.histplot, "y_rep", bins=30, kde=False, color="blue", stat="density")

# Overlay actual y
for ax, site in zip(g.axes.flatten(), data["g"].cat.categories):
    subset = data[data["g"] == site]
    sns.kdeplot(subset["y"], ax=ax, color="black", lw=1.5, label="Actual")
    ax.set_title(f"Site: {site}")
    ax.legend()

plt.suptitle("Posterior Predictive Check by Site", y=1.02)
plt.tight_layout()
plt.savefig("figs/ppc_by_site.pdf")
