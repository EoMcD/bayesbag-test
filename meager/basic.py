import pyreadr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel
import numpy as np
import pickle
import os

os.makedirs("figs", exist_ok=True)
os.makedirs("out", exist_ok=True)

# load .RData file
# result = pyreadr.read_r("C:/Users/emcdo/Documents/Artificial Intelligence/DSA/Project/Meager/code/data/microcredit/microcredit-profit-independent.Rdata")
result = pyreadr.read_r("microcredit-profit-independent.Rdata")
data = result["data"]  # The name in the R file is 'data'

# tidy data
data = data.rename(columns={"site": "g", "treatment": "t", "profit": "y"})
data["g"] = data["g"].astype("category")
data["t"] = data["t"].astype("category")

# initial plotting
sns.displot(
    data=data[(data["y"] > -5000) & (data["y"] < 5000)],
    x="y",
    hue="t",
    kind="hist",
    bins=50,
    col="g",
    col_wrap=2,
    stat="density"
)
plt.suptitle("Profit distribution by treatment and site")
plt.tight_layout()
plt.savefig("figs/microcredit-exploratory.pdf")

print("Initial plots made")

# model via Stan
# model = CmdStanModel(stan_file="C:/Users/emcdo/Documents/Artificial Intelligence/DSA/Project/Meager/code/stan/microcredit-independent-model-ss.stan")
model = CmdStanModel(stan_file="microcredit-independent-model-ss.stan")

def fit_meager(data, n_samp=4000, n_warmup=2000, thin=1):
    stan_data = {
        "K": data["g"].nunique(),
        "N": len(data),
        "P": 2,
        "site": data["g"].cat.codes + 1,     # 1-indexed for Stan
        "y": data["y"].values,
        "ITT": data["t"].cat.codes.values
    }

    fit = model.sample(
        data=stan_data,
        chains=4,
        iter_sampling=n_samp,
        iter_warmup=n_warmup,
        thin=thin,
        refresh=1
    )

    print(fit.diagnose())

    draws = fit.draws_pd()

    # Diagnostics
    print("Draws shape:", draws.shape)
    for var in ["mu", "tau", "sigma_mu", "sigma_tau"]:
        print(f"{var} len:", len(draws[var]))

    for var_prefix in ["mu_k", "tau_k", "sigma_y_k"]:
        cols = draws.filter(regex=fr"^{var_prefix}\[\d+\]$").columns
        print(f"{var_prefix} cols:", list(cols))
        for col in cols:
            print(f"{col} len:", len(draws[col]))

    # Extract lambda
    lambda_vars = ["mu", "tau", "sigma_mu", "sigma_tau"]
    lambda_df = draws[lambda_vars].copy()
    # lambda_df["j"] = np.arange(len(lambda_df)) + 1
    n_draws = draws.shape[0]
    lambda_df["j"] = np.arange(1, n_draws + 1)

    # Extract theta
    theta_vars = ["mu_k", "tau_k", "sigma_y_k"]
    K = stan_data["K"]
    
    # theta_df = pd.DataFrame({
    #     "j": np.repeat(np.arange(1, n_draws + 1), K),
    #     "k": np.tile(np.arange(1, K + 1), n_draws),
    #     "mu_k": draws.filter(regex=r"mu_k\\[\\d+\\]").values.flatten(),
    #     "tau_k": draws.filter(regex=r"tau_k\\[\\d+\\]").values.flatten(),
    #     "sigma_k": draws.filter(regex=r"sigma_y_k\\[\\d+\\]").values.flatten(),
    # })
    
    # theta_df = pd.DataFrame({"j": np.arange(1, n_draws + 1)}) # alternative approach to avoid data length mismatches

    mu_k = draws.filter(regex=r"^mu_k\[\d+\]$").values
    tau_k = draws.filter(regex=r"^tau_k\[\d+\]$").values
    sigma_k = draws.filter(regex=r"^sigma_y_k\[\d+\]$").values

    assert mu_k.shape == (n_draws, K), f"mu_k shape mismatch: {mu_k.shape} vs expected ({n_draws}, {K})"
    assert tau_k.shape == (n_draws, K), f"tau_k shape mismatch: {tau_k.shape}"
    assert sigma_k.shape == (n_draws, K), f"sigma_k shape mismatch: {sigma_k.shape}"

    theta_df = pd.DataFrame({
        "j": np.repeat(np.arange(1, n_draws + 1), K),
        "k": np.tile(np.arange(1, K + 1), n_draws),
        "mu_k": mu_k.flatten(),
        "tau_k": tau_k.flatten(),
        "sigma_k": sigma_k.flatten()
    })
    
    return {"lambda": lambda_df, "theta": theta_df}

print("Models made")

# --- BAYESBAG PART, WIP ---
# n_boot = 20
# boot_fits = []
# boot_data = []

# for i in range(n_boot):
#     print(f"Running bootstrap {i+1} of {n_boot}")

#     boot_i = (
#         data.groupby("g", group_keys=False)
#         .apply(lambda g: resample(g, replace=True))
#         .reset_index(drop=True)
#     )

#     boot_data.append(boot_i)
#     fit = fit_meager(boot_i)
#     boot_fits.append(fit)

with open("out/microcredit.pkl", "wb") as f:
    pickle.dump({"mg": fit_meager(data)}, f)

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
