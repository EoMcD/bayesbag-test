import pyreadr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel
import numpy as np
import pickle
import os
from sklearn.utils import resample

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
    data=data[(data["y"] > -2000) & (data["y"] < 2000)],
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
from sklearn.utils import resample

n_boot = 20
boot_fits, boot_data = [], []

for i in range(n_boot):
    print(f"Running bootstrap {i+1} of {n_boot}")
    # stratify by (site, treatment)
    boot_i = (
        data.groupby(["g", "t"], group_keys=False)
            .apply(lambda df: resample(df, replace=True, n_samples=len(df), random_state=10_000 + i))
            .reset_index(drop=True)
    )
    boot_data.append(boot_i)
    fit = fit_meager(boot_i)
    boot_fits.append(fit)

with open("out/microcredit_bayesbag.pkl", "wb") as f:
    pickle.dump({"bayesbag": boot_fits, "boot_data": boot_data}, f)

with open("out/microcredit.pkl", "wb") as f:
    pickle.dump({"mg": fit_meager(data)}, f)
