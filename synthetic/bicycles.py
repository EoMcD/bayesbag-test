import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os

output_dir = "figs"
os.makedirs(output_dir, exist_ok=True)

def load_bike_data():
    data = {
        "category": [],
        "block": [],
        "bikes": [],
        "total": [],
    }

    categories = {
        "residential_yes": ["16/58", "9/90", "10/48", "13/57", "19/103",
                            "20/57", "18/86", "17/112", "35/273", "55/64"],
        "residential_no": ["12/113", "1/18", "2/14", "4/44", "9/208",
                           "7/67", "9/29", "8/154"],
        "fairly_busy_yes": ["8/29", "35/415", "31/425", "19/42", "38/180",
                            "47/675", "44/620", "44/437", "29/47", "18/462"],
        "fairly_busy_no": ["10/557", "43/1258", "5/499", "14/601", "58/1163",
                           "15/700", "0/90", "47/1093", "51/1459", "32/1086"],
        "busy_yes": ["60/1545", "51/1499", "58/1598", "59/503", "53/407",
                     "68/1494", "68/1558", "60/1706", "71/476", "63/752"],
        "busy_no": ["8/1248", "9/1246", "6/1596", "9/1765", "19/1290",
                    "61/2498", "31/2346", "75/3101", "14/1918", "25/2318"],
    }

    for category, entries in categories.items():
        for i, val in enumerate(entries):
            bikes, total = map(int, val.split("/"))
            data["category"].append(category)
            data["block"].append(f"{category}_block_{i+1}")
            data["bikes"].append(bikes)
            data["total"].append(total)

    return pd.DataFrame(data)

def generate_synthetic_from_counts(df):
    synthetic_data = []
    for _, row in df.iterrows():
        ones = np.ones(row["bicycles"])
        zeros = np.zeros(row["non_bicycles"])
        group_data = np.concatenate([ones, zeros])
        np.random.shuffle(group_data)
        synthetic_data.append(group_data)
    return synthetic_data

def contaminate_data(synthetic_data, flip_prob=0.1): # WIP
    contaminated = []
    for group_data in synthetic_data:
        contaminated_group = group_data.copy()
        flip_mask = np.random.rand(len(group_data)) < flip_prob
        contaminated_group[flip_mask] = 1 - contaminated_group[flip_mask]
        contaminated.append(contaminated_group)
    return contaminated

def fit_pymc_model(y_obs, n_obs, block_idx, num_blocks, return_model=False):
    with pm.Model() as model:
        log_alpha = pm.Normal("log_alpha", mu=0, sigma=1.5)
        log_beta = pm.Normal("log_beta", mu=0, sigma=1.5)

        alpha = pm.Deterministic("alpha", pm.math.exp(log_alpha))
        beta = pm.Deterministic("beta", pm.math.exp(log_beta))

        theta = pm.Beta("theta", alpha=alpha, beta=beta, shape=num_blocks)
        y_likelihood = pm.Binomial("y_likelihood", n=n_obs, p=theta[block_idx], observed=y_obs)

        trace = pm.sample(4000, tune=2000, chains=4, cores=2, target_accept=0.95, return_inferencedata=True)

    return (trace, model) if return_model else trace

def bayesbag(y_obs, n_obs, block_idx, num_blocks, b=100, mfactor=1):
    bagged_thetas = []
    m = int(mfactor * len(y_obs))
    for _ in tqdm(range(b), desc="BayesBag iterations"):
        indices = np.random.choice(len(y_obs), size=m, replace=True)
        y_boot = y_obs[indices]
        n_boot = n_obs[indices]
        idx_boot = block_idx[indices]

        trace = fit_pymc_model(y_boot, n_boot, idx_boot, num_blocks)
        mean_theta = trace.posterior["theta"].mean(dim=["chain", "draw"]).values
        bagged_thetas.append(mean_theta)
    return np.array(bagged_thetas)

def plot_bayesbag_results(bagged_ps, label, filename, group_labels=None):
    mean_estimates = np.mean(bagged_ps, axis=0)
    std_estimates = np.std(bagged_ps, axis=0)

    plt.figure(figsize=(12, 5))
    x = np.arange(len(mean_estimates))
    plt.errorbar(x, mean_estimates, yerr=std_estimates, fmt='o', label=label, capsize=5)
    plt.ylim(0, 1)
    plt.xlabel("Group")
    plt.ylabel("Estimated p (bike rate)")
    plt.title(f"BayesBag Posterior Means ± Std Dev — {label}")
    if group_labels:
        plt.xticks(x, group_labels, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

    # ---- Plot Posterior Means ----
    plt.figure(figsize=(12, 6))
    plt.plot(theta_standard_mean, 'o-', label="Standard Posterior Mean", color='blue')
    plt.plot(theta_bagged_mean, 's--', label="BayesBag Posterior Mean", color='red')
    plt.xlabel("Street Segment Index")
    plt.ylabel("Estimated Bicycle Proportion (θ)")
    plt.title("Standard vs BayesBag Posterior Means")
    plt.axvline(9.5, color='gray', linestyle='--', label='Category Divider')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ---- Plot Posterior Std Devs ----
    plt.figure(figsize=(12, 6))
    plt.plot(theta_standard_std, 'o-', label="Standard Posterior Std", color='blue')
    plt.plot(theta_bagged_std, 's--', label="BayesBag Posterior Std", color='red')
    plt.xlabel("Street Segment Index")
    plt.ylabel("Posterior Std Dev")
    plt.title("Posterior Spread: Standard vs BayesBag")
    plt.axvline(9.5, color='gray', linestyle='--', label='Category Divider')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    df = load_bike_data()

    y_obs = np.array(df["bikes"])
    n_obs = np.array(df["total"])
    block_idx = np.array(df["block_idx"])
    num_obs = df["block_idx"].nunique()

    # ---- Standard Fit ----
    trace, model = fit_pymc_model(y_obs, n_obs, block_idx, num_obs, return_model=True)
    theta_standard_mean = trace.posterior["theta"].mean(dim=["chain", "draw"]).values
    theta_standard_std = trace.posterior["theta"].std(dim=["chain", "draw"]).values

    # ---- BayesBag ----
    bagged_theta_samples = bayesbag(y_obs, n_obs, block_idx, num_obs, b=100, mfactor=1)
    theta_bagged_mean = bagged_theta_samples.mean(axis=0)
    theta_bagged_std = bagged_theta_samples.std(axis=0)

    plot_bayesbag_results

    # --------------------------
    # Evaluation / Model Criticism
    # --------------------------
    NvN = len(y_obs) * np.sum(theta_standard_std ** 2)
    MvsM = len(y_obs) * np.sum(theta_bagged_std ** 2)
    mismatch_index = np.nan
    if MvsM > NvN:
        mismatch_index = 1 - ((2 * NvN) / MvsM)
    print(f"\nMISMATCH INDEX = {mismatch_index:.4f}")

    eps = 1e-8
    rse = (theta_bagged_mean - theta_standard_mean) ** 2 / (theta_bagged_std ** 2 + eps)
    rse_mean = np.mean(rse)
    print(f"RELATIVE SQUARED ERROR = {rse_mean:.4f}")

if __name__ == "__main__":
    main()
