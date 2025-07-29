import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import os

output_dir = "figs"
os.makedirs(output_dir, exist_ok=True)
os.makedirs("out", exist_ok=True)

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

def contaminate_data(df, frac=0.1, seed=42):
    np.random.seed(seed)
    df_contaminated = df.copy()
    n_rows = len(df)
    contam_idx = np.random.choice(n_rows, size=int(frac * n_rows), replace=False)

    for i in contam_idx:
        y = df_contaminated.at[i, "bikes"]
        n = df_contaminated.at[i, "total"]
        df_contaminated.at[i, "bikes"] = n - y  # flip count
    return df_contaminated

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

def plot_bayesbag_results(
    bagged_ps,
    theta_standard_mean,
    theta_standard_std,
    theta_bagged_mean,
    theta_bagged_std,
    label,
    base_filename,
    group_labels=None
):
    x = np.arange(len(theta_standard_mean))

    # --- Plot 1: BayesBag error bars ---
    plt.figure(figsize=(12, 5))
    mean_estimates = np.mean(bagged_ps, axis=0)
    std_estimates = np.std(bagged_ps, axis=0)
    plt.errorbar(x, mean_estimates, yerr=std_estimates, fmt='o', label=label, capsize=5)
    plt.ylim(0, 1)
    plt.xlabel("Group")
    plt.ylabel("Estimated p (bike rate)")
    plt.title(f"BayesBag Posterior Means ± Std Dev — {label}")
    if group_labels:
        plt.xticks(x, group_labels, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_filename}_bagged_errorbars.png"))
    plt.close()

    # --- Plot 2: Posterior Means Comparison ---
    plt.figure(figsize=(12, 6))
    plt.plot(theta_standard_mean, 'o-', label="Standard Posterior Mean", color='blue')
    plt.plot(theta_bagged_mean, 's--', label="BayesBag Posterior Mean", color='red')
    plt.xlabel("Street Segment Index")
    plt.ylabel("Estimated Bicycle Proportion (θ)")
    plt.title(f"{label} — Standard vs BayesBag Posterior Means")
    plt.axvline(9.5, color='gray', linestyle='--')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_filename}_means_comparison.png"))
    plt.close()

    # --- Plot 3: Posterior Std Devs Comparison ---
    plt.figure(figsize=(12, 6))
    plt.plot(theta_standard_std, 'o-', label="Standard Posterior Std", color='blue')
    plt.plot(theta_bagged_std, 's--', label="BayesBag Posterior Std", color='red')
    plt.xlabel("Street Segment Index")
    plt.ylabel("Posterior Std Dev")
    plt.title(f"{label} — Posterior Spread Comparison")
    plt.axvline(9.5, color='gray', linestyle='--')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{base_filename}_stddev_comparison.png"))
    plt.close()

def run_pipeline(df, label_prefix="Clean"):
    y_obs = np.array(df["bikes"])
    n_obs = np.array(df["total"])
    block_idx = np.array(df["block_idx"])
    num_blocks = df["block_idx"].nunique()

    trace, model = fit_pymc_model(y_obs, n_obs, block_idx, num_blocks, return_model=True)
    theta_standard_mean = trace.posterior["theta"].mean(dim=["chain", "draw"]).values
    theta_standard_std = trace.posterior["theta"].std(dim=["chain", "draw"]).values

    # --- Save trace ---
    trace_path = os.path.join("out", f"{label_prefix.lower()}_trace.nc")
    az.to_netcdf(trace, trace_path)

    # --- BayesBag Posterior ---
    bagged_theta_samples = bayesbag(y_obs, n_obs, block_idx, num_blocks, b=100)
    theta_bagged_mean = bagged_theta_samples.mean(axis=0)
    theta_bagged_std = bagged_theta_samples.std(axis=0)

    # --- Evaluation ---
    NvN = len(y_obs) * np.sum(theta_standard_std ** 2)
    MvsM = len(y_obs) * np.sum(theta_bagged_std ** 2)
    mismatch_index = 1 - ((2 * NvN) / MvsM) if MvsM > NvN else np.nan

    eps = 1e-8
    rse = (theta_bagged_mean - theta_standard_mean) ** 2 / (theta_bagged_std ** 2 + eps)
    rse_mean = np.mean(rse)

    print(f"\n---- {label_prefix} Data ----")
    print(f"MISMATCH INDEX = {mismatch_index:.4f}")
    print(f"RELATIVE SQUARED ERROR = {rse_mean:.4f}")

    # --- Plotting ---
    plot_bayesbag_results(
        bagged_ps=bagged_theta_samples,
        theta_standard_mean=theta_standard_mean,
        theta_standard_std=theta_standard_std,
        theta_bagged_mean=theta_bagged_mean,
        theta_bagged_std=theta_bagged_std,
        label=label_prefix,
        base_filename=label_prefix.lower(),
        group_labels=None
    )

def main():
    clean_df = load_bike_data()
    run_pipeline(clean_df, label_prefix="Clean")

    contaminated_df = contaminate_data(clean_df, frac=0.1)
    run_pipeline(contaminated_df, label_prefix="Contaminated")
