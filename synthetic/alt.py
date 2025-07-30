import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

output_dir = "figs_synthetic"
os.makedirs(output_dir, exist_ok=True)

# data generation

def generate_synthetic_data(num_groups=20, n_per_group=100, alpha=2, beta=5, seed=42):
    np.random.seed(seed)
    true_theta = np.random.beta(alpha, beta, size=num_groups)
    y_obs = np.random.binomial(n_per_group, true_theta)
    n_obs = np.full(num_groups, n_per_group)
    return y_obs, n_obs, true_theta

# MISSPECIFICATION: contaminated data

def contaminate_synthetic_data(y_obs, n_obs, frac=0.1, seed=42):
    np.random.seed(seed)
    y_contaminated = y_obs.copy()
    n_contaminated = n_obs.copy()
    num_groups = len(y_obs)
    contam_idx = np.random.choice(num_groups, size=int(frac * num_groups), replace=False)
    for i in contam_idx:
        y_contaminated[i] = n_obs[i] - y_obs[i]
    return y_contaminated, n_contaminated, contam_idx

# MISSPECIFICATION: incorrect model

def fit_normal_model(y_obs, n_obs, num_groups):
    y_prop = y_obs / n_obs
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.5, sigma=0.3)
        sigma = pm.HalfNormal("sigma", sigma=0.2)
        theta = pm.Normal("theta", mu=mu, sigma=sigma, shape=num_groups)
        theta_clipped = pm.Deterministic("theta_clipped", pm.math.clip(theta, 1e-6, 1 - 1e-6))
        y_like = pm.Binomial("y_like", n=n_obs, p=theta_clipped, observed=y_obs)
        trace = pm.sample(2000, tune=1000, chains=4, target_accept=0.95, return_inferencedata=True)
    return trace

# fit matching model

def fit_beta_binomial(y_obs, n_obs, num_groups):
    with pm.Model() as model:
        log_alpha = pm.Normal("log_alpha", mu=0, sigma=1.5)
        log_beta = pm.Normal("log_beta", mu=0, sigma=1.5)
        alpha = pm.Deterministic("alpha", pm.math.exp(log_alpha))
        beta = pm.Deterministic("beta", pm.math.exp(log_beta))
        theta = pm.Beta("theta", alpha=alpha, beta=beta, shape=num_groups)
        y_like = pm.Binomial("y_like", n=n_obs, p=theta, observed=y_obs)
        trace = pm.sample(2000, tune=1000, chains=4, target_accept=0.95, return_inferencedata=True)
    return trace

# bayesbag

def bayesbag(y_obs, n_obs, num_groups, b=100, mfactor=1.0):
    bagged_thetas = []
    m = int(mfactor * len(y_obs))
    for _ in tqdm(range(b), desc="BayesBag iterations"):
        indices = np.random.choice(len(y_obs), size=m, replace=True)
        y_boot = y_obs[indices]
        n_boot = n_obs[indices]
        trace = fit_beta_binomial(y_boot, n_boot, num_groups)
        theta_mean = trace.posterior["theta"].mean(dim=["chain", "draw"]).values
        bagged_thetas.append(theta_mean)
    return np.array(bagged_thetas)

# eval

# def evaluate_and_plot(y_obs, n_obs, true_theta, label_prefix="Clean", contam_idx=None):  # contam version
def evaluate_and_plot(y_obs, n_obs, true_theta, label_prefix="Clean", contam_idx=None, model_type="beta"): # wrong model version
  
    num_groups = len(y_obs)

    # Standard Posterior
    if model_type == "beta":
        trace = fit_beta_binomial(y_obs, n_obs, num_groups)
    elif model_type == "normal":
        trace = fit_normal_model(y_obs, n_obs, num_groups)
    else:
        raise ValueError("Invalid model_type")
    
    theta_mean = trace.posterior["theta"].mean(dim=["chain", "draw"]).values
    theta_std = trace.posterior["theta"].std(dim=["chain", "draw"]).values

    # BayesBag
    def bayesbag_model(y_obs, n_obs, num_groups):
        if model_type == "beta":
            return fit_beta_binomial(y_obs, n_obs, num_groups)
        elif model_type == "normal":
            return fit_normal_model(y_obs, n_obs, num_groups)

    bagged_samples = bayesbag(y_obs, n_obs, num_groups, b=100)
  
    # theta_bagged_mean = bagged_samples.mean(axis=0)
    # theta_bagged_std = bagged_samples.std(axis=0)

    theta_bagged_mean = np.mean(bagged_samples, axis=0)
    theta_bagged_std = np.std(bagged_samples, axis=0)

    # Metrics
    eps = 1e-8
    mse_standard = np.mean((theta_mean - true_theta) ** 2)
    mse_bagged = np.mean((theta_bagged_mean - true_theta) ** 2)

    rse = (theta_bagged_mean - theta_mean) ** 2 / (theta_bagged_std ** 2 + eps)
    rse_mean = np.mean(rse)

    NvN = len(y_obs) * np.sum(theta_std ** 2)
    MvsM = len(y_obs) * np.sum(theta_bagged_std ** 2)
    mismatch_index = 1 - ((2 * NvN) / MvsM) if MvsM > NvN else np.nan

    print(f"\n--- {label_prefix} Results ---")
    print(f"Standard MSE: {mse_standard:.4f}")
    print(f"BayesBag MSE: {mse_bagged:.4f}")
    print(f"RSE Mean: {rse_mean:.4f}")
    print(f"Mismatch Index: {mismatch_index:.4f}")

    x = np.arange(num_groups)
    plt.figure(figsize=(12, 6))
    plt.plot(true_theta, 'k*-', label="True θ")
    plt.plot(theta_mean, 'bo-', label="Standard Posterior Mean")
    plt.plot(theta_bagged_mean, 'ro--', label="BayesBag Mean")
    if contam_idx is not None:
        for i in contam_idx:
            plt.axvline(i, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel("Group Index")
    plt.ylabel("Estimated θ")
    plt.title(f"{label_prefix}: Posterior Estimates")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{label_prefix.lower()}_posterior_plot.png"))
    plt.close()

# -----------------------------
# 6. Main
# -----------------------------

def main():
    y_clean, n_clean, true_theta = generate_synthetic_data()
    
    # evaluate_and_plot(y_clean, n_clean, true_theta, label_prefix="Clean")

    y_contaminated, n_contaminated, contam_idx = contaminate_synthetic_data(y_clean, n_clean, frac=0.2)
    
    # evaluate_and_plot(y_contaminated, n_contaminated, true_theta, label_prefix="Contaminated", contam_idx=contam_idx)

    # print("\n=== Fitting Normal model to clean data ===")
    # evaluate_and_plot(y_clean, n_clean, true_theta, label_prefix="Clean_Normal", model_type="normal")

    print("\n=== Fitting Normal model to contaminated data ===")
    evaluate_and_plot(y_contaminated, n_contaminated, true_theta, label_prefix="Contaminated_Normal", contam_idx=contam_idx, model_type="normal")


if __name__ == "__main__":
    main()
