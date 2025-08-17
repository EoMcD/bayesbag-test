import argparse
import os
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# --------------------------
# Paths
# --------------------------
FIGS_DIR = "figs_gamma"
OUT_DIR = "out_gamma"
os.makedirs(FIGS_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# --------------------------
# Data generation & contamination
# --------------------------
def generate_data_gamma(num_groups=20, n_per_group=100, shape=2.0, scale=3.0, seed=42):
    """
    Draw true per-group means mu_k ~ Gamma(shape, scale)  (between-group variability).
    Then within each group, draw y_ik ~ Gamma(alpha_sim, beta_k) where mean(mu_k)=alpha_sim/beta_k.
    """
    rng = np.random.default_rng(seed)
    true_means = rng.gamma(shape=shape, scale=scale, size=num_groups)

    alpha_sim = 2.0  # fixed positive shape for within-group simulation (can change)
    rows = []
    for k in range(num_groups):
        mu_k = true_means[k]
        beta_k = alpha_sim / mu_k  # rate so that mean = alpha_sim/beta_k = mu_k
        y = rng.gamma(shape=alpha_sim, scale=1.0 / beta_k, size=n_per_group)  # numpy uses scale=1/rate
        for yi in y:
            rows.append({"g": int(k), "y": float(yi)})

    df = pd.DataFrame(rows)
    return df, true_means


def contaminate_data(df, frac_groups=1.0, frac_points=0.2, factor=0.1, seed=42):
    """
    Multiplicative contamination on a fraction of points within a fraction of groups.
    This creates within-group outliers/attenuation.
    """
    rng = np.random.default_rng(seed)
    df_cont = df.copy()
    groups = df["g"].unique()
    m = max(1, int(frac_groups * len(groups)))
    contam_groups = rng.choice(groups, size=m, replace=False)

    for g in contam_groups:
        grp_idx = df_cont.index[df_cont["g"] == g].to_numpy()
        k = max(1, int(np.floor(len(grp_idx) * frac_points)))
        contam_idx = rng.choice(grp_idx, size=k, replace=False)
        df_cont.loc[contam_idx, "y"] *= factor

    return df_cont, np.array(contam_groups, dtype=int)

# --------------------------
# Models
# --------------------------
def fit_model_gamma(df, draws=2000, tune=1000, chains=4, target_accept=0.9):
    groups = np.sort(df["g"].unique())
    K = len(groups)
    site_idx = df["g"].values.astype("int64")
    y = df["y"].values.astype("float64")

    with pm.Model() as model:
        alpha = pm.HalfNormal("alpha", sigma=10.0)
        mu_group_raw = pm.Normal("mu_group_raw", mu=0.0, sigma=3.0, shape=K)
        mu_group = pm.Deterministic("mu_group", pm.math.exp(mu_group_raw))
        mu_obs = pm.Deterministic("mu_obs", mu_group[site_idx])
        beta_obs = pm.Deterministic("beta_obs", alpha / mu_obs)
        y_like = pm.Gamma("y_like", alpha=alpha, beta=beta_obs, observed=y)
        trace = pm.sample(draws=draws, tune=tune, chains=chains,
                          target_accept=target_accept, return_inferencedata=True)
    return trace


def fit_model_normal(df, draws=2000, tune=1000, chains=4, target_accept=0.9):
    groups = np.sort(df["g"].unique())
    K = len(groups)
    site_idx = df["g"].values.astype("int64")
    y = df["y"].values.astype("float64")

    with pm.Model() as model:
        mu_k = pm.Normal("mu", mu=0.0, sigma=10.0, shape=K)
        sigma_k = pm.HalfNormal("sigma", sigma=10.0, shape=K)
        y_like = pm.Normal("y_like", mu=mu_k[site_idx], sigma=sigma_k[site_idx], observed=y)
        trace = pm.sample(draws=draws, tune=tune, chains=chains,
                          target_accept=target_accept, return_inferencedata=True)
    return trace

# --------------------------
# BayesBag: cluster bootstrap
# --------------------------
def _cluster_bootstrap(df, mfactor=1.0, seed=None):
    rng = np.random.default_rng(seed)
    groups = np.sort(df["g"].unique())
    K = len(groups)

    m_groups = max(1, int(np.ceil(mfactor * K)))  # number of groups per bootstrap sample
    boot_groups = rng.choice(groups, size=m_groups, replace=True)

    parts = []
    for g in boot_groups:
        grp = df[df["g"] == g]
        parts.append(grp.sample(n=len(grp), replace=True,
                                random_state=int(rng.integers(0, 2**31 - 1))))
    boot_df = pd.concat(parts, ignore_index=True)

    # Remap group IDs to compact 0..K_boot-1 for model indexing
    unique_boot = np.sort(boot_df["g"].unique())
    old_to_new = {old: i for i, old in enumerate(unique_boot)}
    boot_df["g"] = boot_df["g"].map(old_to_new).astype(int)

    return boot_df, unique_boot, old_to_new, groups


def _align_group_vector_to_full(means_boot, unique_boot, old_to_new, groups):
    K = len(groups)
    aligned = np.full(K, np.nan)
    for old_g in unique_boot:
        aligned_idx = np.where(groups == old_g)[0][0]   # index in original ordering
        compact_idx = old_to_new[old_g]                 # index in boot's compact ordering
        aligned[aligned_idx] = means_boot[compact_idx]
    return aligned


def bayesbag_gamma_cluster(df, b=50, mfactor=1.0, draws=2000, tune=1000, chains=4, target_accept=0.9, seed=42):
    bagged = []
    rng = np.random.default_rng(seed)
    for _ in tqdm(range(b), desc="BayesBag (Gamma, cluster)"):
        boot_df, unique_boot, old_to_new, groups = _cluster_bootstrap(
            df, mfactor=mfactor, seed=int(rng.integers(0, 2**31 - 1))
        )
        trace = fit_model_gamma(boot_df, draws=draws, tune=tune, chains=chains, target_accept=target_accept)
        means_boot = trace.posterior["mu_group"].mean(dim=("chain", "draw")).values
        aligned = _align_group_vector_to_full(means_boot, unique_boot, old_to_new, groups)
        bagged.append(aligned)
    return np.array(bagged)


def bayesbag_normal_cluster(df, b=50, mfactor=1.0, draws=2000, tune=1000, chains=4, target_accept=0.9, seed=42):
    bagged = []
    rng = np.random.default_rng(seed)
    for _ in tqdm(range(b), desc="BayesBag (Normal, cluster)"):
        boot_df, unique_boot, old_to_new, groups = _cluster_bootstrap(
            df, mfactor=mfactor, seed=int(rng.integers(0, 2**31 - 1))
        )
        trace = fit_model_normal(boot_df, draws=draws, tune=tune, chains=chains, target_accept=target_accept)
        means_boot = trace.posterior["mu"].mean(dim=("chain", "draw")).values
        aligned = _align_group_vector_to_full(means_boot, unique_boot, old_to_new, groups)
        bagged.append(aligned)
    return np.array(bagged)

# --------------------------
# Summaries & evaluation helpers
# --------------------------
def posterior_ci(trace, varname, alpha=0.05):
    post = trace.posterior[varname]
    mean = post.mean(dim=("chain", "draw")).values
    low = post.quantile(alpha / 2.0, dim=("chain", "draw")).values
    high = post.quantile(1.0 - alpha / 2.0, dim=("chain", "draw")).values
    std = post.std(dim=("chain", "draw")).values
    return mean, low, high, std


def summarize_bagged(bag_matrix):
    med = np.nanmedian(bag_matrix, axis=0)
    mean = np.nanmean(bag_matrix, axis=0)
    low = np.nanpercentile(bag_matrix, 2.5, axis=0)
    high = np.nanpercentile(bag_matrix, 97.5, axis=0)
    n_eff = np.sum(~np.isnan(bag_matrix), axis=0)
    return {"median": med, "mean": mean, "low": low, "high": high, "n_eff": n_eff}


def mse_to_truth(est, truth):
    return float(np.nanmean((est - truth) ** 2))


def eval_and_plot_groupwise(true_means, std_mean, std_low, std_high, bag_summary, title_prefix, figs_dir=FIGS_DIR):
    K = len(true_means)

    # Coverage (95%)
    std_cover = np.mean((true_means >= std_low) & (true_means <= std_high))
    bag_cover = np.mean((true_means >= bag_summary["low"]) & (true_means <= bag_summary["high"]))

    # CI widths
    std_width = np.median(std_high - std_low)
    bag_width = np.median(bag_summary["high"] - bag_summary["low"])

    # MSE of point estimates vs truth (use standard mean vs bagged median)
    std_mse = mse_to_truth(std_mean, true_means)
    bag_mse = mse_to_truth(bag_summary["median"], true_means)

    print(f"\n[{title_prefix}] Coverage 95% - Standard: {std_cover:.3f}, Bagged: {bag_cover:.3f}")
    print(f"[{title_prefix}] Median CI width - Standard: {std_width:.3f}, Bagged: {bag_width:.3f}")
    print(f"[{title_prefix}] MSE to true means - Standard: {std_mse:.4f}, Bagged: {bag_mse:.4f}")
    print(f"[{title_prefix}] Bagged per-group effective bootstraps (min/median/max): "
          f"{int(np.nanmin(bag_summary['n_eff']))}/"
          f"{int(np.nanmedian(bag_summary['n_eff']))}/"
          f"{int(np.nanmax(bag_summary['n_eff']))}")

    # Plot: group index vs mean with 95% intervals
    x = np.arange(K)
    plt.figure(figsize=(9, 5))
    plt.errorbar(x, std_mean, yerr=[std_mean - std_low, std_high - std_mean],
                 fmt='o', capsize=2, label="Standard")
    plt.errorbar(x, bag_summary["median"],
                 yerr=[bag_summary["median"] - bag_summary["low"],
                       bag_summary["high"] - bag_summary["median"]],
                 fmt='x', capsize=2, label="Bagged (median)")
    plt.plot(x, true_means, linestyle='--', marker='.', label="True μ")
    plt.xlabel("Group index")
    plt.ylabel("Mean")
    plt.title(f"{title_prefix}: group means ± 95% intervals")
    plt.legend()
    f1 = os.path.join(figs_dir, f"{title_prefix.lower().replace(' ', '_')}_means_intervals.png")
    plt.tight_layout(); plt.savefig(f1, dpi=150); plt.close()

    # Plot: true vs estimated (no error bars)
    plt.figure(figsize=(6, 6))
    plt.scatter(true_means, std_mean, label="Standard")
    plt.scatter(true_means, bag_summary["median"], label="Bagged (median)", marker='x')
    lo = np.nanmin([true_means.min(), std_mean.min(), bag_summary["median"].min()])
    hi = np.nanmax([true_means.max(), std_mean.max(), bag_summary["median"].max()])
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("True group mean")
    plt.ylabel("Estimated group mean")
    plt.title(f"{title_prefix}: True vs Estimated")
    plt.legend()
    f2 = os.path.join(figs_dir, f"{title_prefix.lower().replace(' ', '_')}_true_vs_est.png")
    plt.tight_layout(); plt.savefig(f2, dpi=150); plt.close()

# --------------------------
# Optional: tiny predictive checks
# --------------------------
def simple_posterior_predictive_rmse_gamma(df, trace, var_mu_name="mu_group", holdout_frac=0.2, seed=0):
    """
    Very lightweight: hold out a small fraction per group and compare observed to predicted mean (mu_g).
    """
    rng = np.random.default_rng(seed)
    df = df.copy()
    preds, obs = [], []
    mu_group = trace.posterior[var_mu_name].mean(dim=("chain", "draw")).values

    for g, grp in df.groupby("g"):
        idx = grp.index.to_numpy()
        n_hold = max(1, int(np.floor(len(idx) * holdout_frac)))
        hold_idx = rng.choice(idx, size=n_hold, replace=False)
        y_hold = df.loc[hold_idx, "y"].values
        mu_g = mu_group[int(g)]
        preds.append(np.full_like(y_hold, fill_value=mu_g, dtype=float))
        obs.append(y_hold)

    preds = np.concatenate(preds)
    obs = np.concatenate(obs)
    rmse = float(np.sqrt(np.mean((preds - obs) ** 2)))
    return rmse


def simple_posterior_predictive_rmse_normal(df, trace, var_mu_name="mu", holdout_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    df = df.copy()
    preds, obs = [], []
    mu = trace.posterior[var_mu_name].mean(dim=("chain", "draw")).values
    for g, grp in df.groupby("g"):
        idx = grp.index.to_numpy()
        n_hold = max(1, int(np.floor(len(idx) * holdout_frac)))
        hold_idx = rng.choice(idx, size=n_hold, replace=False)
        y_hold = df.loc[hold_idx, "y"].values
        preds.append(np.full_like(y_hold, fill_value=mu[int(g)], dtype=float))
        obs.append(y_hold)
    preds = np.concatenate(preds)
    obs = np.concatenate(obs)
    rmse = float(np.sqrt(np.mean((preds - obs) ** 2)))
    return rmse

# --------------------------
# Main driver
# --------------------------
def run_branch(label, df, true_means, b, draws, tune, chains, target_accept):
    """
    Runs both Gamma (correct) and Normal (mis-specified) on a given dataset df,
    plus BayesBag for each, then prints/plots evaluations.
    """
    K = df["g"].nunique()

    # ----- Gamma (correct) -----
    print(f"\n=== [{label}] Fitting Gamma (correct) ===")
    trace_gamma = fit_model_gamma(df, draws=draws, tune=tune, chains=chains, target_accept=target_accept)
    az.to_netcdf(trace_gamma, os.path.join(OUT_DIR, f"trace_gamma_{label}.nc"))

    std_mean_g, std_low_g, std_high_g, _ = posterior_ci(trace_gamma, "mu_group", alpha=0.05)

    print(f"[{label}] BayesBag (Gamma, cluster bootstrap)")
    bag_gamma = bayesbag_gamma_cluster(df, b=b, mfactor=1.0, draws=draws, tune=tune,
                                       chains=chains, target_accept=target_accept)
    bag_sum_gamma = summarize_bagged(bag_gamma)

    eval_and_plot_groupwise(
        true_means=true_means[:K],
        std_mean=std_mean_g,
        std_low=std_low_g,
        std_high=std_high_g,
        bag_summary=bag_sum_gamma,
        title_prefix=f"{label} - Gamma (correct)"
    )

    rmse_g = simple_posterior_predictive_rmse_gamma(df, trace_gamma, holdout_frac=0.2, seed=123)
    print(f"[{label}] Gamma: simple posterior-predictive mean RMSE (hold-out) = {rmse_g:.4f}")

    # ----- Normal (mis-specified) -----
    print(f"\n=== [{label}] Fitting Normal (mis-specified) ===")
    trace_norm = fit_model_normal(df, draws=draws, tune=tune, chains=chains, target_accept=target_accept)
    az.to_netcdf(trace_norm, os.path.join(OUT_DIR, f"trace_normal_{label}.nc"))

    std_mean_n, std_low_n, std_high_n, _ = posterior_ci(trace_norm, "mu", alpha=0.05)

    print(f"[{label}] BayesBag (Normal, cluster bootstrap)")
    bag_norm = bayesbag_normal_cluster(df, b=b, mfactor=1.0, draws=draws, tune=tune,
                                       chains=chains, target_accept=target_accept)
    bag_sum_norm = summarize_bagged(bag_norm)

    eval_and_plot_groupwise(
        true_means=true_means[:K],
        std_mean=std_mean_n,
        std_low=std_low_n,
        std_high=std_high_n,
        bag_summary=bag_sum_norm,
        title_prefix=f"{label} - Normal (mis-specified)"
    )

    rmse_n = simple_posterior_predictive_rmse_normal(df, trace_norm, holdout_frac=0.2, seed=123)
    print(f"[{label}] Normal: simple posterior-predictive mean RMSE (hold-out) = {rmse_n:.4f}")


def main(b=50, draws=1000, tune=1000, chains=4, target_accept=0.9,
         num_groups=20, n_per_group=100, shape=2.0, scale=3.0, contam=True,
         contam_frac_groups=1.0, contam_frac_points=0.2, contam_factor=0.1, seed=42):
    # Generate base (clean) data with true per-group means
    df_clean, true_means = generate_data_gamma(
        num_groups=num_groups, n_per_group=n_per_group, shape=shape, scale=scale, seed=seed
    )

    # Optionally create contaminated version
    if contam:
        df_cont, contam_groups = contaminate_data(
            df_clean, frac_groups=contam_frac_groups, frac_points=contam_frac_points,
            factor=contam_factor, seed=seed
        )
        print(f"Contaminated groups: {sorted(map(int, contam_groups))}")
    else:
        df_cont = df_clean.copy()

    # Run both branches (clean and contaminated)
    run_branch("CLEAN", df_clean, true_means, b, draws, tune, chains, target_accept)
    run_branch("CONTAM", df_cont, true_means, b, draws, tune, chains, target_accept)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--b", type=int, default=50, help="Number of BayesBag bootstraps")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target_accept", type=float, default=0.9)
    parser.add_argument("--num_groups", type=int, default=20)
    parser.add_argument("--n_per_group", type=int, default=100)
    parser.add_argument("--shape", type=float, default=2.0, help="Between-group Gamma shape for true means")
    parser.add_argument("--scale", type=float, default=3.0, help="Between-group Gamma scale for true means")
    parser.add_argument("--no_contam", action="store_true", help="Disable contamination")
    parser.add_argument("--contam_frac_groups", type=float, default=1.0)
    parser.add_argument("--contam_frac_points", type=float, default=0.2)
    parser.add_argument("--contam_factor", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(
        b=args.b,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        target_accept=args.target_accept,
        num_groups=args.num_groups,
        n_per_group=args.n_per_group,
        shape=args.shape,
        scale=args.scale,
        contam=not args.no_contam,
        contam_frac_groups=args.contam_frac_groups,
        contam_frac_points=args.contam_frac_points,
        contam_factor=args.contam_factor,
        seed=args.seed,
    )
