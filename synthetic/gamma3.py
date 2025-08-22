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
FIGS_DIR = "figs_gamma3"
OUT_DIR  = "out_gamma3"
os.makedirs(FIGS_DIR, exist_ok=True)
os.makedirs(OUT_DIR,  exist_ok=True)

def _make_io_dirs(base_figs, base_out, label):
    sub = label.lower()
    figs_dir = os.path.join(base_figs, sub)
    out_dir  = os.path.join(base_out,  sub)
    os.makedirs(figs_dir, exist_ok=True)
    os.makedirs(out_dir,  exist_ok=True)
    return figs_dir, out_dir

# ==========================================================
# Data generation & contamination
# ==========================================================
def generate_data_gamma(num_groups=20, n_per_group=100, shape=2.0, scale=3.0, alpha_sim=2.0, seed=42):
    """
    Draw true per-group means mu_k ~ Gamma(shape, scale)  (between-group variability).
    Then within each group, draw y_ik ~ Gamma(alpha_sim, beta_k) where E[y|g]=mu_k.
    Smaller alpha_sim => heavier-tailed within-group data (harder for Normal).
    """
    rng = np.random.default_rng(seed)
    true_means = rng.gamma(shape=shape, scale=scale, size=num_groups)

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

def contaminate_groups_mixed(df, frac_groups=0.5, atten_range=(0.05, 0.7),
                             inflate_range=(1.5, 6.0), p_inflate=0.5, seed=42):
    rng = np.random.default_rng(seed)
    dfc = df.copy()
    groups = np.array(sorted(dfc["g"].unique()))
    m = max(1, int(np.round(frac_groups * len(groups))))
    chosen = rng.choice(groups, size=m, replace=False)

    for g in chosen:
        if rng.random() < p_inflate:
            f = rng.uniform(*inflate_range)   # > 1
        else:
            f = rng.uniform(*atten_range)     # < 1
        dfc.loc[dfc["g"] == g, "y"] *= f

    return dfc, chosen

def contaminate_points_mixed(df, frac_groups=0.5, frac_points_range=(0.05, 0.4),
                             atten_range=(0.05, 0.7), inflate_range=(1.5, 8.0),
                             seed=42):
    rng = np.random.default_rng(seed)
    dfc = df.copy()
    groups = np.array(sorted(dfc["g"].unique()))
    m = max(1, int(np.round(frac_groups * len(groups))))
    chosen = rng.choice(groups, size=m, replace=False)

    for g in chosen:
        grp_idx = dfc.index[dfc["g"] == g].to_numpy()
        frac = rng.uniform(*frac_points_range)
        k = max(1, int(np.floor(frac * len(grp_idx))))
        idx = rng.choice(grp_idx, size=k, replace=False)
        # split into down vs up
        k_down = rng.integers(0, k + 1)
        idx_down = idx[:k_down]
        idx_up = idx[k_down:]
        if len(idx_down):
            dfc.loc[idx_down, "y"] *= rng.uniform(*atten_range, size=len(idx_down))
        if len(idx_up):
            dfc.loc[idx_up, "y"] *= rng.uniform(*inflate_range, size=len(idx_up))

    return dfc, chosen

def contaminate_heavy_tail(df, frac_groups=0.5, p_outlier=0.15,
                           lognorm_sigma=2.0, p_small=0.3, small_min=0.02, small_max=0.3,
                           seed=42):
    rng = np.random.default_rng(seed)
    dfc = df.copy()
    groups = np.array(sorted(dfc["g"].unique()))
    m = max(1, int(np.round(frac_groups * len(groups))))
    chosen = rng.choice(groups, size=m, replace=False)

    for g in chosen:
        grp_idx = dfc.index[dfc["g"] == g].to_numpy()
        mask = rng.random(len(grp_idx)) < p_outlier
        idx = grp_idx[mask]
        if len(idx) == 0:
            continue
        is_small = rng.random(len(idx)) < p_small
        big_idx = idx[~is_small]
        if len(big_idx):
            multipliers = np.exp(rng.normal(0.0, lognorm_sigma, size=len(big_idx)))
            dfc.loc[big_idx, "y"] *= multipliers
        small_idx = idx[is_small]
        if len(small_idx):
            dfc.loc[small_idx, "y"] *= rng.uniform(small_min, small_max, size=len(small_idx))

    return dfc, chosen

# ==========================================================
# Models (return both trace and model for PPC)
# ==========================================================
def fit_model_gamma(df, draws=2000, tune=1000, chains=4, target_accept=0.9):
    groups = np.sort(df["g"].unique())
    K = len(groups)
    site_idx = df["g"].values.astype("int64")
    y = df["y"].values.astype("float64")

    with pm.Model() as model:
        alpha = pm.HalfNormal("alpha", sigma=10.0)
        mu_group_raw = pm.Normal("mu_group_raw", mu=0.0, sigma=3.0, shape=K)
        mu_group = pm.Deterministic("mu_group", pm.math.exp(mu_group_raw))
        mu_obs = mu_group[site_idx]
        beta_obs = alpha / mu_obs
        _ = pm.Gamma("y_like", alpha=alpha, beta=beta_obs, observed=y)
        trace = pm.sample(draws=draws, tune=tune, chains=chains,
                          target_accept=target_accept, return_inferencedata=True)
    return trace, model

def fit_model_gamma_hier(df, draws=2000, tune=1000, chains=4, target_accept=0.9):
    groups = np.sort(df["g"].unique())
    K = len(groups)
    site_idx = df["g"].values.astype("int64")
    y = df["y"].values.astype("float64")

    with pm.Model() as model:
        mu0 = pm.Normal("mu0", 0.0, 2.5)
        sigma_mu = pm.HalfNormal("sigma_mu", 1.0)
        mu_group_raw = pm.Normal("mu_group_raw", mu=mu0, sigma=sigma_mu, shape=K)
        mu_group = pm.Deterministic("mu_group", pm.math.exp(mu_group_raw))
        alpha = pm.HalfNormal("alpha", 10.0)
        mu_obs = mu_group[site_idx]
        beta_obs = alpha / mu_obs
        _ = pm.Gamma("y_like", alpha=alpha, beta=beta_obs, observed=y)
        trace = pm.sample(draws=draws, tune=tune, chains=chains,
                          target_accept=target_accept, return_inferencedata=True)
    return trace, model

def fit_model_normal_shared_sigma(df, draws=2000, tune=1000, chains=4, target_accept=0.9):
    """
    Normal likelihood with *shared* sigma across groups.
    This makes Gamma->Normal misspecification visible in uncertainty/tails.
    """
    groups = np.sort(df["g"].unique())
    K = len(groups)
    site_idx = df["g"].values.astype("int64")
    y = df["y"].values.astype("float64")

    with pm.Model() as model:
        mu_k = pm.Normal("mu", mu=0.0, sigma=10.0, shape=K)
        sigma = pm.HalfNormal("sigma", sigma=10.0)
        _ = pm.Normal("y_like", mu=mu_k[site_idx], sigma=sigma, observed=y)
        trace = pm.sample(draws=draws, tune=tune, chains=chains,
                          target_accept=target_accept, return_inferencedata=True)
    return trace, model

# ==========================================================
# BayesBag: cluster bootstrap (fixed) + posterior mixture
# ==========================================================
def cluster_bootstrap(df, mfactor=1.0, within_frac=1.0, within_replace=True, seed=None):
    """
    Cluster bootstrap that PRESERVES multiplicities:
    - Sample groups with replacement.
    - Each drawn copy becomes its own new group (g_boot).
    - Optionally sub-sample within groups (within_frac < 1) for robustness.
    """
    rng = np.random.default_rng(seed)
    groups = np.sort(df["g"].unique())
    K = len(groups)

    m_groups = max(1, int(np.ceil(mfactor * K)))
    boot_groups = rng.choice(groups, size=m_groups, replace=True)

    parts = []
    for j, g in enumerate(boot_groups):
        grp = df[df["g"] == g]
        m = max(1, int(np.ceil(within_frac * len(grp))))
        grp_samp = grp.sample(
            n=m,
            replace=within_replace,
            random_state=int(rng.integers(0, 2**31 - 1))
        ).copy()
        grp_samp["g_boot"] = j          # NEW group id per copy
        grp_samp["g_orig"] = int(g)     # remember where it came from
        parts.append(grp_samp)

    boot_df = pd.concat(parts, ignore_index=True)
    return boot_df

def _summarize_bagged_dict(bag_mu_by_orig, groups_order, alpha=0.05):
    """
    bag_mu_by_orig: dict {g -> 1D np.array of draws}, groups_order: sorted array of original group ids
    """
    K = len(groups_order)
    med = np.full(K, np.nan)
    mean = np.full(K, np.nan)
    low = np.full(K, np.nan)
    high = np.full(K, np.nan)
    n_eff = np.zeros(K, dtype=int)
    for i, g in enumerate(groups_order):
        draws = np.asarray(bag_mu_by_orig[g])
        if draws.size == 0:
            continue
        med[i]  = np.median(draws)
        mean[i] = np.mean(draws)
        low[i]  = np.percentile(draws, 100*alpha/2)
        high[i] = np.percentile(draws, 100*(1-alpha/2))
        n_eff[i] = draws.size
    return {"median": med, "mean": mean, "low": low, "high": high, "n_eff": n_eff}

def bayesbag_cluster_posterior(
    df,
    fit_fn,                # function that returns (trace, model)
    varname_mu,            # 'mu_group' for gamma, 'mu' for normal
    b=100,
    mfactor=0.7,
    within_frac=0.5,
    within_replace=False,
    draws=1000,
    tune=1000,
    chains=2,
    target_accept=0.9,
    seed=42,
    hyper_vars=None,
    return_predictive=False,
    ppc_var="y_like",
):
    """
    Generic BayesBag: builds a posterior mixture for group means (and optionally hyperparameters and PPC).
    Returns:
      bagged_mu: dict {original_group_id -> concatenated draws}
      bagged_hyper: DataFrame of concatenated hyperparameter draws (optional)
      bagged_ppc: np.ndarray of concatenated posterior predictive draws (optional)
    """
    rng = np.random.default_rng(seed)
    all_groups = np.sort(df["g"].unique())

    bag_mu_by_orig = {g: [] for g in all_groups}
    bag_hyper = []
    bag_ppc = []

    for _ in tqdm(range(b), desc=f"BayesBag ({varname_mu})"):
        boot_df = cluster_bootstrap(
            df, mfactor=mfactor, within_frac=within_frac,
            within_replace=within_replace, seed=int(rng.integers(0, 2**31 - 1))
        )
        trace, model = fit_fn(boot_df, draws=draws, tune=tune, chains=chains, target_accept=target_accept)

        # Map boot groups back to original ids
        ordering = (boot_df[["g_boot","g_orig"]]
                    .drop_duplicates()
                    .sort_values("g_boot"))
        g_orig = ordering["g_orig"].to_numpy()

        # Collect draws of group means
        mu = trace.posterior[varname_mu]                 # (chain, draw, K_boot)
        mu = mu.stack(sample=("chain","draw"))           # (sample, K_boot)
        mu_np = np.asarray(mu.values)                    # to numpy
        for j, g in enumerate(g_orig):
            bag_mu_by_orig[int(g)].append(mu_np[:, j])

        # Collect hyperparameters (if provided)
        if hyper_vars:
            bag_hyper.append(az.extract(trace, var_names=list(hyper_vars)).to_dataframe())

        # Optional: posterior predictive (mixture across bags)
        if return_predictive:
            ppc_idata = pm.sample_posterior_predictive(
                trace, var_names=[ppc_var], model=model,
                random_seed=int(rng.integers(0, 2**31 - 1))
            )
            # Extract to numpy: (chain, draw, N) -> stack chains/draws
            y_pp = az.extract(ppc_idata, group="posterior_predictive", var_names=[ppc_var])[ppc_var].to_numpy()
            bag_ppc.append(y_pp)

    # Concatenate across bags
    bagged_mu = {}
    for g, parts in bag_mu_by_orig.items():
        if len(parts):
            bagged_mu[g] = np.concatenate(parts, axis=0)
        else:
            bagged_mu[g] = np.array([])

    bagged_hyper = pd.concat(bag_hyper, ignore_index=True) if len(bag_hyper) else None
    bagged_ppc = np.concatenate(bag_ppc, axis=0) if len(bag_ppc) else None
    return bagged_mu, bagged_hyper, bagged_ppc

# ==========================================================
# Summaries & evaluation helpers
# ==========================================================
def posterior_ci(trace, varname, alpha=0.05):
    post = trace.posterior[varname]
    mean = post.mean(dim=("chain", "draw")).values
    low = post.quantile(alpha / 2.0, dim=("chain", "draw")).values
    high = post.quantile(1.0 - alpha / 2.0, dim=("chain", "draw")).values
    std = post.std(dim=("chain", "draw")).values
    return mean, low, high, std

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

    # MSE of point estimates vs truth
    std_mse = mse_to_truth(std_mean, true_means)
    bag_mse = mse_to_truth(bag_summary["median"], true_means)

    print(f"\n[{title_prefix}] Coverage 95% - Standard: {std_cover:.3f}, Bagged: {bag_cover:.3f}")
    print(f"[{title_prefix}] Median CI width - Standard: {std_width:.3f}, Bagged: {bag_width:.3f}")
    print(f"[{title_prefix}] MSE to true means - Standard: {std_mse:.4f}, Bagged: {bag_mse:.4f}")
    if "n_eff" in bag_summary:
        print(f"[{title_prefix}] Bagged draws per group (min/median/max): "
              f"{int(np.nanmin(bag_summary['n_eff']))}/"
              f"{int(np.nanmedian(bag_summary['n_eff']))}/"
              f"{int(np.nanmax(bag_summary['n_eff']))}")

    # Plot: group index vs mean with 95% intervals
    x = np.arange(K)
    plt.figure(figsize=(10, 5))
    plt.errorbar(x, std_mean, yerr=[std_mean - std_low, std_high - std_mean],
                 fmt='o', capsize=2, label="Standard")
    plt.errorbar(x, bag_summary["median"],
                 yerr=[bag_summary["median"] - bag_summary["low"],
                       bag_summary["high"] - bag_summary["median"]],
                 fmt='x', capsize=2, label="BayesBag (median)")
    plt.plot(x, true_means, linestyle='--', marker='.', label="True μ")
    plt.xlabel("Group index"); plt.ylabel("Mean")
    plt.title(f"{title_prefix}: group means ± 95% intervals")
    plt.legend()
    f1 = os.path.join(figs_dir, f"{title_prefix.lower().replace(' ', '_')}_means_intervals.png")
    plt.tight_layout(); plt.savefig(f1, dpi=150); plt.close()

    # Plot: true vs estimated
    plt.figure(figsize=(6, 6))
    plt.scatter(true_means, std_mean, label="Standard")
    plt.scatter(true_means, bag_summary["median"], label="BayesBag (median)", marker='x')
    lo = np.nanmin([true_means.min(), std_mean.min(), bag_summary["median"].min()])
    hi = np.nanmax([true_means.max(), std_mean.max(), bag_summary["median"].max()])
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("True group mean"); plt.ylabel("Estimated group mean")
    plt.title(f"{title_prefix}: True vs Estimated")
    plt.legend()
    f2 = os.path.join(figs_dir, f"{title_prefix.lower().replace(' ', '_')}_true_vs_est.png")
    plt.tight_layout(); plt.savefig(f2, dpi=150); plt.close()

# Simple predictive check (kept lightweight, mean-based)
def simple_posterior_predictive_rmse_gamma(df, trace, var_mu_name="mu_group", holdout_frac=0.2, seed=0):
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

# ==========================================================
# Orchestration
# ==========================================================
def run_branch(label, df, true_means, b, draws, tune, chains, target_accept, models=("gamma", "normal")):
    """
    Runs selected models on df and evaluates/plots.
    models: iterable with any of {"gamma","hier","normal"} where "normal" uses SHARED sigma.
    """
    figs_dir, out_dir = _make_io_dirs(FIGS_DIR, OUT_DIR, label)
    K = df["g"].nunique()
    groups_order = np.arange(K)

    # ----- Gamma (simple) -----
    if "gamma" in models:
        print(f"\n=== [{label}] Fitting Gamma (simple) ===")
        trace_gamma, model_gamma = fit_model_gamma(df, draws=draws, tune=tune, chains=chains, target_accept=target_accept)
        az.to_netcdf(trace_gamma, os.path.join(out_dir, f"trace_gamma_simple_{label}.nc"))

        std_mean_g, std_low_g, std_high_g, _ = posterior_ci(trace_gamma, "mu_group", alpha=0.05)

        print(f"[{label}] BayesBag (Gamma simple) — posterior mixture of draws")
        bag_mu_g, _, _ = bayesbag_cluster_posterior(
            df, fit_model_gamma, varname_mu="mu_group",
            b=b, mfactor=0.7, within_frac=0.5, within_replace=False,
            draws=draws//2, tune=tune//2, chains=max(2, chains//2), target_accept=target_accept,
            seed=123, hyper_vars=None, return_predictive=False
        )
        bag_sum_gamma = _summarize_bagged_dict(bag_mu_g, groups_order, alpha=0.05)

        eval_and_plot_groupwise(
            true_means=true_means[:K],
            std_mean=std_mean_g,
            std_low=std_low_g,
            std_high=std_high_g,
            bag_summary=bag_sum_gamma,
            title_prefix=f"{label} - Gamma (simple)",
            figs_dir=figs_dir,
        )

        rmse_g = simple_posterior_predictive_rmse_gamma(df, trace_gamma, holdout_frac=0.2, seed=123)
        print(f"[{label}] Gamma (simple): simple posterior-predictive mean RMSE (hold-out) = {rmse_g:.4f}")

    # ----- Gamma (hierarchical) -----
    if "hier" in models:
        print(f"\n=== [{label}] Fitting Gamma (hierarchical) ===")
        trace_gamma_h, model_gamma_h = fit_model_gamma_hier(df, draws=draws, tune=tune, chains=chains, target_accept=target_accept)
        az.to_netcdf(trace_gamma_h, os.path.join(out_dir, f"trace_gamma_hier_{label}.nc"))

        std_mean_gh, std_low_gh, std_high_gh, _ = posterior_ci(trace_gamma_h, "mu_group", alpha=0.05)

        print(f"[{label}] BayesBag (Gamma hier) — posterior mixture of draws")
        bag_mu_gh, bag_hyper_gh, _ = bayesbag_cluster_posterior(
            df, fit_model_gamma_hier, varname_mu="mu_group",
            b=b, mfactor=0.7, within_frac=0.5, within_replace=False,
            draws=draws//2, tune=tune//2, chains=max(2, chains//2), target_accept=target_accept,
            seed=456, hyper_vars=["mu0","sigma_mu","alpha"], return_predictive=False
        )
        bag_sum_gamma_h = _summarize_bagged_dict(bag_mu_gh, groups_order, alpha=0.05)

        eval_and_plot_groupwise(
            true_means=true_means[:K],
            std_mean=std_mean_gh,
            std_low=std_low_gh,
            std_high=std_high_gh,
            bag_summary=bag_sum_gamma_h,
            title_prefix=f"{label} - Gamma (hier)",
            figs_dir=figs_dir,
        )

        rmse_gh = simple_posterior_predictive_rmse_gamma(df, trace_gamma_h, holdout_frac=0.2, seed=123)
        print(f"[{label}] Gamma (hier): simple posterior-predictive mean RMSE (hold-out) = {rmse_gh:.4f}")

    # ----- Normal (mis-specified; shared sigma) -----
    if "normal" in models:
        print(f"\n=== [{label}] Fitting Normal (mis-specified, shared σ) ===")
        trace_norm, model_norm = fit_model_normal_shared_sigma(df, draws=draws, tune=tune, chains=chains, target_accept=target_accept)
        az.to_netcdf(trace_norm, os.path.join(out_dir, f"trace_normal_shared_{label}.nc"))

        std_mean_n, std_low_n, std_high_n, _ = posterior_ci(trace_norm, "mu", alpha=0.05)

        print(f"[{label}] BayesBag (Normal, shared σ) — posterior mixture of draws")
        bag_mu_n, _, _ = bayesbag_cluster_posterior(
            df, fit_model_normal_shared_sigma, varname_mu="mu",
            b=b, mfactor=0.7, within_frac=0.5, within_replace=False,
            draws=draws//2, tune=tune//2, chains=max(2, chains//2), target_accept=target_accept,
            seed=789, hyper_vars=["sigma"], return_predictive=False
        )
        bag_sum_norm = _summarize_bagged_dict(bag_mu_n, groups_order, alpha=0.05)

        eval_and_plot_groupwise(
            true_means=true_means[:K],
            std_mean=std_mean_n,
            std_low=std_low_n,
            std_high=std_high_n,
            bag_summary=bag_sum_norm,
            title_prefix=f"{label} - Normal (mis-specified, shared σ)",
            figs_dir=figs_dir,
        )

        rmse_n = simple_posterior_predictive_rmse_normal(df, trace_norm, holdout_frac=0.2, seed=123)
        print(f"[{label}] Normal (shared σ): simple posterior-predictive mean RMSE (hold-out) = {rmse_n:.4f}")

# ==========================================================
# Main
# ==========================================================
def main(b=50, draws=4000, tune=2000, chains=4, target_accept=0.95,
         num_groups=20, n_per_group=100, shape=2.0, scale=3.0,
         contam=True, contam_frac_groups=1.0, contam_frac_points=0.2, contam_factor=0.1, seed=42,
         scenarios=("clean", "contam"), models=("gamma", "normal"),
         alpha_sim=2.0, contam_type="uniform"):
    """
    Recommended choices to make differences pop:
      num_groups ~ 60, n_per_group ~ 8, alpha_sim ~ 0.6 (heavier tails).
      For BayesBag: b ~ 100, mfactor ~ 0.7, within_frac ~ 0.5, within_replace=False.
    """
    # 1) Generate CLEAN data
    df_clean, true_means = generate_data_gamma(
        num_groups=num_groups,
        n_per_group=n_per_group,
        shape=shape,
        scale=scale,
        alpha_sim=alpha_sim,
        seed=seed,
    )

    # 2) Optional CONTAMINATED dataset
    df_cont = None
    contam_label = None
    if "contam" in scenarios:
        if not contam:
            print("[warn] 'contam' requested but contamination disabled via --no_contam; skipping 'contam' scenario.")
        else:
            if contam_type == "uniform":
                df_cont, contam_groups = contaminate_data(
                    df_clean,
                    frac_groups=contam_frac_groups,
                    frac_points=contam_frac_points,
                    factor=contam_factor,
                    seed=seed,
                )
            elif contam_type == "groups_mixed":
                df_cont, contam_groups = contaminate_groups_mixed(
                    df_clean,
                    frac_groups=contam_frac_groups,
                    seed=seed,
                )
            elif contam_type == "points_mixed":
                df_cont, contam_groups = contaminate_points_mixed(
                    df_clean,
                    frac_groups=contam_frac_groups,
                    seed=seed,
                )
            elif contam_type == "heavy_tail":
                df_cont, contam_groups = contaminate_heavy_tail(
                    df_clean,
                    frac_groups=contam_frac_groups,
                    seed=seed,
                )
            else:
                raise ValueError(f"Unknown contam_type: {contam_type}")

            contam_label = f"CONTAM_{contam_type}"
            print(f"Contaminated groups ({contam_type}): {sorted(map(int, contam_groups))}")

    # 3) Run scenarios/models
    if "clean" in scenarios:
        run_branch("CLEAN", df_clean, true_means, b, draws, tune, chains, target_accept, models=models)

    if "contam" in scenarios:
        if df_cont is not None:
            run_branch(contam_label, df_cont, true_means, b, draws, tune, chains, target_accept, models=models)
        else:
            print("[info] Skipping CONTAM scenario because no contaminated dataset was constructed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--b", type=int, default=50, help="Number of BayesBag bootstraps")
    parser.add_argument("--draws", type=int, default=2000)
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
    parser.add_argument("--scenarios", nargs="+", choices=["clean", "contam"], default=["clean", "contam"],
                    help="Which data scenarios to run")
    parser.add_argument("--models", nargs="+", choices=["gamma", "normal", "hier"], default=["gamma", "normal", "hier"],
                    help="Which models to run for each scenario")
    parser.add_argument("--contam_type", choices=["uniform", "groups_mixed", "points_mixed", "heavy_tail"],
                    default="uniform", help="Contamination mechanism to use when contamination is enabled")
    parser.add_argument("--alpha_sim", type=float, default=2.0,
                    help="Within-group Gamma shape used to GENERATE data (smaller => heavier tails)")
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
        scenarios=tuple(args.scenarios),
        models=tuple(args.models),
        alpha_sim=args.alpha_sim,
        contam_type=args.contam_type,
    )
