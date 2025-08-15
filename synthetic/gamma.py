import argparse
import os
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import argparse

figs_dir = "figs_synthetic"
output_dir = "out_synthetic"
os.makedirs(figs_dir,exist_ok=True)
os.makedirs(output_dir,exist_ok=True)

# -- SYNTHESISE DATA --
def generate_data_gamma(num_groups=20,n_per_group=100,shape=2,scale=3,seed=42):
    np.random.seed(seed)
    true_means = np.random.gamma(shape=shape,scale=scale,size=num_groups)
    rows=[]
    for k in range(num_groups):
        y = np.random.gamma(shape=shape,scale=scale,size=n_per_group)
        for yi in y:
            rows.append({"g": k, "y": float(yi)})
    df = pd.DataFrame(rows)
    return df,true_means

# -- CONTAMINATE DATA
# def contaminate_data(df,frac=0.1,factor=0.1,seed=42):
#     np.random.seed(seed)
#     groups=df["g"].unique()
#     contam_k = np.random.choice(groups,size=int(frac*len(groups)),replace=False)
#     df_cont = df.copy()
#     df_cont.loc[df_cont["g"].isin(contam_k),"y"] = df_cont.loc[df_cont["g"].isin(contam_k),"y"]*factor # break this down
#     # return df_cont, np.array(contam_k)
#     return df_cont, contam_k

def contaminate_data(df, frac_groups=1, frac_points=0.2, factor=0.1, seed=42):
    np.random.seed(seed)
    df_cont = df.copy()
    groups = df["g"].unique()

    contam_groups = np.random.choice(groups, size=int(frac_groups * len(groups)), replace=False)

    for g in contam_groups:
        idx = df_cont[df_cont["g"] == g].sample(frac=frac_points, random_state=seed).index
        df_cont.loc[idx, "y"] *= factor

    return df_cont, contam_groups

# -- MODEL (CORRECT) SPECIFICATION
def fit_model_gamma(df,draws=4000,tune=2000,chains=4,target_accept=0.95):
    groups=np.sort(df["g"].unique())
    K = len(groups)
    site_idx = df["g"].values.astype("int64")
    y = df["y"].values.astype("float64")

    with pm.Model() as model:
        alpha = pm.HalfNormal("alpha",sigma=10)

        mu_group_raw = pm.Normal("mu_group_raw",mu=0,sigma=3,shape=K)
        mu_group = pm.Deterministic("mu_group", pm.math.exp(mu_group_raw))
        mu_obs = pm.Deterministic("mu_obs", mu_group[site_idx])
        
        beta_obs = pm.Deterministic("beta_obs",alpha/mu_obs)
        y_like = pm.Gamma("y_like",alpha=alpha,beta=beta_obs,observed=y)

        trace = pm.sample(draws=draws,tune=tune,chains=chains,target_accept=target_accept,return_inferencedata=True)
        return trace

# -- MODEL MISSPECIFICATION --
def fit_model_normal(df,draws=4000,tune=2000,chains=4,target_accept=0.95):
    groups=np.sort(df["g"].unique())
    K = len(groups)
    site_idx = df["g"].values.astype("int64")
    y = df["y"].values # why not as floats?

    with pm.Model() as model:
        mu_k = pm.Normal("mu",mu=0,sigma=10,shape=K)
        sigma_k = pm.HalfNormal("sigma",sigma=10,shape=K)
        y_like = pm.Normal("y_like",mu=mu_k[site_idx],sigma=sigma_k[site_idx],observed=y)
        trace = pm.sample(draws=draws,tune=tune,chains=chains,target_accept=target_accept,return_inferencedata=True)
        return trace

# -- BAYESBAG PART --
# def bayesbag(df,fit_func,extract_group_var,b=100,mfactor=1,sample_kwargs=None):
#     sample_kwargs=sample_kwargs or {}
#     groups = np.sort(df["g"].unique())
#     K = len(groups)
#     bagged = []

#     for _ in tqdm(range(b),desc="BayesBag"):
#         boot_groups = np.random.choice(groups,size=max(1,int(mfactor*K)),replace=True)
#         rows=[]
#         for g in boot_groups:
#             grp = df[df["g"]==g]
#             boot_rows = grp.sample(n=len(grp),replace=True)
#             rows.append(boot_rows)
#         boot_df = pd.concat(rows,ignore_index=True)

#         # unique_boot = np.unique(boot_groups)
#         # mapping = {old: new for new, old in enumerate(unique_boot)}
#         # boot_df["g"] = boot_df["g"].map(mapping).astype(int)
#         boot_df["g"] = boot_df["g"].astype(int)

#         trace = fit_func(boot_df,**sample_kwargs)

#         # find the group var to extract
#         if extract_group_var in trace.posterior:
#             var = trace.posterior[extract_group_var]
#         else:
#             possible = [n for n in trace.posterior.data_vars if n.startswith(extract_group_var)]
#             if not possible:
#                 raise KeyError(f"Could not find group var '{extract_group_var}' in trace; have: {list(trace.posterior.data_vars)}")
#             var = trace.posterior[possible[0]]

#         means = var.mean(dim=("chain", "draw")).values  # (K_boot,)
#         # pad to length K for consistent stacking
#         if means.shape[-1] < K:
#             padded = np.full(K, np.nan)
#             padded[: means.shape[-1]] = means
#             means = padded
#         bagged.append(means)

#     return np.array(bagged)  # (b, K)

def bayesbag_gamma(df, num_groups, b=50, mfactor=1.0, draws=4000, tune=2000, chains=4, target_accept=0.95):
    bagged_means = []
    m = int(mfactor * len(df))
    for _ in tqdm(range(b), desc="BayesBag iterations"):
        # Bootstrap rows
        boot_df = df.sample(n=m, replace=True)
        # Fit model to bootstrap
        trace = fit_model_gamma(
            boot_df,
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept
        )
        # Extract posterior mean of group means
        theta_mean = trace.posterior["mu_group"].mean(dim=["chain", "draw"]).values
        bagged_means.append(theta_mean)
    return np.array(bagged_means)

def bayesbag_normal(df, num_groups, b=50, mfactor=1.0, draws=4000, tune=2000, chains=4, target_accept=0.95):
    bagged_means = []
    m = int(mfactor * len(df))
    for _ in tqdm(range(b), desc="BayesBag iterations"):
        boot_df = df.sample(n=m, replace=True)
        trace = fit_model_normal(
            boot_df,
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept
        )
        theta_mean = trace.posterior["mu"].mean(dim=["chain", "draw"]).values
        bagged_means.append(theta_mean)
    return np.array(bagged_means)

def summarize_trace_group(trace: az.InferenceData, varname: str):
    """Return (mean, std) across chains/draws for a group-level var of shape (K,)."""
    mu = trace.posterior[varname].mean(dim=("chain", "draw")).values
    sd = trace.posterior[varname].std(dim=("chain", "draw")).values
    return mu, sd


def evaluate_results(theta_mean, theta_std, theta_bagged_mean, theta_bagged_std, n_obs: int, true_theta=None, label_prefix=""):
    """Compute metrics. If true_theta is None, skip MSEs (for real-data cases)."""
    eps = 1e-8

    # Only compute MSE if truth provided
    if true_theta is not None:
        mse_standard = np.mean((theta_mean - true_theta) ** 2)
        mse_bagged = np.mean((theta_bagged_mean - true_theta) ** 2)
    else:
        mse_standard = None
        mse_bagged = None

    # RSE mean
    rse = (theta_bagged_mean - theta_mean) ** 2 / (theta_bagged_std ** 2 + eps)
    rse_mean = np.nanmean(rse)

    # Mismatch Index
    NvN = n_obs * np.nansum(theta_std ** 2)
    MvsM = n_obs * np.nansum(theta_bagged_std ** 2)
    mismatch_index = 1 - ((2 * NvN) / MvsM) if MvsM > NvN else np.nan

    print(f"\n--- {label_prefix} Results ---")
    if mse_standard is not None:
        print(f"Standard MSE: {mse_standard:.4f}")
        print(f"BayesBag MSE: {mse_bagged:.4f}")
    print(f"RSE Mean: {rse_mean:.4f}")
    print(f"Mismatch Index: {mismatch_index:.4f}")

    return {
        "mse_standard": mse_standard,
        "mse_bagged": mse_bagged,
        "rse_mean": rse_mean,
        "mismatch_index": mismatch_index,
    }


def plot_true_vs_est(true_means, est_mean, est_bagged_mean, title: str, out_path: str):
    plt.figure(figsize=(7, 6))
    plt.scatter(true_means, est_mean, label="Standard posterior mean")
    plt.scatter(true_means, est_bagged_mean, label="BayesBag posterior mean", marker="x")
    lo = min(np.min(true_means), np.min(est_mean), np.min(est_bagged_mean))
    hi = max(np.max(true_means), np.max(est_mean), np.max(est_bagged_mean))
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("True group mean")
    plt.ylabel("Estimated group mean")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# -- MAIN --

def main(run: str = "both", contaminate: bool = True, b: int = 50, draws: int = 4000, tune: int = 2000, chains: int = 4, target_accept: float = 0.95):
    # 1) Generate and (optionally) contaminate data
    df, true_means = generate_data_gamma()
    if contaminate:
        df_cont, contam_k = contaminate_data(df)
        print(f"Contaminated groups: {sorted(map(int, contam_k))}")
    else:
        df_cont = df

    K = df_cont['g'].nunique()
    N = len(df_cont)

    # 2) Correct Gamma model
    if run in ("gamma", "both"):
        print("\nFitting correct Gamma model")
        trace_gamma = fit_model_gamma(df_cont, draws=draws, tune=tune, chains=chains, target_accept=target_accept)
        az.to_netcdf(trace_gamma, os.path.join(output_dir, "trace_gamma.nc"))

        # BayesBag for Gamma
        print("Running BayesBag for Gamma model")
        bag_gamma = bayesbag_gamma(
            df_cont,
            num_groups=K,
            b=b,
            mfactor=1.0,
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept
        )

        mu_mean_gamma, mu_std_gamma = summarize_trace_group(trace_gamma, "mu_group")
        mu_bag_mean_gamma = np.nanmean(bag_gamma, axis=0)
        mu_bag_std_gamma = np.nanstd(bag_gamma, axis=0)

        # Metrics + plot (synthetic has truth)
        _ = evaluate_results(
            theta_mean=mu_mean_gamma,
            theta_std=mu_std_gamma,
            theta_bagged_mean=mu_bag_mean_gamma,
            theta_bagged_std=mu_bag_std_gamma,
            n_obs=N,
            true_theta=true_means[:K],
            label_prefix="Synthetic - Gamma (correct)",
        )

        plot_true_vs_est(
            true_means[:K],
            mu_mean_gamma,
            mu_bag_mean_gamma,
            title="True vs Estimated (Gamma)",
            out_path=os.path.join(figs_dir, "gamma_true_vs_est.png"),
        )

    # 3) Incorrect Normal model
    if run in ("normal", "both"):
        print("\nFitting incorrect Normal model")
        trace_norm = fit_model_normal(df_cont, draws=draws, tune=tune, chains=chains, target_accept=target_accept)
        az.to_netcdf(trace_norm, os.path.join(output_dir, "trace_normal.nc"))

        print("Running BayesBag for Normal model")
        bag_norm = bayesbag_normal(
            df_cont,
            num_groups=K,
            b=b,
            mfactor=1.0,
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=target_accept
        )

        mu_mean_norm, mu_std_norm = summarize_trace_group(trace_norm, "mu")
        mu_bag_mean_norm = np.nanmean(bag_norm, axis=0)
        mu_bag_std_norm = np.nanstd(bag_norm, axis=0)

        _ = evaluate_results(
            theta_mean=mu_mean_norm,
            theta_std=mu_std_norm,
            theta_bagged_mean=mu_bag_mean_norm,
            theta_bagged_std=mu_bag_std_norm,
            n_obs=N,
            true_theta=true_means[:K],
            label_prefix="Synthetic - Normal (mis-specified)",
        )

        plot_true_vs_est(
            true_means[:K],
            mu_mean_norm,
            mu_bag_mean_norm,
            title="True vs Estimated (Normal)",
            out_path=os.path.join(figs_dir, "normal_true_vs_est.png"),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=["gamma", "normal", "both"], default="both", help="Which model(s) to run")
    parser.add_argument("--contaminate", dest="contaminate", action="store_true", help="Apply multiplicative contamination to a fraction of groups")
    parser.add_argument("--no-contaminate", dest="contaminate", action="store_false", help="Do not contaminate the data")
    parser.set_defaults(contaminate=True)
    parser.add_argument("--b", type=int, default=50, help="Number of BayesBag bootstraps")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target_accept", type=float, default=0.9)
    args = parser.parse_args()

    main(
        run=args.run,
        contaminate=args.contaminate,
        b=args.b,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        target_accept=args.target_accept,
    )
