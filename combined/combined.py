import argparse
import os
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

output_dir = "figs_synthetic"
os.makedirs(output_dir, exist_ok=True)

# -- SYNTHETIC DATA --
def generate_gamma_long(num_groups=20, n_per_group=100, shape=2.0, scale=3.0, seed=42):
    np.random.seed(seed)
    # true group means (mu_k)
    true_means = np.random.gamma(shape, scale, size=num_groups)
    rows = []
    for k in range(num_groups):
        y = np.random.gamma(shape, scale, size=n_per_group)
        for yi in y:
            rows.append({"g": k, "y": yi})
    df = pd.DataFrame(rows)
    return df, true_means

# DATA CONTAM.
def contaminate_multiplicative(df, frac=0.1, factor=0.1, seed=42):
    np.random.seed(seed)
    groups = df['g'].unique()
    contam_k = np.random.choice(groups, size=int(frac * len(groups)), replace=False)
    df_cont = df.copy()
    df_cont.loc[df_cont['g'].isin(contam_k), 'y'] = df_cont.loc[df_cont['g'].isin(contam_k), 'y'] * factor
    return df_cont, contam_k

# MODEL SPECIFICATION
def fit_gamma_model_long(df, draws=4000, tune=2000, chains=4, target_accept=0.95):
    groups = np.sort(df['g'].unique())
    K = len(groups)
    site_idx = df['g'].values.astype(int)
    y = df['y'].values

    with pm.Model() as model:
        alpha = pm.HalfNormal('alpha', sigma=10)
        mu_group = pm.Normal('mu_group', mu=0, sigma=10, shape=K)

        # Convert mu_group to positive mean (Gamma mean must be > 0)
        mu_obs = pm.Deterministic('mu', pm.math.exp(mu_group)[groups])

        # Convert mean+shape to shape+rate
        beta = alpha / mu_obs

        y_like = pm.Gamma('y_like', alpha=alpha, beta=beta, observed=y)
        trace = pm.sample(draws=draws, tune=tune, chains=chains, target_accept=target_accept, return_inferencedata=True)

    return trace

# MODEL MISSPECIFICATION
def fit_normal_model_long(df, draws=4000, tune=2000, chains=4, target_accept=0.95):
    groups = np.sort(df['g'].unique())
    K = len(groups)
    site_idx = df['g'].values.astype(int)
    y = df['y'].values

    with pm.Model() as model:
        mu_k = pm.Normal('mu', mu=0.0, sigma=10.0, shape=K)
        sigma_k = pm.HalfNormal('sigma', sigma=10.0, shape=K)
        y_like = pm.Normal('y_like', mu=mu_k[site_idx], sigma=sigma_k[site_idx], observed=y)
        trace = pm.sample(draws=draws, tune=tune, chains=chains, target_accept=target_accept, return_inferencedata=True)

    return trace

# -- MEAGER --
def fit_meager_pymc(data_df, draws=4000, tune=2000, chains=4, target_accept=0.95):
    df = data_df.copy()
    df = df.reset_index(drop=True)
    K = df['g'].nunique()
    site_idx = df['g'].astype(int).values
    y = df['y'].values
    ITT = df['t'].astype(int).values if 't' in df.columns else np.zeros(len(df), dtype=int)

    with pm.Model() as model:
        mu = pm.Normal('mu', mu=0.0, sigma=100)
        tau = pm.Normal('tau', mu=0.0, sigma=100)

        sigma_mu = pm.HalfNormal('sigma_mu', sigma=100)
        sigma_tau = pm.HalfNormal('sigma_tau', sigma=100)

        mu_k = pm.Normal('mu_k', mu=mu, sigma=sigma_mu, shape=K)
        tau_k = pm.Normal('tau_k', mu=tau, sigma=sigma_tau, shape=K)

        sigma_y_k = pm.HalfNormal('sigma_y_k', sigma=100, shape=K)

        mu_obs = mu_k[site_idx] + tau_k[site_idx] * ITT
        y_like = pm.Normal('y_like', mu=mu_obs, sigma=sigma_y_k[site_idx], observed=y)

        trace = pm.sample(draws=draws, tune=tune, chains=chains, target_accept=target_accept, return_inferencedata=True)

    return trace

# BayesBag
# def bayesbag(y_obs, model_func, b=100, mfactor=1.0):
#     bagged_means = []
#     m = int(mfactor * len(y_obs))
#     for _ in tqdm(range(b), desc="BayesBag iterations"):
#         indices = np.random.choice(len(y_obs), size=m, replace=True)
#         boot_data = [y_obs[i] for i in indices]
#         trace = model_func(boot_data)
#         mu_mean = trace.posterior["mu"].mean(dim=["chain", "draw"]).values
#         bagged_means.append(mu_mean)
#     return np.array(bagged_means)

def bayesbag_long(df, fit_func, extract_group_var='mu', b=100, mfactor=1.0, sample_kwargs=None):
    sample_kwargs = sample_kwargs or {}
    groups = np.sort(df['g'].unique())
    K = len(groups)
    bagged = []

    for i in tqdm(range(b), desc='BayesBag'):
        # resample groups and within-group rows
        boot_groups = np.random.choice(groups, size=int(mfactor * K), replace=True)
        rows = []
        for g in boot_groups:
            grp = df[df['g'] == g]
            # resample rows within group (same size)
            boot_rows = grp.sample(n=len(grp), replace=True)
            # assign a new group index to keep K consistent later: map boot_groups unique order to 0..K-1
            rows.append(boot_rows)
        boot_df = pd.concat(rows, ignore_index=True)

        # reindex groups so they are 0..K-1 in the bootstrap
        old_groups = np.unique(boot_groups)
        mapping = {old: new for new, old in enumerate(old_groups)}
        boot_df['g'] = boot_df['g'].map(mapping).astype(int)

        trace = fit_func(boot_df, **sample_kwargs)

        # extract group means; try a few common names
        if extract_group_var in trace.posterior:
            var = trace.posterior[extract_group_var]
        else:
            # try fallback names
            possible = [n for n in trace.posterior.data_vars if n.startswith(extract_group_var)]
            if not possible:
                raise KeyError(f"Could not find group var '{extract_group_var}' in trace posterior. Available: {list(trace.posterior.data_vars)}")
            var = trace.posterior[possible[0]]

        # compute posterior mean for each group
        means = var.mean(dim=("chain", "draw")).values
        # if fewer groups were present in a bootstrap than K, pad with nan
        if means.shape[-1] < K:
            padded = np.full(K, np.nan)
            padded[: means.shape[-1]] = means
            means = padded
        bagged.append(means)

    return np.array(bagged)

# Model eval
def evaluate_results(theta_mean, theta_std, theta_bagged_mean, theta_bagged_std, y_obs, true_theta=None, label_prefix=""):
    eps = 1e-8
    # Only compute MSE if true_theta is provided
    if true_theta is not None:
        mse_standard = np.mean((theta_mean - true_theta) ** 2)
        mse_bagged = np.mean((theta_bagged_mean - true_theta) ** 2)
    else:
        mse_standard = mse_bagged = None

    # RSE mean
    rse = (theta_bagged_mean - theta_mean) ** 2 / (theta_bagged_std ** 2 + eps)
    rse_mean = np.mean(rse)

    # Mismatch Index
    NvN = len(y_obs) * np.sum(theta_std ** 2)
    MvsM = len(y_obs) * np.sum(theta_bagged_std ** 2)
    mismatch_index = 1 - ((2 * NvN) / MvsM) if MvsM > NvN else np.nan

    # Print results
    print(f"\n--- {label_prefix} Results ---")
    if mse_standard is not None:
        print(f"Standard MSE: {mse_standard:.4f}")
        print(f"BayesBag MSE: {mse_bagged:.4f}")
    print(f"RSE Mean: {rse_mean:.4f}")
    print(f"Mismatch Index: {mismatch_index:.4f}")

def compare_group_means(true_means, trace_std, bagged_means, varname='mu', out_prefix='synthetic'):
    """Plot comparison of true means, standard posterior means, and BayesBag means."""
    mu_std = trace_std.posterior[varname].mean(dim=("chain", "draw")).values
    mu_bag = np.nanmean(bagged_means, axis=0)

    K = len(true_means)
    plt.figure(figsize=(8, 6))
    plt.scatter(true_means, mu_std[:K], label='Standard posterior mean')
    plt.scatter(true_means, mu_bag[:K], label='BayesBag posterior mean', marker='x')
    plt.plot([true_means.min(), true_means.max()], [true_means.min(), true_means.max()], 'k--', alpha=0.6)
    plt.xlabel('True group mean')
    plt.ylabel('Estimated group mean')
    plt.legend()
    plt.title('True vs Estimated group means')
    plt.tight_layout()
    plt.savefig(f"figs/{out_prefix}_group_means.png", dpi=150)
    plt.close()

# -- MAIN --
def main_synthetic(contaminate=False):
    # Generate data
    df, true_means = generate_gamma_long(num_groups=20, n_per_group=100, shape=2.0, scale=3.0)

    if contaminate:
        df_cont, contam_k = contaminate_multiplicative(df, frac=0.2, factor=0.1)
    else:
        df_cont = df.copy()

    # Fit correct model (Gamma) on contaminated data
    print('\nFitting correct Gamma model (contaminated data)')
    trace_gamma = fit_gamma_model_long(df_cont, draws=1000, tune=1000)

    # BayesBag for Gamma model
    print('\nRunning BayesBag for Gamma model')
    bagged_gamma = bayesbag_long(df_cont, fit_gamma_model_long, extract_group_var='mu', b=20, sample_kwargs={"draws":500, "tune":500})

    # Fit incorrect model (Normal) on contaminated data
    print('\nFitting incorrect Normal model (contaminated data)')
    trace_norm = fit_normal_model_long(df_cont, draws=1000, tune=1000)

    # BayesBag for Normal model
    print('\nRunning BayesBag for Normal model')
    bagged_norm = bayesbag_long(df_cont, fit_normal_model_long, extract_group_var='mu', b=20, sample_kwargs={"draws":500, "tune":500})

    # Compare (use true_means available for synthetic)
    compare_group_means(true_means, trace_gamma, bagged_gamma, varname='mu', out_prefix='gamma_correct')
    compare_group_means(true_means, trace_norm, bagged_norm, varname='mu', out_prefix='normal_incorrect')
    
    theta_mean = az.extract(trace_gamma, var_names=['mu']).mean(dim=("chain", "draw")).values
    theta_std = az.extract(trace_gamma, var_names=['mu']).std(dim=("chain", "draw")).values

    theta_bagged_mean = bagged_gamma["mu"]["mean"]
    theta_bagged_std = bagged_gamma["mu"]["std"]

    y_obs = df_cont['y'].values
    true_theta = true_means.values

    evaluate_results(theta_mean, theta_std, theta_bagged_mean, theta_bagged_std,
                 y_obs, true_theta=true_theta,
                 label_prefix="Synthetic - Correct Model")
    # Save traces
    az.to_netcdf(trace_gamma, "out/trace_gamma.nc")
    az.to_netcdf(trace_norm, "out/trace_norm.nc")

    print('Synthetic experiment completed. Plots in figs/, traces in out/.')

def main_meager(rdata_path='microcredit-profit-independent.Rdata', stan_like_df=None):
    # If stan_like_df is provided (pre-loaded), use it; otherwise try to read Rdata
    if stan_like_df is None:
        try:
            import pyreadr
            res = pyreadr.read_r(rdata_path)
            data = res['data']
            data = data.rename(columns={"site": "g", "treatment": "t", "profit": "y"})
            data['g'] = data['g'].astype('category')
            data['t'] = data['t'].astype('category')
            data['g'] = data['g'].cat.codes.astype(int)
            data['t'] = data['t'].cat.codes.astype(int)
        except Exception as e:
            raise RuntimeError('Could not load Rdata. Provide a pandas DataFrame via stan_like_df or install pyreadr.')
    else:
        data = stan_like_df.copy()

    # Fit Meager PyMC model
    print('\nFitting Meager PyMC model')
    trace_meager = fit_meager_pymc(data, draws=1000, tune=1000)

    # BayesBag on Meager model (bootstrap by site)
    print('\nRunning BayesBag for Meager model')
    bagged_meager = bayesbag_long(data, fit_meager_pymc, extract_group_var='mu_k', b=20, sample_kwargs={"draws":500, "tune":500})

    compare_group_means(None, trace_meager, bagged_meager, varname='mu_k', out_prefix='meager_model')

    theta_mean = az.extract(trace_meager, var_names=['mu_k']).mean(dim=("chain", "draw")).values
    theta_std = az.extract(trace_meager, var_names=['mu_k']).std(dim=("chain", "draw")).values

    theta_bagged_mean = bagged_meager["mu_k"]["mean"]
    theta_bagged_std = bagged_meager["mu_k"]["std"]

    y_obs = data['y'].values

    evaluate_results(theta_mean, theta_std, theta_bagged_mean, theta_bagged_std,
                 y_obs, true_theta=None,
                 label_prefix="Meager Model")
    # Save trace
    az.to_netcdf(trace_meager, "out/trace_meager.nc")
    print('Meager experiment completed. Traces in out/.')


if __name__ == "__main__":
    import argparse, os
    
    parser = argparse.ArgumentParser(description="Run BayesBag experiments.")
    parser.add_argument("--mode", choices=["synthetic", "meager"], required=True,
                        help="Choose experiment type.")
    parser.add_argument("--contaminate", action="store_true",
                        help="If set, contaminate synthetic data.")
    args = parser.parse_args()

    # Make sure output dirs exist
    os.makedirs("figs", exist_ok=True)
    os.makedirs("out", exist_ok=True)

    if args.mode == "synthetic":
        main_synthetic(contaminate=args.contaminate)
    elif args.mode == "meager":
        main_meager()
