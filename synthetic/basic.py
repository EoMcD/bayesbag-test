import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
from scipy.stats import norm as sp_norm
from scipy.special import logsumexp
import pymc as pm
import arviz as az
from tqdm.auto import tqdm
import os, json
from datetime import datetime

# data = gamma.rvs(a=3,scale=2,size=1000)

def data_generation(groups,per_group,seed):
    rng = np.random.default_rng(seed)
    alpha,theta = 3,2
    alpha_sd,theta_sd = 0.3,0.3

    # group level params
    alphas = np.exp(norm.rvs(loc=np.log(alpha),scale=alpha_sd,size=groups,random_state=rng))
    thetas = np.exp(norm.rvs(loc=np.log(theta),scale=theta_sd,size=groups,random_state=rng))

    # synthetic data for each group
    data = np.empty((groups, per_group), dtype=float)
    for g in range(groups):
        data[g, :] = gamma.rvs(a=alphas[g],scale=thetas[g],size=per_group,random_state=rng)
    return data, alphas, thetas

X,a,theta = data_generation(10,100,42)

# # verification
# sample_mean = X.mean(axis=1)
# sample_var = X.var(axis=1,ddof=1)
# theo_mean = a*theta
# theo_var = a*(theta**2)

# print("grp   k      θ      E[X]   mean̂     Var[X]  var̂")
# for g in range(len(a)):
#     print(f"{g:>3}  {a[g]:6.3f} {theta[g]:6.3f} {theo_mean[g]:6.3f} "
#           f"{sample_mean[g]:7.3f} {theo_var[g]:7.3f} {sample_var[g]:7.3f}")

# print(f"\nMean abs error of group means: {np.mean(np.abs(sample_mean - theo_mean)):.3f}")
# print(f"Mean abs error of group variances: {np.mean(np.abs(sample_var - theo_var)):.3f}")

# # z-scores for group means
# se_mean = np.sqrt(theo_var / X.shape[1])
# z = (sample_mean - theo_mean) / se_mean
# print("Mean z  :", z.mean().round(3))
# print("Std z   :", z.std(ddof=1).round(3))
# print("Max |z| :", np.abs(z).max().round(3))

# BAYESIAN INFERENCE
def fit_gamma(X,draws=4000,tune=2000,chains=4,prior_a=3,prior_theta=2,prior_sd=0.5,seed=42):
    X = np.asarray(X)
    groups = X.shape[0]

    with pm.Model() as model:
        # hyperparams
        mu_log_a = pm.Normal("mu_log_a",mu=np.log(prior_a),sigma=prior_sd)
        sigma_log_a = pm.HalfNormal("sigma_log_a",sigma=prior_sd)
        mu_log_theta = pm.Normal("mu_log_theta",mu=np.log(prior_theta),sigma=prior_sd)
        sigma_log_theta = pm.HalfNormal("sigma_log_theta",sigma=prior_sd)

        # group params
        alpha = pm.LogNormal("alpha",mu=mu_log_a,sigma=sigma_log_a,shape=groups)
        theta = pm.LogNormal("theta",mu=mu_log_theta,sigma=sigma_log_theta,shape=groups)

        for g in range(groups):
            pm.Gamma(f"y_{g}",alpha=alpha[g],beta=1/theta[g],observed=X[g])
        
        trace = pm.sample(draws=draws,tune=tune,chains=chains,random_seed=seed,target_accept=0.95,return_inferencedata=True)
    return trace

def eval_gamma(trace,hdi=0.9):
    post=trace.posterior
    keys = ["mu_log_a", "sigma_log_a", "mu_log_theta", "sigma_log_theta", "alpha", "theta"]
    return az.summary(post[keys],hdi_prob=hdi)

def bayesbag_gamma(X, B=50, draws=4000, tune=2000, chains=4,
                   prior_a=3, prior_theta=2, prior_sd=0.5, hdi=0.9,
                   show_progress=True, mfactor=1, show_health=True):
    rng = np.random.default_rng(42)
    X = np.asarray(X)
    groups, per_group = X.shape

    bagged = []
    iterator = range(B)
    if show_progress:
        iterator = tqdm(iterator, desc="BayesBag bootstraps", unit="fit")

    for b in iterator:
        Xb = np.empty_like(X)
        for g in range(groups):
            m=mfactor*per_group
            idx = rng.integers(0, per_group, size=m)
            # pad = rng.integers(0, m, size=per_group - m)
            # Xb[g] = np.concatenate([X[g, idx], X[g, idx][pad]])
            Xb[g] = X[g, idx]

        trace_b = fit_gamma(
            Xb, draws=draws, tune=tune, chains=chains,
            prior_a=prior_a, prior_theta=prior_theta, prior_sd=prior_sd, seed=rng.integers(1_000_000_000)
        )
        bagged.append(trace_b)

        if show_progress and show_health:
            try:
                div = int(trace_b.sample_stats["diverging"].values.sum())
                iterator.set_postfix(divergences=div)
            except Exception:
                pass

    trace_bagged = az.concat(bagged, dim="draw")
    summary = eval_gamma(trace_bagged, hdi=hdi)
    return {"trace": trace_bagged, "summary": summary}

# NORMAL MODEL
def fit_normal(
    X, draws=4000, tune=200, chains=4, seed=None, progressbar=False,
    prior_mu_loc=6.0, prior_mu_scale=2.0,
    prior_tau_scale=2.0,
    prior_sigma_median=3.5, prior_sigma_log_sd=0.5,
):
    X = np.asarray(X); G = X.shape[0]

    with pm.Model(coords={"group": np.arange(G)}) as m:
        # hyperpriors
        mu_mu = pm.Normal("mu_mu", mu=prior_mu_loc, sigma=prior_mu_scale)
        tau_mu = pm.HalfNormal("tau_mu", sigma=prior_tau_scale)

        mu_log_sigma    = pm.Normal("mu_log_sigma",    mu=np.log(prior_sigma_median), sigma=prior_sigma_log_sd)
        sigma_log_sigma = pm.HalfNormal("sigma_log_sigma", sigma=prior_sigma_log_sd)

        # group-level params
        mu    = pm.Normal("mu",    mu=mu_mu,    sigma=tau_mu,          dims="group")
        sigma = pm.LogNormal("sigma", mu=mu_log_sigma, sigma=sigma_log_sigma, dims="group")

        # likelihood
        for g in range(G):
            pm.Normal(f"y_{g}", mu=mu[g], sigma=sigma[g], observed=X[g])

        idata = pm.sample(
            draws=draws, tune=tune, chains=chains,
            random_seed=seed, target_accept=0.9,
            progressbar=progressbar, return_inferencedata=True
        )
    return idata

def eval_normal(idata, hdi=0.90):
    post = idata.posterior
    keys = ["mu_mu","tau_mu","mu_log_sigma","sigma_log_sigma","mu","sigma"]
    return az.summary(post[keys], hdi_prob=hdi)

def bayesbag_normal(
    X, B=50, draws=400, tune=2000, chains=4, seed=0,
    show_progress=True, show_health=True, m_frac=1.0, **fit_kwargs
):
    rng = np.random.default_rng(seed)
    X = np.asarray(X); G, n = X.shape

    bagged = []
    iterator = range(B)
    if show_progress:
        iterator = tqdm(iterator, desc="BayesBag (Normal) bootstraps", unit="fit")

    for b in iterator:
        Xb = np.empty_like(X)
        m = max(1, int(round(m_frac * n)))
        for g in range(G):
            # sample m with replacement from the group, then pad back to length n
            idx_m = rng.integers(0, n, size=m)
            if m < n:
                pad = rng.integers(0, m, size=n - m)
                idx_full = np.concatenate([idx_m, idx_m[pad]])
            else:
                # ordinary bootstrap of length n
                idx_full = rng.integers(0, n, size=n)
            Xb[g] = X[g, idx_full]

        id_b = fit_normal(
            Xb, draws=draws, tune=tune, chains=chains,
            seed=rng.integers(1_000_000_000), progressbar=False, **fit_kwargs
        )
        bagged.append(id_b)

        if show_progress and show_health:
            try:
                div = int(id_b.sample_stats["diverging"].values.sum())
                iterator.set_postfix(divergences=div)
            except Exception:
                pass

    id_bag = az.concat(bagged, dim="draw")  # mixture along draws
    return {"trace": id_bag, "summary": eval_normal(id_bag)}


# CONTAM
def contaminate_scale_inflate(X,a,theta,groups=(0,),eps=0.1,scale_mult=5,seed=0):
    rng = np.random.default_rng(seed)
    Xc = X.copy()
    G,n = X.shape
    for g in groups:
        m = rng.random(n) < eps
        Xc[g,m] = gamma.rvs(a=a[g],scale=theta[g]*scale_mult,size=m.sum(),random_state=rng)
    return Xc

# SAVING
def save_bundle(outdir,
                X_clean, X_contam,
                trace_std_clean, trace_bag_clean,
                trace_std_contam, trace_bag_contam,
                trace_std_norm, trace_bag_norm,
                true_alpha, true_theta, contam_idx,
                meta=None):
    os.makedirs(outdir, exist_ok=True)

    # arrays
    np.savez(os.path.join(outdir, "data_arrays.npz"),
             X_clean=X_clean, X_contam=X_contam,
             true_alpha=true_alpha, true_theta=true_theta,
             contam_idx=np.array(contam_idx, dtype=bool))

    # traces
    trace_std_clean.to_netcdf(os.path.join(outdir, "trace_std_clean.nc"))
    trace_bag_clean.to_netcdf(os.path.join(outdir, "trace_bag_clean.nc"))
    trace_std_contam.to_netcdf(os.path.join(outdir, "trace_std_contam.nc"))
    trace_bag_contam.to_netcdf(os.path.join(outdir, "trace_bag_contam.nc"))
    trace_std_norm.to_netcdf(os.path.join(outdir, "trace_std_norm.nc"))
    trace_bag_norm.to_netcdf(os.path.join(outdir, "trace_bag_norm.nc"))

    # metadata
    meta = {} if meta is None else dict(meta)
    meta.setdefault("created_at", datetime.now().isoformat(timespec="seconds"))
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

def default_run_dir(root="runs"):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(root, ts)
    return outdir

# EVAL
def flatten_draws(trace, var):
    da = trace.posterior[var]            
    da = da.stack(sample=("chain", "draw"))
    other = [d for d in da.dims if d != "sample"]
    if len(other) == 0:
        return da.values[:, None]
    if len(other) > 1:
        da = da.stack(group=other)
        return da.transpose("sample", "group").values
    return da.transpose("sample", other[0]).values

def param_recovery(trace,true_alpha,true_theta,hdi_prob=0.9):
    alpha_draws = flatten_draws(trace,"alpha")
    theta_draws = flatten_draws(trace,"theta")
    alpha_mean = alpha_draws.mean(axis=0)
    theta_mean = theta_draws.mean(axis=0)

    alpha_hdi = az.hdi(alpha_draws,hdi_prob=hdi_prob)
    theta_hdi = az.hdi(theta_draws,hdi_prob=hdi_prob)

    alpha_cover = np.mean((true_alpha >= alpha_hdi[:,0]) & (true_alpha <= alpha_hdi[:,1]))
    theta_cover = np.mean((true_theta >= theta_hdi[:,0]) & (true_theta <= theta_hdi[:,1]))
    alpha_width = np.mean(alpha_hdi[:,1] - alpha_hdi[:,0])
    theta_width = np.mean(theta_hdi[:,1] - theta_hdi[:,0])

    alpha_rmse = np.sqrt(np.mean((alpha_mean - true_alpha)**2))
    theta_rmse = np.sqrt(np.mean((theta_mean - true_theta)**2))

    return {
        "alpha_rmse":alpha_rmse,
        "theta_rmse":theta_rmse,
        "alpha_cover":alpha_cover,
        "theta_cover":theta_cover,
        "alpha_hdi_width":alpha_width,
        "theta_hdi_width":theta_width,
        "alpha_mean":alpha_mean,
        "theta_mean":theta_mean,
        "alpha_hdi":alpha_hdi,
        "theta_hdi":theta_hdi,
    }

def predictive_density(trace,X,samples=2000,rng=None):
    alpha = flatten_draws(trace,"alpha")
    theta = flatten_draws(trace,"theta")
    S,G=alpha.shape
    if samples and S > samples:
        rng = np.random.default_rng(None if rng is None else rng)
        idx = rng.choice(S,samples,replace=False)
        alpha,theta = alpha[idx],theta[idx]
        S=samples
    total = 0
    for g in range(G):
        y = X[g]
        logp = gamma.logpdf(y[:,None],a=alpha[:,g],scale=theta[:,g])
        total += (logsumexp(logp,axis=1)-np.log(S)).sum()
    return total / X.size

def predictive_density_normal(trace, X, samples=2000, rng=None):
    mu = flatten_draws(trace, "mu")      # (S, G)
    sg = flatten_draws(trace, "sigma")   # (S, G)
    S, G = mu.shape
    if samples and S > samples:
        rng = np.random.default_rng(None if rng is None else rng)
        idx = rng.choice(S, samples, replace=False)
        mu, sg = mu[idx], sg[idx]
        S = samples

    total = 0.0
    X = np.asarray(X)
    for g in range(G):
        y = X[g]
        logp = sp_norm.logpdf(y[:, None], loc=mu[:, g], scale=sg[:, g])  # (n, S)
        total += (logsumexp(logp, axis=1) - np.log(S)).sum()
    return total / X.size

def posterior_means(trace):
    alpha = flatten_draws(trace,"alpha").mean(axis=0)
    theta = flatten_draws(trace,"theta").mean(axis=0)
    return alpha,theta

def stability_delta(trace_clean,trace_contam,clean_idx,which="theta"):
    alpha_clean,theta_clean = posterior_means(trace_clean)
    alpha_contam,theta_contam = posterior_means(trace_contam)
    if which == "theta":
        return float(np.mean(np.abs(theta_contam[clean_idx]-theta_clean[clean_idx])))
    elif which == "alpha":
        return float(np.mean(np.abs(alpha_contam[clean_idx]-alpha_clean[clean_idx])))
    else:
        raise ValueError("hich must be 'theta' or 'alpha")
    
def hyper_stability(trace_clean, trace_contam):
    def get(idata, name):
        x = idata.posterior[name].stack(sample=("chain","draw")).values
        return float(x.mean())
    keys = ["mu_log_a","mu_log_theta","sigma_log_a","sigma_log_theta"]
    return {k: abs(get(trace_contam,k) - get(trace_clean,k)) for k in keys}

# MAIN
def main():
    # FITTING
    trace_std_clean = fit_gamma(X,draws=200,tune=100,chains=4)
    # print(eval_gamma(trace_std_clean))

    trace_bb_clean = bayesbag_gamma(X,B=50,draws=200,tune=100,chains=4)
    # print(trace_bb_clean["summary"])
    trace_bag_clean = trace_bb_clean["trace"]

    # CONTAMINATION
    Xc = contaminate_scale_inflate(X,a,theta,groups=(1,8),eps=0.1,scale_mult=6,seed=42)

    trace_std_contam = fit_gamma(Xc,draws=200,tune=100,chains=4)
    trace_bb_contam = bayesbag_gamma(Xc,B=50,draws=200,tune=100,chains=4)
    trace_bag_contam = trace_bb_contam["trace"]

    # NORMAL
    trace_std_norm = fit_normal(X, draws=200, tune=100, chains=4, seed=42)
    norm_bag = bayesbag_normal(X, B=50, draws=200, tune=100, chains=4, seed=42, m_frac=1.0)
    trace_bag_norm = norm_bag["trace"]

    # SAVING
    # Example meta you might want to record:
    meta = dict(
        groups=int(X.shape[0]),
        per_group=int(X.shape[1]),
        eps=0.1, scale_mult=6, contam_groups=[1, 8],
        draws=200, tune=100, chains=4, B=50, m_frac=1.0
    )

    outdir = default_run_dir()  # e.g., runs/20250824_153210
    save_bundle(outdir,
            X_clean=X, X_contam=Xc,
            trace_std_clean=trace_std_clean,
            trace_bag_clean=trace_bag_clean,
            trace_std_contam=trace_std_contam,
            trace_bag_contam=trace_bag_contam,
            trace_std_norm=trace_std_norm,
            trace_bag_norm=trace_bag_norm,
            true_alpha=a, true_theta=theta,
            contam_idx=[False, True, False, False, False, False, False, False, True, False],
            meta=meta)
    
    print("Saved bundle to:", outdir)

if __name__ == "__main__":
    main()
    
