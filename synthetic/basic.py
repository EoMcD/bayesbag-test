import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma,norm
import pymc as pm
import arviz as az
from tqdm.auto import tqdm

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
def fit_gamma(X,draws=4000,tune=2000,chains=4,prior_a=3,prior_theta=2,prior_sd=0.5):
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
        
        trace = pm.sample(draws=draws,tune=tune,chains=chains,random_seed=42,target_accept=0.95,progressbar=True,return_inferencedata=True)
    return trace

def eval_gamma(trace,hdi=0.9):
    post=trace.posterior
    keys = ["mu_log_a", "sigma_log_a", "mu_log_theta", "sigma_log_theta", "alpha", "theta"]
    return az.summary(post[keys],hdi_prob=hdi)

def bayesbag_gamma(X, B=50, draws=4000, tune=2000, chains=4,
                   prior_a=3, prior_theta=2, prior_sd=0.5, hdi=0.9,
                   show_progress=True, show_health=True):
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
            idx = rng.integers(0, per_group, size=per_group)
            Xb[g] = X[g, idx]

        trace_b = fit_gamma(
            Xb, draws=draws, tune=tune, chains=chains,
            prior_a=prior_a, prior_theta=prior_theta, prior_sd=prior_sd, progressbar=False
        )
        bagged.append(trace_b)

        if show_progress and show_health:
            try:
                div = int(trace_b.sample_stats["diverging"].values.sum())
                iterator.set_postfix(divergences=div)
            except Exception:
                pass

    trace_bagged = az.concat(bagged, dim="chain")
    summary = eval_gamma(trace_bagged, hdi=hdi)
    return {"trace": trace_bagged, "summary": summary}

# MAIN
def main():
    trace_std = fit_gamma(X,draws=2000,tune=1000,chains=4)
    print(eval_gamma(trace_std))

    trace_bb = bayesbag_gamma(X,B=50,draws=2000,tune=1000,chains=4)
    print(trace_bb["summary"])

if __name__ == "__main__":
    main()
