import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma,norm
from scipy.special import logsumexp
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

# EVAL

def flatten_draws(trace,var):
    x = trace.posterior[var]
    x = x.stack(sample=("chain","draw")).transpose("sample","group").values
    return x

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
        logp = gamma.logpdf(y[:,None],alpha=alpha[:,g],scale=theta[:,g])
        total += (logsumexp(logp,axis=1)-np.log(S)).sum()
    return total / X.size

# MAIN
def main():
    # FITTING
    trace_std = fit_gamma(X,draws=1000,tune=500,chains=4)
    print(eval_gamma(trace_std))

    trace_bb = bayesbag_gamma(X,B=50,draws=1000,tune=500,chains=4)
    print(trace_bb["summary"])
    trace_bag = trace_bb["trace"]

    # EVALUATION
    lp_std = predictive_density(trace_std,X)
    lp_bag = predictive_density(trace_bag,X)
    print(f"Avg log pred density – std: {lp_std:.4f}, bagged: {lp_bag:.4f}, Δ={lp_bag - lp_std:.4f}")

    m_std = param_recovery(trace_std,a,theta,hdi_prob=0.9)
    m_bag = param_recovery(trace_bag,a,theta,hdi_prob=0.9)
    print(f"RMSE α: std={m_std['alpha_rmse']:.3f}, bag={m_bag['alpha_rmse']:.3f}")
    print(f"RMSE θ: std={m_std['theta_rmse']:.3f}, bag={m_bag['theta_rmse']:.3f}")
    print(f"Cover90 α: std={m_std['alpha_cover']:.2f}, bag={m_bag['alpha_cover']:.2f}")
    print(f"Cover90 θ: std={m_std['theta_cover']:.2f}, bag={m_bag['theta_cover']:.2f}")
    print(f"HDI width α (bag/std): {m_bag['alpha_hdi_width']/m_std['alpha_hdi_width']:.2f}")
    print(f"HDI width θ (bag/std): {m_bag['theta_hdi_width']/m_std['theta_hdi_width']:.2f}")

if __name__ == "__main__":
    main()
