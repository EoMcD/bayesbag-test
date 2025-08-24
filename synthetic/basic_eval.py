import os, json
import numpy as np
import arviz as az
from scipy.stats import gamma as sp_gamma
from scipy.stats import norm as sp_norm
from scipy.special import logsumexp
import xarray as xr

# IMPORT
def load_bundle(outdir):
    arrays = np.load(os.path.join(outdir, "data_arrays.npz"), allow_pickle=True)
    X_clean   = arrays["X_clean"]
    X_contam  = arrays["X_contam"]
    true_a    = arrays["true_alpha"]
    true_t    = arrays["true_theta"]
    contam_idx= arrays["contam_idx"].astype(bool)

    trace_std_clean   = az.from_netcdf(os.path.join(outdir, "trace_std_clean.nc"))
    trace_bag_clean   = az.from_netcdf(os.path.join(outdir, "trace_bag_clean.nc"))
    trace_std_contam  = az.from_netcdf(os.path.join(outdir, "trace_std_contam.nc"))
    trace_bag_contam  = az.from_netcdf(os.path.join(outdir, "trace_bag_contam.nc"))

    path_std_norm = os.path.join(outdir, "trace_std_norm.nc")
    path_bag_norm = os.path.join(outdir, "trace_bag_norm.nc")
    trace_std_norm = az.from_netcdf(path_std_norm) if os.path.exists(path_std_norm) else None
    trace_bag_norm = az.from_netcdf(path_bag_norm) if os.path.exists(path_bag_norm) else None

    meta_path = os.path.join(outdir, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

    return (X_clean, X_contam,
            trace_std_clean, trace_bag_clean, trace_std_contam, trace_bag_contam,
            trace_std_norm, trace_bag_norm,
            true_a, true_t, contam_idx, meta)

# EVAL
def flatten_draws(trace, var):
    da = trace.posterior[var]
    da = da.stack(sample=("chain","draw"))
    other = [d for d in da.dims if d != "sample"]
    if len(other) == 0:
        return da.values[:, None]
    if len(other) > 1:
        da = da.stack(group=other)
        return da.transpose("sample","group").values
    return da.transpose("sample", other[0]).values

def hdi_min_width(draws_2d, prob=0.90):
    """
    Robust HDI for each group via sliding-window min-width interval.
    draws_2d: array of shape (S, G) with S posterior draws.
    Returns: (G, 2) array of [lower, upper] bounds.
    """
    S, G = draws_2d.shape
    k = max(1, int(np.ceil(prob * S)))
    out = np.empty((G, 2), dtype=float)
    for g in range(G):
        s = np.sort(draws_2d[:, g])
        if k >= S:
            out[g] = [s[0], s[-1]]
            continue
        widths = s[k-1:] - s[:S-k+1]
        j = int(np.argmin(widths))
        out[g] = [s[j], s[j+k-1]]
    return out

def mismatch_index_theta(trace_std, trace_bag, X_like):
    tS = flatten_draws(trace_std, "theta")  # (S,G)
    tB = flatten_draws(trace_bag, "theta")
    stdS = tS.std(axis=0, ddof=1)
    stdB = tB.std(axis=0, ddof=1)
    N = X_like.size
    NvN = N * float(np.sum(stdS**2))
    MvsM = N * float(np.sum(stdB**2))
    return 1 - ((2 * NvN) / MvsM) if MvsM > NvN else np.nan

def posterior_rank(draws_2d, truth_1d):
    return (draws_2d <= truth_1d).mean(axis=0)

def predictive_density(trace, X, samples=2000, rng=None):
    a = flatten_draws(trace, "alpha"); t = flatten_draws(trace, "theta")
    S, G = a.shape
    if samples and S > samples:
        rng = np.random.default_rng(None if rng is None else rng)
        idx = rng.choice(S, samples, replace=False)
        a, t = a[idx], t[idx]; S = samples
    total = 0.0
    for g in range(G):
        y = X[g]
        logp = sp_gamma.logpdf(y[:, None], a=a[:, g], scale=t[:, g])
        total += (logsumexp(logp, axis=1) - np.log(S)).sum()
    return total / X.size

def predictive_density_normal(trace, X, samples=2000, rng=None):
    mu = flatten_draws(trace, "mu")      # (S,G)
    sg = flatten_draws(trace, "sigma")   # (S,G)
    S, G = mu.shape
    if samples and S > samples:
        rng = np.random.default_rng(None if rng is None else rng)
        idx = rng.choice(S, samples, replace=False)
        mu, sg = mu[idx], sg[idx]; S = samples
    total = 0.0
    for g in range(G):
        y = X[g]
        logp = sp_norm.logpdf(y[:, None], loc=mu[:, g], scale=sg[:, g])  # (n, S)
        total += (logsumexp(logp, axis=1) - np.log(S)).sum()
    return total / X.size

def param_recovery_gamma(trace, true_alpha, true_theta, hdi_prob=0.90):
    """
    Per-group recovery for the hierarchical Gamma model.
    Mirrors param_recovery_normal's structure and keys (α ↔ alpha, θ ↔ theta).
    Returns a dict with RMSE, coverage, HDI width, and posterior ranks.
    """
    # posterior draws (S, G)
    a_draws = flatten_draws(trace, "alpha")
    t_draws = flatten_draws(trace, "theta")

    # posterior means
    a_mean = a_draws.mean(axis=0)
    t_mean = t_draws.mean(axis=0)

    # robust HDIs (minimum-width)
    a_hdi = hdi_min_width(a_draws, prob=hdi_prob)  # (G,2)
    t_hdi = hdi_min_width(t_draws, prob=hdi_prob)

    # coverage & widths
    a_cover = float(np.mean((true_alpha >= a_hdi[:,0]) & (true_alpha <= a_hdi[:,1])))
    t_cover = float(np.mean((true_theta >= t_hdi[:,0]) & (true_theta <= t_hdi[:,1])))
    a_width = float(np.mean(a_hdi[:,1] - a_hdi[:,0]))
    t_width = float(np.mean(t_hdi[:,1] - t_hdi[:,0]))

    # RMSE
    a_rmse = float(np.sqrt(np.mean((a_mean - true_alpha) ** 2)))
    t_rmse = float(np.sqrt(np.mean((t_mean - true_theta) ** 2)))

    # posterior ranks (percentiles) of the true values
    a_rank = (a_draws <= true_alpha).mean(axis=0)
    t_rank = (t_draws <= true_theta).mean(axis=0)

    return dict(
        alpha_rmse=a_rmse, theta_rmse=t_rmse,
        alpha_cover=a_cover, theta_cover=t_cover,
        alpha_hdi_width=a_width, theta_hdi_width=t_width,
        alpha_rank=a_rank, theta_rank=t_rank,
        alpha_mean=a_mean, theta_mean=t_mean,
        alpha_hdi=a_hdi, theta_hdi=t_hdi,
    )

def param_recovery_normal(trace, true_mean, true_sd, hdi_prob=0.90):
    mu  = flatten_draws(trace, "mu")      # (S,G)
    sig = flatten_draws(trace, "sigma")   # (S,G)
    mu_mean  = mu.mean(axis=0)
    sig_mean = sig.mean(axis=0)

    # robust HDIs (same algorithm you use for Gamma)
    mu_hdi  = hdi_min_width(mu,  prob=hdi_prob)
    sg_hdi  = hdi_min_width(sig, prob=hdi_prob)

    # coverage wrt Gamma truths: μ_true = αθ, σ_true = sqrt(α) θ
    mu_cover = np.mean((true_mean >= mu_hdi[:,0]) & (true_mean <= mu_hdi[:,1]))
    sg_cover = np.mean((true_sd   >= sg_hdi[:,0]) & (true_sd   <= sg_hdi[:,1]))
    mu_width = np.mean(mu_hdi[:,1] - mu_hdi[:,0])
    sg_width = np.mean(sg_hdi[:,1] - sg_hdi[:,0])

    mu_rmse  = float(np.sqrt(np.mean((mu_mean  - true_mean)**2)))
    sg_rmse  = float(np.sqrt(np.mean((sig_mean - true_sd)**2)))

    # ranks (percentiles) of truth under the posterior
    mu_rank  = (mu  <= true_mean).mean(axis=0)
    sg_rank  = (sig <= true_sd).mean(axis=0)

    return dict(mu_rmse=mu_rmse, sg_rmse=sg_rmse,
                mu_cover=mu_cover, sg_cover=sg_cover,
                mu_width=mu_width, sg_width=sg_width,
                mu_rank=mu_rank, sg_rank=sg_rank)


# ---------- evaluation ----------
# def evaluate_methods(
#     X_clean, X_contam,
#     trace_std_clean, trace_bag_clean,
#     trace_std_cont, trace_bag_cont,
#     true_alpha, true_theta,
#     contam_idx, hdi_prob=0.90, print_table=True
# ):
#     contam_idx = np.array(contam_idx, dtype=bool)
#     clean_idx = ~contam_idx

#     # Predictive
#     lp_std_clean = predictive_density(trace_std_clean, X_clean)
#     lp_std_cont  = predictive_density(trace_std_cont,  X_contam)
#     lp_bag_clean = predictive_density(trace_bag_clean, X_clean)
#     lp_bag_cont  = predictive_density(trace_bag_cont,  X_contam)
#     print("\n=== Predictive accuracy (avg log pred density; higher is better) ===")
#     print(f"Standard: clean={lp_std_clean:.4f}, contam={lp_std_cont:.4f}, Δ={lp_std_cont - lp_std_clean:+.4f}")
#     print(f"BayesBag: clean={lp_bag_clean:.4f}, contam={lp_bag_cont:.4f}, Δ={lp_bag_cont - lp_bag_clean:+.4f}")

#     # Stability on unaffected groups
#     def post_means(trace):
#         return (flatten_draws(trace,"alpha").mean(axis=0),
#                 flatten_draws(trace,"theta").mean(axis=0))
#     a_sc, t_sc = post_means(trace_std_clean)
#     a_bc, t_bc = post_means(trace_bag_clean)
#     a_sx, t_sx = post_means(trace_std_cont)
#     a_bx, t_bx = post_means(trace_bag_cont)
#     stab_std = float(np.mean(np.abs(t_sx[clean_idx] - t_sc[clean_idx])))
#     stab_bag = float(np.mean(np.abs(t_bx[clean_idx] - t_bc[clean_idx])))
#     print("\n=== Stability on unaffected groups (mean |Δ θ_g|) ===")
#     print(f"Standard: {stab_std:.4f}   BayesBag: {stab_bag:.4f}")

#     # Contaminated groups: coverage & width (θ)
#     tS = flatten_draws(trace_std_cont, "theta")
#     tB = flatten_draws(trace_bag_cont, "theta")
#     hS = hdi_min_width(tS, hdi_prob); hB = hdi_min_width(tB, hdi_prob)
#     cover_std = np.mean((true_theta[contam_idx] >= hS[contam_idx,0]) &
#                         (true_theta[contam_idx] <= hS[contam_idx,1]))
#     cover_bag = np.mean((true_theta[contam_idx] >= hB[contam_idx,0]) &
#                         (true_theta[contam_idx] <= hB[contam_idx,1]))
#     wS = float(np.mean(hS[contam_idx,1] - hS[contam_idx,0])) if contam_idx.any() else np.nan
#     wB = float(np.mean(hB[contam_idx,1] - hB[contam_idx,0])) if contam_idx.any() else np.nan
#     print("\n=== Contaminated groups: coverage & width (θ) ===")
#     print(f"Cover{int(hdi_prob*100)} θ: std={cover_std:.2f}, bag={cover_bag:.2f}")
#     if np.isfinite(wS) and wS>0:
#         print(f"HDI width ratio (bag/std): {wB/wS:.2f}")

#     # Posterior ranks for θ
#     rS = posterior_rank(tS, true_theta); rB = posterior_rank(tB, true_theta)
#     print("\n=== Posterior rank of θ_true on contaminated groups (0=low, 0.5=center, 1=high) ===")
#     for g in np.where(contam_idx)[0]:
#         print(f"g={g}: rank_std={rS[g]:.3f}, rank_bag={rB[g]:.3f}")

#     mi = mismatch_index_theta(trace_std_cont, trace_bag_cont, X_contam)
#     print(f"\n=== Mismatch index (θ, contaminated fit) ===")
#     print(f"Mismatch index: {mi:.4f}" if np.isfinite(mi) else "Mismatch index: nan")

#     # Optional per-group table
#     if print_table and contam_idx.any():
#         tS_mean = tS.mean(axis=0); tB_mean = tB.mean(axis=0)
#         print("\n=== Per-group (contaminated) θ summaries ===")
#         for g in np.where(contam_idx)[0]:
#             covS = int(hS[g,0] <= true_theta[g] <= hS[g,1])
#             covB = int(hB[g,0] <= true_theta[g] <= hB[g,1])
#             print(f"g={g} | θ_true={true_theta[g]:.3f} | "
#                   f"std: {tS_mean[g]:.3f} [{hS[g,0]:.3f},{hS[g,1]:.3f}] cov={covS} | "
#                   f"bag: {tB_mean[g]:.3f} [{hB[g,0]:.3f},{hB[g,1]:.3f}] cov={covB}")

def evaluate_methods(
    X_clean, X_contam,
    trace_std_clean, trace_bag_clean,
    trace_std_cont, trace_bag_cont,
    true_alpha, true_theta,
    contam_idx, hdi_prob=0.90, print_table=True,
    # optional Normal traces on clean data:
    trace_std_norm=None, trace_bag_norm=None
):
    contam_idx = np.array(contam_idx, dtype=bool)
    clean_idx  = ~contam_idx

    # GAMMA
    lp_std_clean = predictive_density(trace_std_clean, X_clean)
    lp_std_cont  = predictive_density(trace_std_cont,  X_contam)
    lp_bag_clean = predictive_density(trace_bag_clean, X_clean)
    lp_bag_cont  = predictive_density(trace_bag_cont,  X_contam)
    print("\n=== [Gamma] Predictive accuracy (avg log pred density; higher is better) ===")
    print(f"Standard: clean={lp_std_clean:.4f}, contam={lp_std_cont:.4f}, Δ={lp_std_cont - lp_std_clean:+.4f}")
    print(f"BayesBag: clean={lp_bag_clean:.4f}, contam={lp_bag_cont:.4f}, Δ={lp_bag_cont - lp_bag_clean:+.4f}")

    # [Gamma] Parameter recovery on CLEAN data
    met_std_clean = param_recovery_gamma(trace_std_clean, true_alpha, true_theta, hdi_prob)
    met_bag_clean = param_recovery_gamma(trace_bag_clean, true_alpha, true_theta, hdi_prob)
    print("\n=== [Gamma, clean data] Parameter recovery ===")
    print(f"RMSE α: std={met_std_clean['alpha_rmse']:.3f}, bag={met_bag_clean['alpha_rmse']:.3f}")
    print(f"RMSE θ: std={met_std_clean['theta_rmse']:.3f}, bag={met_bag_clean['theta_rmse']:.3f}")
    print(f"Cover{int(hdi_prob*100)} α: std={met_std_clean['alpha_cover']:.2f}, bag={met_bag_clean['alpha_cover']:.2f}")
    print(f"Cover{int(hdi_prob*100)} θ: std={met_std_clean['theta_cover']:.2f}, bag={met_bag_clean['theta_cover']:.2f}")
    if met_std_clean['alpha_hdi_width'] > 0:
        print(f"HDI width α (bag/std): {met_bag_clean['alpha_hdi_width']/met_std_clean['alpha_hdi_width']:.2f}")
    if met_std_clean['theta_hdi_width'] > 0:
        print(f"HDI width θ (bag/std): {met_bag_clean['theta_hdi_width']/met_std_clean['theta_hdi_width']:.2f}")

    # Parameter recovery on contam
    met_std_cont = param_recovery_gamma(trace_std_cont, true_alpha, true_theta, hdi_prob)
    met_bag_cont = param_recovery_gamma(trace_bag_cont, true_alpha, true_theta, hdi_prob)
    print("\n=== [Gamma, contaminated data] Parameter recovery ===")
    print(f"RMSE α: std={met_std_cont['alpha_rmse']:.3f}, bag={met_bag_cont['alpha_rmse']:.3f}")
    print(f"RMSE θ: std={met_std_cont['theta_rmse']:.3f}, bag={met_bag_cont['theta_rmse']:.3f}")
    print(f"Cover{int(hdi_prob*100)} α: std={met_std_cont['alpha_cover']:.2f}, bag={met_bag_cont['alpha_cover']:.2f}")
    print(f"Cover{int(hdi_prob*100)} θ: std={met_std_cont['theta_cover']:.2f}, bag={met_bag_cont['theta_cover']:.2f}")
    if met_std_cont['alpha_hdi_width'] > 0:
        print(f"HDI width α (bag/std): {met_bag_cont['alpha_hdi_width']/met_std_cont['alpha_hdi_width']:.2f}")
    if met_std_cont['theta_hdi_width'] > 0:
        print(f"HDI width θ (bag/std): {met_bag_cont['theta_hdi_width']/met_std_cont['theta_hdi_width']:.2f}")

    # Stability on unaffected groups (θ)
    def post_means_gamma(trace):
        return (flatten_draws(trace,"alpha").mean(axis=0),
                flatten_draws(trace,"theta").mean(axis=0))
    a_sc, t_sc = post_means_gamma(trace_std_clean)
    a_bc, t_bc = post_means_gamma(trace_bag_clean)
    a_sx, t_sx = post_means_gamma(trace_std_cont)
    a_bx, t_bx = post_means_gamma(trace_bag_cont)
    stab_std = float(np.mean(np.abs(t_sx[clean_idx] - t_sc[clean_idx])))
    stab_bag = float(np.mean(np.abs(t_bx[clean_idx] - t_bc[clean_idx])))
    print("\n=== [Gamma] Stability on unaffected groups (mean |Δ θ_g|) ===")
    print(f"Standard: {stab_std:.4f}   BayesBag: {stab_bag:.4f}")

    # Contaminated groups: coverage & width (θ)
    tS = flatten_draws(trace_std_cont, "theta")
    tB = flatten_draws(trace_bag_cont, "theta")
    hS = hdi_min_width(tS, prob=hdi_prob); hB = hdi_min_width(tB, prob=hdi_prob)
    cover_std = np.mean((true_theta[contam_idx] >= hS[contam_idx,0]) &
                        (true_theta[contam_idx] <= hS[contam_idx,1]))
    cover_bag = np.mean((true_theta[contam_idx] >= hB[contam_idx,0]) &
                        (true_theta[contam_idx] <= hB[contam_idx,1]))
    wS = float(np.mean(hS[contam_idx,1] - hS[contam_idx,0])) if contam_idx.any() else np.nan
    wB = float(np.mean(hB[contam_idx,1] - hB[contam_idx,0])) if contam_idx.any() else np.nan
    print("\n=== [Gamma] Contaminated groups: coverage & width (θ) ===")
    print(f"Cover{int(hdi_prob*100)} θ: std={cover_std:.2f}, bag={cover_bag:.2f}")
    if np.isfinite(wS) and wS>0:
        print(f"HDI width ratio (bag/std): {wB/wS:.2f}")

    # Posterior ranks for θ (contam groups)
    rS = posterior_rank(tS, true_theta); rB = posterior_rank(tB, true_theta)
    print("\n=== [Gamma] Posterior rank of θ_true on contaminated groups (0=low, 0.5=center, 1=high) ===")
    for g in np.where(contam_idx)[0]:
        print(f"g={g}: rank_std={rS[g]:.3f}, rank_bag={rB[g]:.3f}")

    # Mismatch index (θ) on contaminated fit
    mi = mismatch_index_theta(trace_std_cont, trace_bag_cont, X_contam)
    print(f"\n=== [Gamma] Mismatch index (θ, contaminated fit) ===")
    print(f"Mismatch index: {mi:.4f}" if np.isfinite(mi) else "Mismatch index: nan")

    # Optional per-group table
    if print_table and contam_idx.any():
        tS_mean = tS.mean(axis=0); tB_mean = tB.mean(axis=0)
        print("\n=== [Gamma] Per-group (contaminated) θ summaries ===")
        for g in np.where(contam_idx)[0]:
            covS = int(hS[g,0] <= true_theta[g] <= hS[g,1])
            covB = int(hB[g,0] <= true_theta[g] <= hB[g,1])
            print(f"g={g} | θ_true={true_theta[g]:.3f} | "
                  f"std: {tS_mean[g]:.3f} [{hS[g,0]:.3f},{hS[g,1]:.3f}] cov={covS} | "
                  f"bag: {tB_mean[g]:.3f} [{hB[g,0]:.3f},{hB[g,1]:.3f}] cov={covB}")

    # NORMAL
    if (trace_std_norm is not None) and (trace_bag_norm is not None):
        true_mean = true_alpha * true_theta          # Gamma mean
        true_sd   = np.sqrt(true_alpha) * true_theta # Gamma SD

        lpd_stdN = predictive_density_normal(trace_std_norm, X_clean)
        lpd_bagN = predictive_density_normal(trace_bag_norm, X_clean)
        print("\n=== [Normal, clean data] Predictive accuracy (avg log pred density) ===")
        print(f"Standard: {lpd_stdN:.4f}   BayesBag: {lpd_bagN:.4f}   Δ={lpd_bagN - lpd_stdN:+.4f}")

        m_stdN = param_recovery_normal(trace_std_norm, true_mean, true_sd, hdi_prob)
        m_bagN = param_recovery_normal(trace_bag_norm, true_mean, true_sd, hdi_prob)

        print("\n=== [Normal, clean data] Parameter recovery vs Gamma truth ===")
        print(f"RMSE μ: std={m_stdN['mu_rmse']:.3f}, bag={m_bagN['mu_rmse']:.3f}")
        print(f"RMSE σ: std={m_stdN['sg_rmse']:.3f}, bag={m_bagN['sg_rmse']:.3f}")
        print(f"Cover{int(hdi_prob*100)} μ: std={m_stdN['mu_cover']:.2f}, bag={m_bagN['mu_cover']:.2f}")
        print(f"Cover{int(hdi_prob*100)} σ: std={m_stdN['sg_cover']:.2f}, bag={m_bagN['sg_cover']:.2f}")
        if m_stdN['mu_width'] > 0:
            print(f"HDI width ratio μ (bag/std): {m_bagN['mu_width']/m_stdN['mu_width']:.2f}")
        if m_stdN['sg_width'] > 0:
            print(f"HDI width ratio σ (bag/std): {m_bagN['sg_width']/m_stdN['sg_width']:.2f}")

        # (optional) posterior rank summaries for μ and σ
        print("\n=== [Normal, clean data] Posterior rank of Gamma truths ===")
        print(f"μ ranks (mean ± sd): std={np.mean(m_stdN['mu_rank']):.2f}±{np.std(m_stdN['mu_rank']):.2f} | "
              f"bag={np.mean(m_bagN['mu_rank']):.2f}±{np.std(m_bagN['mu_rank']):.2f}")
        print(f"σ ranks (mean ± sd): std={np.mean(m_stdN['sg_rank']):.2f}±{np.std(m_stdN['sg_rank']):.2f} | "
              f"bag={np.mean(m_bagN['sg_rank']):.2f}±{np.std(m_bagN['sg_rank']):.2f}")

        # Mismatch index for Normal model (μ and σ)
        mi_mu = mismatch_index_var(trace_std_norm, trace_bag_norm, "mu", X_clean)
        mi_sg = mismatch_index_var(trace_std_norm, trace_bag_norm, "sigma", X_clean)
        print("\n=== [Normal, clean data] Mismatch index (μ, σ) ===")
        print(f"Mismatch μ: {mi_mu:.4f}" if np.isfinite(mi_mu) else "Mismatch μ: nan")
        print(f"Mismatch σ: {mi_sg:.4f}" if np.isfinite(mi_sg) else "Mismatch σ: nan")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <bundle_dir>")
        sys.exit(1)
    outdir = sys.argv[1]
    (X, Xc,trace_std_clean, trace_bag_clean, trace_std_cont, trace_bag_cont,trace_std_norm, trace_bag_norm,a, theta, contam_idx, meta) = load_bundle(outdir)
    evaluate_methods(X, Xc,trace_std_clean, trace_bag_clean,trace_std_cont, trace_bag_cont,a, theta, contam_idx,hdi_prob=0.90, print_table=True,
                     trace_std_norm=trace_std_norm, trace_bag_norm=trace_bag_norm
    )

