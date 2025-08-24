import os, json
import numpy as np
import arviz as az
from scipy.stats import gamma as sp_gamma
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
    meta_path = os.path.join(outdir, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    return (X_clean, X_contam, trace_std_clean, trace_bag_clean,
            trace_std_contam, trace_bag_contam, true_a, true_t, contam_idx, meta)

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

# ---------- evaluation ----------
def evaluate_methods(
    X_clean, X_contam,
    trace_std_clean, trace_bag_clean,
    trace_std_cont, trace_bag_cont,
    true_alpha, true_theta,
    contam_idx, hdi_prob=0.90, print_table=True
):
    contam_idx = np.array(contam_idx, dtype=bool)
    clean_idx = ~contam_idx

    # Predictive
    lp_std_clean = predictive_density(trace_std_clean, X_clean)
    lp_std_cont  = predictive_density(trace_std_cont,  X_contam)
    lp_bag_clean = predictive_density(trace_bag_clean, X_clean)
    lp_bag_cont  = predictive_density(trace_bag_cont,  X_contam)
    print("\n=== Predictive accuracy (avg log pred density; higher is better) ===")
    print(f"Standard: clean={lp_std_clean:.4f}, contam={lp_std_cont:.4f}, Δ={lp_std_cont - lp_std_clean:+.4f}")
    print(f"BayesBag: clean={lp_bag_clean:.4f}, contam={lp_bag_cont:.4f}, Δ={lp_bag_cont - lp_bag_clean:+.4f}")

    # Stability on unaffected groups
    def post_means(trace):
        return (flatten_draws(trace,"alpha").mean(axis=0),
                flatten_draws(trace,"theta").mean(axis=0))
    a_sc, t_sc = post_means(trace_std_clean)
    a_bc, t_bc = post_means(trace_bag_clean)
    a_sx, t_sx = post_means(trace_std_cont)
    a_bx, t_bx = post_means(trace_bag_cont)
    stab_std = float(np.mean(np.abs(t_sx[clean_idx] - t_sc[clean_idx])))
    stab_bag = float(np.mean(np.abs(t_bx[clean_idx] - t_bc[clean_idx])))
    print("\n=== Stability on unaffected groups (mean |Δ θ_g|) ===")
    print(f"Standard: {stab_std:.4f}   BayesBag: {stab_bag:.4f}")

    # Contaminated groups: coverage & width (θ)
    tS = flatten_draws(trace_std_cont, "theta")
    tB = flatten_draws(trace_bag_cont, "theta")
    hS = hdi_min_width(tS, hdi_prob); hB = hdi_min_width(tB, hdi_prob)
    cover_std = np.mean((true_theta[contam_idx] >= hS[contam_idx,0]) &
                        (true_theta[contam_idx] <= hS[contam_idx,1]))
    cover_bag = np.mean((true_theta[contam_idx] >= hB[contam_idx,0]) &
                        (true_theta[contam_idx] <= hB[contam_idx,1]))
    wS = float(np.mean(hS[contam_idx,1] - hS[contam_idx,0])) if contam_idx.any() else np.nan
    wB = float(np.mean(hB[contam_idx,1] - hB[contam_idx,0])) if contam_idx.any() else np.nan
    print("\n=== Contaminated groups: coverage & width (θ) ===")
    print(f"Cover{int(hdi_prob*100)} θ: std={cover_std:.2f}, bag={cover_bag:.2f}")
    if np.isfinite(wS) and wS>0:
        print(f"HDI width ratio (bag/std): {wB/wS:.2f}")

    # Posterior ranks for θ
    rS = posterior_rank(tS, true_theta); rB = posterior_rank(tB, true_theta)
    print("\n=== Posterior rank of θ_true on contaminated groups (0=low, 0.5=center, 1=high) ===")
    for g in np.where(contam_idx)[0]:
        print(f"g={g}: rank_std={rS[g]:.3f}, rank_bag={rB[g]:.3f}")

    mi = mismatch_index_theta(trace_std_cont, trace_bag_cont, X_contam)
    print(f"\n=== Mismatch index (θ, contaminated fit) ===")
    print(f"Mismatch index: {mi:.4f}" if np.isfinite(mi) else "Mismatch index: nan")

    # Optional per-group table
    if print_table and contam_idx.any():
        tS_mean = tS.mean(axis=0); tB_mean = tB.mean(axis=0)
        print("\n=== Per-group (contaminated) θ summaries ===")
        for g in np.where(contam_idx)[0]:
            covS = int(hS[g,0] <= true_theta[g] <= hS[g,1])
            covB = int(hB[g,0] <= true_theta[g] <= hB[g,1])
            print(f"g={g} | θ_true={true_theta[g]:.3f} | "
                  f"std: {tS_mean[g]:.3f} [{hS[g,0]:.3f},{hS[g,1]:.3f}] cov={covS} | "
                  f"bag: {tB_mean[g]:.3f} [{hB[g,0]:.3f},{hB[g,1]:.3f}] cov={covB}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <bundle_dir>")
        sys.exit(1)
    outdir = sys.argv[1]
    (X, Xc, trace_std_clean, trace_bag_clean, trace_std_cont, trace_bag_cont,
     a, theta, contam_idx, meta) = load_bundle(outdir)
    evaluate_methods(X, Xc, trace_std_clean, trace_bag_clean, trace_std_cont, trace_bag_cont,
                     a, theta, contam_idx, hdi_prob=0.90, print_table=True)
