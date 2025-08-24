import os, sys, glob, json
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from scipy.special import logsumexp
from scipy.stats import gamma as sp_gamma, norm as sp_norm

from basic_eval import (
    load_bundle, flatten_draws, hdi_min_width
)

# ---------- helpers ----------
def find_bundles(paths: List[str]) -> List[str]:
    out = []
    for p in paths:
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "data_arrays.npz")):
            out.append(p)
        elif os.path.isdir(p):
            for d in sorted(glob.glob(os.path.join(p, "*"))):
                if os.path.isdir(d) and os.path.exists(os.path.join(d, "data_arrays.npz")):
                    out.append(d)
    if not out:
        raise SystemError("No bundle dirs found (expected folders containing data_arrays.npz).")
    return out

def bootstrap_mean_ci(x: np.ndarray, B=2000, alpha=0.05, rng=None) -> Tuple[float,float,float]:
    x = np.asarray(x)
    rng = np.random.default_rng(rng)
    if x.size == 0:
        return np.nan, np.nan, np.nan
    bs = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, x.size, x.size)
        bs[b] = x[idx].mean()
    m = float(np.mean(bs))
    lo = float(np.quantile(bs, alpha/2))
    hi = float(np.quantile(bs, 1-alpha/2))
    return m, lo, hi

def predictive_density_by_group_gamma(trace, X) -> np.ndarray:
    a = flatten_draws(trace, "alpha")  # (S,G)
    t = flatten_draws(trace, "theta")  # (S,G)
    S, G = a.shape
    out = np.zeros(G, dtype=float)
    for g in range(G):
        y = X[g][:, None]                                   # (n,1)
        lp = sp_gamma.logpdf(y, a=a[:, g], scale=t[:, g])   # (n,S)
        out[g] = float((logsumexp(lp, axis=1) - np.log(S)).mean())
    return out  # per-group ALPD

def predictive_density_by_group_normal(trace, X) -> np.ndarray:
    if trace is None:
        return None
    mu = flatten_draws(trace, "mu")      # (S,G)
    sg = flatten_draws(trace, "sigma")   # (S,G)
    S, G = mu.shape
    out = np.zeros(G, dtype=float)
    for g in range(G):
        y = X[g][:, None]
        lp = sp_norm.logpdf(y, loc=mu[:, g], scale=sg[:, g])  # (n,S)
        out[g] = float((logsumexp(lp, axis=1) - np.log(S)).mean())
    return out

def pit_values_gamma(trace, X) -> np.ndarray:
    a = flatten_draws(trace, "alpha")  # (S,G)
    t = flatten_draws(trace, "theta")  # (S,G)
    S, G = a.shape
    pits = []
    for g in range(G):
        y = X[g]
        # posterior predictive CDF is mean of Gamma CDFs over draws
        F = sp_gamma.cdf(y[:, None], a=a[:, g], scale=t[:, g])  # (n, S)
        pits.append(F.mean(axis=1))
    return np.concatenate(pits)

def pit_values_normal(trace, X) -> np.ndarray:
    if trace is None:
        return np.array([])
    mu = flatten_draws(trace, "mu")
    sg = flatten_draws(trace, "sigma")
    S, G = mu.shape
    pits = []
    for g in range(G):
        y = X[g]
        F = sp_norm.cdf(y[:, None], loc=mu[:, g], scale=sg[:, g])
        pits.append(F.mean(axis=1))
    return np.concatenate(pits)

def post_mean_theta(trace) -> np.ndarray:
    return flatten_draws(trace, "theta").mean(axis=0)

# ---------- main metric extraction per bundle ----------
def metrics_for_bundle(bundle_dir: str) -> Dict:
    (Xc, Xx,
     tr_sc, tr_bc, tr_sx, tr_bx,
     tr_sn, tr_bn,
     true_a, true_t, contam_idx, meta) = load_bundle(bundle_dir)

    contam_idx = np.asarray(contam_idx, bool)
    G = true_t.shape[0]
    n_per_group = np.array([len(x) for x in Xc])

    # ALPD per group (clean data; universal)
    alpd_sc = predictive_density_by_group_gamma(tr_sc, Xc)
    alpd_bc = predictive_density_by_group_gamma(tr_bc, Xc)
    delta_alpd_gamma = alpd_bc - alpd_sc                     # per-group Δ

    # Normal if available (still on clean data)
    alpd_sn = predictive_density_by_group_normal(tr_sn, Xc) if tr_sn is not None else None
    alpd_bn = predictive_density_by_group_normal(tr_bn, Xc) if tr_bn is not None else None
    delta_alpd_norm = (alpd_bn - alpd_sn) if (alpd_sn is not None and alpd_bn is not None) else None

    # PIT (clean fits, clean data)
    pit_sc = pit_values_gamma(tr_sc, Xc)
    pit_bc = pit_values_gamma(tr_bc, Xc)
    pit_sn = pit_values_normal(tr_sn, Xc) if tr_sn is not None else np.array([])
    pit_bn = pit_values_normal(tr_bn, Xc) if tr_bn is not None else np.array([])

    # Synthetic-only truth comparisons (theta)
    # We assume truth exists if finite and positive (Gamma θ > 0)
    has_truth = np.isfinite(true_t).all() and (true_t > 0).all()
    err_std = err_bag = None
    if has_truth:
        m_sc = post_mean_theta(tr_sc)
        m_bc = post_mean_theta(tr_bc)
        err_std = np.abs(m_sc - true_t)   # use clean fit for universal comparability
        err_bag = np.abs(m_bc - true_t)

    # HDI width ratio (Bag/Std) on clean fits (universal)
    theta_sc = flatten_draws(tr_sc, "theta")
    theta_bc = flatten_draws(tr_bc, "theta")
    h_sc = hdi_min_width(theta_sc, prob=0.90)  # (G,2)
    h_bc = hdi_min_width(theta_bc, prob=0.90)
    width_ratio = (h_bc[:,1] - h_bc[:,0]) / np.maximum(1e-12, (h_sc[:,1] - h_sc[:,0]))

    return dict(
        dir=bundle_dir,
        group_n=n_per_group, contam_idx=contam_idx,
        delta_alpd_gamma=delta_alpd_gamma,
        delta_alpd_norm=delta_alpd_norm,
        pit_sc=pit_sc, pit_bc=pit_bc, pit_sn=pit_sn, pit_bn=pit_bn,
        has_truth=has_truth, err_std=err_std, err_bag=err_bag,
        width_ratio=width_ratio
    )

# ---------- plotting across bundles ----------
def plot_forest_delta_alpd(results: List[Dict], save_dir: str):
    labels, means, lo, hi, win = [], [], [], [], []
    for r in results:
        d = r["delta_alpd_gamma"]
        m, l, h = bootstrap_mean_ci(d, B=2000, alpha=0.05, rng=0)
        labels.append(os.path.basename(r["dir"]))
        means.append(m); lo.append(l); hi.append(h)
        win.append(float(np.mean(d > 0)))
    order = np.argsort(means)[::-1]
    y = np.arange(len(results))
    plt.figure(figsize=(7.6, 0.5 + 0.4*len(results)))
    plt.hlines(y, np.array(lo)[order], np.array(hi)[order], lw=2)
    plt.plot(np.array(means)[order], y, 'o', ms=6)
    # annotate win-rate
    for i, idx in enumerate(order):
        plt.text(hi[idx] + 0.001, i, f" win={win[idx]*100:.0f}%", va='center', fontsize=8)
    plt.axvline(0, color='k', lw=1, ls='--')
    plt.yticks(y, [labels[i] for i in order])
    plt.xlabel(r"$\Delta$ALPD (Bag - Std)  [per obs]")
    plt.title("Predictive lift by experiment (with 95% bootstrap CI across groups)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "forest_delta_alpd.png"), dpi=150)
    plt.close()

def plot_delta_vs_n(results: List[Dict], save_dir: str):
    xs, ys = [], []
    for r in results:
        n = r["group_n"]
        d = r["delta_alpd_gamma"]
        xs.append(n); ys.append(d)
    x = np.concatenate(xs); y = np.concatenate(ys)
    plt.figure(figsize=(7.0, 4.2))
    plt.scatter(x, y, s=16, alpha=0.7)
    # annotate correlation
    if x.size > 1:
        corr = np.corrcoef(x, y)[0,1]
        plt.text(0.02, 0.95, f"r = {corr:.2f}", transform=plt.gca().transAxes, va='top')
    plt.axhline(0, color='k', lw=1, ls='--')
    plt.xlabel("group size n")
    plt.ylabel(r"$\Delta$ALPD (Bag - Std)  [per obs]")
    plt.title("Predictive lift vs group size (all experiments)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "delta_alpd_vs_group_size.png"), dpi=150)
    plt.close()

def plot_pit(results: List[Dict], save_dir: str):
    bins = 20
    plt.figure(figsize=(7.0, 4.0))
    # stack PITs across experiments
    pit_sc = np.concatenate([r["pit_sc"] for r in results])
    pit_bc = np.concatenate([r["pit_bc"] for r in results])
    plt.hist(pit_sc, bins=bins, range=(0,1), histtype='step', label="Std (clean)", lw=1.6)
    plt.hist(pit_bc, bins=bins, range=(0,1), histtype='step', ls='--', label="Bag (clean)", lw=1.8)
    plt.xlabel("PIT")
    plt.ylabel("count")
    plt.title("Calibration via PIT (aggregated across experiments)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pit_hist_clean.png"), dpi=150)
    plt.close()

def plot_param_scatter(results: List[Dict], save_dir: str):
    # pooled
    xs, ys = [], []
    for r in results:
        if r["has_truth"]:
            xs.append(r["err_std"]); ys.append(r["err_bag"])
    if xs:
        x = np.concatenate(xs); y = np.concatenate(ys)
        plt.figure(figsize=(4.6, 4.6))
        plt.scatter(x, y, s=16, alpha=0.7)
        lim = (0, max(np.max(x), np.max(y)) * 1.05 + 1e-9)
        plt.plot(lim, lim, 'k--', lw=1)
        plt.xlim(lim); plt.ylim(lim)
        plt.xlabel(r"|$\hat\theta_{\mathrm{Std}}-\theta_{\mathrm{true}}$|")
        plt.ylabel(r"|$\hat\theta_{\mathrm{Bag}}-\theta_{\mathrm{true}}$|")
        plt.title("Who’s closer to truth? (pooled synthetic)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "synthetic_param_scatter_pooled.png"), dpi=150)
        plt.close()
    # per experiment
    for r in results:
        if not r["has_truth"]:
            continue
        x = r["err_std"]; y = r["err_bag"]
        plt.figure(figsize=(4.6, 4.6))
        plt.scatter(x, y, s=20, alpha=0.8)
        lim = (0, max(np.max(x), np.max(y)) * 1.05 + 1e-9)
        plt.plot(lim, lim, 'k--', lw=1)
        plt.xlim(lim); plt.ylim(lim)
        plt.xlabel(r"|$\hat\theta_{\mathrm{Std}}-\theta_{\mathrm{true}}$|")
        plt.ylabel(r"|$\hat\theta_{\mathrm{Bag}}-\theta_{\mathrm{true}}$|")
        base = os.path.basename(r["dir"])
        plt.title(f"Who’s closer?  ({base})")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"synthetic_param_scatter_{base}.png"), dpi=150)
        plt.close()

def plot_width_vs_lift(results: List[Dict], save_dir: str):
    xs, ys = [], []
    for r in results:
        xs.append(r["width_ratio"])
        ys.append(r["delta_alpd_gamma"])
    x = np.concatenate(xs); y = np.concatenate(ys)
    plt.figure(figsize=(7.0, 4.2))
    plt.scatter(x, y, s=14, alpha=0.7)
    plt.axvline(1.0, color='k', ls=':', lw=1)
    plt.axhline(0.0, color='k', ls='--', lw=1)
    plt.xlabel("HDI width ratio (Bag / Std)  [θ, clean fit]")
    plt.ylabel(r"$\Delta$ALPD (Bag - Std)  [per obs]")
    plt.title("Uncertainty change vs predictive lift (all experiments)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "width_ratio_vs_lift.png"), dpi=150)
    plt.close()

# ---------- CLI ----------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:\n  python plots_universal.py <bundle_dir or runs_dir> [more_dirs ...] [--out OUTDIR]")
        sys.exit(1)
    args = sys.argv[1:]
    out_idx = [i for i,a in enumerate(args) if a == "--out"]
    if out_idx:
        i = out_idx[0]
        outdir = args[i+1]
        inputs = args[:i]
    else:
        inputs = args
        outdir = os.path.join(os.path.dirname(inputs[0]), "figs_universal")
    bundles = find_bundles(inputs)
    os.makedirs(outdir, exist_ok=True)

    # compute metrics
    results = [metrics_for_bundle(b) for b in bundles]

    # plots
    plot_forest_delta_alpd(results, outdir)
    plot_delta_vs_n(results, outdir)
    plot_pit(results, outdir)
    plot_param_scatter(results, outdir)
    plot_width_vs_lift(results, outdir)

    print("Saved figures to:", outdir)
