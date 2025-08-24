# plots_combo_from_bundle.py
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from basic_eval import (
    load_bundle, flatten_draws, posterior_rank, hdi_min_width
)

# -----------------------
# Utilities
# -----------------------

def _find_bundle_dir(path: str) -> str:
    """
    Convenience: allow passing the parent 'runs' dir. If 'path' itself
    contains a bundle (has data_arrays.npz), use it; else search its
    immediate subdirs for one containing data_arrays.npz and pick the
    most recently modified.
    """
    wanted = os.path.join(path, "data_arrays.npz")
    if os.path.exists(wanted):
        return path

    candidates = []
    for d in glob.glob(os.path.join(path, "*")):
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "data_arrays.npz")):
            candidates.append((d, os.path.getmtime(d)))
    if not candidates:
        raise FileNotFoundError(
            f"Could not find a bundle under {path!r}. Expected a folder with data_arrays.npz."
        )
    # newest first
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


def _theta_samples_gamma(trace) -> np.ndarray:
    # (S, G)
    return flatten_draws(trace, "theta")


def _alpha_samples_gamma(trace) -> np.ndarray:
    return flatten_draws(trace, "alpha")


def _theta_from_normal(trace) -> np.ndarray:
    """
    Map Normal posterior draws (mu, sigma) to Gamma-parameter space via
      theta_tilde = sigma^2 / mu
    Invalid (mu<=0 or sigma<=0) draws are dropped.
    Returns: (S_eff, G)
    """
    if trace is None:
        return None
    mu = flatten_draws(trace, "mu")      # (S, G)
    sg = flatten_draws(trace, "sigma")   # (S, G)
    valid = (mu > 0.0) & (sg > 0.0)
    # keep rows (across groups) that have at least one valid entry; invalid entries are masked out later
    row_valid = valid.any(axis=1)
    mu, sg, valid = mu[row_valid], sg[row_valid], valid[row_valid]
    # avoid /0
    eps = 1e-12
    theta_tilde = (sg ** 2) / np.maximum(mu, eps)
    # for invalid entries, set NaN and let np.nan* aggregations handle
    theta_tilde[~valid] = np.nan
    return theta_tilde


def _post_mean(arr_2d: np.ndarray) -> np.ndarray:
    # nanmean so Normal-mapped invalid entries don't break things
    return np.nanmean(arr_2d, axis=0)


def _coverage_from_draws(draws_2d: np.ndarray, truth: np.ndarray, prob=0.90, mask=None) -> float:
    """
    Coverage of central prob interval vs truth. Supports NaNs and optional boolean mask on groups.
    """
    if draws_2d is None:
        return np.nan
    if mask is None:
        mask = np.ones_like(truth, dtype=bool)
    # HDI using provided helper
    hdi = hdi_min_width(draws_2d, prob=prob)  # (G, 2)
    lo, hi = hdi[:, 0], hdi[:, 1]
    ok = (truth >= lo) & (truth <= hi)
    if mask is not None:
        ok = ok[mask]
    return float(np.nanmean(ok))


def _abs_error_to_truth(post_mean: np.ndarray, truth: np.ndarray, mask=None) -> np.ndarray:
    e = np.abs(post_mean - truth)
    if mask is not None:
        return e[mask]
    return e


# -----------------------
# PLOTS
# -----------------------

def plot_all(outdir: str, save_dir: str = None):
    if save_dir is None:
        save_dir = os.path.join(outdir, "figs_all")
    os.makedirs(save_dir, exist_ok=True)

    # Load bundle
    (X_clean, X_contam,
     tr_std_clean, tr_bag_clean, tr_std_cont, tr_bag_cont,
     tr_std_norm, tr_bag_norm,
     true_alpha, true_theta, contam_idx, meta) = load_bundle(outdir)

    contam_idx = np.array(contam_idx, dtype=bool)
    clean_idx  = ~contam_idx
    G = true_theta.shape[0]
    x = np.arange(G)

    # Precompute draws
    tS_clean = _theta_samples_gamma(tr_std_clean)     # (S, G)
    tB_clean = _theta_samples_gamma(tr_bag_clean)
    tS_cont  = _theta_samples_gamma(tr_std_cont)
    tB_cont  = _theta_samples_gamma(tr_bag_cont)

    # Normal mapped -> theta_tilde
    tS_norm = _theta_from_normal(tr_std_norm) if tr_std_norm is not None else None
    tB_norm = _theta_from_normal(tr_bag_norm) if tr_bag_norm is not None else None

    # Colors by model; linestyle by Std/Bag
    C_gamma_cont = "C0"
    C_gamma_clean= "C1"
    C_normal     = "C2"

    # 1) Posterior θ for each contaminated group — combine Gamma(clean), Gamma(contam), Normal(θ̃)
    for g in np.where(contam_idx)[0]:
        plt.figure(figsize=(7.0, 4.2))
        # choose bins based on *largest* available sample set
        sample_sizes = [tS_cont.shape[0], tB_cont.shape[0], tS_clean.shape[0], tB_clean.shape[0]]
        if tS_norm is not None: sample_sizes.append(tS_norm.shape[0])
        bins = max(25, int(np.sqrt(max(sample_sizes)) // 2))

        # Gamma (contaminated fit)
        plt.hist(tS_cont[:, g], bins=bins, density=True, histtype='step', linewidth=1.8,
                 label="Gamma-contam Std", color=C_gamma_cont)
        plt.hist(tB_cont[:, g], bins=bins, density=True, histtype='step', linestyle='--', linewidth=2.0,
                 label="Gamma-contam Bag", color=C_gamma_cont)

        # Gamma (clean fit) — acts as “counterfactual baseline”
        plt.hist(tS_clean[:, g], bins=bins, density=True, histtype='step', linestyle=':', linewidth=1.5,
                 label="Gamma-clean Std", color=C_gamma_clean)
        plt.hist(tB_clean[:, g], bins=bins, density=True, histtype='step', linestyle='--', linewidth=1.8,
                 label="Gamma-clean Bag", color=C_gamma_clean)

        # Normal mapped θ̃ (clean fit)
        if tS_norm is not None:
            validS = ~np.isnan(tS_norm[:, g])
            validB = ~np.isnan(tB_norm[:, g])
            if validS.any():
                plt.hist(tS_norm[validS, g], bins=bins, density=True, histtype='step', linestyle='-.', linewidth=1.8,
                         label="Normal (θ̃) Std", color=C_normal)
            if validB.any():
                plt.hist(tB_norm[validB, g], bins=bins, density=True, histtype='step', linestyle=(0,(3,1,1,1)), linewidth=2.0,
                         label="Normal (θ̃) Bag", color=C_normal)

        plt.axvline(true_theta[g], linestyle="-.", linewidth=2, color="k", label=r"$\theta_{\mathrm{true}}$")
        plt.title(f"Posterior of $\\theta$ — group {g} (contaminated group)")
        plt.xlabel(r"$\\theta$")
        plt.ylabel("density")
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"theta_posterior_g{g}_COMBINED.png"), dpi=150)
        plt.close()

    # 2) Posterior ranks — contaminated fits (Gamma only; this is the apples-to-apples view)
    rS = posterior_rank(tS_cont, true_theta)
    rB = posterior_rank(tB_cont, true_theta)
    plt.figure(figsize=(7.2, 4.2))
    plt.scatter(x[clean_idx], rS[clean_idx], marker="o", s=22, label="Gamma-contam Std (clean groups)", color=C_gamma_cont)
    plt.scatter(x[clean_idx], rB[clean_idx], marker="x", s=28, label="Gamma-contam Bag (clean groups)", color=C_gamma_cont)
    plt.scatter(x[contam_idx], rS[contam_idx], marker="o", s=70, label="Gamma-contam Std (contam)", color=C_gamma_cont)
    plt.scatter(x[contam_idx], rB[contam_idx], marker="x", s=90, label="Gamma-contam Bag (contam)", color=C_gamma_cont)
    plt.axhline(0.5, color="k", linestyle="--", linewidth=1)
    plt.fill_between([-0.5, G-0.5], 0.05, 0.95, alpha=0.08, step="pre")
    plt.xlim(-0.5, G-0.5); plt.ylim(-0.02, 1.02)
    plt.xlabel("group"); plt.ylabel(r"posterior rank of $\theta_{\mathrm{true}}$")
    plt.title("Posterior rank of $\\theta_{true}$ by group — Gamma (contaminated fits)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "posterior_ranks_theta_GAMMA_CONTAM.png"), dpi=150)
    plt.close()

    # 3) Posterior ranks — clean fits (Gamma-clean vs Normal mapped θ̃)
    rS_gcl = posterior_rank(tS_clean, true_theta)
    rB_gcl = posterior_rank(tB_clean, true_theta)
    plt.figure(figsize=(7.2, 4.2))
    plt.scatter(x, rS_gcl, marker="o", s=24, label="Gamma-clean Std", color=C_gamma_clean)
    plt.scatter(x, rB_gcl, marker="x", s=30, label="Gamma-clean Bag", color=C_gamma_clean)
    if tS_norm is not None:
        rS_nrm = posterior_rank(tS_norm, true_theta)
        rB_nrm = posterior_rank(tB_norm, true_theta)
        plt.scatter(x, rS_nrm, marker="o", facecolors="none", s=40, label="Normal (θ̃) Std", color=C_normal)
        plt.scatter(x, rB_nrm, marker="x", s=40, label="Normal (θ̃) Bag", color=C_normal)
    plt.axhline(0.5, color="k", linestyle="--", linewidth=1)
    plt.fill_between([-0.5, G-0.5], 0.05, 0.95, alpha=0.08, step="pre")
    plt.xlim(-0.5, G-0.5); plt.ylim(-0.02, 1.02)
    plt.xlabel("group"); plt.ylabel(r"posterior rank of $\theta_{\mathrm{true}}$")
    plt.title("Posterior rank — clean fits (Gamma vs Normal mapped to θ̃)")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "posterior_ranks_theta_CLEAN_COMBINED.png"), dpi=150)
    plt.close()

    # 4) Stability on unaffected groups (Gamma only): |Δθ| between contaminated vs clean fit
    d_std = np.abs(_post_mean(tS_cont)[clean_idx] - _post_mean(tS_clean)[clean_idx])
    d_bag = np.abs(_post_mean(tB_cont)[clean_idx] - _post_mean(tB_clean)[clean_idx])
    idxs  = np.where(clean_idx)[0]
    width = 0.38
    base = np.arange(len(idxs))
    plt.figure(figsize=(7.2, 4.0))
    plt.bar(base - width/2, d_std, width, label="Gamma Std")
    plt.bar(base + width/2, d_bag, width, label="Gamma BayesBag")
    plt.xticks(base, idxs)
    plt.ylabel("|Δθ| (clean vs contaminated fit)")
    plt.xlabel("clean groups")
    plt.title("Stability on unaffected groups — smaller is better (Gamma)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "stability_theta_clean_groups_GAMMA.png"), dpi=150)
    plt.close()

    # 5) Coverage on clean groups under contaminated fit (Gamma) + Normal-on-clean mapped θ̃
    target = 0.90
    cov_gS = _coverage_from_draws(tS_cont, true_theta, prob=target, mask=clean_idx)
    cov_gB = _coverage_from_draws(tB_cont, true_theta, prob=target, mask=clean_idx)
    # Normal (clean) θ̃ coverage vs θ_true on clean groups
    cov_nS = _coverage_from_draws(tS_norm, true_theta, prob=target, mask=clean_idx) if tS_norm is not None else np.nan
    cov_nB = _coverage_from_draws(tB_norm, true_theta, prob=target, mask=clean_idx) if tB_norm is not None else np.nan

    labels = ["Gamma Std (contam fit)", "Gamma Bag (contam fit)", "Normal Std (θ̃ clean)", "Normal Bag (θ̃ clean)"]
    covs   = [cov_gS, cov_gB, cov_nS, cov_nB]
    plt.figure(figsize=(7.0, 4.0))
    pos = np.arange(len(labels))
    plt.bar(pos, covs)
    plt.axhline(target, color="k", linestyle="--", linewidth=1, label=f"target {target:.2f}")
    plt.xticks(pos, labels, rotation=15)
    plt.ylim(0, 1)
    plt.ylabel(f"coverage on clean groups ({int(target*100)}% PI)")
    plt.title("Nominal coverage comparison on clean groups")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "coverage_clean_groups_COMBINED.png"), dpi=150)
    plt.close()

    # 6) Absolute posterior-mean error on contaminated groups (Gamma)
    cont_ids = np.where(contam_idx)[0]
    if len(cont_ids) > 0:
        e_std = _abs_error_to_truth(_post_mean(tS_cont), true_theta, mask=contam_idx)
        e_bag = _abs_error_to_truth(_post_mean(tB_cont), true_theta, mask=contam_idx)
        base = np.arange(len(cont_ids))
        width = 0.38
        plt.figure(figsize=(7.0, 4.0))
        plt.bar(base - width/2, e_std, width, label="Gamma Std")
        plt.bar(base + width/2, e_bag, width, label="Gamma BayesBag")
        plt.xticks(base, cont_ids)
        plt.xlabel("contaminated groups")
        plt.ylabel(r"$|\mathbb{E}[\theta\mid y] - \theta_\mathrm{true}|$")
        plt.title("Absolute posterior-mean error on contaminated groups — smaller is better (Gamma)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "abs_error_contaminated_groups_GAMMA.png"), dpi=150)
        plt.close()

    # 7) Rank-uniformity histogram on clean groups (combine Gamma-clean and Normal θ̃)
    bins = 10
    plt.figure(figsize=(6.8, 4.0))
    plt.hist(posterior_rank(tS_clean, true_theta)[clean_idx], bins=bins, range=(0,1),
             histtype='step', linewidth=1.6, label="Gamma-clean Std")
    plt.hist(posterior_rank(tB_clean, true_theta)[clean_idx], bins=bins, range=(0,1),
             histtype='step', linestyle='--', linewidth=1.8, label="Gamma-clean Bag")
    if tS_norm is not None:
        plt.hist(posterior_rank(tS_norm, true_theta)[clean_idx], bins=bins, range=(0,1),
                 histtype='step', linestyle='-.', linewidth=1.6, label="Normal (θ̃) Std")
        plt.hist(posterior_rank(tB_norm, true_theta)[clean_idx], bins=bins, range=(0,1),
                 histtype='step', linestyle=(0,(3,1,1,1)), linewidth=1.8, label="Normal (θ̃) Bag")
    plt.axhline(np.sum(clean_idx) / bins, color='k', linestyle=':', linewidth=1)
    plt.xlabel(r"rank of $\theta_\mathrm{true}$ (clean groups; should be ~uniform)")
    plt.ylabel("count")
    plt.title("Rank-uniformity diagnostic — clean groups (combined)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "rank_hist_clean_COMBINED.png"), dpi=150)
    plt.close()

    # 8) (Optional) Coverage curve on clean groups under contaminated fit (Gamma)
    probs = np.linspace(0.5, 0.95, 10)
    cov_std_curve, cov_bag_curve = [], []
    for p in probs:
        cov_std_curve.append(_coverage_from_draws(tS_cont, true_theta, prob=p, mask=clean_idx))
        cov_bag_curve.append(_coverage_from_draws(tB_cont, true_theta, prob=p, mask=clean_idx))
    plt.figure(figsize=(6.6, 4.0))
    plt.plot(probs, cov_std_curve, marker="o", label="Gamma-contam Std")
    plt.plot(probs, cov_bag_curve, marker="x", label="Gamma-contam Bag")
    plt.plot([0.5, 0.95], [0.5, 0.95], linestyle="--", color="k", label="ideal")
    plt.xlabel("nominal interval probability")
    plt.ylabel("empirical coverage (clean groups)")
    plt.title("Coverage curve — contaminated fit on clean groups (Gamma)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "coverage_curve_clean_GAMMA.png"), dpi=150)
    plt.close()

    print("Saved figures to:", save_dir)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python plots_combo_from_bundle.py <bundle_or_runs_dir> [<save_dir>]")
        raise SystemExit(1)
    in_path = _find_bundle_dir(sys.argv[1])
    save_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(in_path, "figs_all")
    os.makedirs(save_dir, exist_ok=True)
    print("Bundle dir:", in_path)
    print("Save dir  :", save_dir)
    plot_all(in_path, save_dir)
