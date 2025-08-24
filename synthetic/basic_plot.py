import os
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from basic_eval import load_bundle, flatten_draws, posterior_rank

def plot_evaluation(outdir, save_dir="figs"):
    os.makedirs(save_dir, exist_ok=True)

    (X, Xc, id_std_clean, id_bag_clean, id_std_cont, id_bag_cont,
     a, theta, contam_idx, meta) = load_bundle(outdir)
    contam_idx = np.array(contam_idx, dtype=bool)
    clean_idx  = ~contam_idx
    G = theta.shape[0]

    # 1) θ posterior densities for each contaminated group
    tS = flatten_draws(id_std_cont, "theta")
    tB = flatten_draws(id_bag_cont, "theta")
    for g in np.where(contam_idx)[0]:
        plt.figure(figsize=(5,3.5))
        bins = max(20, int(np.sqrt(tS.shape[0])//2))
        plt.hist(tS[:,g], bins=bins, density=True, alpha=0.4, label="Std (contam)")
        plt.hist(tB[:,g], bins=bins, density=True, alpha=0.4, label="BayesBag (contam)")
        plt.axvline(theta[g], linestyle="--", linewidth=2, label="θ true")
        plt.title(f"Posterior θ for group {g} (contaminated)")
        plt.xlabel("θ"); plt.ylabel("density"); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"theta_posterior_g{g}.png"), dpi=150)
        plt.close()

    # 2) Posterior ranks scatter (θ)
    rS = posterior_rank(tS, theta)
    rB = posterior_rank(tB, theta)
    x = np.arange(G)
    plt.figure(figsize=(6.5,3.5))
    plt.scatter(x[clean_idx], rS[clean_idx], marker="o", label="Std (clean groups)")
    plt.scatter(x[clean_idx], rB[clean_idx], marker="x", label="Bag (clean groups)")
    plt.scatter(x[contam_idx], rS[contam_idx], marker="o", s=80, label="Std (contam)")
    plt.scatter(x[contam_idx], rB[contam_idx], marker="x", s=80, label="Bag (contam)")
    plt.axhline(0.5, color="k", linestyle="--", linewidth=1)
    plt.fill_between([-0.5, G-0.5], 0.05, 0.95, alpha=0.1, step="pre")
    plt.xlim(-0.5, G-0.5); plt.ylim(-0.02, 1.02)
    plt.xlabel("group"); plt.ylabel("posterior rank of θ_true")
    plt.title("Posterior rank of θ_true by group (contam fits)")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "posterior_ranks_theta.png"), dpi=150)
    plt.close()

    # 3) Stability bars: |Δθ| on unaffected groups
    def post_mean_theta(trace): return flatten_draws(trace, "theta").mean(axis=0)
    t_sc = post_mean_theta(id_std_clean);  t_sx = post_mean_theta(id_std_cont)
    t_bc = post_mean_theta(id_bag_clean);  t_bx = post_mean_theta(id_bag_cont)
    d_std = np.abs(t_sx[clean_idx] - t_sc[clean_idx])
    d_bag = np.abs(t_bx[clean_idx] - t_bc[clean_idx])
    idxs  = np.where(clean_idx)[0]
    width = 0.35
    plt.figure(figsize=(6.5,3.5))
    plt.bar(np.arange(len(idxs)) - width/2, d_std, width, label="Std")
    plt.bar(np.arange(len(idxs)) + width/2, d_bag, width, label="BayesBag")
    plt.xticks(np.arange(len(idxs)), idxs)
    plt.ylabel("|Δθ| (clean vs contaminated fit)")
    plt.xlabel("clean groups")
    plt.title("Stability on unaffected groups")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "stability_theta_clean_groups.png"), dpi=150)
    plt.close()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python plots.py <bundle_dir> [<save_dir>]")
        raise SystemExit(1)
    outdir = sys.argv[1]
    save_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(outdir, "figs")
    plot_evaluation(outdir, save_dir)
    print("Saved figures to:", save_dir)
