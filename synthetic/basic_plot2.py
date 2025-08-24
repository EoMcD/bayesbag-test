import os
import re
import numpy as np
import arviz as az  # noqa: F401 (kept for compatibility)
import matplotlib.pyplot as plt
from typing import Dict, Tuple

# Your helpers
from basic_eval import load_bundle, flatten_draws, posterior_rank

"""
BayesBag evaluation plots — multi-model version

This script extends your original plotting code to:
  • Combine results from *multiple model variants* (e.g., Normal, Gamma (clean), Gamma (contam)).
  • Plot Std vs BayesBag for *each* model on the same axes where it makes sense.
  • Add several new plots that highlight BayesBag effects: stability, coverage, error on contaminated groups,
    and rank uniformity.

USAGE EXAMPLES
--------------
1) Backward compatible (single bundle):
   python plots_bayesbag_multi.py /path/to/normal_run [ /path/to/save_dir ]

2) Three models with auto labels (order used for legend):
   python plots_bayesbag_multi.py /path/to/normal /path/to/gamma_clean /path/to/gamma_cont [ /path/to/save_dir ]

3) Explicit labels:
   python plots_bayesbag_multi.py \
       Normal=/path/to/normal \
       "Gamma (clean)=/path/to/gamma_clean" \
       "Gamma (contam)=/path/to/gamma_cont" \
       /path/to/save_dir

Each bundle dir must be loadable by basic_eval.load_bundle and should refer to
runs on the SAME underlying dataset (same theta, contam_idx). The script checks
for consistency and warns if there is a mismatch.
"""

# ------------------------------
# Helpers
# ------------------------------

def _sanitize_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")


def _theta_samples(trace, name="theta"):
    return flatten_draws(trace, name)  # (draws, G)


def _post_mean_theta(trace) -> np.ndarray:
    return _theta_samples(trace).mean(axis=0)


def _post_ci(samples: np.ndarray, q=(0.05, 0.95)) -> Tuple[np.ndarray, np.ndarray]:
    lo, hi = np.quantile(samples, q, axis=0)
    return lo, hi


def _coverage(samples: np.ndarray, theta: np.ndarray, alpha: float = 0.10) -> np.ndarray:
    """Return a boolean array (G,) whether theta is inside the (1-alpha) central interval."""
    lo, hi = _post_ci(samples, (alpha/2, 1-alpha/2))
    return (theta >= lo) & (theta <= hi)


def _ensure_consistent(bundles: Dict[str, tuple]):
    """Check that theta and contam_idx agree across bundles."""
    thetas = []
    contams = []
    for _label, b in bundles.items():
        X, Xc, id_std_clean, id_bag_clean, id_std_cont, id_bag_cont, a, theta, contam_idx, meta = b
        thetas.append(theta)
        contams.append(np.asarray(contam_idx, dtype=bool))
    ref_theta = thetas[0]
    ref_contam = contams[0]
    ok = True
    for t in thetas[1:]:
        ok &= (np.allclose(t, ref_theta))
    for c in contams[1:]:
        ok &= (np.array_equal(c, ref_contam))
    return ok


# ------------------------------
# Core plotting
# ------------------------------

def plot_evaluation_multi(model_dirs: Dict[str, str], save_dir: str = "figs_basic"):
    os.makedirs(save_dir, exist_ok=True)

    # Load all bundles
    bundles = {label: load_bundle(path) for label, path in model_dirs.items()}
    if not _ensure_consistent(bundles):
        print("[WARN] theta / contam_idx differ across bundles. Proceeding anyway; plots may be misaligned.")

    # Use the first bundle as reference for G, theta, contam_idx
    any_label = next(iter(bundles.keys()))
    X, Xc, id_std_clean, id_bag_clean, id_std_cont, id_bag_cont, a, theta, contam_idx, meta = bundles[any_label]
    contam_idx = np.array(contam_idx, dtype=bool)
    clean_idx = ~contam_idx
    G = theta.shape[0]

    # Assign a distinct color to each model label (use Matplotlib default cycler)
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', None)
    labels = list(bundles.keys())
    colors = {label: color_cycle[i % len(color_cycle)] if color_cycle else None for i, label in enumerate(labels)}

    # Convenience dict of theta draw matrices for each model
    # Each entry maps to a dict with keys: S_clean, B_clean, S_cont, B_cont
    draws = {}
    for label, b in bundles.items():
        X, Xc, id_std_clean, id_bag_clean, id_std_cont, id_bag_cont, a, _theta, _contam_idx, meta = b
        draws[label] = {
            'S_clean': _theta_samples(id_std_clean),
            'B_clean': _theta_samples(id_bag_clean),
            'S_cont':  _theta_samples(id_std_cont),
            'B_cont':  _theta_samples(id_bag_cont),
        }

    # 1) Posterior densities for θ, by contaminated group — ALL MODELS together
    for g in np.where(contam_idx)[0]:
        plt.figure(figsize=(6.8, 4))
        # choose a reasonable bin count based on one model's sample size
        any_S_cont = draws[any_label]['S_cont']
        bins = max(25, int(np.sqrt(any_S_cont.shape[0]) // 2))
        for label in labels:
            tS = draws[label]['S_cont'][:, g]
            tB = draws[label]['B_cont'][:, g]
            c = colors[label]
            # Std = solid step; Bag = dashed step
            plt.hist(tS, bins=bins, density=True, histtype='step', linewidth=1.6, label=f"{label} Std", color=c)
            plt.hist(tB, bins=bins, density=True, histtype='step', linestyle='--', linewidth=1.8, label=f"{label} Bag", color=c)
        plt.axvline(theta[g], linestyle="-.", linewidth=2, color='k', label=r"$\theta_\mathrm{true}$")
        plt.title(f"Posterior of $\\theta$ — group {g} (contaminated)")
        plt.xlabel(r"$\\theta$")
        plt.ylabel("density")
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        fname = os.path.join(save_dir, f"theta_posterior_g{g}_ALL.png")
        plt.savefig(fname, dpi=150)
        plt.close()

    # 2) Posterior ranks scatter (using contaminated fits)
    #    Color = model, Marker = Std/Bag, size emphasizes contaminated groups
    x = np.arange(G)
    offsets = np.linspace(-0.18, 0.18, num=max(2, len(labels)))  # spread models slightly

    plt.figure(figsize=(7.2, 4.2))
    for i, label in enumerate(labels):
        tS = draws[label]['S_cont']
        tB = draws[label]['B_cont']
        rS = posterior_rank(tS, theta)
        rB = posterior_rank(tB, theta)
        xo = x + offsets[i]
        plt.scatter(xo[clean_idx], rS[clean_idx], marker="o", s=20, label=f"{label} Std (clean)", color=colors[label], alpha=0.9)
        plt.scatter(xo[clean_idx], rB[clean_idx], marker="x", s=28, label=f"{label} Bag (clean)", color=colors[label], alpha=0.9)
        plt.scatter(xo[contam_idx], rS[contam_idx], marker="o", s=70, label=f"{label} Std (contam)", color=colors[label], alpha=0.9)
        plt.scatter(xo[contam_idx], rB[contam_idx], marker="x", s=90, label=f"{label} Bag (contam)", color=colors[label], alpha=0.9)
    plt.axhline(0.5, color="k", linestyle="--", linewidth=1)
    plt.fill_between([-0.5, G-0.5], 0.05, 0.95, alpha=0.08, step="pre")
    plt.xlim(-0.5, G-0.5)
    plt.ylim(-0.02, 1.02)
    plt.xlabel("group")
    plt.ylabel(r"posterior rank of $\theta_\mathrm{true}$")
    plt.title(r"Posterior rank of $\theta_\mathrm{true}$ by group — all models (contaminated fits)")
    plt.legend(loc="best", fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "posterior_ranks_theta_ALL.png"), dpi=150)
    plt.close()

    # 3) Stability on unaffected groups: |Δθ| between contaminated vs clean fit — ALL MODELS
    clean_ids = np.where(clean_idx)[0]
    n_clean = len(clean_ids)
    base = np.arange(n_clean)
    width = 0.8 / (2 * max(1, len(labels)))  # two bars per model (Std, Bag)

    plt.figure(figsize=(7.6, 4.2))
    for i, label in enumerate(labels):
        t_sc = _post_mean_theta(bundles[label][2])  # id_std_clean
        t_bc = _post_mean_theta(bundles[label][3])  # id_bag_clean
        t_sx = _post_mean_theta(bundles[label][4])  # id_std_cont
        t_bx = _post_mean_theta(bundles[label][5])  # id_bag_cont
        d_std = np.abs(t_sx[clean_idx] - t_sc[clean_idx])
        d_bag = np.abs(t_bx[clean_idx] - t_bc[clean_idx])
        off = (i - (len(labels)-1)/2) * (2*width)
        plt.bar(base + off - width/2, d_std, width, label=f"{label} Std", color=colors[label], alpha=0.6)
        plt.bar(base + off + width/2, d_bag, width, label=f"{label} Bag", color=colors[label], alpha=0.95)
    plt.xticks(base, clean_ids)
    plt.ylabel(r"$|\Delta \theta|$ (clean vs contaminated fit)")
    plt.xlabel("clean groups")
    plt.title("Stability on unaffected groups — smaller is better")
    plt.legend(fontsize=7, ncol=3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "stability_theta_clean_groups_ALL.png"), dpi=150)
    plt.close()

    # 4) Coverage on clean groups (contaminated fits): P(theta_true in 90% PI)
    target = 0.90
    cov_rows = []  # (label, method, coverage)
    for label in labels:
        S_cont = draws[label]['S_cont']
        B_cont = draws[label]['B_cont']
        cov_S = _coverage(S_cont, theta, alpha=0.10)[clean_idx].mean()
        cov_B = _coverage(B_cont, theta, alpha=0.10)[clean_idx].mean()
        cov_rows.append((label, 'Std', cov_S))
        cov_rows.append((label, 'Bag', cov_B))

    # Plot grouped bars by label
    plt.figure(figsize=(6.8, 3.8))
    base = np.arange(len(labels))
    width = 0.35
    cov_S_vals = [v for (lab, m, v) in cov_rows if m == 'Std']
    cov_B_vals = [v for (lab, m, v) in cov_rows if m == 'Bag']
    plt.bar(base - width/2, cov_S_vals, width, label="Std")
    plt.bar(base + width/2, cov_B_vals, width, label="Bag")
    plt.axhline(target, color='k', linestyle='--', linewidth=1, label="target 0.90")
    plt.xticks(base, labels, rotation=15)
    plt.ylim(0, 1)
    plt.ylabel("coverage on clean groups (contaminated fits)")
    plt.title("Nominal 90% posterior interval coverage — higher ≈ better")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "coverage_clean_groups_ALL.png"), dpi=150)
    plt.close()

    # 5) Error on contaminated groups (contaminated fits): |E(theta|y) - theta_true|
    cont_ids = np.where(contam_idx)[0]
    if len(cont_ids) > 0:
        base = np.arange(len(cont_ids))
        width = 0.8 / (2 * max(1, len(labels)))
        plt.figure(figsize=(7.2, 4.0))
        for i, label in enumerate(labels):
            t_sx = _post_mean_theta(bundles[label][4])  # id_std_cont
            t_bx = _post_mean_theta(bundles[label][5])  # id_bag_cont
            e_std = np.abs(t_sx[contam_idx] - theta[contam_idx])
            e_bag = np.abs(t_bx[contam_idx] - theta[contam_idx])
            off = (i - (len(labels)-1)/2) * (2*width)
            plt.bar(base + off - width/2, e_std, width, label=f"{label} Std", color=colors[label], alpha=0.6)
            plt.bar(base + off + width/2, e_bag, width, label=f"{label} Bag", color=colors[label], alpha=0.95)
        plt.xticks(base, cont_ids)
        plt.xlabel("contaminated groups")
        plt.ylabel(r"$|\,\mathbb{E}[\theta \mid y] - \theta_\mathrm{true}\,|$")
        plt.title("Absolute posterior-mean error on contaminated groups — smaller is better")
        plt.legend(fontsize=7, ncol=3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "abs_error_contaminated_groups_ALL.png"), dpi=150)
        plt.close()

    # 6) Rank histogram (clean groups) to check uniformity (contaminated fits)
    bins = 10
    plt.figure(figsize=(6.6, 3.8))
    for label in labels:
        rS = posterior_rank(draws[label]['S_cont'], theta)[clean_idx]
        rB = posterior_rank(draws[label]['B_cont'], theta)[clean_idx]
        # overlay step histograms
        plt.hist(rS, bins=bins, range=(0,1), histtype='step', linewidth=1.5, label=f"{label} Std")
        plt.hist(rB, bins=bins, range=(0,1), histtype='step', linestyle='--', linewidth=1.8, label=f"{label} Bag")
    plt.axhline(len(np.where(clean_idx)[0]) / bins, color='k', linestyle=':', linewidth=1)
    plt.xlabel(r"rank of $\theta_\mathrm{true}$ (should be uniform on clean)")
    plt.ylabel("count (clean groups)")
    plt.title("Rank-uniformity diagnostic on clean groups — all models")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "rank_hist_clean_ALL.png"), dpi=150)
    plt.close()

    print(f"Saved figures to: {save_dir}")


# ---------------------------------
# Backward compatible single-model wrapper
# ---------------------------------

def plot_evaluation(outdir: str, save_dir: str = "figs_basic"):
    """Preserves your original interface for a single bundle directory."""
    label = "Model"
    plot_evaluation_multi({label: outdir}, save_dir=save_dir)


# ---------------------------------
# CLI
# ---------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage:\n  python plots_bayesbag_multi.py <bundle_dir> [<save_dir>]\n  python plots_bayesbag_multi.py <dir1> <dir2> <dir3> [<save_dir>]\n  python plots_bayesbag_multi.py Label1=<dir1> Label2=<dir2> ... [<save_dir>]")
        raise SystemExit(1)

    *args, = sys.argv[1:]
    # Parse possible labels (Label=Path). If last arg is a path that doesn't contain '=',
    # treat it as save_dir *only if* it doesn't look like a bundle dir in key=value mode.
    raw = list(args)

    has_equals = any('=' in a for a in raw)
    save_dir = None

    if has_equals:
        # Last arg might be save_dir if it has no '='
        if '=' not in raw[-1]:
            save_dir = raw.pop()
        model_dirs = {}
        for item in raw:
            if '=' not in item:
                raise SystemExit(f"Unexpected arg without '=': {item}")
            label, path = item.split('=', 1)
            model_dirs[label.strip()] = path.strip()
        if not save_dir:
            # default under first model dir
            first_dir = next(iter(model_dirs.values()))
            save_dir = os.path.join(first_dir, "figs_all")
    else:
        # No labels provided. If 1–3 dirs given, auto-label.
        if len(raw) == 1:
            model_dirs = {"Model": raw[0]}
            save_dir = os.path.join(raw[0], "figs")
        elif len(raw) in (2, 3, 4):
            # If final arg is an existing path and we already have >=2 dirs, use it as save_dir
            candidate = raw[-1]
            if len(raw) >= 2 and os.path.splitext(candidate)[1] == "":
                # heuristically treat the last as save_dir only if it doesn't look like a bundle leaf (weak heuristic)
                # safer: if path doesn't exist yet, that's fine; we'll create it.
                # We'll reserve the last arg as save_dir *only if* we have 4 args total.
                if len(raw) == 4:
                    save_dir = raw.pop()
            if len(raw) == 2:
                model_dirs = {"Model": raw[0]}
                save_dir = raw[1]
            else:
                labels_auto = ["Normal", "Gamma (clean)", "Gamma (contam)"]
                model_dirs = {labels_auto[i]: raw[i] for i in range(min(3, len(raw)))}
                if not save_dir:
                    save_dir = os.path.join(raw[0], "figs_all")
        else:
            raise SystemExit("Provide 1–3 bundle dirs (optionally followed by save_dir).")

    # Sanitize save_dir a bit
    os.makedirs(save_dir, exist_ok=True)

    print("Models:")
    for label, path in model_dirs.items():
        print(f"  - {label}: {path}")
    print(f"Saving plots to: {save_dir}")

    plot_evaluation_multi(model_dirs, save_dir=save_dir)
