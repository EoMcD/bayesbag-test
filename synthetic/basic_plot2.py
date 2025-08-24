import os
import numpy as np
import matplotlib.pyplot as plt

from basic_eval import (
    load_bundle,
    flatten_draws,
    posterior_rank,
)

# --- local helpers (kept here so this file is self-contained) ---

def hdi_min_width(draws_2d, prob=0.90):
    """Minimum-width HDI per column; draws_2d shape (S, G). Returns (G,2)."""
    S, G = draws_2d.shape
    k = max(1, int(np.ceil(prob * S)))
    out = np.empty((G, 2), dtype=float)
    for g in range(G):
        s = np.sort(draws_2d[:, g])
        if k >= S:
            out[g] = [s[0], s[-1]]
            continue
        w = s[k-1:] - s[:S-k+1]
        j = int(np.argmin(w))
        out[g] = [s[j], s[j+k-1]]
    return out

def predictive_density_gamma(trace, X, samples=2000, rng=None):
    from scipy.stats import gamma as sp_gamma
    from scipy.special import logsumexp
    a = flatten_draws(trace, "alpha"); t = flatten_draws(trace, "theta")
    S, G = a.shape
    if samples and S > samples:
        rng = np.random.default_rng(None if rng is None else rng)
        idx = rng.choice(S, samples, replace=False)
        a, t = a[idx], t[idx]; S = samples
    total = 0.0
    for g in range(G):
        y = X[g]
        logp = sp_gamma.logpdf(y[:, None], a=a[:, g], scale=t[:, g])  # (n,S)
        total += (np.log(np.exp(logp).mean(axis=1))).sum()            # log posterior mean density
    return total / X.size

def predictive_density_normal(trace, X, samples=2000, rng=None):
    from scipy.stats import norm as sp_norm
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
        logp = sp_norm.logpdf(y[:, None], loc=mu[:, g], scale=sg[:, g])
        total += (np.log(np.exp(logp).mean(axis=1))).sum()
    return total / X.size

def mismatch_index_var(trace_std, trace_bag, var_name, X_like):
    """Mismatch index for a named posterior variable ('theta','alpha','mu','sigma')."""
    Sstd = flatten_draws(trace_std, var_name).std(axis=0, ddof=1)
    Sbag = flatten_draws(trace_bag, var_name).std(axis=0, ddof=1)
    N = X_like.size
    NvN = N * float(np.sum(Sstd**2))
    MvsM = N * float(np.sum(Sbag**2))
    return 1 - ((2 * NvN) / MvsM) if MvsM > NvN else np.nan

def mismatch_index_gamma_func(trace_std, trace_bag, X_like, func):
    """Mismatch for a derived Gamma quantity: func(alpha_draws, theta_draws)->(S,G)."""
    a_std = flatten_draws(trace_std, "alpha"); t_std = flatten_draws(trace_std, "theta")
    a_bag = flatten_draws(trace_bag, "alpha"); t_bag = flatten_draws(trace_bag, "theta")
    q_std = func(a_std, t_std); q_bag = func(a_bag, t_bag)
    Sstd = q_std.std(axis=0, ddof=1)
    Sbag = q_bag.std(axis=0, ddof=1)
    N = X_like.size
    NvN = N * float(np.sum(Sstd**2))
    MvsM = N * float(np.sum(Sbag**2))
    return 1 - ((2 * NvN) / MvsM) if MvsM > NvN else np.nan

def variance_ratio_from_mismatch(I):
    """Return V_bag / V_std given mismatch I = 1 - 2*V_std/V_bag."""
    if not np.isfinite(I): return np.nan
    return 2.0 / (1.0 - I) if (1.0 - I) != 0 else np.inf


# --- plotting ---

def plot_evaluation(outdir, save_dir=None, hdi_prob=0.90, samples_lpd=2000):
    if save_dir is None:
        save_dir = os.path.join(outdir, "figs_basic")
    os.makedirs(save_dir, exist_ok=True)

    # Load everything (Normal traces may be None if you didn't save them)
    (X_clean, X_contam,
     id_std_clean, id_bag_clean,
     id_std_cont,  id_bag_cont,
     id_std_norm,  id_bag_norm,
     a_true, t_true, contam_idx, meta) = load_bundle(outdir)

    contam_idx = np.array(contam_idx, dtype=bool)
    clean_idx  = ~contam_idx
    G = t_true.shape[0]

    # ---------- 1) Predictive accuracy: 6 bars (Gamma clean/contam + Normal clean; Std & Bag) ----------
    lpd_vals = []
    labels   = []

    # Gamma — clean
    lpd_vals.append(predictive_density_gamma(id_std_clean, X_clean, samples=samples_lpd)); labels.append("Γ clean – Std")
    lpd_vals.append(predictive_density_gamma(id_bag_clean, X_clean, samples=samples_lpd)); labels.append("Γ clean – Bag")

    # Gamma — contaminated
    lpd_vals.append(predictive_density_gamma(id_std_cont,  X_contam, samples=samples_lpd)); labels.append("Γ contam – Std")
    lpd_vals.append(predictive_density_gamma(id_bag_cont,  X_contam, samples=samples_lpd)); labels.append("Γ contam – Bag")

    # Normal — clean (optional)
    if (id_std_norm is not None) and (id_bag_norm is not None):
        lpd_vals.append(predictive_density_normal(id_std_norm, X_clean, samples=samples_lpd)); labels.append("Normal clean – Std")
        lpd_vals.append(predictive_density_normal(id_bag_norm, X_clean, samples=samples_lpd)); labels.append("Normal clean – Bag")

    x = np.arange(len(lpd_vals))
    plt.figure(figsize=(9, 4.2))
    plt.bar(x, lpd_vals, width=0.6)
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Avg log predictive density (higher is better)")
    plt.title("Predictive accuracy across methods/datasets")
    plt.grid(axis="y", alpha=0.25)
    for i, v in enumerate(lpd_vals):
        plt.text(i, v, f"{v:.3f}", va="bottom", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "predictive_accuracy_all.png"), dpi=150)
    plt.close()

    # ---------- 2) Variance inflation ratio (scale-like): Γ θ (clean/contam), Normal σ (clean) ----------
    bars_scale = []
    labels_scale = []

    # Gamma θ mismatch (clean & contaminated)
    I_g_clean  = mismatch_index_var(id_std_clean, id_bag_clean, "theta", X_clean)
    I_g_cont   = mismatch_index_var(id_std_cont,  id_bag_cont,  "theta", X_contam)
    r_g_clean  = variance_ratio_from_mismatch(I_g_clean)
    r_g_cont   = variance_ratio_from_mismatch(I_g_cont)
    bars_scale += [r_g_clean, r_g_cont]; labels_scale += ["Γ clean: θ", "Γ contam: θ"]

    # Normal σ mismatch (clean), if available
    if (id_std_norm is not None) and (id_bag_norm is not None):
        I_n_sigma  = mismatch_index_var(id_std_norm, id_bag_norm, "sigma", X_clean)
        r_n_sigma  = variance_ratio_from_mismatch(I_n_sigma)
        bars_scale += [r_n_sigma]; labels_scale += ["Normal clean: σ"]

    x = np.arange(len(bars_scale))
    plt.figure(figsize=(8.5, 4.0))
    plt.bar(x, bars_scale, width=0.55)
    plt.axhline(2.0, color="k", ls="--", lw=1, alpha=0.6)
    plt.xticks(x, labels_scale, rotation=15, ha="right")
    plt.ylabel(r"Variance inflation $V_{\rm bag}/V_{\rm std}$")
    plt.title("Variance inflation (scale-like parameters)")
    plt.grid(axis="y", alpha=0.25)
    # annotate mismatch index on bars
    for i, r in enumerate(bars_scale):
        I = [I_g_clean, I_g_cont, I_n_sigma][i] if (i < 2 or (id_std_norm is not None and id_bag_norm is not None)) else np.nan
        txt = f"r={r:.2f}" if np.isfinite(r) else "r=NA"
        if np.isfinite(I): txt += f"\nI={I:.3f}"
        plt.text(i, r if np.isfinite(r) else 0.0, txt, ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "variance_inflation_scale.png"), dpi=150)
    plt.close()

    # ---------- 3) Variance inflation ratio (location-like): Γ mean αθ (clean/contam), Normal μ (clean) ----------
    bars_loc = []
    labels_loc = []
    I_gm_clean = mismatch_index_gamma_func(id_std_clean, id_bag_clean, X_clean, lambda a, t: a * t)
    I_gm_cont  = mismatch_index_gamma_func(id_std_cont,  id_bag_cont,  X_contam, lambda a, t: a * t)
    r_gm_clean = variance_ratio_from_mismatch(I_gm_clean)
    r_gm_cont  = variance_ratio_from_mismatch(I_gm_cont)
    bars_loc  += [r_gm_clean, r_gm_cont]; labels_loc += ["Γ clean: mean (αθ)", "Γ contam: mean (αθ)"]

    if (id_std_norm is not None) and (id_bag_norm is not None):
        I_n_mu = mismatch_index_var(id_std_norm, id_bag_norm, "mu", X_clean)
        r_n_mu = variance_ratio_from_mismatch(I_n_mu)
        bars_loc += [r_n_mu]; labels_loc += ["Normal clean: μ"]

    x = np.arange(len(bars_loc))
    plt.figure(figsize=(8.5, 4.0))
    plt.bar(x, bars_loc, width=0.55, color="#6aa9ff")
    plt.axhline(2.0, color="k", ls="--", lw=1, alpha=0.6)
    plt.xticks(x, labels_loc, rotation=15, ha="right")
    plt.ylabel(r"Variance inflation $V_{\rm bag}/V_{\rm std}$")
    plt.title("Variance inflation (location-like parameters)")
    plt.grid(axis="y", alpha=0.25)
    for i, r in enumerate(bars_loc):
        I = [I_gm_clean, I_gm_cont, I_n_mu][i] if (i < 2 or (id_std_norm is not None and id_bag_norm is not None)) else np.nan
        txt = f"r={r:.2f}" if np.isfinite(r) else "r=NA"
        if np.isfinite(I): txt += f"\nI={I:.3f}"
        plt.text(i, r if np.isfinite(r) else 0.0, txt, ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "variance_inflation_location.png"), dpi=150)
    plt.close()

    # ---------- 4) θ posterior densities for contaminated groups ----------
    tS_cont = flatten_draws(id_std_cont, "theta")
    tB_cont = flatten_draws(id_bag_cont, "theta")
    for g in np.where(contam_idx)[0]:
        plt.figure(figsize=(5.2, 3.6))
        bins = max(20, int(np.sqrt(tS_cont.shape[0]) // 2))
        plt.hist(tS_cont[:, g], bins=bins, density=True, alpha=0.45, label="Std (contam)")
        plt.hist(tB_cont[:, g], bins=bins, density=True, alpha=0.45, label="BayesBag (contam)")
        plt.axvline(t_true[g], color="k", ls="--", lw=2, label="θ true")
        plt.title(f"Posterior θ for group {g} (contaminated)")
        plt.xlabel("θ"); plt.ylabel("density"); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"theta_posterior_g{g}.png"), dpi=150)
        plt.close()

    # ---------- 5) Posterior rank scatter (θ), contaminated fits ----------
    rS = posterior_rank(tS_cont, t_true)
    rB = posterior_rank(tB_cont, t_true)
    xg = np.arange(G)
    plt.figure(figsize=(7, 3.8))
    plt.scatter(xg[clean_idx],  rS[clean_idx],  marker="o",  label="Std (clean groups)")
    plt.scatter(xg[clean_idx],  rB[clean_idx],  marker="x",  label="Bag (clean groups)")
    plt.scatter(xg[contam_idx], rS[contam_idx], marker="o",  s=80, label="Std (contam)")
    plt.scatter(xg[contam_idx], rB[contam_idx], marker="x",  s=80, label="Bag (contam)")
    plt.axhline(0.5, color="k", linestyle="--", linewidth=1)
    plt.fill_between([-0.5, G-0.5], 0.05, 0.95, alpha=0.08, step="pre")
    plt.xlim(-0.5, G-0.5); plt.ylim(-0.02, 1.02)
    plt.xlabel("group"); plt.ylabel("posterior rank of θ_true")
    plt.title("Posterior rank of θ_true (contaminated fits)")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "posterior_ranks_theta.png"), dpi=150)
    plt.close()

    # ---------- 6) Stability on unaffected groups: |Δθ| bars ----------
    def post_mean_theta(trace): return flatten_draws(trace, "theta").mean(axis=0)
    t_sc = post_mean_theta(id_std_clean);  t_sx = post_mean_theta(id_std_cont)
    t_bc = post_mean_theta(id_bag_clean);  t_bx = post_mean_theta(id_bag_cont)
    d_std = np.abs(t_sx[clean_idx] - t_sc[clean_idx])
    d_bag = np.abs(t_bx[clean_idx] - t_bc[clean_idx])
    idxs  = np.where(clean_idx)[0]
    width = 0.38
    plt.figure(figsize=(7, 3.8))
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

    # ---------- 7) (Optional) Normal model per-group HDIs vs Gamma truth ----------
    if (id_std_norm is not None) and (id_bag_norm is not None):
        mu_true = a_true * t_true
        sg_true = np.sqrt(a_true) * t_true

        muS = flatten_draws(id_std_norm, "mu");    muB = flatten_draws(id_bag_norm, "mu")
        sgS = flatten_draws(id_std_norm, "sigma"); sgB = flatten_draws(id_bag_norm, "sigma")

        muS_h = hdi_min_width(muS, prob=hdi_prob)
        muB_h = hdi_min_width(muB, prob=hdi_prob)
        sgS_h = hdi_min_width(sgS, prob=hdi_prob)
        sgB_h = hdi_min_width(sgB, prob=hdi_prob)

        xg = np.arange(G)

        # μ intervals
        plt.figure(figsize=(8.5, 4.0))
        yS = muS.mean(axis=0); yB = muB.mean(axis=0)
        yerrS = np.vstack([yS - muS_h[:, 0], muS_h[:, 1] - yS])
        yerrB = np.vstack([yB - muB_h[:, 0], muB_h[:, 1] - yB])
        plt.errorbar(xg - 0.06, yS, yerr=yerrS, fmt='o', capsize=3, label='Std')
        plt.errorbar(xg + 0.06, yB, yerr=yerrB, fmt='s', capsize=3, label='BayesBag')
        plt.plot(xg, mu_true, 'k--', lw=1.2, label='μ (Gamma truth)')
        plt.xlabel("group"); plt.ylabel("μ")
        plt.title("Normal model: μ posterior 90% intervals vs Gamma truth")
        plt.xticks(xg); plt.grid(alpha=0.2); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "normal_mu_intervals.png"), dpi=150)
        plt.close()

        # σ intervals
        plt.figure(figsize=(8.5, 4.0))
        yS = sgS.mean(axis=0); yB = sgB.mean(axis=0)
        yerrS = np.vstack([yS - sgS_h[:, 0], sgS_h[:, 1] - yS])
        yerrB = np.vstack([yB - sgB_h[:, 0], sgB_h[:, 1] - yB])
        plt.errorbar(xg - 0.06, yS, yerr=yerrS, fmt='o', capsize=3, label='Std')
        plt.errorbar(xg + 0.06, yB, yerr=yerrB, fmt='s', capsize=3, label='BayesBag')
        plt.plot(xg, sg_true, 'k--', lw=1.2, label='σ (Gamma truth)')
        plt.xlabel("group"); plt.ylabel("σ")
        plt.title("Normal model: σ posterior 90% intervals vs Gamma truth")
        plt.xticks(xg); plt.grid(alpha=0.2); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "normal_sigma_intervals.png"), dpi=150)
        plt.close()

    print("Saved figures to:", save_dir)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python plots_synth.py <bundle_dir> [<save_dir>]"); raise SystemExit(1)
    outdir = sys.argv[1]
    save_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(outdir, "figs_basic")
    plot_evaluation(outdir, save_dir)
