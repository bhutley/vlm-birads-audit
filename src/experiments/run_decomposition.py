"""Experiment E4: the make-or-break per-case test (RQ5).

Author: Brett Hutley
Question: is a case's **clinical share** s(z) = |clin| / (|clin| + |resid|)
informative about whether the frozen classifier gets it wrong, BEYOND what the
classifier's own confidence already says? (clin = (P_C w).z is the BI-RADS-nameable
part of the decision; resid = w_perp.z is the un-nameable part.)

WP5 (2026-07-22) audit-unit-aware repair (plan §11, review M-4/SC-2):

  * BUSI has no verified patient map, so each consistent-label **audit group**
    (near-duplicate cluster ∪ singleton, from the WP1 manifest) is aggregated to a
    mean L2-normalised embedding — one **duplicate-safe audit unit** per group;
  * predictions are **out-of-fold w.r.t. the probe**: the frozen probe is refit on
    four of the five canonical group-disjoint folds and error/confidence/s(z) are
    read on the held-out fold, so no unit is scored by a probe that trained on it;
  * the likelihood-ratio test is an **unpenalized MLE** nested-model test (not the
    old L2-penalized sklearn fit, which is not the LRT reference distribution);
  * uncertainty **bootstraps audit units**, not images.

The distance-to-flip is a monotone function of confidence (§5), so the only
admissible question is *incremental* over confidence.

VERDICT (unchanged, two-part): s adds value only if the bootstrap delta-AUC CI
lower bound > 0 AND a valid unpenalized LR test has p < 0.05. Otherwise the
per-case claim is DROPPED and the paper keeps model-level RQ2-RQ4 from E2/E3.

Usage:
    PYTHONHASHSEED=0 python -m src.experiments.run_decomposition
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cohort import CohortManifest
from src.evaluation import concept_directions as cd
from src.evaluation import decomposition as dec
from src.experiments.compute_rho_geometry import _load_model, _load_site_embeddings
from src.utils.config import load_experiment_config
from src.utils.reproducibility import set_all_seeds
from src.utils.results import save_results

EXPERIMENT_NAME = "decomposition"


# --- audit-unit aggregation --------------------------------------------------
def _aggregate_audit_units(se):
    """Aggregate each consistent-label audit group to one mean-embedding unit.

    Returns (Zu, yu, foldu). Asserts each group is single-label (the WP1 audit
    excluded label-conflict clusters) and fold-indivisible (WP1 folds).
    """
    rows = defaultdict(list)
    for i, g in enumerate(se.group_ids):
        rows[g].append(i)
    Zu, yu, foldu = [], [], []
    for g, idx in rows.items():
        idx = np.array(idx)
        labels = np.unique(se.y[idx])
        assert len(labels) == 1, f"group {g} is not single-label"
        folds = np.unique(se.folds[idx])
        assert len(folds) == 1, f"group {g} spans folds {folds}"
        z = se.Z[idx].mean(axis=0)
        z = z / (np.linalg.norm(z) + 1e-12)          # re-normalise the mean
        Zu.append(z)
        yu.append(int(labels[0]))
        foldu.append(int(folds[0]))
    order = np.argsort([g for g in rows])            # deterministic order
    Zu = np.array(Zu)[order]
    yu = np.array(yu)[order]
    foldu = np.array(foldu)[order]
    return Zu, yu, foldu


def _fit_probe(Z, y, C):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(Z, y)
    return clf.coef_.ravel().astype(np.float64), float(clf.intercept_[0])


def _case_features(Z, y, w, b, B):
    """Per-case error, confidence (|p-0.5|), and clinical share s(z)."""
    g = Z @ w + b
    p = 1.0 / (1.0 + np.exp(-g))
    pred = (p >= 0.5).astype(int)
    error = (pred != y).astype(int)
    confidence = np.abs(p - 0.5)
    s = dec.decompose(Z, w, b, B).clinical_share
    return error, confidence, s, p


def _out_of_fold_features(Zu, yu, foldu, B, C):
    """Cross-fit the probe over canonical folds; return OOF (err, conf, s).

    For each fold f, fit the probe on the other folds' units and read the case
    features on fold f, so every unit is scored by a probe that did not train on
    it (plan §11.1). Folds are group-disjoint by construction.
    """
    err = np.zeros(len(yu), dtype=int)
    conf = np.zeros(len(yu))
    s = np.zeros(len(yu))
    for f in sorted(set(foldu.tolist())):
        te = foldu == f
        tr = ~te
        if len(np.unique(yu[tr])) < 2 or te.sum() == 0:
            continue
        w, b = _fit_probe(Zu[tr], yu[tr], C)
        e, c, si, _ = _case_features(Zu[te], yu[te], w, b, B)
        err[te], conf[te], s[te] = e, c, si
    return err, conf, s


def _zscore(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + 1e-12)


def _cv_auc(X, err, seed, k):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    if err.sum() < k or (1 - err).sum() < k:
        return float("nan")
    oof = np.zeros(len(err))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, err):
        m = LogisticRegression(max_iter=1000).fit(X[tr], err[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(err, oof))


def _incremental_auc(conf, s, err, seed, k):
    a0 = _cv_auc(_zscore(conf)[:, None], err, seed, k)
    a1 = _cv_auc(np.column_stack([_zscore(conf), _zscore(s)]), err, seed, k)
    delta = (a1 - a0) if (np.isfinite(a0) and np.isfinite(a1)) else float("nan")
    return a0, a1, delta


def _ci(vals, lo=2.5, hi=97.5):
    a = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if a.size == 0:
        return {"lo": float("nan"), "median": float("nan"), "hi": float("nan"), "frac_valid": 0.0}
    return {"lo": float(np.percentile(a, lo)), "median": float(np.median(a)),
            "hi": float(np.percentile(a, hi)), "frac_valid": float(a.size / len(vals))}


def _fit_logit_mle(X, err, max_iter=5000):
    """Unpenalized MLE logistic fit; returns (loglik, converged, in-sample accuracy).

    Log-likelihood is computed from clipped true-class probabilities (no log(0)
    warning). In-sample accuracy exposes (quasi-)separation, which lbfgs may reach
    by tolerance without the coefficients formally diverging.
    """
    import warnings
    from sklearn.linear_model import LogisticRegression
    # C=inf == unpenalized MLE (sklearn's forward-compatible spelling of the
    # deprecated penalty=None); this is the LRT reference fit, not an L2 estimate.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        m = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=max_iter).fit(X, err)
    n_iter = int(np.max(m.n_iter_))
    p = m.predict_proba(X)
    p_true = p[np.arange(len(err)), err]
    ll = float(np.log(np.clip(p_true, 1e-300, 1.0)).sum())
    acc = float((m.predict(X) == err).mean())
    return ll, bool(n_iter < max_iter), acc


def _lr_test(conf, s, err):
    """VALID unpenalized-MLE likelihood-ratio test for adding s (plan §11.2).

    Returns log-likelihoods, the statistic, df, p, and a ``valid`` flag. If either
    fit fails to converge, or (quasi-)separation drives the alternative
    log-likelihood to ~0, the test is marked invalid — which, by the gate, forces
    the E4 verdict to remain negative rather than trusting a chi-square reference
    that does not apply.
    """
    out = {"ll0": float("nan"), "ll1": float("nan"), "stat": float("nan"),
           "df": 1, "p": float("nan"), "valid": False, "reason": ""}
    if len(np.unique(err)) < 2:
        out["reason"] = "single class in error"
        return out
    X0 = _zscore(conf)[:, None]
    X1 = np.column_stack([_zscore(conf), _zscore(s)])
    ll0, c0, _ = _fit_logit_mle(X0, err)
    ll1, c1, acc1 = _fit_logit_mle(X1, err)
    out.update(ll0=ll0, ll1=ll1)
    if not (c0 and c1):
        out["reason"] = "MLE did not converge"
        return out
    if acc1 >= 0.999 or abs(ll1) < 1e-6:
        out["reason"] = "separation (alt model perfectly classifies error)"
        return out
    stat = 2.0 * (ll1 - ll0)
    if stat < -1e-6:
        out["reason"] = "negative statistic (numerical)"
        return out
    out.update(stat=float(stat), p=float(stats.chi2.sf(max(stat, 0.0), df=1)), valid=True)
    return out


def _matched_confidence_gap(conf, s, err, n_bins):
    edges = np.quantile(conf, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    bins = []
    lo_err_w, hi_err_w, n_lo, n_hi = 0.0, 0.0, 0, 0
    for i in range(n_bins):
        m = (conf >= edges[i]) & (conf < edges[i + 1])
        if m.sum() < 4:
            continue
        med = np.median(s[m])
        low = m & (s <= med)
        high = m & (s > med)
        if low.sum() == 0 or high.sum() == 0:
            continue
        el, eh = float(err[low].mean()), float(err[high].mean())
        bins.append({"bin": i, "n": int(m.sum()),
                     "err_low_s": el, "err_high_s": eh, "gap": el - eh})
        lo_err_w += err[low].sum(); n_lo += low.sum()
        hi_err_w += err[high].sum(); n_hi += high.sum()
    pooled_gap = (lo_err_w / max(n_lo, 1)) - (hi_err_w / max(n_hi, 1))
    return {"per_bin": bins, "pooled_low_s_err": lo_err_w / max(n_lo, 1),
            "pooled_high_s_err": hi_err_w / max(n_hi, 1), "pooled_gap": float(pooled_gap)}


def _bootstrap_units(conf, s, err, n_boot, k, n_bins, seed):
    """Bootstrap AUDIT UNITS for delta-AUC, matched gap, corr(s,err) (plan §11.3)."""
    rng = np.random.default_rng(seed)
    n = len(err)
    deltas, gaps, corrs = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        c, ss, e = conf[idx], s[idx], err[idx]
        if e.sum() >= k and (1 - e).sum() >= k:
            deltas.append(_incremental_auc(c, ss, e, seed=0, k=k)[2])
        else:
            deltas.append(np.nan)
        gaps.append(_matched_confidence_gap(c, ss, e, n_bins)["pooled_gap"])
        corrs.append(float(np.corrcoef(ss, e)[0, 1]) if e.std() > 0 else np.nan)
    finite = np.asarray([v for v in deltas if np.isfinite(v)], dtype=float)
    return {"delta_auc": _ci(deltas), "delta_auc_se": float(finite.std()) if finite.size else float("nan"),
            "matched_gap": _ci(gaps), "corr_s_error": _ci(corrs),
            "valid_frac": float(finite.size / n_boot)}


def _power_equivalence(delta_ci, delta_se, power=0.80, alpha=0.05):
    if not np.isfinite(delta_se):
        return {"mde": float("nan"), "equiv_margin": float("nan"), "power": power, "alpha": alpha}
    mde = float((stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)) * delta_se)
    return {"mde": mde, "equiv_margin": float(max(abs(delta_ci["lo"]), abs(delta_ci["hi"]))),
            "power": power, "alpha": alpha}


def _safe_corr(a, b) -> float:
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def run_experiment(config: dict) -> dict:
    repro = config["reproducibility"]
    set_all_seeds(repro["global_seed"], deterministic=repro["deterministic_algorithms"])
    cfg = config["decomposition"]
    bank = config["birads_concept_bank"]
    manifest = CohortManifest.load(PROJECT_ROOT / cfg["cohort_manifest"])

    print(f"backbone: {cfg['backbone']}")
    model = _load_model(cfg["backbone"], config)
    embed_fn = lambda texts: cd.encode_texts(model, texts)
    directions = cd.build_text_directions(cd.concept_prompts_from_config(bank), embed_fn)
    B, _ = cd.stack_basis(directions, orthonormalize=True)
    C = cfg["probe_C"]

    # --- BUSI audit units + out-of-fold case features (primary) ---
    se = _load_site_embeddings(model, cfg["backbone"], "busi", config, manifest)
    Zu, yu, foldu = _aggregate_audit_units(se)
    err, conf, s = _out_of_fold_features(Zu, yu, foldu, B, C)
    print(f"  BUSI: {se.Z.shape[0]} images -> {len(yu)} audit units; "
          f"OOF errors={int(err.sum())} base_err={err.mean():.3f}")

    # seed-averaged incremental CV-AUC on the OOF features
    a0s, a1s, deltas = [], [], []
    for sd in range(cfg["cv_seeds"]):
        a0, a1, dd = _incremental_auc(conf, s, err, sd, cfg["meta_cv_folds"])
        a0s.append(a0); a1s.append(a1); deltas.append(dd)
    auc_conf, auc_both = float(np.nanmean(a0s)), float(np.nanmean(a1s))
    delta_mean, delta_std = float(np.nanmean(deltas)), float(np.nanstd(deltas))

    boot = _bootstrap_units(conf, s, err, cfg["n_boot"], cfg["meta_cv_folds"],
                            cfg["n_conf_bins"], seed=cfg["boot_seed"])
    lr = _lr_test(conf, s, err)
    matched = _matched_confidence_gap(conf, s, err, cfg["n_conf_bins"])
    power = _power_equivalence(boot["delta_auc"], boot["delta_auc_se"])

    # VERDICT: bootstrap delta-AUC CI strictly above 0 AND a VALID LR test p<0.05.
    passed = bool(np.isfinite(boot["delta_auc"]["lo"]) and boot["delta_auc"]["lo"] > 0
                  and lr["valid"] and lr["p"] < 0.05)

    in_dist = {
        "n_images": int(se.Z.shape[0]), "n_audit_units": int(len(yu)),
        "n_errors": int(err.sum()), "base_error": float(err.mean()),
        "corr_s_error": _safe_corr(s, err), "corr_conf_error": _safe_corr(conf, err),
        "corr_s_error_ci": boot["corr_s_error"],
        "auc_conf": auc_conf, "auc_conf_plus_s": auc_both,
        "delta_auc_mean": delta_mean, "delta_auc_std": delta_std,
        "delta_auc_ci": boot["delta_auc"], "delta_auc_se": boot["delta_auc_se"],
        "delta_auc_valid_frac": boot["valid_frac"],
        "mde_80": power["mde"], "equiv_margin": power["equiv_margin"],
        "lr_test": lr,
        "matched_confidence": matched, "matched_gap_ci": boot["matched_gap"],
        "cv_seeds": cfg["cv_seeds"], "n_boot": cfg["n_boot"],
        "VERDICT_s_adds_value": passed,
    }

    # --- cross-site fragility (RQ5b): EXPLORATORY, image-level + GROUP bootstrap ---
    # External sites keep native/subject groups that can be MIXED-label (e.g. a
    # BUS-UCLM subject with both benign and malignant scans), so we do NOT
    # aggregate them to single-label units; we keep image-level cases and resample
    # GROUPS (cluster bootstrap) for the uncertainty (plan §11.4).
    w_full, b_full = _fit_probe(Zu, yu, C)   # one BUSI probe applied to externals
    transfer = {}
    for site in cfg.get("sites_transfer", []):
        try:
            sx = _load_site_embeddings(model, cfg["backbone"], site, config, manifest)
        except (KeyError, FileNotFoundError) as e:
            print(f"  [skip] {site}: {e}")
            continue
        ex, cx, sxs, _ = _case_features(sx.Z, sx.y, w_full, b_full, B)
        m = _matched_confidence_gap(cx, sxs, ex, cfg["n_conf_bins"])
        rows_by_group = defaultdict(list)
        for i, g in enumerate(sx.group_ids):
            rows_by_group[g].append(i)
        uniq = np.array(sorted(rows_by_group))
        rng = np.random.default_rng(cfg["boot_seed"])
        gaps = []
        for _ in range(cfg["n_boot_transfer"]):
            sel = rng.choice(uniq, size=len(uniq), replace=True)
            idx = np.fromiter((i for g in sel for i in rows_by_group[g]), dtype=int)
            gaps.append(_matched_confidence_gap(cx[idx], sxs[idx], ex[idx],
                                                cfg["n_conf_bins"])["pooled_gap"])
        gci = _ci(gaps)
        transfer[site] = {"n_images": int(len(sx.y)), "n_groups": int(sx.n_groups),
                          "base_error": float(ex.mean()),
                          "pooled_low_s_err": m["pooled_low_s_err"],
                          "pooled_high_s_err": m["pooled_high_s_err"],
                          "pooled_gap": m["pooled_gap"], "pooled_gap_ci": gci,
                          "exploratory": True}
        print(f"  transfer {site:<10} n={len(sx.y)} g={sx.n_groups} "
              f"base_err={ex.mean():.3f} gap={m['pooled_gap']:+.3f} "
              f"CI[{gci['lo']:+.3f},{gci['hi']:+.3f}] (exploratory)")

    return {"backbone": cfg["backbone"], "in_distribution": in_dist, "transfer": transfer}


def format_summary(results: dict, config: dict) -> list[str]:
    d = results["in_distribution"]
    da, cse = d["delta_auc_ci"], d["corr_s_error_ci"]
    mg, mgci = d["matched_confidence"], d["matched_gap_ci"]
    lr = d["lr_test"]
    L = [
        "=" * 72,
        f"E4 (WP5) — per-case test (RQ5), audit-unit + out-of-fold   backbone={results['backbone']}",
        "=" * 72,
        f"BUSI: {d['n_images']} images -> {d['n_audit_units']} audit units; "
        f"OOF errors={d['n_errors']} base error={d['base_error']:.3f}",
        f"(out-of-fold over 5 canonical group-disjoint folds; {d['n_boot']} audit-unit bootstraps)",
        "",
        "Does clinical share s(z) predict error BEYOND confidence?",
        f"  corr(s,error)   = {d['corr_s_error']:+.3f}  95%CI[{cse['lo']:+.3f},{cse['hi']:+.3f}]",
        f"  CV-AUC(conf)    = {d['auc_conf']:.3f}",
        f"  CV-AUC(conf+s)  = {d['auc_conf_plus_s']:.3f}",
        f"  delta-AUC       = {d['delta_auc_mean']:+.3f} +/- {d['delta_auc_std']:.3f}"
        f"   95%CI[{da['lo']:+.3f},{da['hi']:+.3f}]  (valid boot frac {d['delta_auc_valid_frac']:.2f})",
        f"  LR test (MLE, df={lr['df']}): stat={lr['stat'] if np.isfinite(lr['stat']) else float('nan'):.3f}"
        f"  p={lr['p'] if np.isfinite(lr['p']) else float('nan'):.4f}  "
        f"valid={lr['valid']}{('  ['+lr['reason']+']') if lr['reason'] else ''}",
        f"    ll(conf)={lr['ll0']:.2f}  ll(conf+s)={lr['ll1']:.2f}",
        f"  power: min detectable delta-AUC @80% = {d.get('mde_80', float('nan')):.3f}; "
        f"equivalence margin = {d.get('equiv_margin', float('nan')):.3f}",
        "  (null = 'no DETECTABLE gain at this n', not 'provably zero')",
        "",
        "Matched-confidence stratification (error rate by clinical share):",
        f"  low-s err={mg['pooled_low_s_err']:.3f}  high-s err={mg['pooled_high_s_err']:.3f}"
        f"  pooled gap={mg['pooled_gap']:+.3f}  95%CI[{mgci['lo']:+.3f},{mgci['hi']:+.3f}]",
        "",
        f">>> VERDICT: s(z) adds value beyond confidence?  {'YES' if d['VERDICT_s_adds_value'] else 'NO'}",
        "    (PASS = bootstrap delta-AUC CI lower bound > 0 AND VALID unpenalized LR p < 0.05;",
        "     NO => 'no detectable incremental value' — keep model-level RQ2-RQ4 from E2/E3.)",
    ]
    if results["transfer"]:
        L += ["", "Cross-site fragility (EXPLORATORY; BUSI probe -> external audit units):"]
        for site, t in results["transfer"].items():
            ci = t.get("pooled_gap_ci", {})
            L.append(f"  {site:<10} n={t['n_images']} g={t['n_groups']} "
                     f"base_err={t['base_error']:.3f}"
                     f"  gap={t['pooled_gap']:+.3f}  95%CI[{ci.get('lo', float('nan')):+.3f},"
                     f"{ci.get('hi', float('nan')):+.3f}]")
    return L


def main():
    config = load_experiment_config(EXPERIMENT_NAME)
    results = run_experiment(config)
    summary = format_summary(results, config)
    for line in summary:
        print(line)
    save_results(EXPERIMENT_NAME, results, config, summary_lines=summary)


if __name__ == "__main__":
    main()
