"""WP4 global backbone-level core test for BI-RADS-subspace expressibility.

Replaces the manuscript's *verbal* aggregation of per-site significance stars with
one explicit, equal-site-weighted statistic per backbone and a synchronized pair
of nulls (plan §10.3). For backbone ``b`` over its ``S`` sites with group-weighted
decision directions ``w_s`` and concept basis ``B``:

    T_clin = mean_s rho(w_s, B)              # equal site weight

evaluated against two nulls that share the backbone's embedding geometry:

* **random null** — draw one random ``m``-subspace and score it against *every*
  site direction before averaging (retains the dependence induced by the shared
  representation space); 20,000 draws;
* **placebo null** — enumerate *every* full-rank ``m``-subset of the ``P`` placebo
  text directions (C(16,5)=4368) and score the same subspace across all sites
  before averaging.

Because the core claim must clear **both** nulls, the reported p-value is the
intersection-union ``p_core = max(p_random, p_placebo)``. Holm–Bonferroni across
the backbone family, plus the locked magnitude gate ``2m/d`` on the equal-site
mean placebo excess, gives the confirmatory-style verdict.

The exhaustive placebo pass also returns the **mean placebo projector** ``P_bar``:
for any direction ``w`` the exact mean placebo rho is ``wᵀ P_bar w / wᵀw`` — this
removes the nested Monte-Carlo placebo loop from the cluster bootstrap (plan §10.2).

Pure NumPy; unit-tested in ``tests/test_global_inference.py``.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np

from src.evaluation.decomposition import rho

_TOL = 1e-8


def _np(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def rho_from_projector(w: np.ndarray, P_bar: np.ndarray) -> float:
    """Exact mean placebo rho for direction ``w`` via a (mean) projector ``P_bar``.

    ``P_bar = mean_j Q_j Q_jᵀ`` averages orthogonal projectors, so
    ``wᵀ P_bar w / wᵀw = mean_j rho(w, Q_j)`` exactly — no sampling.
    """
    w = _np(w)
    wn = float(w @ w)
    if wn == 0.0:
        return float("nan")
    return float((w @ (P_bar @ w)) / wn)


def _rho_across_sites(W: np.ndarray, Q: np.ndarray, wn: np.ndarray) -> np.ndarray:
    """rho(w_s, col(Q)) for every site row of ``W`` given orthonormal ``Q`` (d,m)."""
    QtW = Q.T @ W.T                      # (m, S)
    return (QtW * QtW).sum(axis=0) / np.maximum(wn, 1e-12)


def placebo_bank(site_dirs: np.ndarray, pool: np.ndarray, m: int) -> dict:
    """Exhaustive placebo pass: equal-site mean-rho null + mean projector ``P_bar``.

    Iterates every full-rank ``m``-subset of the ``P`` pool directions exactly once,
    accumulating (a) the equal-site mean rho per subset — the global placebo null —
    and (b) ``P_bar`` for the bootstrap shortcut. Rank-deficient subsets are skipped
    and counted (a.s. none for real text directions).
    """
    W = _np(site_dirs)
    pool = _np(pool)
    P, d = pool.shape
    if P < m:
        raise ValueError(f"need at least m={m} placebo directions, got P={P}")
    wn = (W * W).sum(axis=1)

    P_bar = np.zeros((d, d), dtype=np.float64)
    null_means: list[float] = []
    dropped = 0
    for sel in combinations(range(P), m):
        Q, R = np.linalg.qr(pool[list(sel)].T)
        if np.count_nonzero(np.abs(np.diag(R)) > _TOL) < m:
            dropped += 1
            continue
        P_bar += Q @ Q.T
        null_means.append(float(_rho_across_sites(W, Q, wn).mean()))
    k = len(null_means)
    if k == 0:
        raise ValueError("no full-rank placebo subsets")
    P_bar /= k
    return {
        "P_bar": P_bar,
        "null_means": np.array(null_means),
        "n_subsets_total": math.comb(P, m),
        "n_full_rank": k,
        "n_dropped": dropped,
    }


def mean_projector(pool: np.ndarray, m: int) -> np.ndarray:
    """Mean orthogonal projector ``P_bar`` over all full-rank ``m``-subsets (pure).

    Site-independent, so it is precomputed once per (backbone, pool) and reused for
    the exact per-case mean placebo/malignancy rho in the cluster bootstrap
    (``rho_from_projector``), avoiding the nested Monte-Carlo loop.
    """
    pool = _np(pool)
    P, d = pool.shape
    if P < m:
        raise ValueError(f"need at least m={m} directions, got P={P}")
    P_bar = np.zeros((d, d), dtype=np.float64)
    k = 0
    for sel in combinations(range(P), m):
        Q, R = np.linalg.qr(pool[list(sel)].T)
        if np.count_nonzero(np.abs(np.diag(R)) > _TOL) < m:
            continue
        P_bar += Q @ Q.T
        k += 1
    if k == 0:
        raise ValueError("no full-rank subsets")
    return P_bar / k


def global_random_null(site_dirs: np.ndarray, m: int, n_trials: int = 20000,
                       seed: int = 0) -> np.ndarray:
    """Equal-site mean-rho null from ``n_trials`` random ``m``-subspaces.

    Each trial scores ONE random subspace across every site before averaging, so
    the null retains the cross-site dependence from the shared embedding space.
    """
    W = _np(site_dirs)
    d = W.shape[1]
    wn = (W * W).sum(axis=1)
    rng = np.random.default_rng(seed)
    out = np.empty(n_trials, dtype=np.float64)
    for i in range(n_trials):
        Q, _ = np.linalg.qr(rng.standard_normal((d, m)))
        out[i] = float(_rho_across_sites(W, Q, wn).mean())
    return out


def _perm_p(null: np.ndarray, stat: float) -> float:
    """One-sided upper-tail p-value with the conservative +1 correction.

    For reference sets that are *sampled* (random null) or drawn from a pool that
    excludes the observed statistic (placebo bank). For an exhaustive enumeration
    that **contains** the observed subset, use :func:`pooled_label_permutation`,
    whose denominator needs no smoothing.
    """
    return float((1 + int((null >= stat).sum())) / (len(null) + 1))


def pooled_label_permutation(site_dirs: np.ndarray, pool: np.ndarray,
                             clinical_idx, m: int) -> dict:
    """Exact label-randomisation test over a **pooled** axis bank (secondary).

    Addresses the exchangeability objection to the fixed-bank placebo rank: rather
    than compare the clinical subspace with subsets of a *separately curated* pool,
    pool the clinical axes with a richness-matched control bank and enumerate every
    ``m``-subset of the union. Under the null "the clinical/control label is
    uninformative about alignment with ``w``", the observed subset is exchangeable
    with the reference subsets, so the rank is an exact randomisation p-value.

    The observed clinical subset **is** one of the enumerated subsets and is scored
    inside the same loop by the same QR construction, so observed and reference
    statistics are identically constructed by design (``rho`` depends only on the
    span, so the orthonormalisation route is immaterial). The denominator is the
    full enumeration and takes **no** ``+1`` smoothing — that correction exists for
    sampled reference sets that exclude the observed value.

    ``clinical_idx`` are the pool rows forming the observed subset; ``pool`` stacks
    the clinical rows and the control rows as ``(K, d)``. See
    ``docs/20260723-pooled-permutation-declaration.md`` for the declared scope: this
    is a **secondary** sensitivity and does not enter the confirmatory family.
    """
    W = _np(site_dirs)
    pool = _np(pool)
    K, _ = pool.shape
    clinical_idx = tuple(sorted(int(i) for i in clinical_idx))
    if len(clinical_idx) != m:
        raise ValueError(f"clinical_idx has {len(clinical_idx)} axes, expected m={m}")
    if K < m or not set(clinical_idx) <= set(range(K)):
        raise ValueError("clinical_idx must index rows of a pool with K >= m rows")
    wn = (W * W).sum(axis=1)

    clin_set = set(clinical_idx)
    stats: list[float] = []
    comps: list[int] = []                        # clinical axes retained per subset
    t_obs = None
    dropped = 0
    for sel in combinations(range(K), m):
        Q, R = np.linalg.qr(pool[list(sel)].T)
        if np.count_nonzero(np.abs(np.diag(R)) > _TOL) < m:
            dropped += 1
            continue
        t = float(_rho_across_sites(W, Q, wn).mean())
        stats.append(t)
        comps.append(len(clin_set.intersection(sel)))
        if sel == clinical_idx:
            t_obs = t
    if t_obs is None:
        raise ValueError("the clinical subset was rank-deficient or not enumerated")

    null = np.array(stats)
    comp = np.array(comps)
    n_ge = int((null >= t_obs).sum())             # includes the observed subset

    # Composition of the subsets that outrank the observed one. A subset retaining
    # k clinical axes is not a "control" result: only k=0 subsets are pure control.
    # Reported so a reader can tell "controls beat the clinical set" (mass at k=0)
    # apart from "the labelled set is not the best subset of the pooled bank"
    # (mass at high k, i.e. a weak individual axis).
    outrank = (null >= t_obs) & (comp < m)        # strictly-other subsets
    hist = {int(k): int(((comp == k) & outrank).sum()) for k in range(m + 1)}
    totals = {int(k): int((comp == k).sum()) for k in range(m + 1)}
    best_by_k = {int(k): (float(null[comp == k].max()) if (comp == k).any() else None)
                 for k in range(m + 1)}
    return {
        "T_clin": t_obs,
        "p_pooled": n_ge / len(null),             # exact; floor 1/len(null)
        "rank": n_ge,                             # 1 == strictly best subset
        "n_reference": len(null),
        "n_subsets_total": math.comb(K, m),
        "n_dropped": dropped,
        "pool_size": K,
        "null_mean": float(null.mean()),
        "null_max": float(null.max()),
        "excess_over_pool_mean": t_obs - float(null.mean()),
        "n_outranking": int(outrank.sum()),
        "outrank_by_n_clinical": hist,            # k -> # outranking subsets keeping k
        "reference_by_n_clinical": totals,        # k -> # subsets in the reference
        "max_T_by_n_clinical": best_by_k,         # best T achievable keeping k
        "n_outranking_pure_control": hist[0],     # k=0: genuine control wins
    }


def global_core_test(site_dirs: np.ndarray, B: np.ndarray, placebo_pool: np.ndarray,
                     n_random: int = 20000, seed: int = 0) -> dict:
    """Equal-site core statistic ``T_clin`` against the two references.

    **The two references are not the same kind of object** (STAT-01):

    * the **random null** is a genuine Monte-Carlo null -- subspaces are *sampled*
      from a stated distribution, so ``p_random`` is a calibrated frequentist
      p-value and is the quantity the confirmatory Holm family corrects;
    * the **placebo bank** is a fixed, separately curated set of non-diagnostic
      axes. Its subsets are not exchangeable with a hand-selected clinical axis
      set, so its upper-tail position is an **empirical reference rank**, reported
      as ``placebo_tail_fraction`` / ``placebo_rank``, *not* a calibrated p-value.
      The ``+1`` in the fraction is rank smoothing, not p-value calibration.

    Both remain *required* criteria of the verdict (:func:`audit_verdict`) -- the
    reframe changes what the placebo quantity is called, never whether it must be
    cleared. See ``docs/20260723-stat01-reframe.md``.

    ``p_placebo`` is retained as a deprecated alias of ``placebo_tail_fraction``
    for existing consumers (``compute_reviewer_addenda``); no new code should read
    it, and nothing forms ``max(p_random, p_placebo)`` any more.
    """
    W = _np(site_dirs)
    B = _np(B)
    m = B.shape[1]
    d = B.shape[0]

    rho_s = np.array([rho(w, B) for w in W], dtype=np.float64)
    T = float(rho_s.mean())

    rand_null = global_random_null(W, m, n_trials=n_random, seed=seed)
    bank = placebo_bank(W, placebo_pool, m)
    plac_null = bank["null_means"]

    p_rand = _perm_p(rand_null, T)                     # sampled null -> a p-value
    plac_tail = _perm_p(plac_null, T)                  # fixed bank -> a rank
    return {
        "T_clin": T,
        "rho_by_site": rho_s.tolist(),
        "n_sites": int(W.shape[0]),
        "random_null_mean": float(rand_null.mean()),
        "placebo_null_mean": float(plac_null.mean()),
        "mean_excess_random": T - float(rand_null.mean()),
        "mean_excess_placebo": T - float(plac_null.mean()),
        "p_random": p_rand,
        # --- fixed-bank reference rank (NOT a calibrated p-value) ---
        "placebo_tail_fraction": plac_tail,
        "placebo_rank": 1 + int((plac_null >= T).sum()),
        "n_placebo_reference": int(len(plac_null)),
        "p_placebo": plac_tail,        # deprecated alias; see docstring
        "P_bar": bank["P_bar"],
        "placebo_meta": {k: bank[k] for k in
                         ("n_subsets_total", "n_full_rank", "n_dropped")},
        "m": m,
        "d": d,
    }


def audit_verdict(holm_reject_random: bool, placebo_tail_fraction: float,
                  mean_excess_placebo: float, excess_ci_low, m: int, d: int,
                  k: float = 2.0, tail_alpha: float = 0.05) -> dict:
    """Expressibility verdict as an explicit conjunction of four criteria.

    Replaces the ``Holm(p_core)`` construction, whose single number mixed a
    calibrated p-value with a fixed-bank rank. Every criterion is stored
    separately so a reader can see which one a checkpoint fails:

    1. ``holm_reject_random`` -- Holm-adjusted ``p_random`` clears alpha (the
       confirmatory family; the only calibrated p-value here);
    2. ``placebo_extreme`` -- ``T_clin`` sits in the upper ``tail_alpha`` of the
       fixed placebo bank (an empirical reference rank, not a p-value);
    3. ``ci_excludes_zero`` -- the cluster-bootstrap CI for the equal-site mean
       placebo excess is strictly above zero (uncertainty in ``w`` and cohort);
    4. ``passes_gate`` -- that excess exceeds the locked magnitude gate ``k*m/d``.

    The conjunction is deliberate and load-bearing: criterion 2 is the *only* one
    separating UniMed-CLIP from the expressible set, so demoting the placebo to a
    purely descriptive quantity would change the paper's headline. Relabelling
    what the placebo quantity *is* must not remove it from the rule.
    """
    delta_min = float(k) * m / d
    passes_gate = bool(mean_excess_placebo > delta_min)
    placebo_extreme = bool(placebo_tail_fraction < tail_alpha)
    ci_excludes_zero = bool(excess_ci_low is not None and excess_ci_low > 0)
    return {
        "holm_reject_random": bool(holm_reject_random),
        "placebo_extreme": placebo_extreme,
        "placebo_tail_fraction": float(placebo_tail_fraction),
        "ci_excludes_zero": ci_excludes_zero,
        "delta_min": delta_min,
        "passes_gate": passes_gate,
        "expressible": bool(holm_reject_random and placebo_extreme
                            and ci_excludes_zero and passes_gate),
        "k": float(k),
        "tail_alpha": float(tail_alpha),
    }
