"""Data-free tests for the WP4 global core test (src/evaluation/global_inference.py).

Small synthetic embedding geometry — no data, no GPU. Guards the WP4 inferential
invariants (plan §10.3, §15.4): exhaustive placebo subset count/rank; the mean
projector P_bar reproduces the exact mean placebo rho; the core test is order
invariant; and the fixed placebo bank yields a reference RANK, never a p-value
(STAT-01), with the verdict an explicit conjunction of four separately stored criteria.

Run:  python tests/test_global_inference.py   (or: python -m pytest tests/)
"""

from __future__ import annotations

import math
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.decomposition import rho
from src.evaluation.global_inference import (
    global_core_test,
    global_random_null,
    audit_verdict,
    placebo_bank,
    pooled_label_permutation,
    rho_from_projector,
)

RNG = np.random.default_rng(0)


def _pool(P=6, d=10):
    M = RNG.standard_normal((P, d))
    return M / np.linalg.norm(M, axis=1, keepdims=True)


def _sites(S=3, d=10):
    W = RNG.standard_normal((S, d))
    return W / np.linalg.norm(W, axis=1, keepdims=True)


def test_placebo_bank_enumerates_all_full_rank_subsets():
    P, m = 6, 2
    bank = placebo_bank(_sites(), _pool(P=P), m)
    assert bank["n_subsets_total"] == math.comb(P, m) == 15
    assert bank["n_full_rank"] == 15      # random real dirs are full rank a.s.
    assert bank["n_dropped"] == 0


def test_P_bar_gives_exact_mean_placebo_rho():
    P, m, d = 6, 2, 10
    pool = _pool(P=P, d=d)
    W = _sites(S=1, d=d)
    w = W[0]
    bank = placebo_bank(W, pool, m)
    # brute-force mean of rho(w, subset) over every full-rank subset
    vals = []
    for sel in combinations(range(P), m):
        Q, _ = np.linalg.qr(pool[list(sel)].T)
        vals.append(rho(w, Q))
    brute = float(np.mean(vals))
    assert np.isclose(rho_from_projector(w, bank["P_bar"]), brute, atol=1e-10)


def test_placebo_null_mean_matches_site_average_of_P_bar():
    # equal-site placebo null mean == mean_s rho_from_projector(w_s, P_bar)
    pool, W = _pool(P=6, d=10), _sites(S=4, d=10)
    bank = placebo_bank(W, pool, 2)
    per_site = [rho_from_projector(w, bank["P_bar"]) for w in W]
    assert np.isclose(bank["null_means"].mean(), float(np.mean(per_site)), atol=1e-10)


def test_core_test_is_site_order_invariant():
    pool, W = _pool(P=6, d=10), _sites(S=4, d=10)
    B, _ = np.linalg.qr(RNG.standard_normal((10, 2)))
    a = global_core_test(W, B, pool, n_random=500, seed=7)
    b = global_core_test(W[[3, 1, 0, 2]], B, pool, n_random=500, seed=7)
    assert np.isclose(a["T_clin"], b["T_clin"])
    assert np.isclose(a["placebo_tail_fraction"], b["placebo_tail_fraction"])
    assert np.isclose(a["p_random"], b["p_random"])   # same seed, mean is symmetric


def test_placebo_is_a_reference_rank_not_a_p_value():
    """STAT-01: the fixed bank yields a rank; no p_core is formed from it."""
    pool, W = _pool(P=6, d=10), _sites(S=3, d=10)
    B, _ = np.linalg.qr(RNG.standard_normal((10, 2)))
    r = global_core_test(W, B, pool, n_random=500, seed=1)
    assert "p_core" not in r                       # the mixed quantity is gone
    assert "placebo_tail_fraction" in r and "placebo_rank" in r
    n = r["n_placebo_reference"]
    assert r["placebo_rank"] >= 1
    # the tail fraction is the smoothed rank, exactly as documented
    assert np.isclose(r["placebo_tail_fraction"], r["placebo_rank"] / (n + 1))
    assert r["p_placebo"] == r["placebo_tail_fraction"]   # deprecated alias only


def test_random_null_is_seed_reproducible():
    W = _sites(S=3, d=10)
    a = global_random_null(W, 2, n_trials=300, seed=42)
    b = global_random_null(W, 2, n_trials=300, seed=42)
    assert np.array_equal(a, b)


def test_audit_verdict_requires_all_four_criteria():
    """Every criterion is necessary; dropping any one changes a verdict."""
    ok = dict(holm_reject_random=True, placebo_tail_fraction=0.001,
              mean_excess_placebo=0.5, excess_ci_low=0.1, m=2, d=10, k=2.0)
    assert audit_verdict(**ok)["expressible"]              # gate = 2*2/10 = 0.4
    for override, failed_flag in [
        (dict(holm_reject_random=False), "holm_reject_random"),
        (dict(placebo_tail_fraction=0.30), "placebo_extreme"),
        (dict(excess_ci_low=-0.01), "ci_excludes_zero"),
        (dict(mean_excess_placebo=0.3), "passes_gate"),
    ]:
        v = audit_verdict(**{**ok, **override})
        assert not v["expressible"], override
        assert not v[failed_flag]                          # and we can say which


def test_audit_verdict_placebo_criterion_is_load_bearing():
    """The UniMed-CLIP case: strong random null + gate, but a weak placebo rank.

    Guards the STAT-01 trap -- relabelling the placebo quantity must not remove it
    from the rule, or this configuration would flip to expressible.
    """
    v = audit_verdict(holm_reject_random=True, placebo_tail_fraction=0.118,
                      mean_excess_placebo=0.021, excess_ci_low=0.004,
                      m=5, d=512, k=2.0)
    assert v["holm_reject_random"] and v["passes_gate"] and v["ci_excludes_zero"]
    assert not v["placebo_extreme"] and not v["expressible"]


def test_grounded_direction_beats_placebo_null():
    # a w lying largely in B should give T_clin above the placebo null mean.
    d = 12
    B, _ = np.linalg.qr(RNG.standard_normal((d, 3)))
    pool = _pool(P=8, d=d)
    w = B @ np.array([1.0, 0.5, 0.3]) + 0.05 * RNG.standard_normal(d)
    r = global_core_test(w[None, :], B, pool, n_random=500, seed=3)
    assert r["mean_excess_placebo"] > 0 and r["placebo_tail_fraction"] < 0.05


# --- pooled-label permutation (SECONDARY sensitivity; declaration doc 20260723) ---
def _pooled_setup(d=24, m=3, n_ctrl=7, align=1.0, seed=0):
    """Pool of (m clinical + n_ctrl control) unit axes and one site direction.

    ``align`` scales how strongly w loads on the clinical axes: 1.0 = clinical
    axes carry w, 0.0 = w is built from the control axes instead.
    """
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m + n_ctrl, d))
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    src = A[:m] if align > 0 else A[m:m + m]
    w = (src * np.linspace(1.0, 0.6, m)[:, None]).sum(axis=0)
    w += 0.02 * rng.standard_normal(d)
    return A, w[None, :], tuple(range(m))


def test_pooled_permutation_enumerates_and_includes_the_observed_subset():
    A, W, idx = _pooled_setup()
    r = pooled_label_permutation(W, A, idx, m=3)
    assert r["n_reference"] == r["n_subsets_total"] == math.comb(10, 3)
    assert r["rank"] >= 1                      # the observed subset counts itself
    assert r["p_pooled"] >= 1.0 / r["n_reference"]
    assert abs(r["excess_over_pool_mean"] - (r["T_clin"] - r["null_mean"])) < 1e-12


def test_pooled_permutation_ranks_aligned_clinical_axes_first():
    A, W, idx = _pooled_setup(align=1.0)
    r = pooled_label_permutation(W, A, idx, m=3)
    assert r["rank"] == 1                      # strictly the best subset
    assert r["p_pooled"] == 1.0 / r["n_reference"]
    assert r["T_clin"] == r["null_max"]
    assert r["n_outranking"] == 0              # nothing beats it, so no composition
    assert all(v == 0 for v in r["outrank_by_n_clinical"].values())


def test_pooled_permutation_composition_bookkeeping():
    """The k-clinical histogram must partition the reference and the outrankers."""
    m, n_ctrl = 3, 7
    A, W, idx = _pooled_setup(m=m, n_ctrl=n_ctrl)
    r = pooled_label_permutation(W, A, idx, m=m)
    totals = r["reference_by_n_clinical"]
    # every subset keeping k clinical axes = C(m,k) * C(n_ctrl, m-k)
    for k in range(m + 1):
        assert totals[k] == math.comb(m, k) * math.comb(n_ctrl, m - k)
    assert sum(totals.values()) == r["n_reference"]
    assert totals[m] == 1                       # only the clinical subset itself
    # outrankers are exactly rank minus the observed subset, and none keeps all m
    assert sum(r["outrank_by_n_clinical"].values()) == r["n_outranking"] == r["rank"] - 1
    assert r["outrank_by_n_clinical"][m] == 0
    assert r["n_outranking_pure_control"] == r["outrank_by_n_clinical"][0]


def test_pooled_permutation_separates_control_wins_from_weak_axis():
    """k=0 mass means controls beat the clinical set; high-k mass means one weak axis."""
    A, W, idx = _pooled_setup(align=0.0)        # w built from control axes
    r = pooled_label_permutation(W, A, idx, m=3)
    assert r["n_outranking_pure_control"] > 0   # genuine control subsets win here
    # the best achievable T should rise as clinical axes are swapped out
    best = r["max_T_by_n_clinical"]
    assert best[0] > best[3]


def test_pooled_permutation_does_not_reject_when_controls_carry_the_signal():
    A, W, idx = _pooled_setup(align=0.0)       # w built from control axes
    r = pooled_label_permutation(W, A, idx, m=3)
    assert r["p_pooled"] > 0.05
    assert r["excess_over_pool_mean"] < 0


def test_pooled_permutation_is_invariant_to_pool_row_order():
    A, W, idx = _pooled_setup()
    base = pooled_label_permutation(W, A, idx, m=3)
    perm = np.random.default_rng(5).permutation(len(A))
    A2 = A[perm]
    idx2 = tuple(int(np.where(perm == i)[0][0]) for i in idx)
    r2 = pooled_label_permutation(W, A2, idx2, m=3)
    assert abs(base["p_pooled"] - r2["p_pooled"]) < 1e-12
    assert abs(base["T_clin"] - r2["T_clin"]) < 1e-12


def test_pooled_permutation_p_is_uniform_under_exchangeability():
    """The validity claim: with every axis drawn identically, p ~ Uniform(0,1).

    This is what makes the test an exact randomisation test rather than an
    empirical rank — under a genuinely exchangeable pool the label carries no
    information, so the observed subset's rank is uniform on the enumeration.
    """
    d, m, K, trials = 16, 3, 8, 400
    ps = []
    rng = np.random.default_rng(7)
    for _ in range(trials):
        A = rng.standard_normal((K, d))
        A /= np.linalg.norm(A, axis=1, keepdims=True)
        w = rng.standard_normal((1, d))        # w independent of the labelling
        ps.append(pooled_label_permutation(w, A, tuple(range(m)), m=m)["p_pooled"])
    ps = np.array(ps)
    assert abs(ps.mean() - 0.5) < 0.06, f"mean p = {ps.mean():.3f}, expected ~0.5"
    # a valid test must not over-reject: P(p <= .05) should sit near .05
    assert (ps <= 0.05).mean() < 0.12, f"rejection rate {(ps <= 0.05).mean():.3f}"


def test_pooled_permutation_rejects_a_bad_clinical_index():
    A, W, _ = _pooled_setup()
    for bad in [(0, 1), (0, 1, 2, 3), (0, 1, 99)]:
        try:
            pooled_label_permutation(W, A, bad, m=3)
            assert False, "expected ValueError"
        except ValueError:
            pass


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed")
