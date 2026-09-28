"""Unit tests for the concept-direction + decomposition formalism (proposal §5).

Runs entirely on synthetic embeddings — no model, no data, no GPU. Run either as
    python -m pytest tests/test_concept_geometry.py
or directly:
    python tests/test_concept_geometry.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation import concept_directions as cd
from src.evaluation import decomposition as dec
from src.evaluation import retrieval as ret
from src.evaluation import validity_stats as vs

RNG = np.random.default_rng(42)
D, M, N = 64, 5, 300


def _setup():
    w = RNG.standard_normal(D)
    b = 0.4
    B, _ = np.linalg.qr(RNG.standard_normal((D, M)))  # orthonormal concept basis
    Z = RNG.standard_normal((N, D))
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)  # unit-sphere embeddings
    return w, b, B, Z


# --------------------------------------------------------------------------- #
# decomposition.py
# --------------------------------------------------------------------------- #
def test_decomposition_is_additive():
    w, b, B, Z = _setup()
    r = dec.decompose(Z, w, b, B)
    # clin + resid must reconstruct the de-biased logit exactly
    assert np.allclose(r.clin + r.resid, Z @ w, atol=1e-9)
    assert np.allclose(r.logit_centered, Z @ w, atol=1e-9)
    # clinical share in [0, 1]
    assert np.all((r.clinical_share >= -1e-9) & (r.clinical_share <= 1 + 1e-9))
    # named contributions have one column per concept
    assert r.named_contributions.shape == (N, M)


def test_rho_matches_definition_and_bounds():
    w, b, B, Z = _setup()
    # orthonormal B => rho == ||B^T w||^2 / ||w||^2
    manual = float((B.T @ w) @ (B.T @ w) / (w @ w))
    assert abs(dec.rho(w, B) - manual) < 1e-9
    # full-space basis => rho == 1 and residual == 0
    full = np.eye(D)
    assert abs(dec.rho(w, full) - 1.0) < 1e-9
    r_full = dec.decompose(Z, w, b, full)
    assert np.allclose(r_full.resid, 0.0, atol=1e-8)


def test_random_subspace_null_is_about_m_over_d():
    w, _, _, _ = _setup()
    null = dec.random_subspace_rho(w, M, n_trials=300, seed=1)
    assert abs(null.mean() - M / D) < 0.02  # E[rho] ~ m/d


def test_grounding_excess_reports_null():
    w, b, B, Z = _setup()
    g = dec.grounding_excess(w, B, n_trials=200, seed=2)
    assert g["m"] == M
    assert np.isfinite(g["excess"]) and np.isfinite(g["excess_z"])
    assert abs(g["null_mean"] - M / D) < 0.03
    # empirical upper-tail p-value in (0, 1], floored at 1/(n+1)
    assert 1.0 / 201 <= g["p_value"] <= 1.0


def test_direction_excess_stats_summarises_fold_directions():
    # P1-5: held-out-fold grounding excess for several probe directions (CV folds).
    # The random null mean is w-independent (a reused scalar); the placebo null mean
    # is w-dependent and recomputed per direction from the pool.
    w, b, B, Z = _setup()
    null_mean = M / D
    Bcol = B[:, 0]
    dirs = []
    for a in (0.2, 0.5, 0.8):  # vary how much of each direction lies in C
        out = RNG.standard_normal(D)
        out -= B @ (B.T @ out)                       # component orthogonal to C
        v = a * Bcol + (1 - a) * out / np.linalg.norm(out)
        dirs.append(v)
    dirs = np.stack(dirs)
    pool = RNG.standard_normal((12, D))
    pool /= np.linalg.norm(pool, axis=1, keepdims=True)
    s = dec.direction_excess_stats(
        dirs, B, null_mean=null_mean, placebo_dirs=pool, n_trials=200, seed=7)
    assert s["n_dirs"] == 3
    for i in range(3):                               # per-direction rho matches def
        assert abs(s["rho"][i] - dec.rho(dirs[i], B)) < 1e-12
    er = np.array(s["rho"]) - null_mean
    assert abs(s["excess_rand_mean"] - er.mean()) < 1e-12
    assert abs(s["excess_rand_min"] - er.min()) < 1e-12
    assert abs(s["excess_rand_max"] - er.max()) < 1e-12
    # the placebo mean is recomputed PER direction from the pool (not one scalar),
    # matching a direct subspace_rho_null call with the same seed/trials
    for i in range(3):
        mu_i = dec.subspace_rho_null(dirs[i], pool, M, n_trials=200, seed=7).mean()
        assert abs(s["placebo_mean"][i] - mu_i) < 1e-12
        assert abs(s["excess_plac"][i] - (s["rho"][i] - mu_i)) < 1e-12
    # the per-direction placebo means genuinely differ (w-dependent null)
    assert max(s["placebo_mean"]) - min(s["placebo_mean"]) > 1e-6
    ep = np.array(s["excess_plac"])
    assert abs(s["excess_plac_mean"] - ep.mean()) < 1e-12
    # 1-D input (a single direction) is accepted; placebo keys absent if not asked
    s1 = dec.direction_excess_stats(dirs[0], B, null_mean=null_mean)
    assert s1["n_dirs"] == 1 and "excess_plac_mean" not in s1


def test_placebo_null_mean_is_w_dependent_not_reusable():
    # Regression guard for the reviewer's finding: unlike the isotropic random null
    # (E[rho]~m/d, w-independent), the placebo subset-null MEAN depends on w, so a
    # bootstrap/fold direction cannot reuse another direction's placebo mean.
    _, _, B, _ = _setup()
    m = B.shape[1]
    # a pool aligned with the first concept axes, so a w inside that block sees a
    # high placebo mean and a w on a barely-covered axis sees a low one.
    pool = np.stack(
        [B[:, 0], B[:, 1], B[:, 2]] + [RNG.standard_normal(D) for _ in range(9)]
    )
    pool /= np.linalg.norm(pool, axis=1, keepdims=True)
    w1 = B[:, 0].copy()                 # lies in the pool-aligned block
    w2 = B[:, 4].copy()                 # a concept axis the pool barely covers
    mu1 = dec.subspace_rho_null(w1, pool, m, n_trials=400, seed=11).mean()
    mu2 = dec.subspace_rho_null(w2, pool, m, n_trials=400, seed=11).mean()
    assert abs(mu1 - mu2) > 0.02        # placebo mean demonstrably moves with w
    # therefore direction_excess_stats must subtract each direction's OWN mean
    s = dec.direction_excess_stats(
        np.stack([w1, w2]), B, null_mean=m / D,
        placebo_dirs=pool, n_trials=400, seed=11)
    assert abs(s["excess_plac"][0] - (dec.rho(w1, B) - mu1)) < 1e-12
    assert abs(s["excess_plac"][1] - (dec.rho(w2, B) - mu2)) < 1e-12
    # reusing w1's mean for w2 (the old bug) would shift its excess non-trivially
    buggy_excess_w2 = dec.rho(w2, B) - mu1
    assert abs(s["excess_plac"][1] - buggy_excess_w2) > 0.02


def test_leave_one_out_placebo_excess_is_stable_and_bounded():
    # P3: dropping any single placebo axis should barely move the excess when no
    # one axis dominates the null. Build a pool of near-orthogonal placebo dirs.
    w, b, B, Z = _setup()
    rng = np.random.default_rng(7)
    pool = rng.standard_normal((8, D))
    pool /= np.linalg.norm(pool, axis=1, keepdims=True)
    names = [f"plac{j}" for j in range(8)]
    out = dec.leave_one_out_placebo_excess(w, B, pool, names=names, n_trials=200, seed=1)
    assert set(out["loo"]) == set(names) and len(out["loo"]) == 8
    # full excess lies within the LOO envelope's neighbourhood, and the envelope is
    # tight (no single placebo axis swings the verdict by a large amount)
    assert out["loo_min"] <= out["full"] + 1e-9
    assert out["full"] <= out["loo_max"] + 1e-9
    assert (out["loo_max"] - out["loo_min"]) < 0.05
    # each LOO excess equals a direct placebo_excess on that 7-axis pool
    import numpy as _np
    sub = _np.delete(pool, 3, axis=0)
    direct = dec.placebo_excess(w, B, sub, n_trials=200, seed=1)["excess"]
    assert abs(out["loo"]["plac3"] - direct) < 1e-9


def test_magnitude_gate_is_two_m_over_d_and_power_independent():
    # threshold is exactly 2 m/d and ignores the null SD (power-independent)
    g = dec.magnitude_gate(0.0, M, D)
    assert abs(g["delta_min"] - 2 * M / D) < 1e-12
    # an excess just above / below 2 m/d flips the gate
    thr = 2 * M / D
    assert dec.magnitude_gate(thr + 1e-6, M, D)["passed"]
    assert not dec.magnitude_gate(thr - 1e-6, M, D)["passed"]
    # the C5 failure mode: a tiny placebo excess that is permutation-significant
    # only because the null is tightly concentrated must NOT clear the gate
    assert not dec.magnitude_gate(0.006, m=5, d=768)["passed"]   # SigLIP BUSI-WHU-like
    # a genuine grounded cell clears it; smaller d => higher bar
    assert dec.magnitude_gate(0.058, m=5, d=512)["passed"]       # BiomedCLIP BUS-BRA-like
    assert (dec.magnitude_gate(0.0, M, 512)["delta_min"]
            > dec.magnitude_gate(0.0, M, 768)["delta_min"])
    assert dec.magnitude_gate(0.05, M, D, k=4.0)["delta_min"] == 4.0 * M / D


def test_subspace_rho_null_bounds_and_guard():
    w, _, _, _ = _setup()
    pool = np.linalg.qr(RNG.standard_normal((D, 8)))[0].T  # 8 orthonormal dirs (8, D)
    null = dec.subspace_rho_null(w, pool, M, n_trials=100, seed=3)
    assert null.shape == (100,)
    assert np.all((null >= -1e-9) & (null <= 1 + 1e-9))   # rho in [0, 1]
    # guard: fewer pool directions than m must raise
    try:
        dec.subspace_rho_null(w, pool[:3], M, n_trials=5)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_orthonormal_basis_rejects_rank_deficiency():
    # full-rank input round-trips to an orthonormal basis spanning the same space
    mat = RNG.standard_normal((D, M))
    Q = dec.orthonormal_basis(mat, "test")
    assert np.allclose(Q.T @ Q, np.eye(M), atol=1e-9)
    # a duplicated column makes the set rank-deficient: reduced QR would silently
    # return m orthonormal cols anyway, so the guard must raise instead.
    bad = mat.copy(); bad[:, -1] = bad[:, 0]
    try:
        dec.orthonormal_basis(bad, "test")
        raised = False
    except ValueError:
        raised = True
    assert raised
    # stack_basis(orthonormalize=True) inherits the guard for collinear concepts
    v = RNG.standard_normal(D)
    try:
        cd.stack_basis({"a": v, "b": 2.0 * v}, orthonormalize=True)
        raised2 = False
    except ValueError:
        raised2 = True
    assert raised2


def test_placebo_excess_separates_clinical_from_placebo():
    """w placed in the clinical subspace must exceed the placebo (text-direction)
    null; a w placed in the placebo span must NOT."""
    # disjoint orthonormal blocks: clinical (M dims) and placebo pool (8 dims)
    Q = np.linalg.qr(RNG.standard_normal((D, M + 8)))[0]
    B = Q[:, :M]                       # clinical basis (d, M)
    placebo = Q[:, M:].T              # placebo pool (8, d), orthogonal to B
    # w lying entirely in the clinical subspace
    w_clin = B @ RNG.standard_normal(M)
    pe = dec.placebo_excess(w_clin, B, placebo, n_trials=200, seed=4)
    assert pe["m"] == M and pe["n_placebo"] == 8
    assert pe["excess"] > 0.5 and pe["excess_z"] > 3        # strongly grounded vs placebo
    assert pe["p_value"] < 0.05                             # clears the placebo null
    # w lying entirely in the placebo span: rho(w,B)~0, placebo null high => excess<0
    w_plac = placebo.T @ RNG.standard_normal(8)
    pe2 = dec.placebo_excess(w_plac, B, placebo, n_trials=200, seed=4)
    assert pe2["excess"] < 0
    assert pe2["p_value"] > 0.5                             # nowhere near significant


def test_malignancy_synonym_null_is_a_stronger_control_than_placebo():
    """The malignancy-synonym null (E3, reviewer P0-1) reuses the pool-subset null
    machinery but with a pool of directions that ALIGN with w (malignancy semantics),
    so it is a strictly harder bar than a placebo pool orthogonal to w. A concept
    basis that partially aligns with w must clear the placebo null by MORE than it
    clears the malignancy-synonym null -- which is why passing the placebo null shows
    grounding vs generic geometry, but not necessarily feature-specificity vs
    malignancy semantics."""
    rng = np.random.default_rng(7)
    # concept basis B partially aligned with w (one basis axis ~ w, rest random)
    cols = np.column_stack([rng.standard_normal(D) for _ in range(M - 1)])
    w = rng.standard_normal(D)
    B = np.linalg.qr(np.column_stack([w, cols]))[0][:, :M]
    # placebo pool: directions unrelated to w (low rho -> high excess for B)
    placebo = cd._row_normalize(rng.standard_normal((16, D)))
    # malignancy-synonym pool: directions clustered around w (high rho -> low excess)
    malig = cd._row_normalize(w[None, :] + 0.15 * rng.standard_normal((10, D)))
    pe = dec.placebo_excess(w, B, placebo, n_trials=200, seed=5)
    me = dec.placebo_excess(w, B, malig, n_trials=200, seed=5)
    # same rho, but the malignancy-loaded null captures w far better than the placebo
    assert np.isclose(pe["rho"], me["rho"])
    assert me["placebo_mean"] > pe["placebo_mean"]          # w-aligned null sits higher
    assert pe["excess"] > me["excess"]                      # harder to beat the malignancy null


def test_holm_bonferroni_step_down():
    from src.evaluation.metrics import holm_bonferroni
    # sorted order a,b,d,c,e with thresholds alpha/(n-rank):
    #   a r0 .010, b r1 .0125, d r2 .0167, c r3 .025, e r4 .05
    pv = {"a": 0.001, "b": 0.004, "c": 0.20, "d": 0.011, "e": 0.9}
    out = holm_bonferroni(pv, alpha=0.05)
    assert out["a"]["reject"] and out["b"]["reject"] and out["d"]["reject"]
    assert not out["c"]["reject"] and not out["e"]["reject"]  # step-down stops at c
    assert all(out[k]["holm_family_size"] == 5 for k in pv)
    assert abs(out["d"]["holm_threshold"] - 0.05 / 3) < 1e-12  # rank 2 -> alpha/(5-2)
    assert abs(out["c"]["holm_threshold"] - 0.05 / 2) < 1e-12  # rank 3 -> alpha/(5-3)


def test_scalar_trap_flip_distance_is_rescaled_margin():
    """d_C / d_full must be constant across cases (= 1/sqrt(rho))."""
    w, b, B, Z = _setup()
    fd = dec.flip_distances(Z, w, b, B)
    ratio = fd["ratio"]
    assert ratio.std() < 1e-6  # constant across all cases
    assert abs(ratio.mean() - 1.0 / np.sqrt(dec.rho(w, B))) < 1e-6


def test_minimal_flip_direction_is_degenerate():
    """Every case's minimum-norm edit is collinear -> identical explanation."""
    w, b, B, Z = _setup()
    edits = dec.minimal_flip_edits(Z, w, b, B)
    u = edits / (np.linalg.norm(edits, axis=1, keepdims=True) + 1e-12)
    cos = np.abs(u[:50] @ u[:50].T)
    assert cos.min() > 1 - 1e-8  # all pairwise |cos| == 1


# --------------------------------------------------------------------------- #
# concept_directions.py
# --------------------------------------------------------------------------- #
def test_paired_text_direction_is_unit_and_oriented():
    # build a clean concept axis e0; pos near +e0, neg near -e0
    e0 = np.zeros(D); e0[0] = 1.0
    pos = e0 + 0.05 * RNG.standard_normal((20, D))
    neg = -e0 + 0.05 * RNG.standard_normal((20, D))
    d = cd.paired_text_direction(pos, neg)
    assert abs(np.linalg.norm(d) - 1.0) < 1e-9
    assert d @ e0 > 0.9  # points along the intended axis


def test_build_text_directions_with_stub_embedder():
    axes = {0: 1.0, 1: 2.0}  # two concepts on two orthogonal axes
    def stub(texts):
        # deterministic embeddings: encode the requested sign on a chosen axis
        out = np.zeros((len(texts), D))
        for i, tx in enumerate(texts):
            ax, sign = tx
            out[i, ax] = sign
        return out
    concept_prompts = {
        "margin": {"pos": [(0, +1)], "neg": [(0, -1)]},
        "shape":  {"pos": [(1, +1)], "neg": [(1, -1)]},
    }
    dirs = cd.build_text_directions(concept_prompts, stub)
    B, names = cd.stack_basis(dirs, orthonormalize=True)
    assert B.shape == (D, 2)
    assert np.allclose(B.T @ B, np.eye(2), atol=1e-9)  # orthonormal


def test_cav_mean_difference_recovers_separation():
    e0 = np.zeros(D); e0[3] = 1.0
    pos = e0 + 0.1 * RNG.standard_normal((40, D))
    neg = -e0 + 0.1 * RNG.standard_normal((40, D))
    embs = np.vstack([pos, neg])
    labels = np.array([1] * 40 + [0] * 40)
    d = cd.cav_mean_difference(embs, labels)
    assert d @ e0 > 0.9


def test_modality_gap_and_projection():
    img = RNG.standard_normal((100, D)) + 5.0  # offset cluster (the "gap")
    txt = RNG.standard_normal((100, D))
    mg = cd.modality_gap(img, txt)
    assert mg["gap_norm"] > 0
    assert -1 <= mg["centroid_cosine"] <= 1
    # a direction inside the image PCA subspace is mostly retained
    U = cd.image_subspace(img, 10)
    inside = U[:, 0]
    proj, retained = cd.project_to_image_subspace(inside, img, 10)
    assert retained > 0.9


# --------------------------------------------------------------------------- #
# retrieval.py
# --------------------------------------------------------------------------- #
def test_nearest_opposite_picks_other_class():
    # two separated clusters on the sphere, labels 0 and 1
    c0 = np.zeros(D); c0[0] = 1.0
    c1 = np.zeros(D); c1[1] = 1.0
    A = c0 + 0.05 * RNG.standard_normal((30, D))
    Bc = c1 + 0.05 * RNG.standard_normal((30, D))
    Z = np.vstack([A, Bc]); Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    y = np.array([0] * 30 + [1] * 30)
    idx, dist = ret.nearest_opposite(Z, y)
    assert np.all(y[idx] != y)              # every match is opposite class
    assert np.all(dist >= 0)


def test_birads_shift_orientation_and_concordance():
    # 2 benign (BI-RADS 2,3), 2 malignant (BI-RADS 4,5)
    y = np.array([0, 0, 1, 1])
    birads = np.array([2.0, 3.0, 4.0, 5.0])
    # faithful retrieval: each benign query -> a malignant prototype, and vice versa
    idx = np.array([2, 3, 0, 1])
    out = ret.birads_shift(y, birads, idx)
    # benign->malignant prototypes are higher BI-RADS, malignant->benign lower:
    # oriented delta = +|change| for every case, concordance = 1
    assert np.all(out["oriented_delta"] > 0)
    assert out["concordance_rate"] == 1.0
    assert out["mean_delta_benign_to_malignant"] > 0
    assert out["mean_delta_malignant_to_benign"] < 0
    # an UNfaithful retrieval (benign->benign-range prototype) flips the sign and
    # drops concordance: send benign queries to a BI-RADS-2 "prototype"
    bad = np.array([0, 0, 0, 0])  # all prototypes are benign-range (BI-RADS 2)
    out_bad = ret.birads_shift(y, birads, bad)
    assert out_bad["concordance_rate"] < out["concordance_rate"]


def test_edit_concept_fraction_bounds():
    B, _ = np.linalg.qr(RNG.standard_normal((D, M)))  # orthonormal subspace
    inside = (B @ RNG.standard_normal((M, 5))).T       # edits entirely in col(B)
    assert np.allclose(ret.edit_concept_fraction(inside, B), 1.0, atol=1e-8)
    # an edit orthogonal to the subspace has ~0 fraction
    v = RNG.standard_normal(D); v -= B @ (B.T @ v)
    assert ret.edit_concept_fraction(v[None, :], B)[0] < 1e-8


def test_directional_counterfactual_flips_and_is_unit():
    w = RNG.standard_normal(D); b = 0.1
    u = w / np.linalg.norm(w)               # move along the decision dir itself
    Z = RNG.standard_normal((40, D)); Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    zc = ret.directional_counterfactual(Z, w, b, u, max_alpha=10.0)
    assert np.allclose(np.linalg.norm(zc, axis=1), 1.0, atol=1e-6)
    # sign of the logit should flip for most cases when moving along w
    flipped = np.sign(Z @ w + b) != np.sign(zc @ w + b)
    assert flipped.mean() > 0.8


def test_directional_counterfactual_flip_is_post_renormalisation():
    """The flip must hold on the returned (renormalised) embedding, not just on
    the un-normalised step. The probe is read as sign((zc/||zc||).w + b): the bias
    b is NOT scaled by ||zc||, so sign(zc.w + b) != sign((zc/||zc||).w + b). The
    pre-renorm flip happens at a smaller alpha than the post-renorm one, so a
    search that tests pre-renorm can stop early and return a vector that is *not*
    actually flipped once renormalised.

    Regime that exposes it: a sizeable bias (b=0.6) and a push direction that
    correlates with w but is not exactly w (drives a genuine flip while ||zc||
    grows). Here the old pre-renorm search returned ~25-35/60 un-flipped cases
    that were in fact reachable; the fixed search must leave ZERO such cases.
    """
    rng = np.random.default_rng(7)
    max_alpha, steps = 2.0, 121
    b = 0.6
    w = rng.standard_normal(D)
    u = w / np.linalg.norm(w) + 0.5 * rng.standard_normal(D)
    u /= np.linalg.norm(u)
    Z = rng.standard_normal((60, D)); Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    zc = ret.directional_counterfactual(Z, w, b, u, max_alpha=max_alpha, steps=steps)
    assert np.allclose(np.linalg.norm(zc, axis=1), 1.0, atol=1e-6)

    cur = np.sign(Z @ w + b)
    returned_flip = np.sign(zc @ w + b) != cur
    alphas = np.linspace(0.0, max_alpha, steps)[1:]
    unflipped_but_reachable = 0
    for i in range(len(Z)):
        cand = Z[i][None, :] + np.outer(-cur[i] * alphas, u)      # (steps-1, d)
        cand /= np.linalg.norm(cand, axis=1, keepdims=True)       # renormalise
        reachable = np.any(np.sign(cand @ w + b) != cur[i])
        if reachable and not returned_flip[i]:
            unflipped_but_reachable += 1
    assert unflipped_but_reachable == 0, (
        f"{unflipped_but_reachable}/60 cases had a reachable post-renorm flip but "
        "the returned vector did not flip — the search is testing the wrong vector")


def test_breast_pole_mapping_matches_concept_bank():
    # E6: lock the BrEaST descriptor -> benign(0)/malignant(1) pole mapping so it
    # cannot silently drift from the concept-bank poles. Pure (no data/openpyxl).
    from src.data.datasets import BrEaSTDataset as B
    # shape: irregular = malignant pole; oval/round = benign; else excluded
    assert B._pole_shape("irregular") == 1
    assert B._pole_shape("oval") == 0 and B._pole_shape("round") == 0
    # margin: circumscribed = benign; any "not circumscribed - ..." = malignant
    assert B._pole_margin("circumscribed") == 0
    assert B._pole_margin("not circumscribed - spiculated&indistinct") == 1
    assert B._pole_margin("not circumscribed - microlobulated") == 1
    # echogenicity: hypoechoic = malignant; anechoic/iso/hyper = benign;
    # heterogeneous & complex cystic/solid are OFF the intensity axis -> None
    assert B._pole_echogenicity("hypoechoic") == 1
    assert {B._pole_echogenicity(v) for v in ("anechoic", "isoechoic", "hyperechoic")} == {0}
    assert B._pole_echogenicity("heterogeneous") is None
    assert B._pole_echogenicity("complex cystic/solid") is None
    # posterior: shadowing/combined = malignant; no/enhancement = benign
    assert B._pole_posterior("shadowing") == 1 and B._pole_posterior("combined") == 1
    assert B._pole_posterior("no") == 0 and B._pole_posterior("enhancement") == 0
    # not-applicable / unknown -> None on every axis (excluded, never mislabelled)
    for fn in (B._pole_shape, B._pole_margin, B._pole_echogenicity, B._pole_posterior):
        assert fn("not applicable") is None


def test_directional_counterfactual_orients_axis_to_probe():
    # Codex #2: an axis pointing the OPPOSITE way to w must still drive flips —
    # the function orients u to w internally. An anti-aligned u previously stepped
    # the wrong way and gave flip_rate 0.
    w = RNG.standard_normal(D); b = 0.1
    u = -w / np.linalg.norm(w)                       # deliberately anti-aligned
    Z = RNG.standard_normal((40, D)); Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    zc = ret.directional_counterfactual(Z, w, b, u, max_alpha=10.0)
    flipped = np.sign(Z @ w + b) != np.sign(zc @ w + b)
    assert flipped.mean() > 0.8
    # orientation is internal: passing u or -u yields identical counterfactuals
    zc_pos = ret.directional_counterfactual(Z, w, b, -u, max_alpha=10.0)
    assert np.allclose(zc, zc_pos, atol=1e-9)


def test_decompose_named_uses_raw_axes_via_b_named():
    # Codex #4: with a QR-orthonormalised subspace basis, named_contributions must
    # be built from the RAW axes (B_named) to keep their BI-RADS names, not from
    # the QR-mixed columns.
    w, b, _, Z = _setup()
    raw = RNG.standard_normal((D, M))                # non-orthonormal raw axes
    Q, _ = np.linalg.qr(raw)                         # orthonormal span, same subspace
    named_raw = dec.decompose(Z, w, b, raw).named_contributions
    named_via = dec.decompose(Z, w, b, Q, B_named=raw).named_contributions
    assert np.allclose(named_raw, named_via, atol=1e-9)          # raw axes recovered
    named_mixed = dec.decompose(Z, w, b, Q).named_contributions  # footgun: Q columns
    assert not np.allclose(named_raw, named_mixed, atol=1e-6)
    # the subspace-invariant clinical share is unaffected by the basis choice
    s_raw = dec.decompose(Z, w, b, raw).clinical_share
    s_q = dec.decompose(Z, w, b, Q).clinical_share
    assert np.allclose(s_raw, s_q, atol=1e-9)


def test_rho_is_basis_agnostic_unlike_the_orthonormal_shortcut():
    # Codex #3: the bootstrap used (B^T w)·(B^T w)/(w·w), valid ONLY for orthonormal
    # B. dec.rho projects onto col(B) and depends only on the subspace.
    w, _, _, _ = _setup()
    raw = RNG.standard_normal((D, M))                # non-orthonormal columns
    Q, _ = np.linalg.qr(raw)                         # orthonormal, same column space
    shortcut = lambda B: float((B.T @ w) @ (B.T @ w) / (w @ w))
    assert abs(dec.rho(w, raw) - dec.rho(w, Q)) < 1e-9   # basis-agnostic
    assert 0.0 <= dec.rho(w, raw) <= 1.0
    assert abs(shortcut(Q) - dec.rho(w, Q)) < 1e-9       # shortcut ok for orthonormal
    assert abs(shortcut(raw) - dec.rho(w, raw)) > 1e-3   # shortcut wrong otherwise


# --------------------------------------------------------------------------- #
# validity_stats.py (E6 diagnosis-controlled feature-specificity)
# --------------------------------------------------------------------------- #
def test_rank_auc_matches_bruteforce_and_ignores_unlabelled():
    rng = np.random.default_rng(0)
    scores = rng.standard_normal(60)
    labels = rng.integers(0, 2, 60)
    pos, neg = scores[labels == 1], scores[labels == 0]
    brute = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])
    assert abs(vs.rank_auc(scores, labels) - brute) < 1e-9
    # all-tied scores => AUC 0.5
    assert abs(vs.rank_auc(np.zeros(10), np.array([0, 1] * 5)) - 0.5) < 1e-9
    # -1 (unlabelled) entries are ignored, so padding with them changes nothing
    s2 = np.concatenate([scores, rng.standard_normal(25)])
    l2 = np.concatenate([labels, -np.ones(25, int)])
    assert abs(vs.rank_auc(s2, l2) - vs.rank_auc(scores, labels)) < 1e-9


def test_conditioned_auc_removes_diagnosis_confound():
    # feature correlates with diagnosis; score is driven ONLY by diagnosis. The
    # marginal AUC is inflated by the confound, the conditioned AUC is not.
    rng = np.random.default_rng(1)
    n = 300
    diag = np.array([0] * n + [1] * n)
    feat = np.empty(2 * n, int)
    feat[:n] = (rng.random(n) < 0.2).astype(int)   # benign: few malignant-pole
    feat[n:] = (rng.random(n) < 0.8).astype(int)   # malignant: many malignant-pole
    score = diag + 0.5 * rng.standard_normal(2 * n)  # tracks diagnosis, not feature
    assert vs.rank_auc(score, feat) > 0.65           # confound inflates marginal AUC
    assert 0.42 < vs.conditioned_auc(score, feat, diag) < 0.58   # removed by conditioning
    assert vs.within_diag_perm_p(score, feat, diag, n_perm=300, rng=rng) > 0.1


def test_conditioned_auc_and_paired_diff_detect_real_signal():
    # score tracks the FEATURE within each diagnosis stratum: conditioned AUC high,
    # permutation significant, and the paired diff over an unrelated score excludes 0.
    rng = np.random.default_rng(2)
    n = 300
    diag = np.array([0] * n + [1] * n)
    feat = rng.integers(0, 2, 2 * n)                 # independent of diagnosis
    score = feat + 0.3 * diag + 0.4 * rng.standard_normal(2 * n)
    assert vs.conditioned_auc(score, feat, diag) > 0.7
    assert vs.within_diag_perm_p(score, feat, diag, n_perm=300, rng=rng) < 0.02
    unrelated = rng.standard_normal(2 * n)
    ci, _ = vs.paired_diff_ci(
        lambda idx: vs.rank_auc(score[idx], feat[idx]),
        lambda idx: vs.rank_auc(unrelated[idx], feat[idx]),
        2 * n, rng, n_boot=300, null=0.0)
    assert ci[0] > 0                                 # own-feature score strictly better


# --- ROB-03: subspace principal angles ---------------------------------------
def test_principal_angles_identical_subspace_is_zero():
    rng = np.random.default_rng(0)
    B = rng.standard_normal((12, 4))
    ang = cd.subspace_principal_angles(B, B)
    assert ang.shape == (4,)
    assert np.allclose(ang, 0.0, atol=1e-8)


def test_principal_angles_invariant_to_basis_within_the_span():
    """Rotating a basis within its own span must not move any angle."""
    rng = np.random.default_rng(1)
    B = rng.standard_normal((12, 3))
    R, _ = np.linalg.qr(rng.standard_normal((3, 3)))     # in-span rotation
    assert np.allclose(cd.subspace_principal_angles(B, B @ R), 0.0, atol=1e-8)


def test_principal_angles_orthogonal_subspaces_are_ninety_degrees():
    d = 10
    I = np.eye(d)
    a = cd.subspace_principal_angles(I[:, :3], I[:, 3:6])
    assert np.allclose(a, np.pi / 2, atol=1e-8)


def test_principal_angles_partial_overlap_and_unequal_dims():
    d = 8
    I = np.eye(d)
    # spans share one axis, differ on the other -> one angle 0, one pi/2
    a = np.sort(cd.subspace_principal_angles(I[:, [0, 1]], I[:, [0, 2]]))
    assert np.allclose(a, [0.0, np.pi / 2], atol=1e-8)
    # unequal dimensions -> min(k1, k2) angles
    assert cd.subspace_principal_angles(I[:, :2], I[:, :5]).shape == (2,)


def test_principal_angles_grow_with_perturbation():
    """A monotone check: a bigger nudge to one axis gives a bigger largest angle."""
    rng = np.random.default_rng(3)
    B = rng.standard_normal((16, 4))
    prev = -1.0
    for eps in (0.0, 0.05, 0.2, 0.6):
        Bp = B.copy()
        Bp[:, 0] += eps * rng.standard_normal(16)
        largest = cd.subspace_principal_angles(B, Bp).max()
        assert largest >= prev - 1e-9
        prev = largest


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
        passed += 1
    print(f"\n{passed}/{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
