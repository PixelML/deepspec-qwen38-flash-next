"""Unit tests for the accepted-prefix accumulator in scripts/eval/agreement_curve.py.

The point of these tests is to show the fix is RIGHT, not merely different.
Each case has an answer worked out by hand:

  * independent per-position acceptance -- the product of marginals IS the
    joint, so the corrected number must equal the old proxy exactly;
  * positively correlated acceptance -- the product UNDERSTATES the joint;
  * negatively correlated acceptance -- the product OVERSTATES it.

Run:  python -m pytest tests/test_agreement_curve.py -q
"""
import importlib.util
import itertools
import os

import pytest
import torch

_SPEC = importlib.util.spec_from_file_location(
    "agreement_curve",
    os.path.join(os.path.dirname(__file__), "..", "scripts", "eval", "agreement_curve.py"),
)
agreement_curve = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(agreement_curve)

PrefixAccumulator = agreement_curve.PrefixAccumulator
_tau_proxy = agreement_curve._tau_proxy


def run(correct_rows, mask_rows=None):
    """Feed a list of per-anchor rows as one [1, n_anchors, K] batch."""
    correct = torch.tensor([correct_rows], dtype=torch.bool)
    if mask_rows is None:
        mask = torch.ones_like(correct)
    else:
        mask = torch.tensor([mask_rows], dtype=torch.bool)
    acc = PrefixAccumulator(correct.shape[-1])
    acc.update(correct, mask)
    return acc.result()


def marginals(res):
    return [e["top1_agreement"] for e in res["agreement_curve"]]


def joint_full(res):
    return [e["p_prefix_accepted"] for e in res["accepted_prefix_curve"]["full_depth_cohort"]]


def joint_var(res):
    return [e["p_prefix_accepted"] for e in res["accepted_prefix_curve"]["variable_cohort"]]


def hazard(res):
    return [e["p_correct_given_prefix"] for e in res["hazard_curve"]]


# ---------------------------------------------------------------------------
# 1. Independent acceptance: the product of marginals is exactly the joint.
# ---------------------------------------------------------------------------

def test_independent_product_equals_joint():
    # All 2^2 outcomes once each == independent Bernoulli(0.5) at both
    # positions with zero correlation.
    rows = [list(c) for c in itertools.product([1, 0], repeat=2)]
    res = run(rows)

    assert marginals(res) == pytest.approx([0.5, 0.5])
    # joint: P(>=1) = 0.5, P(>=2) = 0.5 * 0.5 = 0.25  == product of marginals
    assert joint_full(res) == pytest.approx([0.5, 0.25])
    assert res["expected_accepted_length"]["full_depth_cohort"] == pytest.approx(1.75)
    # and therefore the old proxy is, in this one regime, correct
    assert res["tau_proxy_product_of_marginals"] == pytest.approx(1.75)
    # independence also means hazard == marginal
    assert hazard(res) == pytest.approx(marginals(res))


def test_independent_three_positions_p_two_thirds():
    # 3 positions, independent, p = 2/3 each: enumerate all 27 combinations.
    rows = []
    for combo in itertools.product([1, 1, 0], repeat=3):
        rows.append(list(combo))
    res = run(rows)
    p = 2.0 / 3.0
    assert marginals(res) == pytest.approx([p, p, p])
    assert joint_full(res) == pytest.approx([p, p ** 2, p ** 3])
    assert res["expected_accepted_length"]["full_depth_cohort"] == pytest.approx(
        1 + p + p ** 2 + p ** 3
    )
    assert res["tau_proxy_product_of_marginals"] == pytest.approx(
        res["expected_accepted_length"]["full_depth_cohort"]
    )


# ---------------------------------------------------------------------------
# 2. Correlated acceptance: the product is wrong, in a known direction.
# ---------------------------------------------------------------------------

def test_positive_correlation_product_understates():
    # Anchors are either fully right or fully wrong -- perfect positive
    # correlation. Marginals are still 0.5 at both positions, so the proxy
    # cannot tell this apart from the independent case above.
    res = run([[1, 1], [0, 0]])

    assert marginals(res) == pytest.approx([0.5, 0.5])
    assert res["tau_proxy_product_of_marginals"] == pytest.approx(1.75)

    # Truth: half the anchors accept BOTH tokens.
    assert joint_full(res) == pytest.approx([0.5, 0.5])
    assert res["expected_accepted_length"]["full_depth_cohort"] == pytest.approx(2.0)
    assert res["expected_accepted_length"]["full_depth_cohort"] > res[
        "tau_proxy_product_of_marginals"
    ]
    # the conditional the marginal is mistaken for is 1.0 here, not 0.5
    assert hazard(res) == pytest.approx([0.5, 1.0])


def test_negative_correlation_product_overstates():
    # Exactly one position right per anchor -- no anchor ever accepts a
    # 2-token prefix, yet both marginals are still 0.5.
    res = run([[1, 0], [0, 1]])

    assert marginals(res) == pytest.approx([0.5, 0.5])
    assert res["tau_proxy_product_of_marginals"] == pytest.approx(1.75)
    assert joint_full(res) == pytest.approx([0.5, 0.0])
    assert res["expected_accepted_length"]["full_depth_cohort"] == pytest.approx(1.5)
    assert res["expected_accepted_length"]["full_depth_cohort"] < res[
        "tau_proxy_product_of_marginals"
    ]
    assert hazard(res) == pytest.approx([0.5, 0.0])


def test_latent_difficulty_mixture_matches_closed_form():
    # Half the anchors are "easy" (accept w.p. 1 at every position), half are
    # "hard" (accept w.p. 0). Closed form: marginal p_k = 0.5 for all k, proxy
    # tau = 1 + 0.5 + 0.25 + 0.125 = 1.875; true E[L] = 1 + 3*0.5 = 2.5.
    res = run([[1, 1, 1], [0, 0, 0]])
    assert marginals(res) == pytest.approx([0.5, 0.5, 0.5])
    assert res["tau_proxy_product_of_marginals"] == pytest.approx(1.875)
    assert res["expected_accepted_length"]["full_depth_cohort"] == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# 3. eval_mask handling: validity truncation must not be read as rejection.
# ---------------------------------------------------------------------------

def test_variable_and_full_depth_cohorts_with_truncated_masks():
    # A: valid to depth 3, all correct
    # B: valid to depth 2, correct then wrong (its 3rd slot is padding)
    # C: valid to depth 1 only
    correct = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    mask = [[1, 1, 1], [1, 1, 0], [1, 0, 0]]
    res = run(correct, mask)

    assert res["n_anchors"] == 3
    assert res["n_full_depth_anchors"] == 1

    # marginal: pos1 3/3, pos2 1/2 (B wrong), pos3 1/1 (only A valid)
    assert marginals(res) == pytest.approx([1.0, 0.5, 1.0])

    # variable cohort mixes populations across k and can even rise again --
    # that non-monotonicity is exactly why the fixed cohort is the headline.
    assert joint_var(res) == pytest.approx([1.0, 0.5, 1.0])

    # fixed cohort = anchor A only, which accepts all three
    assert joint_full(res) == pytest.approx([1.0, 1.0, 1.0])
    assert res["expected_accepted_length"]["full_depth_cohort"] == pytest.approx(4.0)

    # hazard denominators: 3, 2 (A,B alive and valid), 1 (A only)
    assert hazard(res) == pytest.approx([1.0, 0.5, 1.0])


def test_masked_positions_never_counted():
    # A wholly invalid anchor must move no counter at all.
    live = run([[1, 0, 1]], [[1, 1, 1]])
    padded = run([[1, 0, 1], [1, 1, 1]], [[1, 1, 1], [0, 0, 0]])
    assert marginals(live) == pytest.approx(marginals(padded))
    assert joint_full(live) == pytest.approx(joint_full(padded))
    assert live["n_anchors"] == padded["n_anchors"] == 1


# ---------------------------------------------------------------------------
# 4. Invariants that must hold on arbitrary data.
# ---------------------------------------------------------------------------

def _random_case(seed, k=7, n=512):
    g = torch.Generator().manual_seed(seed)
    # latent per-anchor difficulty -> positively correlated positions, the
    # regime a real drafter is in
    easiness = torch.rand(1, n, 1, generator=g)
    correct = torch.rand(1, n, k, generator=g) < easiness
    depth = torch.randint(1, k + 1, (1, n, 1), generator=g)
    mask = torch.arange(k).view(1, 1, k) < depth
    return correct, mask


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_invariants_on_random_data(seed):
    correct, mask = _random_case(seed)
    acc = PrefixAccumulator(correct.shape[-1])
    acc.update(correct, mask)
    res = acc.result()

    # position 1 has an empty prefix, so joint == marginal there by definition
    assert joint_var(res)[0] == pytest.approx(marginals(res)[0])
    assert hazard(res)[0] == pytest.approx(marginals(res)[0])

    # a survival function is non-increasing
    full = joint_full(res)
    assert all(full[i] >= full[i + 1] - 1e-12 for i in range(len(full) - 1))

    # accepted prefix can never exceed marginal correctness at the same k
    for j, m in zip(joint_var(res), marginals(res)):
        assert j <= m + 1e-12

    # positively correlated data: the product of marginals must UNDERSTATE
    assert res["expected_accepted_length"]["full_depth_cohort"] > res[
        "tau_proxy_product_of_marginals"
    ]

    # truncated expectations are the partial sums of the survival curve
    trunc = res["expected_length_truncated_at_k"]["full_depth_cohort"]
    assert trunc[-1] == pytest.approx(res["expected_accepted_length"]["full_depth_cohort"])
    assert trunc[0] == pytest.approx(1.0 + full[0])


def test_streaming_matches_single_batch():
    # The accumulator runs over a DataLoader, so batching must not change it.
    correct, mask = _random_case(7)
    one = PrefixAccumulator(correct.shape[-1])
    one.update(correct, mask)

    many = PrefixAccumulator(correct.shape[-1])
    for i in range(0, correct.shape[1], 37):
        many.update(correct[:, i:i + 37], mask[:, i:i + 37])

    assert one.result() == many.result()
