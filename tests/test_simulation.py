import warnings

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.proportion import proportions_ztest

from ab_testing.simulation import _significant_experiments, simulate_false_positive_rates


def test_seed_reproducibility():
    first_rate, first_results = simulate_false_positive_rates(n_experiments=100, seed=42)
    second_rate, second_results = simulate_false_positive_rates(n_experiments=100, seed=42)

    assert first_rate == second_rate
    pd.testing.assert_frame_equal(first_results, second_results)


def test_fixed_horizon_uses_final_accumulated_data():
    seed = 17
    n_experiments = 100
    n_looks = 3
    batch_size = 100
    rate, results = simulate_false_positive_rates(
        n_experiments=n_experiments,
        players_per_group=batch_size * n_looks,
        n_looks=n_looks,
        seed=seed,
    )

    rng = np.random.default_rng(seed)
    control = np.zeros(n_experiments, dtype=int)
    treatment = np.zeros(n_experiments, dtype=int)
    for _ in range(n_looks):
        control += rng.binomial(batch_size, 0.20, size=n_experiments)
        treatment += rng.binomial(batch_size, 0.20, size=n_experiments)
    p_values = [
        proportions_ztest([t, c], [300, 300], alternative='two-sided')[1]
        for t, c in zip(treatment, control, strict=True)
    ]

    assert rate == np.mean(np.array(p_values) < 0.05)
    assert rate == results.iloc[-1]['significant_at_look']


def test_accumulation_properties():
    rate, results = simulate_false_positive_rates(n_experiments=200)
    assert results['players_per_group'].tolist() == list(range(300, 3001, 300))
    rates = results[['significant_at_look', 'significant_by_look']]
    assert rates.ge(0).all().all()
    assert rates.le(1).all().all()
    assert results['significant_by_look'].is_monotonic_increasing
    assert results['significant_by_look'].ge(results['significant_at_look']).all()
    assert results.iloc[0]['significant_by_look'] == results.iloc[0]['significant_at_look']
    assert rate == results.iloc[-1]['significant_at_look']


def test_one_look_has_same_decision_for_both_strategies():
    rate, results = simulate_false_positive_rates(n_experiments=100, n_looks=1)
    assert rate == results.iloc[0]['significant_by_look']


def test_zero_variance_does_not_reject():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        significant = _significant_experiments(
            treatment_returns=np.array([0, 100]),
            control_returns=np.array([0, 100]),
            players_per_group=100,
            alpha=0.05,
        )
    np.testing.assert_array_equal(significant, [False, False])


def test_decisions_match_statsmodels():
    treatment = np.array([10, 20, 0, 100])
    control = np.array([20, 20, 100, 0])
    expected = [
        proportions_ztest([t, c], [100, 100], alternative='two-sided')[1] < 0.05
        for t, c in zip(treatment, control, strict=True)
    ]
    np.testing.assert_array_equal(
        _significant_experiments(treatment, control, 100, 0.05), expected
    )


@pytest.mark.parametrize('parameters', [
    {'n_experiments': 0},
    {'n_experiments': 1.5},
    {'n_experiments': True},
    {'players_per_group': -1},
    {'players_per_group': 301},
    {'n_looks': 0},
    {'n_looks': 2.5},
    {'n_looks': 4000},
    {'baseline_d7': 0},
    {'baseline_d7': 1},
    {'baseline_d7': np.nan},
    {'alpha': 0},
    {'alpha': 1},
    {'alpha': np.nan},
])
def test_invalid_parameters(parameters):
    with pytest.raises(ValueError):
        simulate_false_positive_rates(**parameters)
