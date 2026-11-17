import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest


def _significant_experiments(
    treatment_returns: np.ndarray,
    control_returns: np.ndarray,
    players_per_group: int,
    alpha: float,
) -> np.ndarray:
    significant = np.zeros(treatment_returns.size, dtype=bool)
    for index, (treatment_count, control_count) in enumerate(
        zip(treatment_returns, control_returns, strict=True)
    ):
        # При равных долях тест не отклоняет H₀, даже если дисперсия равна нулю.
        if treatment_count == control_count:
            continue
        _, p_value = proportions_ztest(
            count=[treatment_count, control_count],
            nobs=[players_per_group, players_per_group],
            value=0,
            alternative='two-sided',
        )
        significant[index] = p_value < alpha
    return significant


def simulate_false_positive_rates(
    *,
    n_experiments: int = 10_000,
    players_per_group: int = 3_000,
    n_looks: int = 10,
    baseline_d7: float = 0.20,
    alpha: float = 0.05,
    seed: int = 2026,
) -> tuple[float, pd.DataFrame]:
    '''Возвращает долю ложных срабатываний в контрольном прогоне и таблицу peeking.'''
    if n_experiments < 1 or players_per_group < 1 or n_looks < 1:
        raise ValueError('Число экспериментов, игроков и проверок должно быть положительным')
    if players_per_group % n_looks:
        raise ValueError('Размер группы должен делиться на число проверок без остатка')
    if not 0 < baseline_d7 < 1 or not 0 < alpha < 1:
        raise ValueError('baseline_d7 и alpha должны быть строго между нулём и единицей')

    rng = np.random.default_rng(seed)
    control_returns = rng.binomial(players_per_group, baseline_d7, size=n_experiments)
    treatment_returns = rng.binomial(players_per_group, baseline_d7, size=n_experiments)
    fixed_horizon_rate = _significant_experiments(
        treatment_returns, control_returns, players_per_group, alpha
    ).mean()

    # Каждая порция добавляет новых игроков к тем же экспериментам.
    players_per_batch = players_per_group // n_looks
    control_returns = np.zeros(n_experiments, dtype=int)
    treatment_returns = np.zeros(n_experiments, dtype=int)
    ever_significant = np.zeros(n_experiments, dtype=bool)
    results = []

    for look in range(1, n_looks + 1):
        control_returns += rng.binomial(players_per_batch, baseline_d7, size=n_experiments)
        treatment_returns += rng.binomial(players_per_batch, baseline_d7, size=n_experiments)
        players_so_far = look * players_per_batch

        significant_now = _significant_experiments(
            treatment_returns, control_returns, players_so_far, alpha
        )
        ever_significant |= significant_now
        results.append({
            'players_per_group': players_so_far,
            'significant_at_look': significant_now.mean(),
            'significant_by_look': ever_significant.mean(),
        })

    return float(fixed_horizon_rate), pd.DataFrame(results)


if __name__ == '__main__':
    fixed_rate, peeking_results = simulate_false_positive_rates()
    print(f'Отдельный прогон с одной проверкой в конце: {fixed_rate:.2%}')
    print(peeking_results.to_string(
        index=False,
        formatters={
            'significant_at_look': '{:.2%}'.format,
            'significant_by_look': '{:.2%}'.format,
        },
    ))
