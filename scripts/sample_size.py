from statsmodels.stats.proportion import samplesize_proportions_2indep_onetail
from math import ceil

params = {
    'ratio': 1,
    'diff': 0.03,
    'prop2': 0.2,
    'alpha': 0.05,
    'power': 0.8
}

different = {
    'ratio': 1,
    'diff': 0.02,
    'prop2': 0.3,
    'alpha': 0.01,
    'power': 0.9
}

print('-' * 60)
for new_param, value in different.items():
    new_params = params.copy()
    new_params[new_param] = value
    sample_size = ceil(samplesize_proportions_2indep_onetail(
        **new_params,
        alternative='two-sided'
    ))
    print(
        f'Новый параметр {new_param}: {value}\n' \
        f'Игроков на группу: {sample_size}, игроков на эксперимент: {sample_size * 2}'
    )
    print('-' * 60)