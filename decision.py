"""Exploratory multicriteria scoring and distribution monitoring."""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from core import FEATURES


def normalized_weights(weights=None):
    if weights is None:
        weights = dict.fromkeys(FEATURES, 1.0)
    if set(weights) != set(FEATURES):
        raise ValueError('Необходимо задать веса всех показателей')
    values = np.array([weights[f] for f in FEATURES], dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or values.max() <= 0:
        raise ValueError('Веса должны быть конечными, неотрицательными и не все нулевыми')
    values /= values.max()
    return values / values.sum()


def score_matrix(df, year):
    rows = df.loc[df.year == year].sort_values('university').reset_index(drop=True)
    if len(rows) < 2:
        raise ValueError('Необходимо не менее двух вузов за выбранный год')
    x = rows[FEATURES].to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError('Обнаружены некорректные показатели')
    lower = x.min(axis=0)
    span = x.max(axis=0) - lower
    # Constant indicators contribute no differentiation.
    scaled = np.divide(x-lower, span, out=np.zeros_like(x), where=span != 0)
    return rows, scaled


def rank_universities(df, year, weights=None):
    rows, matrix = score_matrix(df, year)
    weights = normalized_weights(weights)
    result = rows[['university','region','year']].copy()
    result['score_total'] = matrix @ weights * 100
    result['rank'] = result.score_total.rank(method='min', ascending=False).astype(int)
    return result.sort_values(['rank','university']).reset_index(drop=True)


def sensitivity(df, year, weights=None, repetitions=300, seed=42):
    if not 20 <= repetitions <= 5000:
        raise ValueError('Число сценариев должно быть от 20 до 5000')
    rows, matrix = score_matrix(df, year)
    base = normalized_weights(weights)
    rng = np.random.default_rng(seed)
    # Multiplicative perturbation retains zero weights and preserves the simplex.
    sampled = base * rng.uniform(.75, 1.25, size=(repetitions,len(base)))
    sampled /= sampled.sum(axis=1,keepdims=True)
    scores = matrix @ sampled.T
    ranks = pd.DataFrame(scores).rank(axis=0,method='min',ascending=False).to_numpy()
    result = rows[['university','region']].copy()
    result['rank_base'] = pd.Series(matrix @ base).rank(method='min',ascending=False).astype(int)
    result['rank_median'] = np.median(ranks,axis=1)
    result['rank_best'] = ranks.min(axis=1).astype(int)
    result['rank_worst'] = ranks.max(axis=1).astype(int)
    result['top10_share'] = (ranks <= 10).mean(axis=1)*100
    return result.sort_values(['rank_base','university']).reset_index(drop=True)


def distribution_drift(df, reference_year, comparison_year):
    if reference_year >= comparison_year:
        raise ValueError('Базовый год должен предшествовать сравниваемому')
    a = df[df.year == reference_year]
    b = df[df.year == comparison_year]
    common = sorted(set(a.university) & set(b.university))
    if len(common) < 4:
        raise ValueError('Необходимо минимум четыре общих вуза')
    a = a.set_index('university').loc[common]
    b = b.set_index('university').loc[common]
    records = []
    for feature in FEATURES:
        x, y = a[feature].to_numpy(), b[feature].to_numpy()
        # KS distance is descriptive here: observations are paired across years.
        distance = float(ks_2samp(x,y).statistic)
        records.append({'indicator':feature,'matched':len(common),
                        'reference_mean':float(x.mean()),'comparison_mean':float(y.mean()),
                        'median_change':float(np.median(y-x)), 'ks_distance':distance,
                        'flag': 'Проверить' if distance >= .2 else 'Нет сигнала'})
    return pd.DataFrame(records)
