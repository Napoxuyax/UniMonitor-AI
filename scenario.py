"""Сценарные прогнозы и временная калибровка неопределённости."""
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from core import FEATURES
from research import build_pairs


def scenario_analysis(df, changes=None, coverage=.9, seed=42):
    changes = changes or {}
    if not 0 < coverage < 1:
        raise ValueError('Уровень покрытия должен быть между 0 и 1')
    if set(changes)-set(FEATURES):
        raise ValueError('Неизвестный показатель')
    if any(not np.isfinite(v) or abs(v)>100 for v in changes.values()):
        raise ValueError('Изменение должно лежать в диапазоне ±100%')
    pairs = build_pairs(df)
    years = sorted(pairs.target_year.unique())
    if len(years)<4:
        raise ValueError('Нужны четыре целевых года')
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    residuals = []
    for year in years[1:-1]:
        train = pairs[pairs.target_year<year]
        valid = pairs[pairs.target_year==year]
        fitted = clone(model).fit(train[FEATURES],train.target)
        residuals.extend(abs(valid.target-np.clip(fitted.predict(valid[FEATURES]),0,100)))
    residuals = np.sort(residuals)
    index = int(np.ceil((len(residuals)+1)*coverage))-1
    if index>=len(residuals):
        raise ValueError('Недостаточно калибровочных наблюдений для заданного покрытия')
    radius = float(residuals[index])
    train = pairs[pairs.target_year<years[-1]]
    test = pairs[pairs.target_year==years[-1]]
    fitted = clone(model).fit(train[FEATURES],train.target)
    pred = np.clip(fitted.predict(test[FEATURES]),0,100)
    baseline_mae = float(np.mean(abs(test.target-pred)))
    rng = np.random.default_rng(seed)
    importance = []
    for col in FEATURES:
        errors = []
        for _ in range(20):
            shuffled = test[FEATURES].copy()
            shuffled[col] = rng.permutation(shuffled[col].to_numpy())
            errors.append(float(np.mean(abs(test.target-np.clip(fitted.predict(shuffled),0,100))))-baseline_mae)
        importance.append({'feature':col,'delta_mae':np.mean(errors),'std':np.std(errors)})
    observed = float(np.mean(abs(test.target-pred)<=radius))
    fitted.fit(pairs[FEATURES],pairs.target)
    latest = df[df.year==df.year.max()].copy()
    base = np.clip(fitted.predict(latest[FEATURES]),0,100)
    modified = latest[FEATURES].copy()
    for col, amount in changes.items():
        modified[col] *= 1+amount/100
        if col in ('score','employment'):
            modified[col] = modified[col].clip(0,100)
    scenario = np.clip(fitted.predict(modified),0,100)
    result = latest[['university','region','year']].copy()
    result['forecast_year'] = result.year+1
    result['baseline'] = base
    result['scenario'] = scenario
    result['delta'] = scenario-base
    result['lower'] = np.maximum(0,scenario-radius)
    result['upper'] = np.minimum(100,scenario+radius)
    return result,pd.DataFrame(importance).sort_values('delta_mae',ascending=False),{
        'model':'Ridge alpha=1 (фиксированная конфигурация)', 'nominal_coverage':coverage,
        'observed_test_coverage':observed,'radius':radius,'test_year':int(years[-1]),
        'calibration_count':len(residuals),'test_count':len(test),'baseline_mae':baseline_mae,
        'changes_percent':changes,'interval_note':'Эмпирический интервал; покрытие будущих данных и сценариев не гарантировано.'}
