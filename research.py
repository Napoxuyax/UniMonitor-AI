"""Воспроизводимый эксперимент с временной валидацией."""
from itertools import product
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from core import FEATURES


def build_pairs(df):
    pairs = df.sort_values(['university', 'year']).copy()
    grouped = pairs.groupby('university')
    pairs['target_year'] = grouped.year.shift(-1)
    pairs['target'] = grouped.employment.shift(-1)
    return pairs.loc[pairs.target_year.eq(pairs.year + 1)].copy()


def scores(y, pred):
    pred = np.clip(pred, 0, 100)
    return dict(MAE=float(mean_absolute_error(y, pred)),
                RMSE=float(np.sqrt(mean_squared_error(y, pred))),
                R2=float(r2_score(y, pred)) if len(y)>1 and np.std(y)>0 else None)


def candidates():
    for layers, alpha in product([(16,), (32, 16), (32, 16, 8)], [0.01, 0.1]):
        name = f'MLP {layers}, alpha={alpha}'
        yield name, TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=layers, alpha=alpha, solver='lbfgs',
            max_iter=2500, max_fun=30000, random_state=42)), transformer=StandardScaler())
    for alpha in [0.1, 1., 10.]:
        yield f'Ridge alpha={alpha}', make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def experiment(df):
    pairs = build_pairs(df)
    years = sorted(pairs.target_year.unique())
    if len(years)<4:
        raise ValueError('Для временной валидации нужны минимум четыре целевых года')
    test_year = years[-1]
    validation_years = years[1:-1]
    rows = []
    for name, model in candidates():
        for year in validation_years:
            train = pairs[pairs.target_year < year]
            valid = pairs[pairs.target_year == year]
            model.fit(train[FEATURES], train.target)
            rows.append(dict(model=name, year=int(year), **scores(valid.target, model.predict(valid[FEATURES]))))
    folds = pd.DataFrame(rows)
    ranking = folds.groupby('model', as_index=False)[['MAE','RMSE']].mean().sort_values('MAE')
    train = pairs[pairs.target_year < test_year]
    test = pairs[pairs.target_year == test_year]
    models = dict(candidates())
    chosen = [ranking[ranking.model.str.startswith(prefix)].iloc[0].model for prefix in ['MLP','Ridge']]
    comparisons = []
    for name in chosen:
        model = models[name]
        model.fit(train[FEATURES], train.target)
        comparisons.append(dict(model=name, **scores(test.target, model.predict(test[FEATURES]))))
    comparisons.append(dict(model='Наивный прогноз', **scores(test.target, test.employment)))
    winner = ranking.iloc[0].model
    model = models[winner]
    model.fit(train[FEATURES], train.target)
    predictions = test[['university','year','target_year','target']].copy()
    predictions['prediction'] = np.clip(model.predict(test[FEATURES]),0,100)
    predictions['error'] = predictions.prediction-predictions.target
    # После оценки на отложенном годе переобучаем для будущего прогноза.
    model.fit(pairs[FEATURES], pairs.target)
    latest = df[df.year==df.year.max()][['university','year']+FEATURES].copy()
    latest['forecast_year'] = latest.year+1
    latest['forecast_employment'] = np.clip(model.predict(latest[FEATURES]),0,100)
    return {'folds':folds,'ranking':ranking,'comparison':pd.DataFrame(comparisons),
            'predictions':predictions,'forecast':latest,'winner':winner,'test_year':int(test_year)}
