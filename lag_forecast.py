"""Прогноз по трёхлетней истории без заполнения временных разрывов."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from core import FEATURES, load_data
from research import scores


def windows(df, history=3):
    if history < 2:
        raise ValueError('История должна содержать минимум два года')
    if df.duplicated(['university', 'year']).any():
        raise ValueError('Повторяющиеся годы')
    columns = [f'{feature}_lag{lag}' for lag in range(history) for feature in FEATURES]
    result = []
    for name, group in df.groupby('university'):
        records = group.sort_values('year').set_index('year')
        for year in records.index:
            required = list(range(int(year)-history+1, int(year)+1))
            if not all(t in records.index for t in required):
                continue
            row = {'university': name, 'year': int(year), 'target_year': int(year)+1}
            for lag in range(history):
                for feature in FEATURES:
                    row[f'{feature}_lag{lag}'] = records.loc[year-lag, feature]
            row['target'] = records.loc[year+1, 'employment'] if year+1 in records.index else np.nan
            result.append(row)
    return pd.DataFrame(result, columns=['university','year','target_year','target']+columns), columns


def configurations():
    for alpha in (.1, 1., 10.):
        yield f'Ridge {alpha}', make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    for layers in ((16,), (32,16), (32,16,8)):
        yield f'MLP {layers}', TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=layers,
                alpha=.1, solver='lbfgs', max_iter=2500, max_fun=30000, random_state=42)),
            transformer=StandardScaler())


def run(df, history=3):
    all_rows, columns = windows(df, history)
    pairs = all_rows.dropna(subset=['target'])
    years = sorted(pairs.target_year.unique())
    if len(years) < 3:
        raise ValueError('Для эксперимента нужны минимум три целевых года после формирования истории')
    test_year = years[-1]
    train = pairs[pairs.target_year < test_year]
    test = pairs[pairs.target_year == test_year]
    models = dict(configurations())
    validation = []
    for name, prototype in models.items():
        for year in years[1:-1]:
            earlier = train[train.target_year < year]
            valid = train[train.target_year == year]
            model = clone(prototype).fit(earlier[columns], earlier.target)
            validation.append({'model': name, 'year': int(year), **scores(valid.target, model.predict(valid[columns]))})
    folds = pd.DataFrame(validation)
    ranking = folds.groupby('model', as_index=False).agg(MAE=('MAE','mean'),RMSE=('RMSE','mean')).sort_values(['MAE','model'])
    selected = ranking.iloc[0].model
    comparison = []
    predictions = []
    choices = [ranking[ranking.model.str.startswith(prefix)].iloc[0].model for prefix in ('Ridge','MLP')]
    for name in choices + ['Наивный прогноз','Линейный тренд']:
        if name in models:
            model = clone(models[name]).fit(train[columns], train.target)
            values = model.predict(test[columns])
        elif name == 'Наивный прогноз':
            values = test.employment_lag0
        else:
            # МНК-прогноз на один год вперёд для каждого ряда.
            x = np.arange(history)
            values = [np.polyval(np.polyfit(x, [row[f'employment_lag{lag}'] for lag in reversed(range(history))], 1), history) for _,row in test.iterrows()]
        values = np.clip(values, 0, 100)
        comparison.append({'model':name, **scores(test.target, values)})
        out = test[['university','target_year','target']].copy()
        out['model'] = name
        out['prediction'] = values
        predictions.append(out)
    latest = all_rows[all_rows.year == df.year.max()].copy()
    future = latest[['university','target_year']].copy()
    if not latest.empty:
        model = clone(models[selected]).fit(pairs[columns], pairs.target)
        future['forecast'] = np.clip(model.predict(latest[columns]),0,100)
    else:
        future['forecast'] = pd.Series(dtype=float)
    return {'folds':folds, 'ranking':ranking, 'comparison':pd.DataFrame(comparison),
            'predictions':pd.concat(predictions,ignore_index=True), 'forecast':future,
            'manifest':{'history':history,'features':len(columns),'train':len(train),'test':len(test),
                        'test_year':int(test_year),'winner':selected,'validation_years':[int(y) for y in years[1:-1]]}}


def export(df, destination):
    path = Path(destination)
    path.mkdir(parents=True,exist_ok=True)
    results = run(df)
    for name, value in results.items():
        if isinstance(value,pd.DataFrame):
            value.to_csv(path/f'{name}.csv',index=False)
        else:
            (path/f'{name}.json').write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv')
    parser.add_argument('--output',default='lag_results')
    args = parser.parse_args()
    print(export(load_data(args.csv),args.output)['comparison'].to_string(index=False))
