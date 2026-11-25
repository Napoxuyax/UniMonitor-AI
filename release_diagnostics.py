"""Диагностика полноты панели и устойчивости прогнозов по годам."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from core import FEATURES, load_data
from research import build_pairs, scores


def panel_quality(df):
    if df.empty or df.duplicated(['university', 'year']).any():
        raise ValueError('Нужна непустая панель с уникальными ключами вуз–год')
    years = sorted(df.year.unique())
    all_names = set(df.university)
    rows = []
    previous = set()
    for year in years:
        part = df[df.year == year]
        names = set(part.university)
        rows.append({'year': int(year), 'observed': len(names),
                     'panel_coverage': len(names)/len(all_names),
                     'entered': len(names-previous) if previous else None,
                     'exited': len(previous-names) if previous else None,
                     'zero_values': int(part[FEATURES].eq(0).sum().sum())})
        previous = names
    gaps = []
    for name, part in df.groupby('university'):
        observed = set(part.year.astype(int))
        missing = sorted(set(range(min(observed), max(observed)+1))-observed)
        gaps.append({'university': name, 'observations': len(observed),
                     'first_year': min(observed), 'last_year': max(observed),
                     'internal_gaps': len(missing),
                     'missing_years': ', '.join(map(str, missing))})
    return pd.DataFrame(rows), pd.DataFrame(gaps)


def rolling_evaluation(df, minimum_years=2):
    """Фиксированные модели: каждый год оценивается только по прошлым данным."""
    if minimum_years < 1:
        raise ValueError('Минимум один обучающий год')
    pairs = build_pairs(df)
    years = sorted(pairs.target_year.unique())
    if len(years) <= minimum_years:
        raise ValueError('Недостаточно целевых лет')
    models = {'Ridge 1': make_pipeline(StandardScaler(), Ridge(alpha=1)),
              'Ridge 10': make_pipeline(StandardScaler(), Ridge(alpha=10))}
    records = []
    predictions = []
    for year in years[minimum_years:]:
        train = pairs[pairs.target_year < year]
        test = pairs[pairs.target_year == year]
        estimates = {name: clone(model).fit(train[FEATURES], train.target).predict(test[FEATURES])
                     for name, model in models.items()}
        estimates['Наивный прогноз'] = test.employment.to_numpy()
        for name, pred in estimates.items():
            pred = np.clip(pred, 0, 100)
            err = pred-test.target.to_numpy()
            records.append({'year': int(year), 'model': name, 'train_count': len(train),
                            'test_count': len(test), 'max_train_year': int(train.target_year.max()),
                            'bias': float(err.mean()), **scores(test.target, pred)})
            result = test[['university','target_year','target']].copy()
            result['model'] = name
            result['prediction'] = pred
            result['error'] = err
            predictions.append(result)
    folds = pd.DataFrame(records)
    baseline = folds[folds.model=='Наивный прогноз'].set_index('year').MAE
    folds['baseline_mae'] = folds.year.map(baseline)
    folds['improved'] = folds.MAE < folds.baseline_mae
    summary = folds.groupby('model', as_index=False).agg(
        mean_MAE=('MAE','mean'), worst_MAE=('MAE','max'),
        std_MAE=('MAE','std'), improved_years=('improved','sum'), years=('year','size'))
    return folds, summary, pd.concat(predictions, ignore_index=True)


def export(df, destination):
    path = Path(destination)
    path.mkdir(parents=True, exist_ok=True)
    coverage, gaps = panel_quality(df)
    folds, summary, predictions = rolling_evaluation(df)
    tables = dict(coverage=coverage, gaps=gaps, folds=folds, summary=summary, predictions=predictions)
    for name, table in tables.items():
        table.to_csv(path/f'{name}.csv', index=False)
    manifest = {'observations':len(df), 'universities':int(df.university.nunique()),
                'evaluated_years': sorted(folds.year.unique().astype(int).tolist()),
                'method':'fixed-model rolling evaluation; not independent model selection'}
    (path/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return tables


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv')
    parser.add_argument('--output', default='release_diagnostics_results')
    args = parser.parse_args()
    print(export(load_data(args.csv), args.output)['summary'].to_string(index=False))
