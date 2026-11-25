"""Анализ чувствительности прогнозов и выхода за диапазон обучения."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from core import FEATURES, load_data
from research import build_pairs


def analyze_sensitivity(df, year=None, step=10):
    if not np.isfinite(step) or not 0 < step <= 50:
        raise ValueError('Шаг должен быть в диапазоне (0; 50]')
    year = int(df.year.max()) if year is None else int(year)
    history = df[df.year <= year].copy()
    latest = history[history.year == year].copy()
    pairs = build_pairs(history)
    if latest.empty or pairs.target_year.nunique() < 2:
        raise ValueError('Недостаточно последовательных наблюдений')
    x = pairs[FEATURES]
    lower, upper = x.min(), x.max()
    model = make_pipeline(StandardScaler(), Ridge(alpha=1)).fit(x, pairs.target)
    original = latest[FEATURES].astype(float)
    baseline = np.clip(model.predict(original), 0, 100)
    base_outside = (original.lt(lower) | original.gt(upper)).any(axis=1)
    records = []
    for feature in FEATURES:
        for direction in (-1, 1):
            changed = original.copy()
            changed[feature] *= 1 + direction * step / 100
            if feature in ('score', 'employment'):
                changed[feature] = changed[feature].clip(0, 100)
            prediction = np.clip(model.predict(changed), 0, 100)
            outside = changed.lt(lower) | changed.gt(upper)
            for pos, (_, row) in enumerate(latest.iterrows()):
                records.append(dict(university=row.university, year=year,
                    forecast_year=year+1, feature=feature, change_percent=direction*step,
                    original_value=float(original.iloc[pos][feature]),
                    changed_value=float(changed.iloc[pos][feature]),
                    baseline=float(baseline[pos]), forecast=float(prediction[pos]),
                    delta=float(prediction[pos]-baseline[pos]),
                    baseline_outside=bool(base_outside.iloc[pos]),
                    scenario_outside=bool(outside.iloc[pos].any()),
                    changed_feature_outside=bool(outside.iloc[pos][feature])))
    details = pd.DataFrame(records)
    details['absolute_delta'] = details.delta.abs()
    summary = details.groupby(['feature','change_percent'], as_index=False).agg(
        mean_delta=('delta','mean'), mean_absolute_delta=('absolute_delta','mean'),
        max_absolute_delta=('absolute_delta','max'),
        outside_count=('scenario_outside','sum'),
        changed_feature_outside_count=('changed_feature_outside','sum'),
        observations=('delta','size'))
    summary['outside_share'] = summary.outside_count / summary.observations
    ranges = pd.DataFrame({'feature':FEATURES,'minimum':lower.values,'maximum':upper.values})
    manifest = dict(year=year, forecast_year=year+1, step=step, train_count=len(pairs),
        universities=len(latest), baseline_outside_count=int(base_outside.sum()),
        model='Ridge alpha=1', note='Условная чувствительность, не причинный эффект; выход за диапазон — предупреждение, не доказательство ошибки.')
    return details, summary, ranges, manifest


def export(df, destination, year=None, step=10):
    path = Path(destination)
    path.mkdir(parents=True, exist_ok=True)
    details, summary, ranges, manifest = analyze_sensitivity(df, year, step)
    for name, frame in [('details', details), ('summary', summary), ('ranges', ranges)]:
        frame.to_csv(path / f'{name}.csv', index=False)
    (path/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv')
    parser.add_argument('--output', default='sensitivity_results')
    parser.add_argument('--year', type=int)
    parser.add_argument('--step', type=float, default=10)
    args = parser.parse_args()
    print(export(load_data(args.csv), args.output, args.year, args.step).to_string(index=False))
