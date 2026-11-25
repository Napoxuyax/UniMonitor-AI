"""Экспорт двух прогнозных моделей и воспроизводимых контрольных выборок."""
import argparse
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from core import FEATURES, load_data
from research import build_pairs, candidates, scores, experiment
from model_registry import digest


def export_models(dataset, output, source, kind):
    if kind not in ('demo', 'external') or not source.strip():
        raise ValueError('Укажите тип и источник данных')
    frame = load_data(dataset)
    result = experiment(frame)
    pairs = build_pairs(frame)
    year = result['test_year']
    train = pairs[pairs.target_year < year]
    test = pairs[pairs.target_year == year]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    train.to_csv(output / 'train.csv', index=False)
    test.to_csv(output / 'test.csv', index=False)
    result['ranking'].to_csv(output / 'validation.csv', index=False)
    configs = dict(candidates())
    models = []
    predictions = test[['university', 'target_year', 'target']].copy()
    for number, prefix in enumerate(('MLP', 'Ridge'), 1):
        name = result['ranking'].loc[result['ranking'].model.str.startswith(prefix)].iloc[0].model
        model = configs[name]
        model.fit(train[FEATURES], train.target)
        filename = f'model{number}.joblib'
        joblib.dump(model, output / filename)
        pred = np.clip(model.predict(test[FEATURES]), 0, 100)
        restored = joblib.load(output / filename)
        np.testing.assert_allclose(restored.predict(test[FEATURES]), model.predict(test[FEATURES]))
        predictions[f'model{number}'] = pred
        models.append({'file': filename, 'configuration': name, 'metrics': scores(test.target, pred)})
    predictions.to_csv(output / 'predictions.csv', index=False)
    meta = {'source': source, 'kind': kind, 'dataset_sha256': digest(dataset),
            'python': platform.python_version(), 'sklearn': sklearn.__version__,
            'numpy': np.__version__, 'pandas': pd.__version__, 'features': FEATURES,
            'test_year': year, 'train_rows': len(train), 'test_rows': len(test),
            'models': models, 'baseline': scores(test.target, test.employment),
            'files': {p.name: digest(p) for p in output.iterdir()}}
    (output / 'manifest.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--source', required=True)
    parser.add_argument('--kind', choices=['demo', 'external'], required=True)
    args = parser.parse_args()
    print(json.dumps(export_models(args.dataset, args.output, args.source, args.kind), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
