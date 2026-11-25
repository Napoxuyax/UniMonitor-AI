"""Локальный реестр воспроизводимых экспериментов. Модели только собственного обучения."""
import argparse
import hashlib
import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from core import FEATURES, load_data
from research import build_pairs, candidates, scores


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


class Registry:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, run_id):
        if uuid.UUID(run_id).hex != run_id:
            raise ValueError('Некорректный идентификатор эксперимента')
        return self.root / run_id

    def train(self, frame, source, kind, model_name):
        if kind not in ('demo', 'external') or not str(source).strip():
            raise ValueError('Укажите происхождение данных')
        from io import StringIO
        frame = load_data(StringIO(frame.to_csv(index=False)))
        models = dict(candidates())
        if model_name not in models:
            raise ValueError('Неизвестная конфигурация модели')
        pairs = build_pairs(frame)
        years = sorted(pairs.target_year.unique())
        if len(years) < 4:
            raise ValueError('Необходимо минимум четыре целевых года')
        test_year = years[-1]
        train = pairs[pairs.target_year < test_year]
        test = pairs[pairs.target_year == test_year]
        model = models[model_name]
        model.fit(train[FEATURES], train.target)
        prediction = np.clip(model.predict(test[FEATURES]), 0, 100)
        metrics = scores(test.target, prediction)
        run_id = uuid.uuid4().hex
        with tempfile.TemporaryDirectory(dir=self.root) as folder:
            work = Path(folder) / run_id
            work.mkdir()
            frame.to_csv(work / 'dataset.csv', index=False)
            results = test[['university','year','target_year','target']].copy()
            results['prediction'] = prediction
            results.to_csv(work / 'predictions.csv', index=False)
            joblib.dump(model, work / 'evaluated.joblib')
            model.fit(pairs[FEATURES], pairs.target)
            joblib.dump(model, work / 'production.joblib')
            meta = {'id': run_id, 'created_utc': datetime.now(timezone.utc).isoformat(),
                    'model': model_name, 'source': source, 'kind': kind,
                    'features': FEATURES, 'sklearn': sklearn.__version__,
                    'train_rows': len(train), 'test_rows': len(test),
                    'production_rows': len(pairs), 'test_year': int(test_year),
                    'metrics': metrics, 'baseline': scores(test.target,test.employment),
                    'files': {p.name: digest(p) for p in work.iterdir()}}
            (work / 'manifest.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
            os.replace(work, self.path(run_id))
        return meta

    def read(self, run_id):
        path = self.path(run_id)
        meta = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
        if meta['id'] != run_id:
            raise ValueError('Нарушена целостность идентификатора')
        return meta

    def list(self):
        return sorted((self.read(p.name) for p in self.root.iterdir()
                       if p.is_dir() and (p / 'manifest.json').is_file()),
                      key=lambda item: item['created_utc'], reverse=True)

    def verify(self, run_id):
        meta = self.read(run_id)
        path = self.path(run_id)
        expected = {'dataset.csv','predictions.csv','evaluated.joblib','production.joblib'}
        if set(meta['files']) != expected:
            raise ValueError('Неполный состав эксперимента')
        for name, checksum in meta['files'].items():
            if digest(path / name) != checksum:
                raise ValueError('Нарушена целостность файла: '+name)
        return meta

    def predict(self, run_id, frame):
        # joblib не предназначен для загрузки чужих файлов. Реестр должен быть доверенным.
        meta = self.verify(run_id)
        if meta['sklearn'] != sklearn.__version__:
            raise ValueError('Версия scikit-learn отличается от среды обучения')
        model = joblib.load(self.path(run_id) / 'production.joblib')
        latest = frame[frame.year == frame.year.max()].copy()
        latest['forecast_year'] = latest.year + 1
        latest['forecast'] = np.clip(model.predict(latest[FEATURES]), 0, 100)
        return latest[['university','forecast_year','forecast']]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Обучение и регистрация локальной модели')
    parser.add_argument('csv')
    parser.add_argument('--source', required=True)
    parser.add_argument('--kind', choices=['demo','external'], required=True)
    parser.add_argument('--model', choices=list(dict(candidates())), default='Ridge alpha=1.0')
    parser.add_argument('--registry', default='model_runs')
    args = parser.parse_args()
    print(json.dumps(Registry(args.registry).train(load_data(args.csv), args.source, args.kind, args.model), ensure_ascii=False, indent=2))
