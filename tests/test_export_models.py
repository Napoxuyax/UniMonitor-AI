import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from export_models import export_models
from core import FEATURES
from model_registry import digest


def test_export_reproduces_predictions(tmp_path):
    source = Path(__file__).resolve().parents[1] / 'data/demo.csv'
    output = tmp_path / 'models'
    meta = export_models(source, output, 'Синтетические данные', 'demo')
    train = pd.read_csv(output / 'train.csv')
    test = pd.read_csv(output / 'test.csv')
    predictions = pd.read_csv(output / 'predictions.csv')
    assert train.target_year.max() < test.target_year.min()
    assert meta['kind'] == 'demo'
    for i in (1, 2):
        model = joblib.load(output / f'model{i}.joblib')
        np.testing.assert_allclose(np.clip(model.predict(test[FEATURES]), 0, 100), predictions[f'model{i}'])
    for filename, checksum in meta['files'].items():
        assert digest(output / filename) == checksum
    assert json.loads((output / 'manifest.json').read_text()) == meta
    with pytest.raises(FileExistsError):
        export_models(source, output, 'Синтетические данные', 'demo')


def test_requires_provenance(tmp_path):
    with pytest.raises(ValueError):
        export_models('missing.csv', tmp_path, '', 'external')
