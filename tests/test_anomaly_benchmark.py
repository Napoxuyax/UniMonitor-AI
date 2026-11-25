from pathlib import Path
import pytest
from core import load_data
from anomaly_benchmark import inject, benchmark


def frame():
    return load_data(Path(__file__).resolve().parents[1]/'data/demo.csv')


def test_injection():
    df=frame()
    latest=df[df.year==2025]
    out=inject(latest)
    assert len(out)==75 and out.injected.sum()==15
    assert out.employment.between(0,100).all()
    assert out.score.between(0,100).all()
    assert len(df)==420


def test_benchmark():
    metrics, predictions, manifest=benchmark(frame())
    assert len(metrics)==4 and len(predictions)==300
    assert manifest['train']==300 and manifest['test']==60
    assert manifest['calibration_year'] < manifest['test_year']
    assert metrics[['test_ap','roc_auc','precision','recall','f1']].apply(lambda x:x.between(0,1).all()).all()


def test_short_history():
    with pytest.raises(ValueError): benchmark(frame().query('year >= 2024'))
