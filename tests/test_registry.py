import pytest
from core import load_data
from model_registry import Registry

@pytest.fixture(scope='module')
def run(tmp_path_factory):
    registry = Registry(tmp_path_factory.mktemp('models'))
    frame = load_data('data/demo.csv')
    meta = registry.train(frame,'Учебный набор','demo','Ridge alpha=1.0')
    return registry,frame,meta

def test_manifest(run):
    registry,frame,meta=run
    assert registry.verify(meta['id']) == meta
    assert meta['test_year']==2025
    assert meta['train_rows']+meta['test_rows']==meta['production_rows']==360
    assert meta['kind']=='demo'

def test_reload(run):
    registry,frame,meta=run
    result=registry.predict(meta['id'],frame)
    assert len(result)==60 and result.forecast.between(0,100).all()
    assert result.forecast_year.eq(2026).all()

def test_list(run):
    registry,_,meta=run
    assert registry.list()[0]['id']==meta['id']

def test_integrity(run):
    registry,_,meta=run
    file=registry.path(meta['id'])/'predictions.csv'
    original=file.read_bytes()
    try:
        file.write_bytes(original+b'broken')
        with pytest.raises(ValueError,match='целостность'):
            registry.verify(meta['id'])
    finally:
        file.write_bytes(original)

def test_path(run):
    with pytest.raises(ValueError):
        run[0].path('../outside')

def test_bad_model(run):
    with pytest.raises(ValueError):
        run[0].train(run[1],'demo','demo','unknown')
