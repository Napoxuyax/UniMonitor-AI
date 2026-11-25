from io import StringIO
import numpy as np
import pytest
from core import load_data
from storage import Store
from model_registry import Registry
from release_tools import create_backup, restore_backup


def test_empty_input():
    with pytest.raises(ValueError, match='Нет наблюдений'):
        load_data(StringIO('university,region,year,students,score,research,income,employment\n'))


def test_full_backup_with_trained_model(tmp_path):
    db = tmp_path / 'source.db'
    store = Store(db)
    store.bootstrap('admin', 'password-123456')
    frame = load_data('data/demo.csv')
    dataset = store.save('admin', 'Учебный', 'Синтетический набор', 'demo', frame.to_csv(index=False))
    registry = Registry(tmp_path / 'registry')
    run = registry.train(frame, 'Синтетический набор', 'demo', 'Ridge alpha=1.0')
    original = registry.predict(run['id'], frame)
    archive = tmp_path / 'snapshot.zip'
    create_backup(db, registry.root, archive)
    restored = restore_backup(archive, tmp_path / 'restored')
    recovered = Registry(restored / 'model_runs')
    assert recovered.verify(run['id']) == run
    np.testing.assert_allclose(recovered.predict(run['id'], frame).forecast, original.forecast)
    recovered_store = Store(restored / 'unimonitor.db')
    assert recovered_store.authenticate('admin', 'password-123456') == 'admin'
    assert recovered_store.read('admin', dataset)['sha256'] == store.read('admin', dataset)['sha256']
