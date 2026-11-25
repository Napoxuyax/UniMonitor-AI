import sqlite3
import pytest
from operations import TrainingQueue, backup_database, diagnostics
from storage import Store
from core import load_data
from pathlib import Path


def test_backup_and_diagnostics(tmp_path):
    store = Store(tmp_path/'db.sqlite')
    store.bootstrap('admin','password123456')
    store.create_user('admin','viewer','password123456','viewer')
    result = backup_database(store,'admin',tmp_path/'copy.sqlite')
    assert len(result['sha256']) == 64
    with sqlite3.connect(tmp_path/'copy.sqlite') as db:
        assert db.execute('SELECT count(*) FROM users').fetchone()[0] == 2
    assert diagnostics(store,'admin')['Целостность SQLite'] == 'ok'
    with pytest.raises(PermissionError):
        backup_database(store,'viewer',tmp_path/'bad.sqlite')
    with pytest.raises(ValueError):
        backup_database(store,'admin',tmp_path/'db.sqlite')


def test_queue_success_and_failure():
    class Registry:
        def train(self, frame, source, kind, model):
            if model == 'bad':
                raise ValueError('bad model')
            return {'id':'example'}
    queue = TrainingQueue(Registry())
    frame = load_data(Path(__file__).resolve().parents[1]/'data/demo.csv')
    queue.submit('admin',frame,'demo','demo','good')
    queue.submit('admin',frame,'demo','demo','bad')
    queue.close()
    assert [j['status'] for j in queue.list()] == ['failed','completed']
