from concurrent.futures import ThreadPoolExecutor
import pytest
from auth_guard import LoginGuard
from storage import Store
from core import load_data
from pathlib import Path


def test_throttle_window():
    now = [0]
    guard = LoginGuard(attempts=2, window=60, clock=lambda: now[0])
    assert guard.reserve('a') and guard.reserve('a')
    assert not guard.reserve('a')
    assert guard.reserve('b')
    now[0] = 60
    assert guard.reserve('a')


def test_throttle_concurrency():
    guard = LoginGuard(attempts=3)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(guard.reserve, ['a'] * 20))
    assert sum(results) == 3


def test_throttle_capacity_and_clear():
    guard = LoginGuard(capacity=1)
    assert guard.reserve('a')
    assert not guard.reserve('b')
    guard.clear('a')
    assert guard.reserve('b')


def test_dataset_integrity(tmp_path):
    store = Store(tmp_path / 'db')
    store.bootstrap('admin', 'password-admin-123')
    frame = load_data(Path(__file__).parents[1] / 'data/demo.csv')
    key = store.save('admin', 'demo', 'synthetic', 'demo', frame.to_csv(index=False))
    assert store.read('admin', key)['kind'] == 'demo'
    with store.connect() as db:
        db.execute('UPDATE datasets SET csv=? WHERE id=?', ('damaged', key))
    with pytest.raises(ValueError, match='целостность'):
        store.read('admin', key)


def test_invalid_dataset_kind(tmp_path):
    store = Store(tmp_path / 'db')
    store.bootstrap('admin', 'password-admin-123')
    with pytest.raises(ValueError, match='тип'):
        store.save('admin', 'name', 'source', 'unknown', '')
