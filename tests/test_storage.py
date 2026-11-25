import pytest
from storage import Store
from pathlib import Path
CSV=(Path(__file__).resolve().parents[1]/'data/demo.csv').read_text()
@pytest.fixture
def store(tmp_path):
    s=Store(tmp_path/'test.db'); assert s.bootstrap('admin','very-long-password'); return s

def test_auth(store):
    assert store.authenticate('admin','very-long-password')=='admin'
    assert store.authenticate('admin','wrong') is None
    assert not store.bootstrap('other','another-password')

def test_roles(store):
    store.create_user('admin','reader','reader-password','viewer')
    with pytest.raises(PermissionError): store.save('reader','demo','synthetic','demo',CSV)
    with pytest.raises(PermissionError): store.audit('reader')
    with pytest.raises(PermissionError): store.create_user('reader','x','long-password','admin')

def test_persistence(store):
    i=store.save('admin','demo','synthetic','demo',CSV)
    reopened=Store(store.path)
    assert reopened.read('admin',i)['kind']=='demo'
    assert len(reopened.list('admin'))==1
    assert reopened.audit('admin')[0]['action']=='import'

def test_rejected_import(store):
    with pytest.raises(ValueError): store.save('admin','bad','unknown','external','bad,data\n1,2')
    assert store.list('admin')==[]
