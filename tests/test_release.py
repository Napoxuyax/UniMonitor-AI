import zipfile
import pytest
from storage import Store
from model_registry import Registry
from release_tools import create_backup, restore_backup


def test_roundtrip(tmp_path):
    db = tmp_path/'db.sqlite'
    Store(db).bootstrap('admin','password123456')
    reg = Registry(tmp_path/'registry')
    archive = tmp_path/'backup.zip'
    create_backup(db, reg.root, archive)
    restored = restore_backup(archive, tmp_path/'restored')
    assert Store(restored/'unimonitor.db').authenticate('admin','password123456') == 'admin'
    with pytest.raises(ValueError):
        restore_backup(archive, restored)


def test_tampered_backup(tmp_path):
    db = tmp_path/'db.sqlite'
    Store(db)
    archive = tmp_path/'backup.zip'
    create_backup(db, tmp_path/'registry', archive)
    bad = tmp_path/'bad.zip'
    with zipfile.ZipFile(archive) as src, zipfile.ZipFile(bad,'w') as dst:
        for name in src.namelist():
            dst.writestr(name, b'bad' if name == 'unimonitor.db' else src.read(name))
    with pytest.raises(ValueError, match='целостность'):
        restore_backup(bad, tmp_path/'restored')
    assert not (tmp_path/'restored').exists()


def test_path_traversal(tmp_path):
    archive = tmp_path/'bad.zip'
    with zipfile.ZipFile(archive,'w') as z:
        z.writestr('../escape','bad')
    with pytest.raises(ValueError, match='путь'):
        restore_backup(archive, tmp_path/'restored')
    assert not (tmp_path/'escape').exists()
