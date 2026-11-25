"""Офлайн-резервирование и восстановление доверенных локальных данных."""
import argparse
import json
import shutil
import sqlite3
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from model_registry import Registry, digest


def verify_database(path):
    with sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True) as db:
        if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise ValueError('Повреждена база данных')
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {'users', 'datasets', 'audit'} <= tables:
            raise ValueError('Неизвестная схема базы данных')


def create_backup(database, registry, destination):
    database, destination = Path(database), Path(destination)
    if not database.is_file() or destination.exists():
        raise ValueError('База отсутствует или архив уже существует')
    verify_database(database)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temp:
        root = Path(temp) / 'snapshot'
        root.mkdir()
        with sqlite3.connect(database.resolve().as_uri() + '?mode=ro', uri=True) as source:
            with sqlite3.connect(root / 'unimonitor.db') as target:
                source.backup(target)
        verify_database(root / 'unimonitor.db')
        original = Registry(registry)
        copied = Registry(root / 'model_runs')
        for run in original.list():
            run_id = run['id']
            original.verify(run_id)
            shutil.copytree(original.path(run_id), copied.path(run_id))
            copied.verify(run_id)
        files = {p.relative_to(root).as_posix(): digest(p) for p in root.rglob('*') if p.is_file()}
        (root / 'backup.json').write_text(json.dumps({'format': 1, 'files': files}, indent=2), encoding='utf-8')
        archive = Path(temp) / 'backup.zip'
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
            for p in root.rglob('*'):
                if p.is_file():
                    z.write(p, p.relative_to(root).as_posix())
        archive.replace(destination)
    return {'files': len(files), 'sha256': digest(destination)}


def restore_backup(archive, destination):
    destination = Path(destination)
    if destination.exists():
        raise ValueError('Восстановление допускается только в новый каталог')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temp:
        root = Path(temp) / 'restore'
        root.mkdir()
        with zipfile.ZipFile(archive) as z:
            infos = z.infolist()
            names = [i.filename for i in infos]
            if len(names) != len(set(names)) or sum(i.file_size for i in infos) > 2_000_000_000:
                raise ValueError('Некорректный архив')
            for name in names:
                p = PurePosixPath(name)
                if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name:
                    raise ValueError('Недопустимый путь в архиве')
            meta = json.loads(z.read('backup.json'))
            if meta.get('format') != 1 or set(names) != set(meta['files']) | {'backup.json'}:
                raise ValueError('Неполный состав архива')
            for name, checksum in meta['files'].items():
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(name) as source, target.open('wb') as out:
                    shutil.copyfileobj(source, out)
                if digest(target) != checksum:
                    raise ValueError('Нарушена целостность: ' + name)
        verify_database(root / 'unimonitor.db')
        registry = Registry(root / 'model_runs')
        for run in registry.list():
            registry.verify(run['id'])
        root.rename(destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='action', required=True)
    backup = subs.add_parser('backup')
    backup.add_argument('database')
    backup.add_argument('registry')
    backup.add_argument('archive')
    restore = subs.add_parser('restore')
    restore.add_argument('archive')
    restore.add_argument('destination')
    args = parser.parse_args()
    if args.action == 'backup':
        print(create_backup(args.database, args.registry, args.archive))
    else:
        print(restore_backup(args.archive, args.destination))


if __name__ == '__main__':
    main()
