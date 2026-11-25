"""Локальное управление обучением и резервными копиями SQLite."""
import sqlite3
import os
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import json
from uuid import uuid4
from datetime import datetime, timezone
from model_registry import digest


def backup_database(store, actor, destination):
    destination = Path(destination)
    with store.connect() as db:
        store.require(db, actor, ('admin',))
        if destination.resolve() == Path(store.path).resolve():
            raise ValueError('Нельзя заменить рабочую базу')
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, name = tempfile.mkstemp(dir=destination.parent)
        os.close(fd)
        try:
            target = sqlite3.connect(name)
            try:
                db.backup(target)
                if target.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                    raise ValueError('Повреждённая копия')
            finally:
                target.close()
            os.replace(name, destination)
        finally:
            if os.path.exists(name):
                os.unlink(name)
    return {'file': destination.name, 'sha256': digest(destination), 'bytes': destination.stat().st_size}


def diagnostics(store, actor):
    with store.connect() as db:
        store.require(db, actor, ('admin',))
        return {'Целостность SQLite': db.execute('PRAGMA quick_check').fetchone()[0],
                'Пользователи': db.execute('SELECT count(*) FROM users').fetchone()[0],
                'Наборы данных': db.execute('SELECT count(*) FROM datasets').fetchone()[0],
                'Записи журнала': db.execute('SELECT count(*) FROM audit').fetchone()[0]}


class TrainingQueue:
    def __init__(self, registry, limit=16, store=None):
        self.registry = registry
        self.limit = limit
        self.lock = Lock()
        self.jobs = {}
        self.store = store
        if store is not None:
            with store.connect() as db:
                db.execute('CREATE TABLE IF NOT EXISTS training_jobs(id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
                for row in db.execute('SELECT payload FROM training_jobs ORDER BY rowid'):
                    job = json.loads(row[0])
                    if job['status'] in ('queued', 'running'):
                        job.update(status='interrupted', error='Перезапуск сервера; запустите обучение повторно')
                    self.jobs[job['id']] = job
            for job in self.jobs.values():
                self._persist(job)

        self.executor = ThreadPoolExecutor(max_workers=1)

    def _persist(self, job):
        if self.store is not None:
            with self.store.connect() as db:
                db.execute('INSERT INTO training_jobs VALUES (?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload', (job['id'], json.dumps(job, ensure_ascii=False)))

    def submit(self, actor, frame, source, kind, model):
        with self.lock:
            if sum(j['status'] in ('queued', 'running') for j in self.jobs.values()) >= self.limit:
                raise ValueError('Очередь заполнена')
            job_id = uuid4().hex
            self.jobs[job_id] = dict(id=job_id, actor=actor, model=model, status='queued', created=datetime.now(timezone.utc).isoformat())
            self._persist(self.jobs[job_id])
        self.executor.submit(self._run, job_id, frame.copy(deep=True), source, kind, model)
        return job_id

    def _run(self, job_id, frame, source, kind, model):
        with self.lock:
            self.jobs[job_id]['status'] = 'running'
            self._persist(self.jobs[job_id])
        try:
            result = self.registry.train(frame, source, kind, model)
            update = dict(status='completed', experiment=result['id'])
        except Exception as exc:
            update = dict(status='failed', error=str(exc))
        with self.lock:
            self.jobs[job_id].update(update)
            self._persist(self.jobs[job_id])

    def list(self):
        with self.lock:
            return [dict(j) for j in reversed(list(self.jobs.values()))]

    def close(self):
        self.executor.shutdown(wait=True)
