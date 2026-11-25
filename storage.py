"""SQLite storage and local authentication for the teaching prototype."""
import hashlib
import hmac
import secrets
import sqlite3
from contextlib import contextmanager
from io import StringIO
from core import load_data

class Store:
    def __init__(self, path):
        self.path = path
        with self.connect() as db:
            db.executescript('''CREATE TABLE IF NOT EXISTS users(login TEXT PRIMARY KEY, salt TEXT NOT NULL, digest TEXT NOT NULL, role TEXT NOT NULL CHECK(role IN ('admin','analyst','viewer')));
            CREATE TABLE IF NOT EXISTS datasets(id INTEGER PRIMARY KEY, name TEXT NOT NULL, source TEXT NOT NULL, kind TEXT NOT NULL CHECK(kind IN ('demo','external')), csv TEXT NOT NULL, sha256 TEXT NOT NULL, created TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY, login TEXT NOT NULL, action TEXT NOT NULL, detail TEXT NOT NULL, created TEXT DEFAULT CURRENT_TIMESTAMP);''')
    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()
    @staticmethod
    def digest(password, salt):
        return hashlib.pbkdf2_hmac('sha256', password.encode(), bytes.fromhex(salt), 600000).hex()
    def bootstrap(self, login, password):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            if db.execute('SELECT 1 FROM users LIMIT 1').fetchone():
                return False
            self._insert_user(db, login, password, 'admin')
            self._audit(db, login, 'bootstrap', 'initial administrator')
            return True
    def _insert_user(self, db, login, password, role):
        if not login.strip() or len(password)<12 or role not in ('admin','analyst','viewer'):
            raise ValueError('Укажите логин, роль и пароль не короче 12 символов')
        salt = secrets.token_hex(16)
        db.execute('INSERT INTO users VALUES(?,?,?,?)',(login,salt,self.digest(password,salt),role))
    def authenticate(self, login, password):
        with self.connect() as db:
            row = db.execute('SELECT * FROM users WHERE login=?',(login,)).fetchone()
            salt = row['salt'] if row else '00'*16
            valid = hmac.compare_digest(self.digest(password,salt), row['digest'] if row else '0'*64)
            return row['role'] if row and valid else None
    def require(self, db, login, roles):
        row=db.execute('SELECT role FROM users WHERE login=?',(login,)).fetchone()
        if not row or row['role'] not in roles:
            raise PermissionError('Недостаточно прав')
    def _audit(self, db, login, action, detail):
        db.execute('INSERT INTO audit(login,action,detail) VALUES(?,?,?)',(login,action,detail))
    def create_user(self, actor, login, password, role):
        with self.connect() as db:
            self.require(db,actor,('admin',))
            self._insert_user(db,login,password,role)
            self._audit(db,actor,'create_user',login)
    def save(self, actor, name, source, kind, csv):
        if kind not in ('demo','external'):
            raise ValueError('Некорректный тип набора')
        if not name.strip() or not source.strip(): raise ValueError('Укажите название и источник')
        if len(csv.encode())>10_000_000: raise ValueError('Файл превышает 10 МБ')
        frame=load_data(StringIO(csv))
        if frame.empty: raise ValueError('Нет наблюдений')
        normalized=frame.to_csv(index=False)
        checksum=hashlib.sha256(normalized.encode()).hexdigest()
        with self.connect() as db:
            self.require(db,actor,('admin','analyst'))
            cur=db.execute('INSERT INTO datasets(name,source,kind,csv,sha256) VALUES(?,?,?,?,?)',(name,source,kind,normalized,checksum))
            self._audit(db,actor,'import',str(cur.lastrowid))
            return cur.lastrowid
    def list(self, actor):
        with self.connect() as db:
            self.require(db,actor,('admin','analyst','viewer'))
            return [dict(r) for r in db.execute('SELECT id,name,source,kind,sha256,created FROM datasets ORDER BY id DESC')]
    def read(self, actor, dataset):
        with self.connect() as db:
            self.require(db,actor,('admin','analyst','viewer'))
            row=db.execute('SELECT * FROM datasets WHERE id=?',(dataset,)).fetchone()
            if not row: raise ValueError('Набор не найден')
            actual = hashlib.sha256(row['csv'].encode()).hexdigest()
            if not hmac.compare_digest(actual, row['sha256']):
                raise ValueError('Нарушена целостность набора данных')
            return dict(row)
    def audit(self, actor):
        with self.connect() as db:
            self.require(db,actor,('admin',))
            return [dict(r) for r in db.execute('SELECT * FROM audit ORDER BY id DESC LIMIT 500')]
