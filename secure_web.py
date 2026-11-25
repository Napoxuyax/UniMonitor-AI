"""Authenticated local interface; binds only to loopback."""
import argparse
import getpass
import secrets
import time
from threading import Lock
from http.cookies import SimpleCookie
from http.server import ThreadingHTTPServer
from io import StringIO
from urllib.parse import parse_qs, urlsplit, urlencode
from html import escape
import pandas as pd
from storage import Store
from auth_guard import LoginGuard
from operations import TrainingQueue, diagnostics
from research import candidates
from model_registry import Registry
from core import load_data
from cluster_diagnostics import evaluate_clusters
from anomaly_benchmark import benchmark
from lag_forecast import run as lag_run
from peer_analysis import run as peer_run
from university_profile import profile
from release_diagnostics import panel_quality, rolling_evaluation
from sensitivity import analyze_sensitivity
from temporal_analysis import detect_temporal_anomalies, anomaly_summary, trend_profiles
from reporting import build_report
from web_local import Handler, ROOT, layout, table

class SecureHandler(Handler):
    def session(self):
        cookie = SimpleCookie()
        try:
            cookie.load(self.headers.get('Cookie',''))
        except Exception:
            return None, None
        token = cookie.get('session')
        if not token:
            return None, None
        with self.server.lock:
            value = self.server.sessions.get(token.value)
            if value and value['expires'] > time.monotonic():
                return token.value, value.copy()
            self.server.sessions.pop(token.value, None)
        return None, None

    def redirect(self, target, cookie=None):
        self.send_response(303)
        self.send_header('Location', target)
        self.send_header('Cache-Control','no-store')
        if cookie is not None:
            self.send_header('Set-Cookie', cookie)
        self.send_header('Content-Length','0')
        self.end_headers()

    def page(self, title, body, session=None):
        html = layout('/', '<section><h2>'+escape(title)+'</h2>'+body+'</section>')
        if title == 'Фактический мониторинг вузов':
            html = html.replace('<h1>Обзор</h1>', '<h1>Фактический мониторинг вузов</h1>').replace('<h2>'+escape(title)+'</h2>', '', 1)
            html = html.replace('Синтетические учебные данные. Не официальная статистика.', 'Фактические данные мониторинга; экспериментальные оценки не являются официальным рейтингом.')
        else:
            html = html.replace('Синтетические учебные данные. Не официальная статистика.', 'Локальная система мониторинга. Источник данных указан в реестре.')
        return self.send(200, html)

    def login_page(self):
        return self.page('Вход в систему','<form method="post" action="/login"><p>Логин <input name="login" required></p><p>Пароль <input type="password" name="password" required></p><button>Войти</button></form>')

    def form(self, session, action, fields, label):
        return f'<form method="post" action="{action}"><input type="hidden" name="csrf" value="{session["csrf"]}">{fields}<button>{label}</button></form>'

    def do_GET(self):
        route = urlsplit(self.path).path
        token, session = self.session()
        if route == '/login':
            return self.login_page()
        if not session:
            return self.redirect('/login')
        store = self.server.store
        actor = session['login']
        try:
            if route in ('/actual-monitoring','/actual-download'):
                from actual_monitoring import load_panel
                from actual_dashboard import dashboard, select_rows
                frame = load_panel()
                query = parse_qs(urlsplit(self.path).query)
                if route == '/actual-download':
                    rows = select_rows(frame, int(query.get('year',['2025'])[0]), query.get('region',[''])[0], query.get('search',[''])[0][:200])
                    # Guard spreadsheet formula interpretation in text columns.
                    for col in ('name','region'):
                        rows[col] = rows[col].map(lambda v: "'"+v if isinstance(v,str) and v.startswith(('=','+','-','@','\t','\r')) else v)
                    return self.send(200,rows.to_csv(index=False).encode('utf-8-sig'),'text/csv; charset=utf-8','actual-monitoring.csv')
                return self.page('Фактический мониторинг вузов',dashboard(frame,query),session)
            if route == '/operations':
                if session['role'] != 'admin':
                    return self.send(403,'Недостаточно прав')
                options = ''.join('<option>'+escape(name)+'</option>' for name, _ in candidates())
                body = self.form(session,'/train','<select name="model">'+options+'</select>','Обучить на активном наборе')
                body += table(pd.DataFrame(self.server.training.list()))
                body += table(pd.DataFrame([diagnostics(store, actor)]))
                return self.page('Задачи обучения и диагностика',body,session)
            if route == '/experiments':
                if session['role'] != 'admin':
                    return self.send(403,'Недостаточно прав')
                records = self.server.registry.list()
                rows = [dict(Модель=r['model'], Источник=r['source'], Год=r['test_year'], MAE=r['metrics']['MAE'], RMSE=r['metrics']['RMSE']) for r in records]
                return self.page('Реестр экспериментов',table(pd.DataFrame(rows)) if rows else '<p>Экспериментов пока нет. Обучение запускается из командной строки.</p>',session)
            if route == '/datasets':
                rows = store.list(actor)
                body = '<p>Активный набор: '+escape(str(session.get('dataset') or 'демонстрационный'))+'</p>'
                if rows:
                    body += table(pd.DataFrame(rows))
                choices = '<option value="0">Демонстрационный набор</option>'+''.join(f'<option value="{r["id"]}">{escape(r["name"])}</option>' for r in rows)
                body += self.form(session,'/select','<select name="dataset">'+choices+'</select>','Выбрать')
                if session['role'] in ('admin','analyst'):
                    body += self.form(session,'/import','<p>Название <input name="name" required></p><p>Источник <input name="source" required></p><p>CSV (заголовок и строки)</p><textarea name="csv" rows="12" cols="65" required></textarea><p><select name="kind"><option value="external">Внешние данные</option><option value="demo">Учебные данные</option></select></p>','Импортировать')
                return self.page('Реестр наборов данных',body,session)
            if route == '/admin':
                if session['role'] != 'admin':
                    return self.send(403,'Недостаточно прав')
                body = self.form(session,'/users','<p>Логин <input name="login" required></p><p>Пароль <input name="password" type="password" minlength="12" required></p><select name="role"><option value="analyst">Аналитик</option><option value="viewer">Наблюдатель</option><option value="admin">Администратор</option></select>','Создать пользователя')
                return self.page('Администрирование',body+table(pd.DataFrame(store.audit(actor))),session)
            row = store.read(actor,session['dataset']) if session.get('dataset') else None
            self.active_frame = load_data(StringIO(row['csv'])) if row else self.server.frame
            self.active_source = row['source'] if row else 'Синтетический учебный набор'
            self.active_kind = row['kind'] if row else 'demo'
            if route == '/sensitivity':
                query = parse_qs(urlsplit(self.path).query)
                step = float(query.get('step',['10'])[0])
                year = int(query.get('year',[self.active_frame.year.max()])[0])
                details, summary, ranges, manifest = analyze_sensitivity(self.active_frame, year, step)
                body = '<p>Изменение одного фактора при неизменных остальных. Результат не является причинной оценкой.</p>'
                body += f'<form action="/sensitivity"><label>Год <input type="number" name="year" value="{year}"></label><label>Шаг, % <input type="number" name="step" min="1" max="50" value="{step}"></label><button>Рассчитать</button></form>'
                body += '<h2>Чувствительность прогнозов</h2>'+table(summary)
                body += '<h2>Диапазоны обучающих данных</h2>'+table(ranges)
                body += '<h2>Детализация (первые 100 записей)</h2>'+table(details.head(100))
                return self.page('Анализ чувствительности',body,session)
            if route == '/diagnostics':
                coverage, gaps = panel_quality(self.active_frame)
                body = '<p>Охват относительно всех организаций активного набора, а не генеральной совокупности.</p>'+table(coverage)+table(gaps.head(100))
                if parse_qs(urlsplit(self.path).query).get('run') == ['1']:
                    folds, summary, _ = rolling_evaluation(self.active_frame)
                    body += '<h2>Устойчивость прогнозов</h2>'+table(summary)+table(folds)
                else:
                    body += '<a href="/diagnostics?run=1">Оценить прогнозы по годам</a>'
                return self.page('Качество панели данных',body,session)
            if route == '/profile':
                query = parse_qs(urlsplit(self.path).query)
                year = int(query.get('year',[self.active_frame.year.max()])[0])
                names = sorted(self.active_frame.loc[self.active_frame.year.eq(year),'university'].unique())
                if not names:
                    raise ValueError('Нет данных за указанный год')
                name = query.get('university',[names[0]])[0]
                k = int(query.get('k',['10'])[0])
                result = profile(self.active_frame,name,year,k,self.active_source,self.active_kind,query.get('forecast')==['1'])
                years = ''.join(f'<option {"selected" if y==year else ""}>{int(y)}</option>' for y in sorted(self.active_frame.year.unique()))
                options = ''.join(f'<option {"selected" if n==name else ""}>{escape(n)}</option>' for n in names)
                body = '<p>Источник: '+escape(self.active_source)+'; тип: '+('учебные данные' if self.active_kind=='demo' else 'внешние данные')+'</p>'
                body += '<form action="/profile"><select name="year">'+years+'</select><button>Выбрать год</button></form>'
                body += f'<form action="/profile"><input type="hidden" name="year" value="{year}"><select name="university">{options}</select><input type="number" name="k" min="3" max="50" value="{k}"><label><input type="checkbox" name="forecast" value="1"> Рассчитать прогноз</label><button>Показать карточку</button></form>'
                body += '<h2>'+escape(name)+'</h2>'+table(pd.DataFrame(list(result['manifest']['status'].items()),columns=['Блок','Статус']))
                labels = {'indicators':'Показатели','history':'История','trends':'Тренды','anomalies':'Диагностика изменений','rank':'Рейтинг','rank_sensitivity':'Устойчивость рейтинга','peers':'Сравнение с вузами','recommendations':'Рекомендации','forecast':'Прогноз','forecast_metrics':'Контрольные метрики'}
                for key,label in labels.items():
                    body += '<h2>'+label+'</h2>'+(table(result[key]) if not result[key].empty else '<p>Нет результатов</p>')
                return self.page('Карточка вуза',body,session)
            if route == '/peers':
                query = parse_qs(urlsplit(self.path).query)
                year = int(query.get('year',[self.active_frame.year.max()])[0])
                k = int(query.get('k',['10'])[0])
                result = peer_run(self.active_frame,year,k)
                choices = ''.join(f'<option {"selected" if y==year else ""}>{int(y)}</option>' for y in sorted(self.active_frame.year.unique()))
                body = '<p>Сравнение по численности студентов, без учёта профиля. Результаты не доказывают причины отставания.</p>'
                body += f'<form action="/peers"><select name="year">{choices}</select><input type="number" name="k" min="3" max="50" value="{k}"><button>Сравнить</button></form>'
                body += '<h2>Сводка</h2>'+table(result['summary'])+'<h2>Рекомендации</h2>'+table(result['recommendations'])
                return self.page('Сопоставимые вузы',body,session)
            if route == '/lag-forecast':
                body = '<p>Прогноз по истории за три последовательных года. Последний целевой год исключён из выбора модели.</p><a class="btn" href="/lag-forecast?run=1">Выполнить эксперимент</a>'
                if parse_qs(urlsplit(self.path).query).get('run') == ['1']:
                    results = lag_run(self.active_frame)
                    body += '<h2>Контрольные метрики</h2>'+table(results['comparison'])+'<h2>Прогноз</h2>'+table(results['forecast'])
                return self.page('Прогноз по истории',body,session)
            if route == '/anomaly-benchmark':
                metrics, _, manifest = benchmark(self.active_frame)
                body = '<p>Проверка на искусственно внесённых отклонениях. Исходные записи не являются подтверждённо нормальными.</p>'
                return self.page('Оценка обнаружения аномалий',body+table(metrics)+table(pd.DataFrame([manifest])),session)
            if route == '/temporal':
                anomalies = detect_temporal_anomalies(self.active_frame)
                flagged = anomalies[anomalies.anomaly.eq(True)].sort_values('robust_score', key=abs, ascending=False)
                body = '<p>Скачки оцениваются относительно предыдущих годовых изменений. Флаги не доказывают ошибку данных.</p>'
                body += '<h2>Сводка диагностики</h2>'+table(anomaly_summary(anomalies))
                body += '<h2>Аномальные изменения</h2>'+table(flagged.head(100))
                body += '<h2>Тренды (первые 100 записей)</h2>'+table(trend_profiles(self.active_frame).head(100))
                return self.page('Динамика показателей',body,session)
            if route == '/cluster-quality':
                metrics, _, profiles = evaluate_clusters(self.active_frame)
                return self.page('Качество кластеризации', table(metrics)+table(profiles), session)
            if route == '/download':
                _, bundle, _ = build_report(self.active_frame,self.active_source,self.active_kind)
                return self.send(200,bundle,'application/zip','report.zip')
            return super().do_GET()
        except (ValueError,PermissionError) as exc:
            return self.send(400 if isinstance(exc,ValueError) else 403,escape(str(exc)))

    def send(self,status,body,mime='text/html; charset=utf-8',filename=None):
        if isinstance(body,str) and '<aside>' in body:
            _, session = self.session()
            if session:
                nav = '<p>Дополнительные учебные и пользовательские модули:</p><a href="/datasets">Наборы данных</a><a href="/cluster-quality">Качество кластеризации</a><a href="/temporal">Динамика показателей</a><a href="/anomaly-benchmark">Оценка аномалий</a><a href="/lag-forecast">Прогноз по истории</a><a href="/peers">Сопоставимые вузы</a><a href="/profile">Карточка вуза</a><a href="/diagnostics">Диагностика данных</a><a href="/sensitivity">Чувствительность прогнозов</a>'
                if session['role']=='admin':
                    nav += '<a href="/admin">Администрирование</a><a href="/experiments">Реестр экспериментов</a><a href="/operations">Задачи обучения</a>'
                nav += self.form(session,'/logout','','Выйти')
                body = body.replace('<nav>', '<nav><a href="/actual-monitoring"'+(' class="active"' if urlsplit(self.path).path == '/actual-monitoring' else '')+'>Фактический мониторинг</a><p>Учебные и пользовательские наборы:</p>')
                body = body.replace('</nav>',nav+'</nav>')
                if hasattr(self,'active_source'):
                    body = body.replace('Синтетические учебные данные. Не официальная статистика.',escape(('Учебные данные: ' if self.active_kind=='demo' else 'Внешние данные (источник заявлен пользователем): ')+self.active_source))
            else:
                body = body.replace('<nav>','<nav hidden>')
        return super().send(status,body,mime,filename)

    def do_POST(self):
        route = urlsplit(self.path).path
        try:
            length = int(self.headers.get('Content-Length','0'))
            if not 0 < length <= 30_000_000:
                return self.send(413,'Недопустимый размер запроса')
            if self.headers.get_content_type() != 'application/x-www-form-urlencoded':
                return self.send(415,'Неподдерживаемый формат')
            form = parse_qs(self.rfile.read(length).decode('utf-8'),max_num_fields=12)
            def field(name):
                return form.get(name,[''])[0]
            store = self.server.store
            token, session = self.session()
            if route == '/login':
                address = self.client_address[0]
                if not self.server.login_guard.reserve(address):
                    return self.send(429,'Слишком много попыток входа. Повторите через минуту.')
                login = field('login')
                role = store.authenticate(login,field('password'))
                if not role:
                    return self.send(401,'Неверный логин или пароль')
                self.server.login_guard.clear(address)
                old_token = token
                token = secrets.token_urlsafe(32)
                with self.server.lock:
                    if old_token:
                        self.server.sessions.pop(old_token, None)
                    now = time.monotonic()
                    self.server.sessions = {k:v for k,v in self.server.sessions.items() if v['expires']>now}
                    self.server.sessions[token] = dict(login=login,role=role,csrf=secrets.token_urlsafe(32),expires=now+3600,dataset=0)
                return self.redirect('/',f'session={token}; HttpOnly; SameSite=Strict; Path=/; Max-Age=3600')
            if not session or not secrets.compare_digest(field('csrf'),session['csrf']):
                return self.send(403,'Недействительная сессия или CSRF-токен')
            actor = session['login']
            if route == '/logout':
                with self.server.lock:
                    self.server.sessions.pop(token,None)
                return self.redirect('/login','session=; Path=/; Max-Age=0; HttpOnly; SameSite=Strict')
            if route == '/train':
                with store.connect() as db:
                    store.require(db,actor,('admin',))
                if field('model') not in dict(candidates()):
                    raise ValueError('Неизвестная модель')
                row = store.read(actor,session['dataset']) if session.get('dataset') else None
                frame = load_data(StringIO(row['csv'])) if row else self.server.frame
                self.server.training.submit(actor,frame,row['source'] if row else 'Синтетический учебный набор',row['kind'] if row else 'demo',field('model'))
                return self.redirect('/operations')
            if route == '/select':
                dataset = int(field('dataset'))
                if dataset:
                    store.read(actor,dataset)
                with self.server.lock:
                    if token not in self.server.sessions:
                        return self.send(403,'Сессия завершена')
                    self.server.sessions[token]['dataset']=dataset
                return self.redirect('/datasets')
            if route == '/import':
                store.save(actor,field('name'),field('source'),field('kind'),field('csv'))
                return self.redirect('/datasets')
            if route == '/users':
                store.create_user(actor,field('login'),field('password'),field('role'))
                return self.redirect('/admin')
            return self.send(404,'Страница не найдена')
        except PermissionError:
            return self.send(403,'Недостаточно прав')
        except (ValueError,UnicodeError):
            return self.send(400,'Некорректные данные запроса')
        except __import__('sqlite3').IntegrityError:
            return self.send(409,'Конфликт данных')

def create_secure_server(port, database, registry=None, host="127.0.0.1"):
    server = ThreadingHTTPServer((host,port),SecureHandler)
    server.store = Store(database)
    server.registry = Registry(registry or ROOT / "model_runs")
    server.training = TrainingQueue(server.registry, store=server.store)
    server.frame = load_data(ROOT/'data/demo.csv')
    server.sessions = {}
    server.login_guard = LoginGuard()
    server.lock = Lock()
    return server

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8080)
    parser.add_argument('--database',default='unimonitor.db')
    parser.add_argument('--init-admin',action='store_true')
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--registry',default=str(ROOT / 'model_runs'))
    args = parser.parse_args()
    if args.init_admin:
        print('Создан:',Store(args.database).bootstrap(input('Логин: '),getpass.getpass('Пароль (12+ символов): ')))
    else:
        server = create_secure_server(args.port,args.database,args.registry,args.host)
        print(f'http://127.0.0.1:{args.port}',flush=True)
        server.serve_forever()
