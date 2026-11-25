"""Локальный интерфейс для демонстрации аналитического ядра."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, parse_qs
from html import escape
from pathlib import Path
import argparse
from core import load_data, analyze, FEATURES
from eda import explore
from research import experiment
from reporting import build_report
from scenario import scenario_analysis
from decision import rank_universities, sensitivity, distribution_drift
ROOT=Path(__file__).parent
NAV={'/':'Обзор','/compare':'Сравнение вузов','/anomalies':'Кластеры и аномалии','/eda':'Исследование данных','/forecast':'Прогнозирование','/reports':'Отчёты','/ranking':'Рейтинг вузов','/drift':'Изменение данных','/scenario':'Сценарный прогноз'}
LABELS=dict(zip(['university','region','year']+FEATURES+['cluster','anomaly_score','anomaly'],['Вуз','Регион','Год','Студенты','Средний балл','Наука','Доход','Трудоустройство, %','Кластер','Ошибка реконструкции','Аномалия']))
CSS='''*{box-sizing:border-box}body{margin:0;background:#f0f4f8;color:#17334e;font:15px system-ui}aside{position:fixed;inset:0 auto 0 0;width:240px;background:#17334e;color:white;padding:24px 16px}aside{overflow-y:auto}aside h2{margin:0}aside p{color:#b8cce0}nav a{display:block;padding:13px;margin:5px 0;color:white;text-decoration:none;border-radius:8px}nav a:hover,nav a.active{background:#345c7c}main{margin-left:240px;padding:30px}h1{font-size:30px;margin:0}h2{font-size:19px}section,.metric{background:white;border:1px solid #d7e1eb;border-radius:12px;padding:20px;margin:14px 0}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:15px}.metric b{display:block;font-size:34px;color:#22628a}.notice{background:#fff3d6;padding:12px;border-radius:8px}.scroll{overflow:auto;max-height:430px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:9px;border-bottom:1px solid #e0e7ef;text-align:left;white-space:nowrap}th{background:#eaf1f7;position:sticky;top:0}.bar{display:grid;grid-template-columns:150px 1fr 80px;gap:12px;align-items:center;margin:10px 0}.track{background:#e7edf4;height:14px;border-radius:5px}.fill{background:#367ca5;height:14px;border-radius:5px}button,.btn{background:#215d87;color:white;padding:10px 16px;border:0;border-radius:7px;display:inline-block;text-decoration:none;cursor:pointer}select{padding:9px;border:1px solid #bccbd8;border-radius:6px;margin:8px}small{color:#526c80}@media(max-width:700px){aside{position:static;width:auto}main{margin:0;padding:15px}.cards{grid-template-columns:1fr}}'''
def table(df,n=100):
    return '<div class="scroll">'+df.head(n).rename(columns=LABELS).to_html(index=False,escape=True,float_format=lambda v:f'{v:.2f}')+'</div><small>Показано '+str(min(n,len(df)))+' из '+str(len(df))+' строк</small>'
def bars(labels,values):
    maximum=max([float(v) for v in values]+[1])
    return ''.join(f'<div class="bar"><span>{escape(str(l))}</span><div class="track"><div class="fill" style="width:{max(0,float(v)/maximum*100):.2f}%"></div></div><b>{float(v):.2f}</b></div>' for l,v in zip(labels,values))
def layout(route,body):
    nav=''.join(f'<a class="{"active" if route==k else ""}" href="{k}">{v}</a>' for k,v in NAV.items())
    return f'<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>UniMonitor AI</title><style>{CSS}</style></head><body><aside><h2>UniMonitor AI</h2><p>Локальная демонстрация</p><nav>{nav}</nav></aside><main><h1>{NAV[route]}</h1><p class="notice">Синтетические учебные данные. Не официальная статистика.</p>{body}</main></body></html>'
class Handler(BaseHTTPRequestHandler):
    def send(self,status,body,mime='text/html; charset=utf-8',filename=None):
        if isinstance(body,str): body=body.encode()
        self.send_response(status); self.send_header('Content-Type',mime); self.send_header('Content-Length',str(len(body))); self.send_header('X-Content-Type-Options','nosniff'); self.send_header('Cache-Control','no-store')
        self.send_header('Content-Security-Policy',"default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'")
        if filename: self.send_header('Content-Disposition',f'attachment; filename="{filename}"')
        self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        p=urlsplit(self.path); q=parse_qs(p.query); route=p.path; df=getattr(self,"active_frame",self.server.frame)
        try:
            if route=='/download':
                _,bundle,_=build_report(df,'Синтетический учебный набор','demo'); return self.send(200,bundle,'application/zip','report.zip')
            if route not in NAV: return self.send(404,'Страница не найдена')
            if route=='/':
                body='<div class="cards">'+''.join(f'<div class="metric">{l}<b>{v}</b></div>' for l,v in [('Вузов',df.university.nunique()),('Наблюдений',len(df)),('Последний год',int(df.year.max()))])+'</div>'
                annual=df.groupby('year').employment.mean(); body+='<section><h2>Среднее трудоустройство по годам, %</h2>'+bars(annual.index,annual.values)+'</section><section><h2>Последние наблюдения</h2>'+table(df[df.year==df.year.max()],8)+'</section>'
            elif route=='/compare':
                year=int(q.get('year',[df.year.max()])[0]); f=q.get('feature',['employment'])[0]
                if f not in FEATURES or year not in df.year.unique(): raise ValueError('Некорректный показатель или год')
                def opts(values,selected): return ''.join(f'<option value="{v}"'+(' selected' if v==selected else '')+'>'+LABELS.get(v,str(v))+'</option>' for v in values)
                rows=df[df.year==year].sort_values(f,ascending=False); top=rows.head(10)
                body='<form><label>Год <select name="year">'+opts(sorted(df.year.unique(),reverse=True),year)+'</select></label><label>Показатель <select name="feature">'+opts(FEATURES,f)+'</select></label><button>Сравнить</button></form><section><h2>Первые 10 вузов по показателю</h2>'+bars(top.university,top[f])+'</section>'+table(rows)
            elif route=='/anomalies':
                rows=analyze(df).sort_values('anomaly_score',ascending=False); body='<section><h2>Результаты анализа</h2><p>Аномалии — верхние 10% ошибок реконструкции; требуют экспертной проверки.</p>'+table(rows)+'</section>'
            elif route=='/eda':
                r=explore(df); body='<section><h2>Охват наблюдений</h2>'+table(r['coverage'])+'<h2>Корреляции Спирмена</h2>'+table(r['correlation'].rename(index=LABELS,columns=LABELS).reset_index(names='Показатель'))+'<p>Корреляция не означает причинную связь.</p></section>'
            elif route=='/forecast':
                body='<p>Последний целевой год исключён из подбора параметров.</p><a class="btn" href="/forecast?run=1">Выполнить эксперимент</a>'
                if q.get('run')==['1']:
                    r=experiment(df); body+='<section><h2>Контрольная выборка</h2><p>Модель по валидации: '+escape(r['winner'])+'</p>'+table(r['comparison'])+'<h2>Прогноз следующего года</h2>'+table(r['forecast'])+'</section>'
            elif route=='/ranking':
                year=int(q.get('year',[df.year.max()])[0])
                weights={f:float(q.get(f,['1'])[0]) for f in FEATURES}
                options=''.join(f'<option value="{y}"'+(' selected' if y==year else '')+f'>{y}</option>' for y in sorted(df.year.unique(),reverse=True))
                fields=''.join(f'<p>{LABELS[f]} <input type="number" min="0" step="any" name="{f}" value="{weights[f]}"></p>' for f in FEATURES)
                ranked=rank_universities(df,year,weights)
                stability=sensitivity(df,year,weights)
                body='<section><p>Учебная многокритериальная оценка, не официальный рейтинг. Размер вуза и доход считаются положительными критериями.</p><form><select name="year">'+options+'</select>'+fields+'<button>Рассчитать</button></form></section>'
                body+='<section><h2>Итоговая оценка</h2>'+table(ranked.rename(columns={'score_total':'Баллы','rank':'Место'}))+'</section>'
                body+='<section><h2>Устойчивость места: 300 сценариев изменения весов на ±25%</h2>'+table(stability.rename(columns={'rank_base':'Место','rank_median':'Медиана','rank_best':'Лучшее','rank_worst':'Худшее','top10_share':'Доля в топ-10, %'}))+'</section>'
            elif route=='/scenario':
                changes={f:float(q.get(f,['0'])[0]) for f in FEATURES}
                rows, importance, meta=scenario_analysis(df,changes)
                fields=''.join(f'<p>{LABELS[f]}: <input type="number" min="-100" max="100" step="any" name="{f}" value="{changes[f]}"> %</p>' for f in FEATURES)
                body='<section><form>'+fields+'<button>Рассчитать сценарий</button></form><p>Изменения не являются причинными эффектами. Интервал эмпирический, покрытие будущего сценария не гарантировано.</p></section>'
                body+=f'<section><h2>Контрольное покрытие: {meta["observed_test_coverage"]:.1%}</h2><p>Радиус: {meta["radius"]:.2f} п. п.</p>'+table(rows.rename(columns={'forecast_year':'Год прогноза','baseline':'Базовый прогноз','scenario':'Сценарий','delta':'Разница','lower':'Нижняя граница','upper':'Верхняя граница'}))+'</section>'
                importance['feature']=importance.feature.map(LABELS)
                body+='<section><h2>Влияние перестановки признаков на MAE</h2>'+table(importance.rename(columns={'feature':'Показатель','delta_mae':'Изменение MAE','std':'Стандартное отклонение'}))+'</section>'
            elif route=='/drift':
                years=sorted(df.year.unique())
                first=int(q.get('first',[years[0]])[0]); last=int(q.get('last',[years[-1]])[0])
                def options(selected):
                    return ''.join(f'<option value="{y}"'+(' selected' if y==selected else '')+f'>{y}</option>' for y in years)
                drift=distribution_drift(df,first,last)
                drift['indicator']=drift.indicator.map(LABELS)
                body='<section><form>Базовый год <select name="first">'+options(first)+'</select>Сравниваемый год <select name="last">'+options(last)+'</select><button>Проверить</button></form><p>Расстояние Колмогорова — Смирнова используется описательно. Порог 0,2 — эвристический сигнал, не статистическая значимость.</p>'+table(drift.rename(columns={'indicator':'Показатель','matched':'Общие вузы','reference_mean':'Среднее базовое','comparison_mean':'Среднее новое','median_change':'Медиана изменения','ks_distance':'KS-расстояние','flag':'Сигнал'}))+'</section>'
            else: body='<section><h2>Аналитический отчёт</h2><p>HTML, CSV и паспорт JSON в одном архиве.</p><a class="btn" href="/download">Скачать архив отчёта</a></section>'
            self.send(200,layout(route,body))
        except ValueError as e: self.send(400,escape(str(e)))
def create_server(port=8080):
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler); server.frame=load_data(ROOT/'data/demo.csv'); return server
if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--port',type=int,default=8080); args=parser.parse_args(); server=create_server(args.port); print(f'http://127.0.0.1:{args.port}',flush=True); server.serve_forever()
