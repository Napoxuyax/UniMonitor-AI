"""Read-only published-data dashboard, separate from the five-feature sandbox."""
from html import escape
from urllib.parse import urlencode
import numpy as np
import pandas as pd
import joblib
from actual_monitoring import FEATURES, ROOT, transform, annual_quality
from web_local import table

LABELS = {'id':'ID','name':'Организация','region':'Регион','year':'Год мониторинга','students':'Студенты, чел.','score':'Средний ЕГЭ, балл','research':'НИОКР, тыс. руб.','income':'Доход на приведённый контингент, тыс. руб.','employment':'Трудоустройство, %','parent_id':'ID головной организации'}

def select_rows(frame, year, region='', search=''):
    result = frame.loc[frame.year.eq(year)].copy()
    if result.empty:
        raise ValueError('Нет наблюдений за выбранный год')
    if region and region not in result.region.unique():
        raise ValueError('Неизвестный регион для выбранного года')
    if region:
        result = result[result.region.eq(region)]
    if search:
        result = result[result.name.str.contains(search, case=False, regex=False, na=False) | result.id.eq(search)]
    return result

def statistics(rows):
    records=[]
    for feature in FEATURES:
        values=rows[feature]
        if feature=='score':
            values=values[values.gt(0)]
        records.append({'Показатель':LABELS[feature],'Наблюдений':len(values),'Медиана':values.median(),'Среднее':values.mean(),'Первый квартиль':values.quantile(.25),'Третий квартиль':values.quantile(.75)})
    return pd.DataFrame(records)

def actual_flags(rows):
    # The stored detector was calibrated on 2024; never apply it to historical years.
    if not rows.year.eq(2025).all():
        raise ValueError('Сохранённый детектор предназначен для мониторинга 2025 года')
    eligible=rows.loc[rows.score.gt(0)].copy()
    if eligible.empty:
        return eligible.assign(anomaly_score=pd.Series(dtype=float),flagged=pd.Series(dtype=bool))
    artifact=joblib.load(ROOT/'actual_anomalies/model2.joblib')
    x=artifact['scaler'].transform(transform(eligible))
    error=np.mean((x-artifact['model'].predict(x))**2,axis=1)
    return eligible.assign(anomaly_score=error,flagged=error>artifact['threshold']).sort_values('anomaly_score',ascending=False)

def show(df,n=100):
    return table(df.rename(columns=LABELS),n)

def link(query, label, route='/actual-monitoring'):
    return '<a class="btn" href="'+escape(route+'?'+urlencode(query),quote=True)+'">'+escape(label)+'</a>'

def dashboard(frame, query):
    year=int(query.get('year',['2025'])[0]); region=query.get('region',[''])[0]; search=query.get('search',[''])[0].strip()[:200]
    feature=query.get('feature',['score'])[0]; page=int(query.get('page',['1'])[0]); selected=query.get('id',[''])[0]
    if feature not in FEATURES or page<1:
        raise ValueError('Некорректный показатель или номер страницы')
    rows=select_rows(frame,year,region,search)
    params=dict(year=year,region=region,search=search,feature=feature)
    def options(values, chosen):
        return ''.join('<option value="'+escape(str(v),quote=True)+'"'+(' selected' if str(v)==str(chosen) else '')+'>'+escape(str(v) or 'Все регионы')+'</option>' for v in values)
    regions=sorted(frame.loc[frame.year.eq(year),'region'].dropna().unique())
    features=''.join(f'<option value="{f}"'+(' selected' if f==feature else '')+'>'+LABELS[f]+'</option>' for f in FEATURES)
    body='<p>Источник: <a href="https://tochno.st/datasets/performance">tochno.st/datasets/performance</a>. Год мониторинга относится к сведениям за предыдущий год. Нулевые баллы ЕГЭ показаны отдельно; трудоустройство после 2018 года отсутствует.</p>'
    body+='<form action="/actual-monitoring"><label>Год <select name="year">'+options(sorted(frame.year.unique(),reverse=True),year)+'</select></label><label>Регион <select name="region">'+options(['']+regions,region)+'</select></label><label>Название или ID <input name="search" value="'+escape(search,quote=True)+'"></label><label>Показатель <select name="feature">'+features+'</select></label><button>Применить</button></form>'
    body+='<div class="cards">'+''.join(f'<div class="metric">{label}<b>{value}</b></div>' for label,value in [('Наблюдений',len(rows)),('Нулевой ЕГЭ',int(rows.score.eq(0).sum())),('Регионов',rows.region.nunique())])+'</div>'
    body+=link(params,'Скачать выбранные данные','/actual-download')+'<h2>Сравнение показателей</h2>'+show(statistics(rows))
    ranked=rows.loc[rows.score.gt(0)] if feature=='score' else rows
    ranked=ranked.sort_values([feature,'id'],ascending=[False,True])
    maxpage=max(1,(len(ranked)+49)//50)
    if page>maxpage: raise ValueError('Страница отсутствует')
    portion=ranked.iloc[(page-1)*50:page*50]
    body+='<h2>Показатели организаций</h2><p>Сортировка по выбранному показателю, не официальный рейтинг.</p>'+show(portion[['id','name','region']+FEATURES],50)+f'<p>Страница {page} из {maxpage}; всего {len(ranked)} записей.</p>'
    if page>1: body+=link(dict(params,page=page-1),'Назад')+' '
    if page<maxpage: body+=link(dict(params,page=page+1),'Далее')
    body+='<h2>Карточка организации</h2><form action="/actual-monitoring">'+''.join(f'<input type="hidden" name="{k}" value="{escape(str(v),quote=True)}">' for k,v in params.items())+'<label>ID <input name="id" required value="'+escape(selected,quote=True)+'"></label><button>Показать историю</button></form>'
    if selected:
        history=frame.loc[frame.id.eq(selected)&frame.year.le(year)].sort_values('year')
        if history.empty: raise ValueError('Нет истории организации за выбранный период')
        body+='<p>'+escape(history.iloc[-1]['name'])+'</p>'+show(history[['year']+FEATURES+['employment']])
    if year==2025:
        flagged=actual_flags(rows)
        body+=f'<h2>Нейросетевая диагностика</h2><p>Автоэнкодер 4–16–3–16–4; порог по 2024 году. Проверено {len(flagged)} наблюдений, отмечено {int(flagged.flagged.sum())}. Это сигналы для проверки, не доказанные ошибки.</p>'+show(flagged.loc[flagged.flagged,['id','name','anomaly_score']].rename(columns={'anomaly_score':'Ошибка реконструкции'}),30)
    body+='<h2>Полнота данных</h2>'+show(annual_quality(frame.loc[frame.year.le(year)]))
    if year==2025:
        body+='<h2>Оценка детекторов на искусственно внесённых отклонениях</h2>'+show(pd.read_csv(ROOT/'actual_anomalies/metrics.csv'))
        body+='<h2>Прогноз ЕГЭ: эксперимент с исходными нулями</h2>'+show(pd.read_csv(ROOT/'real_score/comparison.csv'))
        body+='<h2>Прогноз ЕГЭ: только положительные значения</h2>'+show(pd.read_csv(ROOT/'actual_score_positive/comparison.csv'))
    return body
