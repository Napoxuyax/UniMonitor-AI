"""Единая карточка вуза на заданную дату с явным статусом доступности блоков."""
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from core import FEATURES, load_data
from decision import rank_universities, sensitivity
from peer_analysis import run as peer_run
from temporal_analysis import detect_temporal_anomalies, trend_profiles
from lag_forecast import run as lag_run

LABELS = dict(students='Численность студентов', score='Средний балл', research='Научная деятельность', income='Доходы', employment='Трудоустройство')


def profile(frame, university, year, neighbors=10, source='Не указан', kind='demo', forecast=False):
    if not 3 <= neighbors <= 50:
        raise ValueError('Число соседей: от 3 до 50')
    year = int(year)
    history = frame.loc[frame.year.le(year)].copy()
    current = history.loc[history.year.eq(year)]
    own = history.loc[history.university.eq(university)].sort_values('year')
    selected = own.loc[own.year.eq(year)]
    if len(selected) != 1:
        raise ValueError('Нет единственной записи выбранного вуза за указанный год')
    status = {}
    results = {'indicators': selected[['year']+FEATURES], 'history': own[['year']+FEATURES],
               'trends': trend_profiles(own), 'anomalies': detect_temporal_anomalies(own)}
    results['anomalies'] = results['anomalies'].loc[results['anomalies'].year.eq(year)]
    for key in ('rank','rank_sensitivity','peers','recommendations','forecast','forecast_metrics'):
        results[key] = pd.DataFrame()
    if len(current) >= 2:
        rank = rank_universities(history,year)
        results['rank'] = rank.loc[rank.university.eq(university)]
        s = sensitivity(history,year)
        results['rank_sensitivity'] = s.loc[s.university.eq(university)]
        status['Рейтинг'] = 'Рассчитан, равные веса; не официальный рейтинг'
    else:
        status['Рейтинг'] = 'Недостаточно организаций'
    if len(current) > neighbors and current.students.gt(0).all():
        peer = peer_run(history,year,neighbors)
        results['peers'] = peer['comparisons'].loc[lambda d:d.university.eq(university)]
        results['recommendations'] = peer['recommendations'].loc[lambda d:d.university.eq(university)]
        status['Сравнение'] = 'Рассчитано по масштабу; профиль и регион не учтены'
    else:
        status['Сравнение'] = 'Недостаточно сопоставимых организаций или некорректная численность'
    status['Прогноз'] = 'Не запрошен'
    if forecast:
        try:
            experiment = lag_run(history)
            results['forecast'] = experiment['forecast'].loc[lambda d:d.university.eq(university)]
            results['forecast_metrics'] = experiment['comparison']
            status['Прогноз'] = ('Модель: '+experiment['manifest']['winner'] if not results['forecast'].empty else 'Нет непрерывной истории выбранного вуза')
        except ValueError as exc:
            status['Прогноз'] = 'Недоступен: '+str(exc)
    results['manifest'] = dict(university=university, year=year, source=source, kind=kind,
                               universities=len(current), neighbors=neighbors, status=status,
                               limitation='Описательный анализ, не доказательство причин; источник внешних данных заявлен пользователем.')
    return results


def export(results, destination):
    path = Path(destination)
    path.mkdir(parents=True, exist_ok=True)
    for name, value in results.items():
        if isinstance(value,pd.DataFrame):
            value.to_csv(path/f'{name}.csv',index=False)
        else:
            (path/f'{name}.json').write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('csv'); p.add_argument('--university'); p.add_argument('--year',type=int)
    p.add_argument('--neighbors',type=int,default=10); p.add_argument('--forecast',action='store_true')
    p.add_argument('--source',required=True); p.add_argument('--kind',choices=['demo','external'],required=True)
    p.add_argument('--output',default='profile_results')
    args=p.parse_args(); df=load_data(args.csv)
    year=args.year if args.year is not None else int(df.year.max())
    name=args.university or sorted(df.loc[df.year.eq(year),'university'])[0]
    export(profile(df,name,year,args.neighbors,args.source,args.kind,args.forecast),args.output)
