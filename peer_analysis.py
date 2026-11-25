"""Descriptive benchmarking against similarly sized institutions, not causal advice."""
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
from core import load_data

INDICATORS = ['score', 'research', 'income', 'employment']
LABELS = {'score':'Средний балл','research':'Научная деятельность','income':'Доходы','employment':'Трудоустройство'}


def benchmark_peers(frame, year=None, k=10):
    if not isinstance(k, int) or not 3 <= k <= 50:
        raise ValueError('Число соседей: от 3 до 50')
    if frame.empty:
        raise ValueError('Нет наблюдений')
    year = int(frame.year.max()) if year is None else int(year)
    data = frame.loc[frame.year.eq(year)].copy().sort_values('university')
    if len(data) < k + 1:
        raise ValueError('Недостаточно вузов для заданного числа соседей')
    if data.university.duplicated().any():
        raise ValueError('Повторяющиеся вузы')
    if not np.isfinite(data[['students']+INDICATORS]).all().all() or (data.students <= 0).any():
        raise ValueError('Некорректные показатели или численность')
    data['log_size'] = np.log(data.students)
    results, memberships = [], []
    for _, target in data.iterrows():
        peers = data.loc[data.university.ne(target.university)].copy()
        peers['distance'] = (peers.log_size-target.log_size).abs()
        peers = peers.sort_values(['distance','university']).head(k)
        for _, peer in peers.iterrows():
            memberships.append({'university':target.university,'peer':peer.university,'year':year,'size_ratio':float(peer.students/target.students)})
        for indicator in INDICATORS:
            values = peers[indicator].to_numpy(dtype=float)
            q1, median, q3 = np.quantile(values,[.25,.5,.75])
            value = float(target[indicator])
            gap = value-median
            relative = 100*gap/median if median != 0 else None
            percentile = 100*(np.sum(values<value)+.5*np.sum(values==value))/k
            status = 'ниже квартиля' if value<q1 else 'выше квартиля' if value>q3 else 'в пределах межквартильного диапазона'
            results.append({'university':target.university,'year':year,'indicator':indicator,'value':value,'peer_median':median,'peer_q1':q1,'peer_q3':q3,'gap':gap,'relative_gap_pct':relative,'peer_percentile':percentile,'status':status,'peer_count':k,'max_size_ratio':float(np.exp(peers.distance.max()))})
    return pd.DataFrame(results), pd.DataFrame(memberships)


def recommendations(results):
    rows = []
    for university, group in results.groupby('university',sort=True):
        below = group[group.status.eq('ниже квартиля')].sort_values(['peer_percentile','indicator'])
        if below.empty:
            rows.append({'university':university,'indicator':'—','priority':0,'recommendation':'Нет показателей ниже первого квартиля; продолжить наблюдение.'})
            continue
        for priority, (_, row) in enumerate(below.iterrows(),1):
            label = LABELS[row.indicator]
            caveat = ' Состав группы неоднороден по численности.' if row.max_size_ratio>2 else ''
            rows.append({'university':university,'indicator':label,'priority':priority,'recommendation':f'{label}: значение {row.value:.2f}, медиана группы {row.peer_median:.2f}. Проверить сопоставимость методики учёта и исследовать причины отставания.{caveat}'})
    return pd.DataFrame(rows)


def run(frame, year=None, k=10):
    results, members = benchmark_peers(frame,year,k)
    advice = recommendations(results)
    summary = results.groupby('indicator').agg(observations=('university','size'), below_q1=('status',lambda s:s.eq('ниже квартиля').sum()), median_percentile=('peer_percentile','median')).reset_index()
    summary['below_pct'] = 100*summary.below_q1/summary.observations
    manifest = {'year':int(results.year.iloc[0]),'universities':int(results.university.nunique()),'neighbors':k,'comparisons':len(results),'method':'nearest absolute log student count; same year; exclude target','limitations':'Size only; not causal; no adjustment for profile or region.'}
    return {'comparisons':results,'memberships':members,'recommendations':advice,'summary':summary,'manifest':manifest}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('csv')
    parser.add_argument('--year',type=int)
    parser.add_argument('--neighbors',type=int,default=10)
    parser.add_argument('--output',default='peer_results')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    for name, value in run(load_data(args.csv),args.year,args.neighbors).items():
        if isinstance(value,pd.DataFrame):
            value.to_csv(output/f'{name}.csv',index=False)
        else:
            (output/f'{name}.json').write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
