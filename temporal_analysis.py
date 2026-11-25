"""Диагностика динамики: только прошлые наблюдения используются для оценки скачка."""
import numpy as np
import pandas as pd
from core import FEATURES


def annual_changes(frame):
    ordered = frame.sort_values(['university', 'year']).copy()
    if ordered.duplicated(['university', 'year']).any():
        raise ValueError('Повторяющиеся наблюдения')
    groups = ordered.groupby('university')
    consecutive = ordered.year.sub(groups.year.shift()).eq(1)
    rows = []
    for metric in FEATURES:
        previous = groups[metric].shift()
        delta = ordered[metric].sub(previous).where(consecutive)
        rate = delta.div(previous.where(previous.ne(0))).mul(100)
        part = ordered[['university', 'region', 'year']].copy()
        part['metric'] = metric
        part['value'] = ordered[metric]
        part['delta'] = delta
        part['relative_change'] = rate
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def detect_temporal_anomalies(frame, threshold=3.5, minimum_history=3):
    if not np.isfinite(threshold) or threshold <= 0 or minimum_history < 3:
        raise ValueError('Некорректные параметры диагностики')
    changes = annual_changes(frame).dropna(subset=['delta']).copy()
    rows = []
    for _, group in changes.groupby(['university', 'metric']):
        group = group.sort_values('year').copy()
        history = []
        for record in group.to_dict('records'):
            value = record['delta']
            record.update(history_size=len(history), baseline=np.nan, mad=np.nan,
                          robust_score=np.nan, anomaly=False, status='Недостаточно истории')
            if len(history) >= minimum_history:
                center = float(np.median(history))
                mad = float(np.median(np.abs(np.asarray(history)-center)))
                record.update(baseline=center, mad=mad)
                if mad > 1e-12:
                    score = .67448975 * (value-center)/mad
                    record.update(robust_score=score, anomaly=abs(score)>threshold,
                                  status='Оценено')
                else:
                    record['status'] = 'Нулевая вариативность истории'
            history.append(value)
            rows.append(record)
    columns = list(changes.columns)+['history_size','baseline','mad','robust_score','anomaly','status']
    return pd.DataFrame(rows, columns=columns)


def trend_profiles(frame):
    rows = []
    for university, group in frame.groupby('university'):
        group = group.sort_values('year')
        for metric in FEATURES:
            years = group.year.to_numpy(dtype=float)
            values = group[metric].to_numpy(dtype=float)
            slope = float(np.polyfit(years-years.min(), values, 1)[0]) if len(years)>1 else np.nan
            span = int(years.max()-years.min())
            growth = ((values[-1]/values[0])**(1/span)-1)*100 if span and values[0]>0 and values[-1]>=0 else np.nan
            rows.append(dict(university=university, metric=metric, observations=len(years),
                             first_year=int(years.min()), last_year=int(years.max()),
                             missing_years=span+1-len(years), slope=slope, annual_growth=growth))
    return pd.DataFrame(rows)


def anomaly_summary(anomalies):
    rows = []
    for metric, group in anomalies.groupby('metric'):
        scored = group[group.status.eq('Оценено')]
        count = int(scored.anomaly.sum())
        rows.append(dict(metric=metric, changes=len(group), assessed=len(scored),
                         anomalies=count, fraction=count/len(scored) if len(scored) else np.nan))
    return pd.DataFrame(rows)


def export_analysis(frame, destination):
    from pathlib import Path
    output = Path(destination)
    output.mkdir(parents=True, exist_ok=True)
    anomalies = detect_temporal_anomalies(frame)
    profiles = trend_profiles(frame)
    summary = anomaly_summary(anomalies)
    for name, data in [('changes', annual_changes(frame)), ('anomalies', anomalies),
                       ('trends', profiles), ('summary', summary)]:
        data.fillna('').to_csv(output/f'{name}.csv', index=False)
    return summary


if __name__ == '__main__':
    import argparse
    from core import load_data
    parser = argparse.ArgumentParser(description='Диагностика временных рядов')
    parser.add_argument('csv')
    parser.add_argument('--output', default='temporal_results')
    args = parser.parse_args()
    print(export_analysis(load_data(args.csv), args.output).to_string(index=False))
