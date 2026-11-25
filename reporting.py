"""Экспорт воспроизводимого аналитического отчёта без внешних сервисов."""
from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED
from html import escape
import hashlib
import json
import numpy as np
import pandas as pd
from eda import explore


def diagnostics(predictions, df):
    """Оценка ошибок по регионам контрольного года и по всей выборке."""
    required = {'university', 'target_year', 'target', 'prediction'}
    if not required.issubset(predictions.columns) or predictions.empty:
        raise ValueError('Нет контрольных прогнозов для оценки')
    p = predictions[list(sorted(required))].copy()
    if p.duplicated(['university', 'target_year']).any():
        raise ValueError('Повторяющиеся контрольные прогнозы')
    for col in ['target', 'prediction']:
        p[col] = pd.to_numeric(p[col], errors='raise')
        if not np.isfinite(p[col]).all() or not p[col].between(0, 100).all():
            raise ValueError('Некорректные значения трудоустройства')
    lookup = df[['university', 'year', 'region']].rename(columns={'year': 'target_year'})
    p = p.merge(lookup, on=['university', 'target_year'], how='left', validate='one_to_one')
    if p.region.isna().any():
        raise ValueError('Не найден регион контрольного наблюдения')
    p['error'] = p.prediction - p.target
    p['absolute_error'] = p.error.abs()
    p['squared_error'] = p.error.pow(2)
    def summarize(group):
        return {'n': len(group), 'MAE': float(group.absolute_error.mean()),
                'RMSE': float(np.sqrt(group.squared_error.mean())),
                'bias': float(group.error.mean()),
                'within_2pp': float(group.absolute_error.le(2).mean()),
                'max_error': float(group.absolute_error.max())}
    overall = summarize(p)
    regional = pd.DataFrame([{'region': region, **summarize(group)}
                             for region, group in p.groupby('region')])
    return overall, regional, p.sort_values('absolute_error', ascending=False)


def build_report(df, source, kind, experiment_result=None):
    """HTML и ZIP с таблицами и паспортом происхождения данных."""
    if kind not in {'demo', 'external'} or not source.strip():
        raise ValueError('Укажите тип и источник данных')
    result = explore(df)
    raw = df.to_csv(index=False).encode('utf-8')
    manifest = {'source': source, 'kind': kind, 'sha256_normalized_csv': hashlib.sha256(raw).hexdigest(),
                'quality': result['quality'], 'experiment_included': experiment_result is not None}
    warning = ('Синтетические учебные данные. Не официальная статистика.' if kind == 'demo'
               else 'Внешний источник указан пользователем; достоверность не проверена.')
    headings = {'coverage':'Охват наблюдений', 'summary':'Описательная статистика',
                'correlation':'Корреляции Спирмена', 'regional':'Региональные медианы',
                'changes':'Изменения между последовательными годами'}
    tables = {k: result[k] for k in headings}
    if experiment_result is not None:
        overall, regional, errors = diagnostics(experiment_result['predictions'], df)
        manifest['control_year'] = int(experiment_result['test_year'])
        manifest['winner'] = str(experiment_result['winner'])
        manifest['diagnostics'] = overall
        tables.update(comparison=experiment_result['comparison'], regional_errors=regional, errors=errors)
        headings.update(comparison='Сравнение моделей на контрольном годе', regional_errors='Ошибки по регионам', errors='Контрольные прогнозы')
    blocks = ['<!doctype html><html lang="ru"><meta charset="utf-8"><title>UniMonitor AI — отчёт</title>',
              '<style>body{font:16px Arial;max-width:1100px;margin:40px auto;color:#183451}table{border-collapse:collapse;width:100%;font-size:13px}td,th{border:1px solid #ccd5df;padding:6px;text-align:left}th{background:#edf3fa}section{overflow:auto} .notice{padding:12px;background:#fff1db}</style>',
              '<h1>Аналитический отчёт UniMonitor AI</h1>',f'<p class="notice">{escape(warning)}</p>',
              f'<p>Источник: {escape(source)}</p>',f'<p>Наблюдений: {len(df)}; вузов: {df.university.nunique()}; период: {int(df.year.min())}–{int(df.year.max())}.</p>']
    if experiment_result is not None:
        blocks.append(f'<p>Модель выбрана по временной валидации: {escape(manifest["winner"])}. Контрольный год: {manifest["control_year"]}. Ошибки измерены в процентных пунктах.</p>')
    for key, table in tables.items():
        blocks.append(f'<h2>{headings[key]}</h2><section>{table.head(100).to_html(index=key=="correlation",escape=True,float_format=lambda x:f"{x:.3f}")}</section>')
        if len(table)>100:
            blocks.append('<p>Первые 100 строк; полная таблица в архиве CSV.</p>')
    blocks.append('<p>Корреляция не означает причинную связь. Региональные ошибки на малых выборках неустойчивы.</p></html>')
    html = '\n'.join(blocks).encode('utf-8')
    buffer = BytesIO()
    with ZipFile(buffer, 'w', ZIP_DEFLATED) as archive:
        archive.writestr('report.html', html)
        archive.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False))
        archive.writestr('normalized.csv', raw)
        for key, table in tables.items():
            archive.writestr(key+'.csv', table.to_csv(index=key=='correlation'))
    return html, buffer.getvalue(), manifest
