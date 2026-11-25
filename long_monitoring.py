"""Convert explicitly mapped long-format monitoring data; no inferred indicator semantics."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
from core import KEY, FEATURES
from import_monitoring import prepare

SOURCE_URL = 'https://tochno.st/datasets/performance'
PARQUET_URL = 'https://storage.yandexcloud.net/tochno-st-catalog/Minobr/data_performance_137_v20260223/data_performance_137_v202602023.parquet'


def convert(raw, config, source):
    columns = config['columns']
    required = set(KEY + ['indicator', 'value'])
    if set(columns) != required or len(set(columns.values())) != len(required):
        raise ValueError('Необходимо явно сопоставить пять исходных столбцов')
    if not set(columns.values()).issubset(raw.columns):
        raise ValueError('Отсутствуют исходные столбцы')
    indicators = config['indicators']
    if set(indicators) != set(FEATURES):
        raise ValueError('Необходимо сопоставить все пять показателей')
    codes = [str(indicators[f]['code']).strip() for f in FEATURES]
    if any(not c for c in codes) or len(set(codes)) != len(codes):
        raise ValueError('Коды показателей должны быть непустыми и различными')
    for field in FEATURES:
        spec = indicators[field]
        if not str(spec.get('unit', '')).strip() or not str(spec.get('definition', '')).strip():
            raise ValueError('Укажите единицу измерения и определение: ' + field)
    data = raw.loc[:, list(columns.values())].rename(columns={v:k for k,v in columns.items()}).copy()
    for field in KEY + ['indicator']:
        data[field] = data[field].astype('string').str.strip()
    invalid_key = data[KEY].isna().any(axis=1) | data[KEY].eq('').any(axis=1)
    selected = data.loc[data.indicator.isin(codes) & ~invalid_key].copy()
    if selected.empty:
        raise ValueError('Не найдено ни одного выбранного показателя')
    selected['field'] = selected.indicator.map(dict(zip(codes, FEATURES)))
    duplicate = selected.duplicated(KEY + ['field'], keep=False)
    conflicts = selected.loc[duplicate].copy()
    # Never silently average competing values or choose an arbitrary record.
    valid = selected.loc[~duplicate]
    keys = selected[KEY].drop_duplicates().set_index(KEY).index
    wide = valid.pivot(index=KEY, columns='field', values='value').reindex(index=keys, columns=FEATURES).reset_index()
    accepted, rejected, report = prepare(wide, {k:k for k in KEY+FEATURES}, source)
    report.update({'input_long_rows':len(raw), 'selected_long_rows':len(selected), 'invalid_key_rows':int(invalid_key.sum()), 'duplicate_indicator_rows':len(conflicts), 'indicator_mapping':indicators, 'source_columns':columns})
    return accepted, rejected, conflicts, report


def read_source(path, separator=';'):
    path = Path(path)
    if path.suffix.lower() == '.parquet':
        return pd.read_parquet(path)
    return pd.read_csv(path, sep=separator, dtype='string', encoding='utf-8-sig')


def main():
    parser = argparse.ArgumentParser(description='Преобразование длинного формата мониторинга в таблицу вуз–год')
    parser.add_argument('input', type=Path)
    parser.add_argument('mapping', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--separator', default=';')
    parser.add_argument('--source', default=SOURCE_URL)
    parser.add_argument('--inspect', action='store_true', help='Вывести структуру файла без преобразования')
    args = parser.parse_args()
    raw = read_source(args.input, args.separator)
    if args.inspect:
        print(json.dumps({'rows':len(raw),'columns':list(raw.columns),'sample':raw.head(3).fillna('').astype(str).to_dict('records')}, ensure_ascii=False, indent=2))
        return
    mapping = json.loads(args.mapping.read_text(encoding='utf-8'))
    accepted, rejected, conflicts, report = convert(raw, mapping, args.source)
    with args.input.open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda:stream.read(1024*1024), b''):
            digest.update(block)
    report['input_sha256'] = digest.hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    for name, frame in [('accepted',accepted),('rejected',rejected),('conflicts',conflicts)]:
        frame.to_csv(args.output/(name+'.csv'),index=False)
    (args.output/'quality.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if accepted.empty:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
