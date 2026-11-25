"""Explicit mapping of external university-year data; never invent missing indicators."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
import numpy as np
from core import KEY, FEATURES, load_data
from io import StringIO

FIELDS = KEY + FEATURES

def prepare(raw, mapping, source, year_offset=0):
    if set(mapping) != set(FIELDS) or len(set(mapping.values())) != len(FIELDS):
        raise ValueError('Укажите отдельный исходный столбец для каждого обязательного поля')
    if not source.strip():
        raise ValueError('Укажите источник и единицы измерения')
    missing = set(mapping.values()) - set(raw.columns)
    if missing:
        raise ValueError('Нет столбцов: ' + ', '.join(sorted(missing)))
    df = raw[list(mapping.values())].rename(columns={v:k for k,v in mapping.items()}).copy()
    for col in KEY[:2]:
        df[col] = df[col].astype('string').str.strip()
    for col in ['year'] + FEATURES:
        df[col] = pd.to_numeric(df[col].astype('string').str.replace(r'[\s\u00a0]', '', regex=True).str.replace(',', '.', regex=False), errors='coerce')
    df['year'] += year_offset
    reasons = pd.Series('', index=df.index, dtype='object')
    def flag(mask, reason):
        reasons.loc[mask.fillna(True)] += reason + '; '
    flag(df[KEY[:2]].isna().any(axis=1) | df[KEY[:2]].eq('').any(axis=1), 'Нет названия или региона')
    flag(~np.isfinite(df[['year']+FEATURES].astype(float)).all(axis=1), 'Нет числового значения')
    flag((df[FEATURES]<0).any(axis=1) | ~df.score.between(0,100) | ~df.employment.between(0,100), 'Вне диапазона')
    flag(~df.year.between(1900,2100) | (df.year%1!=0), 'Некорректный год')
    flag(df.duplicated(['university','year'],keep=False), 'Неоднозначный ключ вуз–год')
    rejected = df.loc[reasons.ne('')].assign(reason=reasons[reasons.ne('')])
    accepted = df.loc[reasons.eq('')]
    if not accepted.empty:
        accepted = load_data(StringIO(accepted.to_csv(index=False)))
    report = {'source':source, 'year_offset':year_offset, 'input_rows':len(raw), 'accepted_rows':len(accepted), 'rejected_rows':len(rejected), 'mapping':mapping, 'missing_by_field':{k:int(v) for k,v in df.isna().sum().items()}, 'years':sorted(int(y) for y in accepted.year.unique()), 'universities':int(accepted.university.nunique())}
    return accepted, rejected, report

def main():
    p=argparse.ArgumentParser(description='Подготовка внешних данных мониторинга')
    p.add_argument('input',type=Path); p.add_argument('mapping',type=Path); p.add_argument('output',type=Path)
    p.add_argument('--source',required=True); p.add_argument('--separator',default=';'); p.add_argument('--year-offset',type=int,default=0)
    args=p.parse_args()
    raw=pd.read_csv(args.input,sep=args.separator,encoding='utf-8-sig',low_memory=False)
    accepted,rejected,report=prepare(raw,json.loads(args.mapping.read_text(encoding='utf-8')),args.source,args.year_offset)
    report['input_sha256']=hashlib.sha256(args.input.read_bytes()).hexdigest()
    args.output.mkdir(parents=True,exist_ok=True)
    accepted.to_csv(args.output/'accepted.csv',index=False); rejected.to_csv(args.output/'rejected.csv',index=False)
    (args.output/'quality.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if accepted.empty: raise SystemExit(2)
if __name__=='__main__': main()
