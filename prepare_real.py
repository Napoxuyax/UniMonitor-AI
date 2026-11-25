"""Extract university-wide indicators from the published CSV archive in bounded memory."""
import argparse, hashlib, json, zipfile
from pathlib import Path
import pandas as pd

MAPPING={
 'students': ('Общая численность студентов, обучающихся по программам бакалавриата, специалитета, магистратуры','человек'),
 'score': ('Средний  ЕГЭ студентов принятых на обучение по программам бакалавриата и специалитета по всем формам обучения','балл'),
 'research': ('Общий объем научно-исследовательских и опытно-конструкторских работ (далее – НИОКР)','тыс.руб.'),
 'income': ('Доходы образовательной организации из всех источников в расчете на численность студентов (приведенный контингент)','тыс.руб.'),
 'employment': ('Удельный вес выпускников, трудоустроившихся в течение календарного года, следующего за годом выпуска, в общей численности выпускников образовательной организации обучавшихся по основным образовательным программам высшего образования','%')}

def prepare(archive, output):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    columns=['id','name','region','year','indicator_name','indicator_unit','indicator_value','area','specialty_code','specialty_section','parent_id']
    lookup={v:k for k,v in MAPPING.items()}; parts=[]; count=0
    with zipfile.ZipFile(archive) as z:
        csvs=[n for n in z.namelist() if n.endswith('.csv') and not n.startswith('__MACOSX/')]
        if len(csvs)!=1: raise ValueError('Expected exactly one source CSV')
        with z.open(csvs[0]) as stream:
            for chunk in pd.read_csv(stream,sep=';',dtype=str,usecols=columns,chunksize=100000):
                count+=len(chunk)
                q=chunk[chunk.area.isna() & chunk.specialty_code.isna() & chunk.specialty_section.isna()].copy()
                q['field']=[lookup.get(pair) for pair in zip(q.indicator_name,q.indicator_unit)]
                parts.append(q[q.field.notna()][['id','name','region','year','parent_id','field','indicator_value']])
    selected=pd.concat(parts,ignore_index=True)
    selected['value']=pd.to_numeric(selected.indicator_value.str.replace(',','.',regex=False).str.replace('\u00a0','',regex=False).str.replace(' ','',regex=False),errors='coerce')
    invalid=selected.value.isna() | (selected.value<0) | (selected.field.isin(['score','employment']) & (selected.value>100))
    valid=selected[~invalid].copy()
    dup=valid.duplicated(['id','year','field'],keep=False)
    conflicts=valid[dup]; valid=valid[~dup]
    wide=valid.pivot(index=['id','year'],columns='field',values='value').reindex(columns=list(MAPPING)).reset_index()
    meta=selected[['id','year','name','region','parent_id']].drop_duplicates()
    ambiguous=meta.duplicated(['id','year'],keep=False)
    wide=wide.merge(meta[~ambiguous],on=['id','year'],how='inner')
    coverage=wide.groupby('year')[list(MAPPING)].count(); coverage['all_five']=wide.dropna(subset=list(MAPPING)).groupby('year').size(); coverage=coverage.fillna(0).astype(int)
    coverage.to_csv(output/'coverage.csv')
    wide.to_csv(output/'panel.csv',index=False)
    accepted=wide.dropna(subset=list(MAPPING)).copy()
    # Stable identifier across years, even if the name changes.
    accepted['university']=accepted.id
    accepted[['university','region','year']+list(MAPPING)].to_csv(output/'accepted.csv',index=False)
    selected[invalid].to_csv(output/'invalid.csv',index=False); conflicts.to_csv(output/'conflicts.csv',index=False)
    with open(archive,'rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
    report={'source':'https://tochno.st/datasets/performance','archive_sha256':digest,'source_rows':count,'selected_rows':len(selected),'invalid_values':int(invalid.sum()),'duplicate_valid_rows':int(dup.sum()),'panel_rows':len(wide),'complete_rows':len(accepted),'complete_universities':accepted.id.nunique(),'mapping':MAPPING,'year_definition':'Год мониторинга; сведения за предыдущий год','income_definition':'Доходы на приведенный контингент, тыс. руб.'}
    (output/'quality.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(coverage.to_string()); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('archive'); p.add_argument('output'); a=p.parse_args(); prepare(a.archive,a.output)
