"""Exploratory analysis of validated university monitoring records."""
from pathlib import Path
import argparse
import json
import pandas as pd
import numpy as np
from core import FEATURES, load_data


def explore(df):
    if df.empty:
        raise ValueError('Набор данных пуст')
    years = sorted(df.year.unique())
    coverage = (df.groupby('year').agg(universities=('university','nunique'), regions=('region','nunique'), observations=('year','size')).reset_index())
    summary = df[FEATURES].describe(percentiles=[.25,.5,.75]).T.reset_index(names='indicator')
    latest = df[df.year.eq(years[-1])]
    correlation = latest[FEATURES].corr(method='spearman')
    regional = latest.groupby('region')[FEATURES].median().reset_index()
    changes = df.sort_values(['university','year']).copy()
    previous = changes.groupby('university').year.shift()
    for feature in FEATURES:
        changes[feature+'_change'] = changes.groupby('university')[feature].diff().where(changes.year.eq(previous+1))
    changes = changes[['university','region','year']+[f+'_change' for f in FEATURES]]
    quality = {'rows':len(df),'universities':int(df.university.nunique()),'years':[int(y) for y in years], 'missing':{c:int(n) for c,n in df.isna().sum().items()},'duplicate_keys':int(df.duplicated(['university','year']).sum()),'latest_year':int(years[-1])}
    return dict(coverage=coverage,summary=summary,correlation=correlation,regional=regional,changes=changes,quality=quality)


def save_analysis(result, output, label):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    for key in ['coverage','summary','correlation','regional','changes']:
        result[key].to_csv(output/f'{key}.csv',index=key=='correlation')
    (output/'quality.json').write_text(json.dumps({**result['quality'],'dataset':label},ensure_ascii=False,indent=2),encoding='utf-8')
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(result['coverage'].year,result['coverage'].universities)
    ax.set(xlabel='Год',ylabel='Число вузов',title=f'Охват наблюдений — {label}')
    fig.tight_layout(); fig.savefig(output/'coverage.png',dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7,6))
    corr = result['correlation']
    names = ['Студенты','Балл','Наука','Доход','Занятость']
    im = ax.imshow(corr,vmin=-1,vmax=1,cmap='coolwarm')
    ax.set_xticks(range(len(names)),names,rotation=35,ha='right')
    ax.set_yticks(range(len(names)),names)
    for i in range(len(names)):
        for j in range(len(names)):
            value=corr.iloc[i,j]
            ax.text(j,i,f'{value:.2f}' if np.isfinite(value) else '—',ha='center',va='center')
    ax.set_title(f'Корреляции Спирмена — {label}')
    fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(output/'correlation.png',dpi=160); plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('input',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--label',required=True)
    args=parser.parse_args()
    save_analysis(explore(load_data(args.input)),args.output,args.label)
