from pathlib import Path
import json
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from core import load_data, FEATURES
from research import experiment

root = Path(__file__).parent
out = root/'research_results'
out.mkdir(exist_ok=True)
df = load_data(root/'data/demo.csv')
with warnings.catch_warnings(record=True) as records:
    warnings.simplefilter('always', ConvergenceWarning)
    result = experiment(df)
for key in ['folds','ranking','comparison','predictions','forecast']:
    result[key].to_csv(out/f'{key}.csv',index=False)
summary = {'dataset':'Синтетический демонстрационный набор', 'rows':len(df),
           'winner':result['winner'],'test_year':result['test_year'],
           'convergence_warnings':sum(issubclass(r.category,ConvergenceWarning) for r in records)}
(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
plt.figure(figsize=(9,5))
r=result['comparison']; plt.barh(r.model,r.MAE); plt.xlabel('MAE, процентные пункты'); plt.title('Ошибка на отложенном году (синтетические данные)'); plt.tight_layout(); plt.savefig(out/'comparison.png',dpi=160); plt.close()
plt.figure(figsize=(6,5)); p=result['predictions']; plt.scatter(p.target,p.prediction,alpha=.7); lo=min(p.target.min(),p.prediction.min()); hi=max(p.target.max(),p.prediction.max()); plt.plot([lo,hi],[lo,hi],'--'); plt.xlabel('Фактическое значение, %'); plt.ylabel('Прогноз, %'); plt.title('Проверка прогноза на демонстрационных данных'); plt.tight_layout(); plt.savefig(out/'predictions.png',dpi=160); plt.close()
print(summary); print(r.to_string(index=False))
