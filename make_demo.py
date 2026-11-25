import numpy as np
import pandas as pd
from pathlib import Path
rng=np.random.default_rng(42)
rows=[]
for i in range(60):
    base=rng.uniform(0,1)
    for year in range(2019,2026):
        t=year-2019
        rows.append([f'Демо-вуз {i+1:02}',f'Демо-регион {i%6+1}',year,round(1500+base*14000+t*100+rng.normal(0,200)),round(55+base*30+t*.4+rng.normal(0,1),2),round(20+base*130+t*2+rng.uniform(0,5),2),round(80+base*300+t*4+rng.uniform(0,8),2),round(60+base*25+t*.6+rng.normal(0,1),2)])
pd.DataFrame(rows,columns=['university','region','year','students','score','research','income','employment']).to_csv(Path(__file__).parent/'data/demo.csv',index=False)
