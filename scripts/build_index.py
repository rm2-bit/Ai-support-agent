from pathlib import Path
import sys, json, pickle
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.retrieval import CaseRetriever

ROOT=Path(__file__).resolve().parents[1]
df=pd.read_csv(ROOT/'data/processed/direct_pairs.csv').dropna(subset=['customer_message','historical_reply'])
# Development index over a bounded sample; change after brand selection.
df=df.head(20000)
cases=df.to_dict('records')
r=CaseRetriever(); r.build(cases)
with open(ROOT/'data/processed/retriever.pkl','wb') as f: pickle.dump(r,f)
print('Built index over',len(cases),'cases.')
