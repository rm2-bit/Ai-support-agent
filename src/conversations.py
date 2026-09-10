from pathlib import Path
import pandas as pd

EXPECTED = {'tweet_id','author_id','inbound','created_at','text','response_tweet_id','in_response_to_tweet_id'}

def find_csv(root: Path):
    files=list(root.glob('*.csv'))
    if not files: raise FileNotFoundError('No CSV found')
    return next((p for p in files if p.name.lower()=='twcs.csv'), files[0])

def load(root: Path, nrows=None):
    p=find_csv(root)
    df=pd.read_csv(p, nrows=nrows)
    missing=EXPECTED-set(df.columns)
    if missing: raise ValueError(f'Missing columns: {missing}')
    return df

def normalize(df):
    df=df.copy()
    df['tweet_id']=df['tweet_id'].astype(str)
    df['in_response_to_tweet_id']=df['in_response_to_tweet_id'].astype('Int64').astype(str)
    df.loc[df['in_response_to_tweet_id'].eq('<NA>'),'in_response_to_tweet_id']=''
    df['text']=df['text'].fillna('').astype(str).str.replace(r'\s+',' ',regex=True).str.strip()
    return df

def direct_pairs(df, brand=None):
    # A first, conservative approximation: inbound customer tweet -> direct outbound brand response.
    d=normalize(df)
    lookup=d.set_index('tweet_id')
    rows=[]
    for _, r in d.iterrows():
        if int(r['inbound']) != 1 or not r['text']:
            continue
        response_ids=str(r['response_tweet_id']).split(',') if r['response_tweet_id'] else []
        for rid in response_ids:
            rid=rid.strip()
            if rid and rid in lookup.index:
                rr=lookup.loc[rid]
                if int(rr['inbound']) == 0 and rr['text']:
                    rows.append({'customer_tweet_id':r['tweet_id'],'brand_tweet_id':rid,'customer_message':r['text'],'historical_reply':rr['text'],'created_at':r['created_at']})
    return pd.DataFrame(rows).drop_duplicates('customer_tweet_id')
