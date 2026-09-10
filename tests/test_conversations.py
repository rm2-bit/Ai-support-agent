import pandas as pd
from src.conversations import direct_pairs

def test_direct_pair():
    df=pd.DataFrame([
      {'tweet_id':1,'author_id':'a','inbound':1,'created_at':'x','text':'help','response_tweet_id':'2','in_response_to_tweet_id':''},
      {'tweet_id':2,'author_id':'b','inbound':0,'created_at':'x','text':'sure','response_tweet_id':'','in_response_to_tweet_id':'1'}])
    out=direct_pairs(df)
    assert len(out)==1 and out.iloc[0].historical_reply=='sure'
