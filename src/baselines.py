import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

def majority_predict(y_train, n):
    return [y_train.value_counts().idxmax()] * n

def train_tfidf(X, y):
    model=Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=50000,sublinear_tf=True)),('clf',LogisticRegression(max_iter=1000,class_weight='balanced'))])
    model.fit(X,y)
    return model

def metrics(y_true,y_pred):
    return {'accuracy':accuracy_score(y_true,y_pred),'macro_f1':f1_score(y_true,y_pred,average='macro')}
