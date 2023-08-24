# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 10:00:00 2019

@author: Lenovo
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn import metrics
from sklearn.model_selection import KFold



path="/home/kesci/input/bytedance/first-round/"
train = pd.read_csv(path+'train.csv', header=None, chunksize=5000000)
test = pd.read_csv(path+'test.csv', header=None)
pre_df = test[[0, 2]]




kf_n = 5
kf = KFold(n_splits=kf_n, shuffle=True, random_state=2019)
pre = 0

model = LogisticRegression(penalty='l1', C=0.2, n_jobs=-1)
x = 1
for df in train:
    
    ct = CountVectorizer(ngram_range=(1, 2), min_df=50)
    print('第%d次训练开始'%x)
    
    #训练集
    d = df[1] + ' ' +df[3]
    y_train = df[4].copy()
    x_train = ct.fit_transform(d)
    
        
    print('训练集处理完成')

    #测试集
    d = test[1] + ' ' +test[3]
    x_test = ct.transform(d)
    
    
    print('测试集处理完成')
    
    del d

        
    for x_tr, x_vail in kf.split(x_train):
        x_t1 = x_train[x_tr]
        y_t1 = y_train.iloc[x_tr]
        
        x_t2 = x_train[x_vail]
        y_t2 = y_train.iloc[x_vail]
        
        model.fit(x_t1, y_t1)
        print(metrics.roc_auc_score(y_t2, model.predict_proba(x_t2)[:, 1]))
        pre += model.predict_proba(x_test)[:, 1]
        
        
    del x_train
    del x_test
    del ct
    
    print('第%d次训练结束'%x)

    x += 1
        

    
pre1 = pre/(kf_n*(x-1))

pre_df = pre_df.rename(columns={0:'query_id', 2:'query_title_id'})
pre_df['prediction'] = pre1
pre_df.to_csv('lr_ct_500wEoch_无cv.csv', index=False, header=False)
