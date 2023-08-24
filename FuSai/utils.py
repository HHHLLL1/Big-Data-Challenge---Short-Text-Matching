# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 08:06:06 2019

@author: Lenovo
"""

import numpy as np
import pandas as pd


train = pd.read_csv('./train.csv', header=None)

#数据处理（同一query类型的title组合在一起）
def tilte_groupby(df):
    d = df.groupby([0])[1].unique().reset_index().rename(columns={0:'id', 1:'query'})
    d['query'] = d['query'].apply(lambda x: x[0])
    d['title_concat'] = df.groupby([0])[3].unique().reset_index().apply(lambda x: ' '.join(x[3]), axis=1)
    #d.drop(['id'], axis=1, inplace=True)       
    d['label'] = df[4]
    d.to_csv('query_titleConcat.txt', index=False)    
    return d
    
df = tilte_groupby(train)