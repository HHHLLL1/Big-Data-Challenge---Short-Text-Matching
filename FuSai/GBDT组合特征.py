# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 16:07:41 2019

@author: Lenovo
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
import lightgbm  as lgb
import feature



train = pd.read_csv('train.csv', header=None, names=['query_id', 'query', 'title_id', 'title', 'label'])
test = pd.read_csv('test.csv', header=None, names=['query_id', 'query', 'title_id', 'title', 'label'])

train['index'] = train.index

y_train = train['label']
y_test = test['label']


#单维度转化率
single_fea_set = ['query', 'title']
#训练集
train = feature.get_single_dimension_rate_train_feature(train, single_fea_set)
#测试集
test = feature.get_single_dimension_rate_test_feature(train, test, single_fea_set)

#交叉维度转化率
jiaoch_fea_set = ['query', 'title']
#训练集
train = feature.get_jiaocha_dimension_rate_train_feature(train, jiaoch_fea_set)
#测试集
test = feature.get_jiaocha_dimension_rate_test_feature(train, test, jiaoch_fea_set)

train.index = train['index']
train.drop('index', axis=1, inplace=True)

df = pd.concat((train, test))
df.fillna(-1, inplace=True)

df = df.rename(columns = {'query_id':0, 'query':1, 'title_id':2, 'title':3, 'label':4})

df = feature.getfeature(df)
df.dropna(axis=1, how='all', inplace=True)
df.fillna(-1, inplace=True)
df.drop([0,1,2,3,4], axis=1, inplace=True)

train = df.iloc[:15000, :]
test = df.iloc[15000:, :]



#训练

#lgb
#lgb_parms = {
#        "boosting_type": "gbdt",
#        "num_leaves": 127,
#        "max_depth": -1,
#        "learning_rate": 0.01,
#        "n_estimators": 10000,
#        'feature_fraction': 0.8,
#        'bagging_fraction': 0.8,
#        "max_bin": 425,
#        "subsample_for_bin": 20000,
#        "objective": 'binary',
#        'metric': ['binary_logloss',"auc"],
#        "min_split_gain": 0,
#        "min_child_weight": 0.001,
#        "min_child_samples": 20,
#        "subsample": 0.8,
#        "subsample_freq": 1,
#        "colsample_bytree": 0.8,
#        "reg_alpha": 3,
#        "reg_lambda": 5,
#        "seed": 2018,
#        "n_jobs": -1,
#        "verbose": 1,
#        "silent": False
#    }


lgb_parms = {
        'boosting_type': 'gbdt',
        'objective': "binary",
        'metric': ['auc', 'binary_logloss'],
        'num_leaves': 30,
        'num_tree': 32,
        'learning_rate': 0.01,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
#        'device':'gpu',
#        'gpu_platform_id':0,
#        'gpu_device_id':0
    }



lgb1 = lgb.Dataset(train, y_train)
lgb2 = lgb.Dataset(test, y_test)

gbm = lgb.train(
            lgb_parms,
            lgb1,
            num_boost_round=2000,
            valid_sets=lgb2,
            verbose_eval=100,
            early_stopping_rounds=300,
            )



#lr

gbdt = GradientBoostingClassifier(n_estimators=20)
onehot = OneHotEncoder()


#使用GBDT模型训练原始特征
gbdt.fit(train, y_train)
#训练OneHot编码， 识别每一列有多少可取值
onehot.fit(gbdt.apply(train)[:, :, 0])


x_train = onehot.transform(gbdt.apply(train)[:, :, 0])
x_test = onehot.transform(gbdt.apply(test)[:, :, 0])


#合并原始特征和组合特征
x_train = np.hstack((train.values, x_train.toarray()))
x_test = np.hstack((test.values, x_test.toarray()))

lr = LogisticRegression(penalty='l1', C=0.3)
lr.fit(x_train, y_train)

print('auc:',metrics.roc_auc_score(y_test, lr.predict_proba(x_test)[:,1]))
print('loss:',metrics.log_loss(y_test, lr.predict_proba(x_test)[:,1]))






'''
GBDT组合特征对lr（线性模型）效果提升比较大，这应该算是一种模型融合，
但是，在树模型上表现差。
'''









