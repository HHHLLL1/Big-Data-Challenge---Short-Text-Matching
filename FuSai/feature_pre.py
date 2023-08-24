import numpy as np
import pandas as pd
import feature
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics


def getfff(df):
    df = feature.getfeature(df)
#    df = feature.getSimilarityFeature111(df)
    return df



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

df = getfff(df)
df.drop([0,1,2,3,4], axis=1, inplace=True)
df = (df-df.min(axis=0))/(df.max(axis=0)-df.min(axis=0))
df.fillna(-1, inplace=True)


#lgb_parms = {
#        "boosting_type": "gbdt",
#        "num_leaves": 127,
#        "max_depth": -1,
#        "learning_rate": 0.01,
#        "n_estimators": 1000,
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
        'num_leaves': 64,
        'learning_rate': 0.01,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 1
#        'device':'gpu',
#        'gpu_platform_id':0,
#        'gpu_device_id':0
    }

train = lgb.Dataset(df.iloc[:15000, :], y_train)
test = lgb.Dataset(df.iloc[15000:, :], y_test)

gbm = lgb.train(
            lgb_parms,
            train,
            num_boost_round=2000,
            valid_sets=test,
            verbose_eval=100,
            early_stopping_rounds=300,
            )



from sklearn.linear_model import LogisticRegression


train = df.iloc[:15000, :]
test = df.iloc[15000:, :]

lr = LogisticRegression(penalty='l1', C=0.3)
lr = lr.fit(train, y_train)
print('auc:',metrics.roc_auc_score(y_test, lr.predict_proba(test)[:,1]))
print('loss:',metrics.log_loss(y_test, lr.predict_proba(test)[:,1]))


a = pd.DataFrame(df.columns, columns=['fea_name'])
a['lgb'] = gbm.feature_importance()
a['lr'] = lr.coef_[0]





