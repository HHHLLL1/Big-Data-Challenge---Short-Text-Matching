# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 19:05:56 2019

@author: Lenovo
"""

import numpy as np
import pandas as  pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from SiameseNet import SiameseNetwork
from gensim.models import Word2Vec
from feature import getSimilarityFeature, getfeature111, getfeature
import tqdm
import warnings
from torchnet import meter



warnings.filterwarnings('ignore')


def getVec(x, model):
    x = x.split(' ')
    a = []
    for i in range(50):
        try:
            a.append(model[x[i]])
        except:
            a.append(np.zeros(model.vector_size))
    return np.array(a)

def getBatchSize(df, model):
    df.index = range(len(df))
    a = []
    for i in range(len(df)):
        x = df.iloc[i]
        a.append(getVec(x, model))
    return np.atleast_2d(a)

train = pd.read_csv('train.csv', header=None)

model = Word2Vec.load('./word2vec.model')

snet = SiameseNetwork(embedding_size=model.vector_size,
                      feature_size=46,
                      out_channels=64,
                      kernel_size=[2,3,4],
                      max_text_len=50,
                      num_class=2,
                      dropout_rate=0.5)

snet.cuda()

op = optim.Adam(snet.parameters(), 0.01)
loss_func = nn.CrossEntropyLoss()


loss_meter = meter.AverageValueMeter()
confusion_meter = meter.ConfusionMeter(2)
previous_loss = 1e100

loss_meter.reset()
confusion_meter.reset()


snet.train()
for i in range(1, len(train)+1):
    n = np.random.choice(range(len(train)), 20)
    x = train.iloc[n, :]
    x_query = Variable(torch.Tensor(getBatchSize(x[1], model))).cuda()
    x_title = Variable(torch.Tensor(getBatchSize(x[3], model))).cuda()
    y = Variable(torch.tensor(x[4].values)).cuda()
    
    feature = getfeature(x)
    feature.drop([0,1,2,3,4], axis=1, inplace=True)
    feature = (feature-feature.min(axis=0))/(feature.max(axis=0)-feature.min(axis=0))
    feature = Variable(torch.Tensor(feature.values)).cuda()
    
    scores, predictions = snet(x_query, x_title, feature)
    loss = loss_func(scores, y)
    op.zero_grad()
    loss.backward()
    op.step()
    
    loss_meter.add(loss.data[0])
    confusion_meter.add(predictions.data, y.data)
    
#    if i%10 == 0:
    print(loss.data.cpu().numpy())
        
    if i>100:
        break
    
    


