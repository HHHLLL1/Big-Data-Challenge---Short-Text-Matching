# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 14:46:36 2019

@author: Lenovo
"""

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from nnModel import TextCNN
#from SiameseNet import SiameseNetwork



def getVec(x, model):
    x = x.split(' ')
    a = []
    for i in range(50):
        try:
            a.append(model[x[i]])
        except:
            a.append(np.zeros(model.vector_size))
    return np.array(a)

def getBatchSize(df, num, model):
    n = np.random.choice(len(df), num, replace=False)
    a = []
    y = []
    for i in n:
        x = df.iloc[i, :]
        x = x[1] + x[3]
        a.append(getVec(x, model))
        y.append(df.iloc[i, 4])
    return np.array(a), y
    
    


                
model = Word2Vec.load('./word2vec.model')

train = pd.read_csv('train.csv', header=None)
test = pd.read_csv('test.csv', header=None)


textcnn = TextCNN(embedding_size=model.vector_size, out_channels=100, kernel_size=[2,3,4], max_text_len=50, num_class=2, dropout_rate=0.5)
textcnn.cuda()

op = optim.Adam(textcnn.parameters(), lr=0.01)
loss_func = nn.CrossEntropyLoss()

textcnn.train()

for e in range(100):
    x, y = getBatchSize(train, 30, model)
    x = Variable(torch.Tensor(x)).cuda()
    y = Variable(torch.tensor(y)).cuda()
    
    
    for i in range(2):
        
        out = textcnn(x)
        loss = loss_func(out, y)
        op.zero_grad()
        loss.backward()
        op.step()
        
    if e%100 == 0:
        print(loss.data.cpu().numpy())
        
y_pre = []
textcnn.cpu()
textcnn.eval()
for i in range(len(test)):
    s = test.iloc[i, :]
    s = s[1] + s[3]
    x_test = getVec(s, model)
    x_test = Variable(torch.Tensor([x_test]))
    
    y_pre.append(textcnn.forward(x_test))
    

        
    
    
    
