# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 09:42:37 2019

@author: Lenovo
"""


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable



class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        # (B, L, H) -> (B , L, 1)
        encoder_outputs = encoder_outputs.permute(0, 2, 1)
        energy = self.projection(encoder_outputs)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        # (B, L, H) * (B, L, 1) -> (B, H)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs, weights


#class AttnClassifier(nn.Module):
#    def __init__(self, input_dim, embedding_dim, hidden_dim):
#        super().__init__()
#        self.input_dim = input_dim
#        self.embedding_dim = embedding_dim
#        self.hidden_dim = hidden_dim
##        self.embedding = nn.Embedding(input_dim, embedding_dim)
#        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
#        self.attention = SelfAttention(hidden_dim)
#        self.fc = nn.Linear(hidden_dim, 1)
#        
#    def set_embedding(self, vectors):
#        self.embedding.weight.data.copy_(vectors)
#        
#    def forward(self, inputs, lengths):
#        batch_size = inputs.size(1)
#        # (L, B)
##        embedded = self.embedding(inputs)
#        # (L, B, E)
##        packed_emb = nn.utils.rnn.pack_padded_sequence(inputs, lengths)
#        out, hidden = self.lstm(inputs)
#        out = nn.utils.rnn.pad_packed_sequence(out)[0]
#        out = out[:, :, :self.hidden_dim] + out[:, :, self.hidden_dim:]
#        # (L, B, H)
#        embedding, attn_weights = self.attention(out.transpose(0, 1))
#        # (B, HOP, H)
#        outputs = self.fc(embedding.view(batch_size, -1))
#        # (B, 1)
#        return outputs, attn_weights
#
#
#a =  [[14,4,5,58, 58],[5,6,63,4, 941]]

#a = np.random.rand(3, 50, 300)
#a = torch.Tensor(a)
#att = SelfAttention(50)
#out, weight = att(a)



class SiameseNetwork(nn.Module):
    
    '''
    embedding_size: 每个词向量的长度
    
    out_channels: 卷积产生的通道数，有多少个out_channels，就需要多少个一维卷
                 积（也就是卷积核的数量）。图中可以看到每类卷积核，或者说filter，
                 分别有两个，所以out_channels=2
                 
    kernel_size: 卷积核的大小，即图中filter的行数。列数和embedding_size相同，因
                为这是一维的卷积。如图所示，分别是2,3,4
                
    max_text_len: the size of the window to take a max over. 我的理解是，MaxPool1d中
                会取图中feature maps中的最大值，为此，得确定一个范围，即多长的feature map中
                的最大值。当长度为feature map全长时，最大值自然只有一个，当长度为feature map全
                长减1时，最大值会有两个（这里步长stride假设为1），以此类推。 在这里我们选可能
                输入的句子中的最大长度（feature map长度最长为句子长度，这是卷积的结果，比较
                简单，就不赘述了），图中只有一条句子，所以最大长度是该句子的长度7.
    '''
    
    def __init__(self, embedding_size, feature_size, out_channels, kernel_size, 
                                             max_text_len, num_class, dropout_rate, is_cuda=True):
        super(SiameseNetwork, self).__init__()
        self.dropout_rate = dropout_rate
        self.num_class = num_class
        self.is_cuda = is_cuda
        fc_feature = out_channels*len(kernel_size)*2 + 1 + max_text_len*2 + feature_size
 
#        self.embedding = nn.Embedding(num_embeddings=vocab_size, 
#                                embedding_dim=embedding_size)  # 创建词向量对象
        self.convs_query = nn.ModuleList([
                			  nn.Sequential(nn.Conv1d(in_channels=embedding_size, 
                                        			out_channels=out_channels, 
                                        			kernel_size=h),  # 卷积层，这里的Sequential只包含了卷积层，感觉这样写有些多余，
                                        									# Sequential可以把卷积层、 激活函数层和池化层都包含进去
                              nn.ReLU(), # 激活函数层
                              nn.MaxPool1d(kernel_size=max_text_len-h+1)) # 池化层，这里kernel_size的值应该不用进行这样多余的计算，我们的目的
                              #是将feature maps转为一行一列的向量，那么只要保证这里的kernel_size大于等于feature maps的行数就可以了
                     for h in kernel_size	# 不同卷积层的kernel_size不一样，注意：这里的kernel_size和池化层的kernel_size是不同的
                    ])			# 创建多个卷积层，包含了 图中的convolution、activation function和 maxPooling
    
    
        self.convs_title = nn.ModuleList([
                			  nn.Sequential(nn.Conv1d(in_channels=embedding_size, 
                                        			out_channels=out_channels, 
                                        			kernel_size=h),  # 卷积层，这里的Sequential只包含了卷积层，感觉这样写有些多余，
                                        									# Sequential可以把卷积层、 激活函数层和池化层都包含进去
                              nn.ReLU(), # 激活函数层
                              nn.MaxPool1d(kernel_size=max_text_len-h+1)) # 池化层，这里kernel_size的值应该不用进行这样多余的计算，我们的目的
                              #是将feature maps转为一行一列的向量，那么只要保证这里的kernel_size大于等于feature maps的行数就可以了
                     for h in kernel_size	# 不同卷积层的kernel_size不一样，注意：这里的kernel_size和池化层的kernel_size是不同的
                    ])
    
        self.similarity_layer = nn.Bilinear(out_channels*len(kernel_size), out_channels*len(kernel_size), 1)
        self.attention = SelfAttention(max_text_len)
        self.fc1 = nn.Sequential(
                            nn.Linear(in_features=fc_feature, out_features=fc_feature), # 把最终的特征向量的大小转换成类别的大小，以此作为前向传播的输出。这里和图中的有些区别，貌似并没有用到softmax和regularization
                            nn.ReLU(),
                            nn.Dropout(p=dropout_rate),
                            nn.Linear(in_features=fc_feature, out_features=num_class)
                            )
    
    
    
    def forward_query(self, x):
        out_atten, _ = self.attention(x)
#        embed_x = self.embedding(x) # 将句子转为词向量矩阵，大小为1*7*5，这里的1是表示只有1条句子
        x = x.permute(0, 2, 1) # 将矩阵转置
        out = [conv(x) for conv in self.convs_query]  #计算每层卷积的结果，这里输出的结果已经经过池化层处理了，对应着图中的6 个univariate vectors
        out = torch.cat(out, dim=1)  # 对6 个univariate vectors进行拼接
        out = out.view(-1, out.size(1))  #按照行优先的顺序排成一个n行1列的数据，这部分在图中没有表现出来，其实在程序运行的过程中6 个univariate vectors的大小可能是1*1*1，进行拼接后是1*1*6，我们在这里将其转换为1*6，可见每个值都没有丢失，只是矩阵的形状变了而已
        out = F.dropout(input=out, p=self.dropout_rate) # 这里也没有在图中的表现出来，这里是随机让一部分的神经元失活，避免过拟合。它只会在train的状态下才会生效。进入train状态可查看nn.Module。train()方法
        return out, out_atten
    
    def forward_title(self, x):
        out_atten, _ = self.attention(x)
#        embed_x = self.embedding(x) # 将句子转为词向量矩阵，大小为1*7*5，这里的1是表示只有1条句子
        x = x.permute(0, 2, 1) # 将矩阵转置
        out = [conv(x) for conv in self.convs_title]  #计算每层卷积的结果，这里输出的结果已经经过池化层处理了，对应着图中的6 个univariate vectors
        out = torch.cat(out, dim=1)  # 对6 个univariate vectors进行拼接
        out = out.view(-1, out.size(1))  #按照行优先的顺序排成一个n行1列的数据，这部分在图中没有表现出来，其实在程序运行的过程中6 个univariate vectors的大小可能是1*1*1，进行拼接后是1*1*6，我们在这里将其转换为1*6，可见每个值都没有丢失，只是矩阵的形状变了而已
        out = F.dropout(input=out, p=self.dropout_rate) # 这里也没有在图中的表现出来，这里是随机让一部分的神经元失活，避免过拟合。它只会在train的状态下才会生效。进入train状态可查看nn.Module。train()方法
        return out, out_atten

    
    
    def forward(self, input1, input2, feature):
        put1, out_atten1 = self.forward_query(input1)
        put2, out_atten2 = self.forward_title(input2)
        sim = self.similarity_layer(put1, put2)
#        print(put1.shape)
#        print(put2.shape)
#        print(sim.shape)
#        print(out_atten1.shape)
#        print(out_atten2.shape)
#        print(feature.shape)
        out = torch.cat([put1, put2, sim, out_atten1, out_atten2, feature], dim=1)
        scores = self.fc1(out)
        values, predictions = torch.max(out, 1)
        
        return scores, predictions
        
    def cosine_similarity(self, vect1, vect2):
        vect1 = vect1.cpu().detach().numpy()
        vect2 = vect2.cpu().detach().numpy()
        vect1mod = np.array([np.sqrt(i.dot(i)) for i in vect1])
        vect2mod = np.array([np.sqrt(i.dot(i)) for i in vect2])
#        if vect1mod!=0 and vect2mod!=0:
        v = np.array([np.sqrt(i.dot(j)) for i, j in zip(vect1,vect2)])
        
        try:
            simla = (v+1)/(vect1mod*vect2mod+1)
        except:
            simla = 0
#        simla = (vect1.dot(vect2))/(vect1mod*vect2mod)
#        else:
#            simla = 0
        if self.is_cuda:
            return Variable(torch.Tensor(np.atleast_2d(simla).T)).cuda()
        else:
            return Variable(torch.Tensor(np.atleast_2d(simla).T))
    
