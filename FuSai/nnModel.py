# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 11:20:01 2019

@author: Lenovo
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable



class TextCNN(nn.Module):
    
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
    
    def __init__(self, embedding_size, out_channels, kernel_size, max_text_len, num_class, dropout_rate, vocab_size=1):
        super(TextCNN, self).__init__()
        self.dropout_rate = dropout_rate
        self.num_class = num_class
 
#        self.embedding = nn.Embedding(num_embeddings=vocab_size, 
#                                embedding_dim=embedding_size)  # 创建词向量对象
        self.convs = nn.ModuleList([
                			  nn.Sequential(nn.Conv1d(in_channels=embedding_size, 
                                        			out_channels=out_channels, 
                                        			kernel_size=h),  # 卷积层，这里的Sequential只包含了卷积层，感觉这样写有些多余，
                                        									# Sequential可以把卷积层、 激活函数层和池化层都包含进去
                              nn.ReLU(), # 激活函数层
                              nn.MaxPool1d(kernel_size=max_text_len-h+1)) # 池化层，这里kernel_size的值应该不用进行这样多余的计算，我们的目的
                              #是将feature maps转为一行一列的向量，那么只要保证这里的kernel_size大于等于feature maps的行数就可以了
                     for h in kernel_size	# 不同卷积层的kernel_size不一样，注意：这里的kernel_size和池化层的kernel_size是不同的
                    ])			# 创建多个卷积层，包含了 图中的convolution、activation function和 maxPooling
        self.fc = nn.Linear(in_features=out_channels*len(kernel_size),
                            out_features=num_class) # 把最终的特征向量的大小转换成类别的大小，以此作为前向传播的输出。这里和图中的有些区别，貌似并没有用到softmax和regularization
    
    def forward(self, x):
#        embed_x = self.embedding(x) # 将句子转为词向量矩阵，大小为1*7*5，这里的1是表示只有1条句子
        x = x.permute(0, 2, 1) # 将矩阵转置
        out = [conv(x) for conv in self.convs]  #计算每层卷积的结果，这里输出的结果已经经过池化层处理了，对应着图中的6 个univariate vectors
        out = torch.cat(out, dim=1)  # 对6 个univariate vectors进行拼接
        out = out.view(-1, out.size(1))  #按照行优先的顺序排成一个n行1列的数据，这部分在图中没有表现出来，其实在程序运行的过程中6 个univariate vectors的大小可能是1*1*1，进行拼接后是1*1*6，我们在这里将其转换为1*6，可见每个值都没有丢失，只是矩阵的形状变了而已
        out = F.dropout(input=out, p=self.dropout_rate) # 这里也没有在图中的表现出来，这里是随机让一部分的神经元失活，避免过拟合。它只会在train的状态下才会生效。进入train状态可查看nn.Module。train()方法
        out = self.fc(out)
        out = F.softmax(out)
        return out
    
    



#多层感知机
class MLP_Wrapper(nn.Module):

    def __init__(self, num_layers, hidden_dim, num_input, num_output,
                 drop_keep_prob=0.6
                 ):
        super(MLP_Wrapper, self).__init__()

        self.num_layers = num_layers
        if type(hidden_dim) != list:
            self.hidden_dim = [hidden_dim for _ in range(num_layers)]
        self.num_input = num_input
        self.num_output = num_output

        self.drop_keep_prob = drop_keep_prob

        self.bulid_model()
        print('buld_model_success.')

    def bulid_model(self):

        self.in_fc = nn.Linear(self.num_input, self.hidden_dim[0])

        self.in_bn = nn.BatchNorm1d(self.hidden_dim[0])

        for i in range(self.num_layers -1):
            self.add_module('hidden_layer_{}'.format(i),
                            nn.Linear(self.hidden_dim[i], self.hidden_dim[i + 1]))

            self.add_module('bn_{}'.format(i), nn.BatchNorm1d(self.hidden_dim[i+1]))

        self.out_fc = nn.Linear(self.hidden_dim[-1], self.num_output)


    def forward(self, x):
        x = x.float()
        x = self.in_fc(x)
        x = self.in_bn(x)
        x = F.relu(x)
        for i in range(self.num_layers-1):
            x = self.__getattr__('hidden_layer_{}'.format(i))(x)
            x = self.__getattr__('bn_{}'.format(i))(x)

            x = F.relu(x)

        x = F.dropout(x, self.drop_keep_prob, self.training)

        out = self.out_fc(x)

        return out

    def compile_optimizer(self, opt, lr,  num_epochs, l2_reg = 0,
                          lr_decay=False, lr_decay_rate=0.9, lr_decay_min = None,
                          lr_decay_every = 1000,
                          ):

        self.num_epochs = num_epochs
        self.lr_decay = lr_decay
        self.lr_decay_rate=  lr_decay_rate
        self.lr_decay_min = lr_decay_min
        self.lr_decay_every = lr_decay_every

        if opt == 'adam':

            self.opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=l2_reg)

    def decay_lr(self):
        for group in self.opt.param_groups:
            old_lr = group['lr']
            new_lr = old_lr* self.lr_decay_rate

            if self.lr_decay_min: new_lr = max(self.lr_decay_min, new_lr)

            group['lr'] = new_lr

    def loss(self, logits, target):

        loss = F.cross_entropy(logits, target)
        return loss


    def fit(self, train_loader, val_loader, print_every=100,val_every= 1000):

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.train()
        num_train_batches = len(train_loader)
        print('training start.')
        for epoch in range(self.num_epochs):
            for i ,(data, target) in enumerate(train_loader):
                step = num_train_batches * epoch + i
                if self.lr_decay and step >0 and step% self.lr_decay_every ==0:
                    self.decay_lr()

                logits= self.forward(data)
                loss = self.loss(logits, target)

                self.opt.zero_grad()
                loss.backward()
               
                self.opt.step()

                acc = (torch.argmax(logits, -1) == target).sum().float() / data.shape[0]
                if step %print_every ==0:

                    print('step:{0}, loss:{1}, acc:{2}'.format(step, loss, acc))

                if step >0 and step % val_every ==0:
                    loss, acc = self.evaluate()
                    print('validate at step {}, loss:{}, acc:{}'.format(step, loss, acc))
                    self.train()

    def evaluate(self):
        self.eval()
        losses, acc_ = [], []
        for data, target in self.val_loader:

            logits = self.forward(data)
            loss = self.loss(logits, target)

            acc = (torch.argmax(logits, -1) == target).sum().float() / data.shape[0]
            losses.append(loss)
            acc_.append(acc)


        return torch.mean(torch.stack(losses)), torch.mean(torch.stack(acc_))

    def predict(self,x):
        logits = self.forward(x)

        pred = torch.argmax(logits, -1)

        return pred

#m = MLP_Wrapper(num_layers=3, num_input=3, num_output=2, hidden_dim=3)



#class TextCNN(nn.Module):
#    def __init__(self, param: dict):
#        super(TextCNN, self).__init__()
#        ci = 1  # input chanel size
#        kernel_num = param['kernel_num'] # output chanel size
#        kernel_size = param['kernel_size']
#        vocab_size = param['vocab_size']
#        embed_dim = param['embed_dim']
#        dropout = param['dropout']
#        class_num = param['class_num']
#        self.param = param
#        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
#        self.conv11 = nn.Conv2d(ci, kernel_num, (kernel_size[0], embed_dim))
#        self.conv12 = nn.Conv2d(ci, kernel_num, (kernel_size[1], embed_dim))
#        self.conv13 = nn.Conv2d(ci, kernel_num, (kernel_size[2], embed_dim))
#        self.dropout = nn.Dropout(dropout)
#        self.fc1 = nn.Linear(len(kernel_size) * kernel_num, class_num)
#
#    def init_embed(self, embed_matrix):
#        self.embed.weight = nn.Parameter(torch.Tensor(embed_matrix))
#
#    @staticmethod
#    def conv_and_pool(x, conv):
#        # x: (batch, 1, sentence_length, embed_dim)
#        x = conv(x)
#        # x: (batch, kernel_num, H_out, 1)
#        x = F.relu(x.squeeze(3))
#        # x: (batch, kernel_num, H_out)
#        x = F.max_pool1d(x, x.size(2)).squeeze(2)
#        #  (batch, kernel_num)
#        return x
#
#    def forward(self, x):
#        # x: (batch, sentence_length)
#        x = self.embed(x)
#        # x: (batch, sentence_length, embed_dim)
#        # TODO init embed matrix with pre-trained
#        x = x.unsqueeze(1)
#        # x: (batch, 1, sentence_length, embed_dim)
#        x1 = self.conv_and_pool(x, self.conv11)  # (batch, kernel_num)
#        x2 = self.conv_and_pool(x, self.conv12)  # (batch, kernel_num)
#        x3 = self.conv_and_pool(x, self.conv13)  # (batch, kernel_num)
#        x = torch.cat((x1, x2, x3), 1)  # (batch, 3 * kernel_num)
#        x = self.dropout(x)
#        logit = F.log_softmax(self.fc1(x), dim=1)
#        return logit
#    
#    
#textcnn_param = {
#        "vocab_size": 50,
#        "embed_dim": 60,
#        "class_num": 2,
#        "kernel_num": 16,
#        "kernel_size": [3, 4, 5],
#        "dropout": 0.5,
#    }
#
#textcnn = TextCNN(textcnn_param)



#Siamese network
class SiameseNetwork(nn.Module):
	    def __init__(self):
	        super(SiameseNetwork, self).__init__()
	        self.cnn1 = nn.Sequential(
	            nn.ReflectionPad2d(1),
	            nn.Conv2d(1, 4, kernel_size=3),
	            nn.ReLU(inplace=True),
	            nn.BatchNorm2d(4),
	            nn.Dropout2d(p=.2),
	            
	            nn.ReflectionPad2d(1),
	            nn.Conv2d(4, 8, kernel_size=3),
	            nn.ReLU(inplace=True),
	            nn.BatchNorm2d(8),
	            nn.Dropout2d(p=.2),
	
	            nn.ReflectionPad2d(1),
	            nn.Conv2d(8, 8, kernel_size=3),
	            nn.ReLU(inplace=True),
	            nn.BatchNorm2d(8),
	            nn.Dropout2d(p=.2),
	        )
	
	        self.fc1 = nn.Sequential(
	            nn.Linear(8*100*100, 500),
	            nn.ReLU(inplace=True),
	
	            nn.Linear(500, 500),
	            nn.ReLU(inplace=True),
	
	            nn.Linear(500, 5)
	        )
	
	    def forward_once(self, x):
	        output = self.cnn1(x)
	        output = output.view(output.size()[0], -1)
	        output = self.fc1(output)
	        return output
	
	    def forward(self, input1, input2):
	        output1 = self.forward_once(input1)
	        output2 = self.forward_once(input2)
	        return output1, output2
        

#对比损失函数
class ContrastiveLoss(torch.nn.Module):
	    """
	    Contrastive loss function.
	    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
	    """
	
	    def __init__(self, margin=2.0):
	        super(ContrastiveLoss, self).__init__()
	        self.margin = margin
	
	    def forward(self, output1, output2, label):
	        euclidean_distance = F.pairwise_distance(output1, output2)
	        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2)  
	                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
	
	        return loss_contrastive