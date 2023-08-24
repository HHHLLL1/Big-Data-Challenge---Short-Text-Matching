# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 08:25:18 2019

@author: Lenovo
"""



"""
句法分析、实体/N元模型/基于词汇的特征、统计特征和词汇嵌入等方法

文本向量作为机器学习的特征向量，然后利用余弦相似性、单词聚类、文本分类等方法来衡量文本的相似性。
"""



import numpy as np
import pandas as pd
import Levenshtein
import math
import time
import tqdm
from gensim.models import Word2Vec, Doc2Vec
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import StratifiedKFold




#
def lcsubstr_lens(s1, s2): 
    m=[[0 for i in range(len(s2)+1)]  for j in range(len(s1)+1)]  #生成0矩阵，为方便后续计算，比字符串长度多了一列
    mmax=0   #最长匹配的长度
    p=0  #最长匹配对应在s1中的最后一位
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i]==s2[j]:
                m[i+1][j+1]=m[i][j]+1
                if m[i+1][j+1]>mmax:
                    mmax=m[i+1][j+1]
                    p=i+1
    return mmax


#
def lcseque_lens(s1, s2): 
     # 生成字符串长度加1的0矩阵，m用来保存对应位置匹配的结果
    m = [ [ 0 for x in range(len(s2)+1) ] for y in range(len(s1)+1) ] 
    # d用来记录转移方向
    d = [ [ None for x in range(len(s2)+1) ] for y in range(len(s1)+1) ] 

    for p1 in range(len(s1)): 
        for p2 in range(len(s2)): 
            if s1[p1] == s2[p2]:            #字符匹配成功，则该位置的值为左上方的值加1
                m[p1+1][p2+1] = m[p1][p2]+1
                d[p1+1][p2+1] = 'ok'          
            elif m[p1+1][p2] > m[p1][p2+1]:  #左值大于上值，则该位置的值为左值，并标记回溯时的方向
                m[p1+1][p2+1] = m[p1+1][p2] 
                d[p1+1][p2+1] = 'left'          
            else:                           #上值大于左值，则该位置的值为上值，并标记方向up
                m[p1+1][p2+1] = m[p1][p2+1]   
                d[p1+1][p2+1] = 'up'         
    (p1, p2) = (len(s1), len(s2)) 
    s = [] 
    while m[p1][p2]:    #不为None时
        c = d[p1][p2]
        if c == 'ok':   #匹配成功，插入该字符，并向左上角找下一个
            s.append(s1[p1-1])
            p1 -= 1
            p2 -= 1 
        if c == 'left':  #根据标记，向左找下一个
            p2 -= 1
        if c == 'up':   #根据标记，向上找下一个
            p1 -= 1
    return len(s)


def levenshtein(first, second):
        ''' 编辑距离算法（LevD） 
            Args: 两个字符串
            returns: 两个字符串的编辑距离 int
        '''
        if len(first) > len(second):
            first, second = second, first
        if len(first) == 0:
            return len(second)
        if len(second) == 0:
            return len(first)
        first_length = len(first) + 1
        second_length = len(second) + 1
        distance_matrix = [list(range(second_length)) for x in range(first_length)]
        # print distance_matrix
        for i in range(1, first_length):
            for j in range(1, second_length):
                deletion = distance_matrix[i - 1][j] + 1
                insertion = distance_matrix[i][j - 1] + 1
                substitution = distance_matrix[i - 1][j - 1]
                if first[i - 1] != second[j - 1]:
                    substitution += 1
                distance_matrix[i][j] = min(insertion, deletion, substitution)
                # print distance_matrix
        return distance_matrix[first_length - 1][second_length - 1]
    
    
#ratio
def ratio(first, second):
        ''' 编辑距离算法（LevD） 
            Args: 两个字符串
            returns: 两个字符串的编辑距离 int
        '''
        if len(first) > len(second):
            first, second = second, first
        if len(first) == 0:
            return len(second)
        if len(second) == 0:
            return len(first)
        first_length = len(first) + 1
        second_length = len(second) + 1
        sum_len = first_length + second_length 
        distance_matrix = [list(range(second_length)) for x in range(first_length)]
        # print distance_matrix
        for i in range(1, first_length):
            for j in range(1, second_length):
                deletion = distance_matrix[i - 1][j] + 1
                insertion = distance_matrix[i][j - 1] + 1
                substitution = distance_matrix[i - 1][j - 1]
                if first[i - 1] != second[j - 1]:
                    substitution += 2
                distance_matrix[i][j] = min(insertion, deletion, substitution)
                # print distance_matrix
        return np.float32((sum_len - distance_matrix[first_length - 1][second_length - 1])/sum_len)
   
#余弦相似度
def compute_cosine(text_a, text_b):
    # 找单词及词频
    words1 = text_a.split(' ')
    words2 = text_b.split(' ')
    # print(words1)
    words1_dict = {}
    words2_dict = {}
    for word in words1:
        # word = word.strip(",.?!;")
        word = word.lower()
        # print(word)
        if word != '' and word in words1_dict: # 这里改动了
            num = words1_dict[word]
            words1_dict[word] = num + 1
        elif word != '':
            words1_dict[word] = 1
        else:
            continue
    for word in words2:
        # word = word.strip(",.?!;")
        word = word.lower()
        if word != '' and word in words2_dict:
            num = words2_dict[word]
            words2_dict[word] = num + 1
        elif word != '':
            words2_dict[word] = 1
        else:
            continue

    # 排序
    dic1 = sorted(words1_dict.items(), key=lambda asd: asd[1], reverse=True)
    dic2 = sorted(words2_dict.items(), key=lambda asd: asd[1], reverse=True)

    # 得到词向量
    words_key = []
    for i in range(len(dic1)):
        words_key.append(dic1[i][0])  # 向数组中添加元素
    for i in range(len(dic2)):
        if dic2[i][0] in words_key:
            # print 'has_key', dic2[i][0]
            pass
        else:  # 合并
            words_key.append(dic2[i][0])
    # print(words_key)
    vect1 = []
    vect2 = []
    for word in words_key:
        if word in words1_dict:
            vect1.append(words1_dict[word])
        else:
            vect1.append(0)
        if word in words2_dict:
            vect2.append(words2_dict[word])
        else:
            vect2.append(0)

    # 计算余弦相似度
    sum = 0
    sq1 = 0
    sq2 = 0
    for i in range(len(vect1)):
        sum += vect1[i] * vect2[i]
        sq1 += pow(vect1[i], 2)
        sq2 += pow(vect2[i], 2)
    try:
        result = round(float(sum) / (math.sqrt(sq1) * math.sqrt(sq2)), 2)
    except ZeroDivisionError:
        result = 0.0
    return result



#皮尔逊系数
def Pehrson(text_a, text_b):
    # 找单词及词频
    words1 = text_a.split(' ')
    words2 = text_b.split(' ')
    # print(words1)
    words1_dict = {}
    words2_dict = {}
    for word in words1:
        # word = word.strip(",.?!;")
        word = word.lower()
        # print(word)
        if word != '' and word in words1_dict: # 这里改动了
            num = words1_dict[word]
            words1_dict[word] = num + 1
        elif word != '':
            words1_dict[word] = 1
        else:
            continue
    for word in words2:
        # word = word.strip(",.?!;")
        word = word.lower()
        if word != '' and word in words2_dict:
            num = words2_dict[word]
            words2_dict[word] = num + 1
        elif word != '':
            words2_dict[word] = 1
        else:
            continue

    # 排序
    dic1 = sorted(words1_dict.items(), key=lambda asd: asd[1], reverse=True)
    dic2 = sorted(words2_dict.items(), key=lambda asd: asd[1], reverse=True)

    # 得到词向量
    words_key = []
    for i in range(len(dic1)):
        words_key.append(dic1[i][0])  # 向数组中添加元素
    for i in range(len(dic2)):
        if dic2[i][0] in words_key:
            # print 'has_key', dic2[i][0]
            pass
        else:  # 合并
            words_key.append(dic2[i][0])
    # print(words_key)
    vect1 = []
    vect2 = []
    for word in words_key:
        if word in words1_dict:
            vect1.append(words1_dict[word])
        else:
            vect1.append(0)
        if word in words2_dict:
            vect2.append(words2_dict[word])
        else:
            vect2.append(0)

    # 计算Pehrson
    x = np.vstack([vect1, vect2])
    return np.corrcoef(x)[0][1]



#
def getBayesSmoothParam(origion_rate):
    origion_rate_mean = origion_rate.mean()
    origion_rate_var = origion_rate.var()
    alpha = origion_rate_mean / origion_rate_var * (origion_rate_mean * (1 - origion_rate_mean) - origion_rate_var)
    beta = (1 - origion_rate_mean) / origion_rate_var * (origion_rate_mean * (1 - origion_rate_mean) - origion_rate_var)
#     print('origion_rate_mean : ', origion_rate_mean)
#     print('origion_rate_var : ', origion_rate_var)
#     print('alpha : ', alpha)
#     print('beta : ', beta)
    return alpha, beta


# 统计单维度的转化率特征
def get_single_dimension_rate_train_feature(train_df, fea_set):
    skf = StratifiedKFold(n_splits=5, random_state=2019, shuffle=True)
    for fea in fea_set:
        train_temp_df = pd.DataFrame()
        for index, (train_index, test_index) in enumerate(skf.split(train_df, train_df['label'])):
            temp_df = train_df[[fea, 'label']].iloc[train_index].copy()
            temp_pivot_table = pd.pivot_table(temp_df, index=fea, values='label', aggfunc={len, np.mean, np.sum})
            temp_pivot_table.reset_index(inplace=True)
            temp_pivot_table.rename(columns={'len':fea + '_count', 'mean':fea + '_rate', 'sum':fea + '_click_number'}, inplace=True)
            alpha, beta = getBayesSmoothParam(temp_pivot_table[fea + '_rate'])
            temp_pivot_table[fea + '_rate'] = (temp_pivot_table[fea + '_click_number'] + alpha) / (temp_pivot_table[fea + '_count'] + alpha + beta)
#             del temp_pivot_table[fea + '_click_number']
            fea_df = train_df.iloc[test_index].copy()
            fea_df = pd.merge(fea_df, temp_pivot_table, on=fea, how='left')
#             print(fea_df.head())
            train_temp_df = pd.concat([train_temp_df, fea_df])
#         temp_df = train_df[[fea, 'label']].copy()
#         temp_pivot_table = pd.pivot_table(temp_df, index=fea, values='label', aggfunc={len, np.mean, np.sum})
#         temp_pivot_table.reset_index(inplace=True)
#         temp_pivot_table.rename(columns={'len':fea + '_count', 'mean':fea + '_rate', 'sum':fea + '_click_number'}, inplace=True)
#         alpha, beta = getBayesSmoothParam(temp_pivot_table[fea + '_rate'])
#         temp_pivot_table[fea + '_rate'] = (temp_pivot_table[fea + '_click_number'] + alpha) / (temp_pivot_table[fea + '_count'] + alpha + beta)
# #             del temp_pivot_table[fea + '_click_number']
#         valid_df = pd.merge(valid_df, temp_pivot_table, on=fea, how='left')
        print(fea + ' : finish!!!')
        train_df = train_temp_df
        train_df.sort_index(by='index', ascending=True, inplace=True)
    return train_df



# 统计双维度交叉转化率
def get_jiaocha_dimension_rate_train_feature(train_df, fea_set):
    skf = StratifiedKFold(n_splits=5, random_state=2019, shuffle=True)
    for i in range(len(fea_set)):
        for j in range((i+1), len(fea_set)):
            fea1 = fea_set[i]
            fea2 = fea_set[j]
            train_temp_df = pd.DataFrame()
            for index, (train_index, test_index) in enumerate(skf.split(train_df, train_df['label'])):
                temp_df = train_df[[fea1, fea2, 'label']].iloc[train_index].copy()
                temp_pivot_table = pd.pivot_table(temp_df, index=[fea1, fea2], values='label', aggfunc={len, np.mean, np.sum})
                temp_pivot_table.reset_index(inplace=True)
                temp_pivot_table.rename(columns={'len':fea1 + '_' + fea2 + '_count', 'mean':fea1 + '_' + fea2 + '_rate', 'sum':fea1 + '_' + fea2 + '_click_number'}, inplace=True)
                alpha, beta = getBayesSmoothParam(temp_pivot_table[fea1 + '_' + fea2 + '_rate'])
                temp_pivot_table[fea1 + '_' + fea2 + '_rate'] = (temp_pivot_table[fea1 + '_' + fea2 + '_click_number'] + alpha) / (temp_pivot_table[fea1 + '_' + fea2 + '_count'] + alpha + beta)
#                 del temp_pivot_table[fea1 + '_' + fea2 + '_click_number']
                fea_df = train_df.iloc[test_index].copy()
                fea_df = pd.merge(fea_df, temp_pivot_table, on=[fea1, fea2], how='left')
                train_temp_df = pd.concat([train_temp_df, fea_df])
#             temp_df = train_df[[fea1, fea2, 'label']].copy()
#             temp_pivot_table = pd.pivot_table(temp_df, index=[fea1, fea2], values='label', aggfunc={len, np.mean, np.sum})
#             temp_pivot_table.reset_index(inplace=True)
#             temp_pivot_table.rename(columns={'len':fea1 + '_' + fea2 + '_count', 'mean':fea1 + '_' + fea2 + '_rate', 'sum':fea1 + '_' + fea2 + '_click_number'}, inplace=True)
#             alpha, beta = getBayesSmoothParam(temp_pivot_table[fea1 + '_' + fea2 + '_rate'])
#             temp_pivot_table[fea1 + '_' + fea2 + '_rate'] = (temp_pivot_table[fea1 + '_' + fea2 + '_click_number'] + alpha) / (temp_pivot_table[fea1 + '_' + fea2 + '_count'] + alpha + beta)
# #             del temp_pivot_table[fea1 + '_' + fea2 + '_click_number']
            print(fea1 + '_' + fea2 + ' : finish!!!')
#             valid_df = pd.merge(valid_df, temp_pivot_table, on=[fea1, fea2], how='left')
            train_df = train_temp_df
            train_df.sort_index(by='index', ascending=True, inplace=True)
    return train_df


# 统计单维度的转化率特征
def get_single_dimension_rate_test_feature(train_df, valid_df, fea_set):
    for fea in fea_set:
        temp_df = train_df[[fea, 'label']].copy()
        temp_pivot_table = pd.pivot_table(temp_df, index=fea, values='label', aggfunc={len, np.mean, np.sum})
        temp_pivot_table.reset_index(inplace=True)
        temp_pivot_table.rename(columns={'len':fea + '_count', 'mean':fea + '_rate', 'sum':fea + '_click_number'}, inplace=True)
        alpha, beta = getBayesSmoothParam(temp_pivot_table[fea + '_rate'])
        temp_pivot_table[fea + '_rate'] = (temp_pivot_table[fea + '_click_number'] + alpha) / (temp_pivot_table[fea + '_count'] + alpha + beta)
#             del temp_pivot_table[fea + '_click_number']
        valid_df = pd.merge(valid_df, temp_pivot_table, on=fea, how='left')
        print(fea + ' : finish!!!')
    return valid_df

# 统计双维度交叉转化率
def get_jiaocha_dimension_rate_test_feature(train_df, valid_df, fea_set):
    for i in range(len(fea_set)):
        for j in range((i+1), len(fea_set)):
            fea1 = fea_set[i]
            fea2 = fea_set[j]
            temp_df = train_df[[fea1, fea2, 'label']].copy()
            temp_pivot_table = pd.pivot_table(temp_df, index=[fea1, fea2], values='label', aggfunc={len, np.mean, np.sum})
            temp_pivot_table.reset_index(inplace=True)
            temp_pivot_table.rename(columns={'len':fea1 + '_' + fea2 + '_count', 'mean':fea1 + '_' + fea2 + '_rate', 'sum':fea1 + '_' + fea2 + '_click_number'}, inplace=True)
            alpha, beta = getBayesSmoothParam(temp_pivot_table[fea1 + '_' + fea2 + '_rate'])
            temp_pivot_table[fea1 + '_' + fea2 + '_rate'] = (temp_pivot_table[fea1 + '_' + fea2 + '_click_number'] + alpha) / (temp_pivot_table[fea1 + '_' + fea2 + '_count'] + alpha + beta)
#             del temp_pivot_table[fea1 + '_' + fea2 + '_click_number']
            print(fea1 + '_' + fea2 + ' : finish!!!')
            valid_df = pd.merge(valid_df, temp_pivot_table, on=[fea1, fea2], how='left')
    return valid_df




def getfeature(df):
    df.drop_duplicates(inplace=True)
    
    word = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ']
    df[1] = df[1].apply(lambda x: ''.join([i for i in x if i in word]))
    df[3] = df[3].apply(lambda x: ''.join([i for i in x if i in word]))

    
    ite = 1
    since = time.time()
    
    #query 长度
    df['query_len'] = df.apply(lambda x: len(x[1].split(' ')), axis=1)
    df['query_len'] = df['query_len'].astype(np.int32)
    
    time_elapsed = time.time() - since#######################################
    print('{} {} complete in {:.0f}m {:.0f}s'.format(ite, df.columns[-1], time_elapsed // 60, time_elapsed % 60))
    ite += 1
    
    #query_unique 长度
    def get_query_unique(x):
        a = len(set(x[1].split(' '))-set(x[3].split(' ')))
        return a
    
    df['query_unique'] = df.apply(get_query_unique, axis=1)
    df['query_unique'] = df['query_unique'].astype(np.int32)
    
    time_elapsed = time.time() - since#######################################
    print('{} {} complete in {:.0f}m {:.0f}s'.format(ite, df.columns[-1], time_elapsed // 60, time_elapsed % 60))
    ite += 1
    
    #title 长度
    df['title_len'] = df.apply(lambda x: len(x[3].split(' ')), axis=1)
    df['title_len'] = df['title_len'].astype(np.int32)
    
    time_elapsed = time.time() - since#######################################
    print('{} {} complete in {:.0f}m {:.0f}s'.format(ite, df.columns[-1], time_elapsed // 60, time_elapsed % 60))
    ite += 1
    
    #title_unique 长度
    def get_query_unique(x):
        a = len(set(x[3].split(' '))-set(x[1].split(' ')))
        return a
    df['title_unique'] = df.apply(get_query_unique, axis=1)
    df['title_unique'] = df['title_unique'].astype(np.int32)
    
    time_elapsed = time.time() - since#######################################
    print('{} {} complete in {:.0f}m {:.0f}s'.format(ite, df.columns[-1], time_elapsed // 60, time_elapsed % 60))
    ite += 1
    
    #长度差、unique长度差
    df['dif'] = (df['query_len'] - df['title_len']).abs()
    df['dif'] = df['dif'].astype(np.int32)
    df['dif_unique'] = (df['query_unique'] - df['title_unique']).abs()
    df['dif_unique'] = df['dif_unique'].astype(np.int32)
    
    time_elapsed = time.time() - since#######################################
    print('{} {} complete in {:.0f}m {:.0f}s'.format(ite, df.columns[-1], time_elapsed // 60, time_elapsed % 60))
    ite += 1
    
    #query_title_same_len query和title相同的长度
    def query_title_same_len(x):
        a = len(set(x[3].split(' ')) & set(x[1].split(' ')))
        return a
    df['query_title_same_len'] = df.apply(query_title_same_len, axis=1)
    df['query_title_same_len'] = df['query_title_same_len'].astype(np.int32)
    
    time_elapsed = time.time() - since#######################################
    print('{} {} complete in {:.0f}m {:.0f}s'.format(ite, df.columns[-1], time_elapsed // 60, time_elapsed % 60))
    ite += 1

    #query和title相同的长度在query的比率
    def samelen_query_rate(x):
        a = len(set(x[1].split(' ')) & set(x[3].split(' ')))
        return a/len(x[1].split(' '))
    df['samelen_query_rate'] = df.apply(samelen_query_rate, axis=1)
    df['samelen_query_rate'] = df['samelen_query_rate'].astype(np.float32)
    
    time_elapsed = time.time() - since#######################################
    print('{} {} complete in {:.0f}m {:.0f}s'.format(ite, df.columns[-1], time_elapsed // 60, time_elapsed % 60))
    ite += 1
    
    #query和title相同的长度在title的比率
    def samelen_title_rate(x):
        a = len(set(x[1].split(' ')) & set(x[3].split(' ')))
        return a/len(x[3].split(' '))
    df['samelen_title_rate'] = df.apply(samelen_title_rate, axis=1)
    df['samelen_title_rate'] = df['samelen_title_rate'].astype(np.float32)
    
    time_elapsed = time.time() - since#######################################
    print('{} {} complete in {:.0f}m {:.0f}s'.format(ite, df.columns[-1], time_elapsed // 60, time_elapsed % 60))
    ite += 1
    
    
    #query和title总长度
    def q_t_all_len(x):
        a = len(x[1].split(' ')) + len(x[3].split(' '))
        return a
    df['q_t_all_len'] = df.apply(q_t_all_len, axis=1)
    df['q_t_all_len'] = df['q_t_all_len'].astype(np.int32)
    
    time_elapsed = time.time() - since#######################################
    print('{} {} complete in {:.0f}m {:.0f}s'.format(ite, df.columns[-1], time_elapsed // 60, time_elapsed % 60))
    ite += 1
    
    #query和title的unique长度
    def q_t_unique_len(x):
        a = len(set(x[1].split(' ') + x[3].split(' ')))
        return a
    df['q_t_unique_len'] = df.apply(q_t_unique_len, axis=1)
    df['q_t_unique_len'] = df['q_t_unique_len'].astype(np.int32)
    
    time_elapsed = time.time() - since#######################################
    print('{} {} complete in {:.0f}m {:.0f}s'.format(ite, df.columns[-1], time_elapsed // 60, time_elapsed % 60))
    ite += 1

    
    #query和title总词数
    def q_t_all_word_len(x):
        a = len(set(x[1].split(' ')) | set(x[3].split(' ')))
        return a
    df['q_t_all_word_len'] = df.apply(q_t_all_word_len, axis=1)
    df['q_t_all_word_len'] = df['q_t_all_word_len'].astype(np.int32)
    
    time_elapsed = time.time() - since#######################################
    print('{} {} complete in {:.0f}m {:.0f}s'.format(ite, df.columns[-1], time_elapsed // 60, time_elapsed % 60))
    ite += 1
    
    #query和title不同词数
    df['q_t_diff_len'] = df.apply(lambda x: len(set(x[1].split(' '))^set(x[3].split(' '))), axis=1)
    df['q_t_diff_len'] = df['q_t_diff_len'].astype(np.int32)
    
    time_elapsed = time.time() - since#######################################
    print('{} {} complete in {:.0f}m {:.0f}s'.format(ite, df.columns[-1], time_elapsed // 60, time_elapsed % 60))
    ite += 1
    
    #query和title不同词数在query的比率
    def q_t_diff_q_rate(x):
        a = len(set(x[1].split(' '))^set(x[3].split(' ')))
        return np.float32(a/len(x[1].split(' ')))
    df['q_t_diff_q_rate'] = df.apply(q_t_diff_q_rate, axis=1)
    df['q_t_diff_q_rate'] = df['q_t_diff_q_rate'].astype(np.float32)
    
    time_elapsed = time.time() - since#######################################
    print('{} {} complete in {:.0f}m {:.0f}s'.format(ite, df.columns[-1], time_elapsed // 60, time_elapsed % 60))
    ite += 1
    
    #query和title不同词数在title的比率
    def q_t_diff_t_rate(x):
        a = len(set(x[1].split(' '))^set(x[3].split(' ')))
        return np.float32(a/len(x[3].split(' ')))
    df['q_t_diff_t_rate'] = df.apply(q_t_diff_t_rate, axis=1)
    df['q_t_diff_t_rate'] = df['q_t_diff_t_rate'].astype(np.float32)
    
    time_elapsed = time.time() - since#######################################
    print('{} {} complete in {:.0f}m {:.0f}s'.format(ite, df.columns[-1], time_elapsed // 60, time_elapsed % 60))
    ite += 1

    
    #query在title中的不同词数
    def query_diff_title(x):
        a = len(set(x[1].split(' ')) - (set(x[1].split(' ')) & set(x[3].split(' '))))
        return np.int32(a)
    df['query_diff_title'] = df.apply(query_diff_title, axis=1)
    df['query_diff_title'] = df['query_diff_title'].astype(np.int32)
    
    time_elapsed = time.time() - since#######################################
    print('{} {} complete in {:.0f}m {:.0f}s'.format(ite, df.columns[-1], time_elapsed // 60, time_elapsed % 60))
    ite += 1
    
    #query在title中的不同词数在query的比率
    def query_diff_title_rate(x):
        a = len(set(x[1].split(' ')) - (set(x[1].split(' ')) & set(x[3].split(' '))))
        return np.float32(a/len(x[1].split(' ')))
    df['query_diff_title_rate'] = df.apply(query_diff_title_rate, axis=1)
    df['query_diff_title_rate'] = df['query_diff_title_rate'].astype(np.float32)
    
    time_elapsed = time.time() - since#######################################
    print('{} {} complete in {:.0f}m {:.0f}s'.format(ite, df.columns[-1], time_elapsed // 60, time_elapsed % 60))
    ite += 1
    
    #query在title中的不同词数在title的比率
    def query_diff_title_rate(x):
        a = len(set(x[1].split(' ')) - (set(x[1].split(' ')) & set(x[3].split(' '))))
        return np.float32(a/len(x[3].split(' ')))
    df['query_diff_title_rate'] = df.apply(query_diff_title_rate, axis=1)
    df['query_diff_title_rate'] = df['query_diff_title_rate'].astype(np.float32)
    
    time_elapsed = time.time() - since#######################################
    print('{} {} complete in {:.0f}m {:.0f}s'.format(ite, df.columns[-1], time_elapsed // 60, time_elapsed % 60))
    ite += 1
    
    #title在qurey中的不同词数
    def title_diff_query(x):
        a = len(set(x[3].split(' ')) - (set(x[1].split(' ')) & set(x[3].split(' '))))
        return np.int32(a)
    df['title_diff_query'] = df.apply(title_diff_query, axis=1)
    df['title_diff_query'] = df['title_diff_query'].astype(np.int32)
    
    time_elapsed = time.time() - since#######################################
    print('{} {} complete in {:.0f}m {:.0f}s'.format(ite, df.columns[-1], time_elapsed // 60, time_elapsed % 60))
    ite += 1
    
    #title在qurey中的不同词数在query的比率
    def title_diff_query_rate(x):
        a = len(set(x[3].split(' ')) - (set(x[1].split(' ')) & set(x[3].split(' '))))
        return np.float32(a/len(x[1].split(' ')))
    df['title_diff_query_rate'] = df.apply(title_diff_query_rate, axis=1)
    df['title_diff_query_rate'] = df['title_diff_query_rate'].astype(np.float32)
    
    time_elapsed = time.time() - since#######################################
    print('{} {} complete in {:.0f}m {:.0f}s'.format(ite, df.columns[-1], time_elapsed // 60, time_elapsed % 60))
    ite += 1
    
    #title在qurey中的不同词数在title的比率
    def title_diff_query_rate(x):
        a = len(set(x[3].split(' ')) - (set(x[1].split(' ')) & set(x[3].split(' '))))
        return np.float32(a/len(x[3].split(' ')))
    df['title_diff_query_rate'] = df.apply(title_diff_query_rate, axis=1)
    df['title_diff_query_rate'] = df['title_diff_query_rate'].astype(np.float32)
    
    time_elapsed = time.time() - since#######################################
    print('{} {} complete in {:.0f}m {:.0f}s'.format(ite, df.columns[-1], time_elapsed // 60, time_elapsed % 60))
    ite += 1
    
    #Dice系数
    def dice(x):
        a1 = len(set(x[1].split(' ')) & set(x[3].split(' ')))
        a2 = len(set(x[1].split(' '))) + len(set(x[3].split(' ')))
        return np.float32(2*a1/a2)
    df['dice'] = df.apply(dice, axis=1)
    df['dice'] = df['dice'].astype(np.float32)
    
    time_elapsed = time.time() - since#######################################
    print('{} {} complete in {:.0f}m {:.0f}s'.format(ite, df.columns[-1], time_elapsed // 60, time_elapsed % 60))
    ite += 1
    
    #jaccord系数
    def jaccord(x):
        a1 = len(set(x[1].split(' ')) & set(x[3].split(' ')))
        a2 = len(set(x[1].split(' ')) | set(x[3].split(' ')))
        return np.float32(a1/a2)
    df['jaccord'] = df.apply(jaccord, axis=1)
    df['jaccord'] = df['jaccord'].astype(np.float32)
    
    time_elapsed = time.time() - since#######################################
    print('{} {} complete in {:.0f}m {:.0f}s'.format(ite, df.columns[-1], time_elapsed // 60, time_elapsed % 60))
    ite += 1
#    
#    #同一个query出现次数
#    df['query_count'] = df.groupby(1)[1].transform('count')
#    df['query_count'] = df['query_count'].astype(np.int32)
#    
#    time_elapsed = time.time() - since#######################################
#    print('{} {} complete in {:.0f}m {:.0f}s'.format(ite, df.columns[-1], time_elapsed // 60, time_elapsed % 60))
#    ite += 1
#    
#    #同一个title出现的次数
#    df['title_count'] = df.groupby(3)[3].transform('count')
#    df['title_count'] = df['title_count'].astype(np.int32)
#    
#    time_elapsed = time.time() - since#######################################
#    print('{} {} complete in {:.0f}m {:.0f}s'.format(ite, df.columns[-1], time_elapsed // 60, time_elapsed % 60))
#    ite += 1
#    
#    
##    #query和labe的交叉点击
##    df['query_label_click'] = df.groupby([1, 4])[1].transform('count')
##    df['query_label_click'] = df['query_label_click'].astype(np.int32)
##    
##    #title和labe的交叉点击
##    df['title_label_click'] = df.groupby([3, 4])[3].transform('count')
##    df['title_label_click'] = df['title_label_click'].astype(np.int32)
#    
#    
    #lcseque_lens
    df['lcseque_lens'] = df.apply(lambda x: lcseque_lens(x[1], x[3]), axis=1)
    df['lcseque_lens'] = df['lcseque_lens'].astype(np.int32)
    
    time_elapsed = time.time() - since#######################################
    print('{} {} complete in {:.0f}m {:.0f}s'.format(ite, df.columns[-1], time_elapsed // 60, time_elapsed % 60))
    ite += 1
    
    #lcsubstr_lens
    df['lcsubstr_lens'] = df.apply(lambda x: lcsubstr_lens(x[1], x[3]), axis=1)
    df['lcsubstr_lens'] = df['lcsubstr_lens'].astype(np.int32)
    
    time_elapsed = time.time() - since#######################################
    print('{} {} complete in {:.0f}m {:.0f}s'.format(ite, df.columns[-1], time_elapsed // 60, time_elapsed % 60))
    ite += 1

    #ratio
    df['ratio'] = df.apply(lambda x: Levenshtein.ratio(x[1], x[3]), axis=1)
    df['ratio'] = df['ratio'].astype(np.float32)
    
    time_elapsed = time.time() - since#######################################
    print('{} {} complete in {:.0f}m {:.0f}s'.format(ite, df.columns[-1], time_elapsed // 60, time_elapsed % 60))
    ite += 1
    
    

    #Jaro距离
    df['jaro'] = df.apply(lambda x: Levenshtein.jaro(x[1], x[3]), axis=1)
    df['jaro'] = df['jaro'].astype(np.float32)
    
    time_elapsed = time.time() - since#######################################
    print('{} {} complete in {:.0f}m {:.0f}s'.format(ite, df.columns[-1], time_elapsed // 60, time_elapsed % 60))
    ite += 1
    
    #jaro_winkler
    df['jaro_winkler'] = df.apply(lambda x: Levenshtein.jaro_winkler(x[1], x[3]), axis=1)
    df['jaro_winkler'] = df['jaro_winkler'].astype(np.float32)
    
    time_elapsed = time.time() - since#######################################
    print('{} {} complete in {:.0f}m {:.0f}s'.format(ite, df.columns[-1], time_elapsed // 60, time_elapsed % 60))
    ite += 1


    #余弦相似度
    #使用tfidf向量求相似度
    ct = TfidfVectorizer()
    def cosine_similarity(x):
        ct.fit([x[1] + ' ' + x[3]])
        vect1 = ct.transform([x[1]]).toarray()[0]
        vect2 = ct.transform([x[3]]).toarray()[0]
        sum = 0
        sq1 = 0
        sq2 = 0
        for i in range(len(vect1)):
            sum += vect1[i] * vect2[i]
            sq1 += pow(vect1[i], 2)
            sq2 += pow(vect2[i], 2)
        try:
            result = round(float(sum) / (math.sqrt(sq1) * math.sqrt(sq2)), 2)
        except ZeroDivisionError:
            result = 0.0
        return result 
    df['cosine_similarity_tfidf'] = df.apply(lambda x: cosine_similarity(x), axis=1)
    df['cosine_similarity_tfidf'] = df['cosine_similarity_tfidf'].astype(np.float32)
    #使用词频向量求相似度
    df['cosine_similarity'] = df.apply(lambda x: compute_cosine(x[1], x[3]), axis=1)
    df['cosine_similarity'] = df['cosine_similarity'].astype(np.float32)
    
    time_elapsed = time.time() - since#######################################
    print('{} {} complete in {:.0f}m {:.0f}s'.format(ite, df.columns[-1], time_elapsed // 60, time_elapsed % 60))
    ite += 1
    
    
    #编辑距离
    df['levenshtein'] = df.apply(lambda x: Levenshtein.distance(x[1], x[3]), axis=1)
    df['levenshtein'] = df['levenshtein'].astype(np.int32)
    
    time_elapsed = time.time() - since#######################################
    print('{} {} complete in {:.0f}m {:.0f}s'.format(ite, df.columns[-1], time_elapsed // 60, time_elapsed % 60))
    ite += 1
    
    
    #皮尔逊相关系数Pehrson
    df['Pehrson'] = df.apply(lambda x:Pehrson(x[1], x[3]), axis=1)
    df['Pehrson'] = df['Pehrson'].fillna(df['Pehrson'].mean())
    df['Pehrson'] = df['Pehrson'].astype(np.float32)
    
    time_elapsed = time.time() - since#######################################
    print('{}complete in {:.0f}m {:.0f}s'.format(ite, time_elapsed // 60, time_elapsed % 60))
    ite += 1
    
    
    #计算两个字符串list的相似度
    df['list_dis'] = df.apply(lambda x: Levenshtein.seqratio(x[1].split(' '), x[3].split(' ')), axis=1)
    df['list_dis'] = df['list_dis'].astype(np.float32)
    
    time_elapsed = time.time() - since#######################################
    print('{} {} complete in {:.0f}m {:.0f}s'.format(ite, df.columns[-1], time_elapsed // 60, time_elapsed % 60))
    ite += 1
    
    
    #Word2vec计算两个list的相似度
    model_word = Word2Vec.load('./word2vec_groupby.model')
    def word2vec_similar(x):
        a = x[1].split(' ')
        a = [i for i in a if i in model_word]
        b = x[3].split(' ')
        b = [i for i in b if i in model_word]  
        if len(a) == 0 or len(b) == 0:
              return 0
        return model_word.n_similarity(a, b)
    df['word_list_similarity'] = df.apply(lambda x: word2vec_similar(x), axis=1)
    
    time_elapsed = time.time() - since#######################################
    print('{} {} complete in {:.0f}m {:.0f}s'.format(ite, df.columns[-1], time_elapsed // 60, time_elapsed % 60))
    ite += 1
    
    
    #使用word2vec求句向量的均值和最大值
    def getvec(x):
        a = 0
        n = 0
        for i in x:
            try:
                a += model_word[i]
                n += 1
            except:
                pass
        a = np.array(a)/n
        a = a.astype(np.float32)
        return a
    #query
    df['word2vec_query_mean'] = df.apply(lambda x: getvec(x[1].split(' ')).mean(), axis=1)
    df['word2vec_query_max'] = df.apply(lambda x: getvec(x[1].split(' ')).max(), axis=1)
    #title
    df['word2vec_title_mean'] = df.apply(lambda x: getvec(x[3].split(' ')).mean(), axis=1)
    df['word2vec_title_max'] = df.apply(lambda x: getvec(x[3].split(' ')).max(), axis=1)
    #query+title
    df['word2vec_query_title_mean'] = df.apply(lambda x: getvec(x[1].split(' ') + x[3].split(' ')).mean(), axis=1)
    df['word2vec_query_title_max'] = df.apply(lambda x: getvec(x[1].split(' ') + x[3].split(' ')).max(), axis=1)

    time_elapsed = time.time() - since#######################################
    print('{} {} complete in {:.0f}m {:.0f}s'.format(ite, df.columns[-1], time_elapsed // 60, time_elapsed % 60))
    ite += 1
    
    
#    #使用doc2vec求句向量的均值和最大值
#    model_doc = Doc2Vec.load('../1.model')
#    #query
#    df['docvec_query_mean'] = df.apply(lambda x: model_doc.infer_vector(x[1].split(' ')).mean(), axis=1)
#    df['docvec_query_max'] = df.apply(lambda x: model_doc.infer_vector(x[1].split(' ')).max(), axis=1)
#    #title
#    df['docvec_title_mean'] = df.apply(lambda x: model_doc.infer_vector(x[3].split(' ')).mean(), axis=1)
#    df['docvec_title_max'] = df.apply(lambda x: model_doc.infer_vector(x[3].split(' ')).max(), axis=1)
#    #query+title
#    df['docvec_query_title_mean'] = df.apply(lambda x: model_doc.infer_vector(x[1].split(' ') + x[3].split(' ')).mean(), axis=1)
#    df['docvec_query_title_max'] = df.apply(lambda x: model_doc.infer_vector(x[1].split(' ') + x[3].split(' ')).max(), axis=1)


#    #Doc2vec计算两个list的相似度
#    def doc2vec_similar(x):
#        a = x[1].split(' ')
#        a = [i for i in a if i in model_doc]
#        b = x[3].split(' ')
#        b = [i for i in b if i in model_doc]  
#        return model_doc.n_similarity(a, b)
#    df['doc_list_similarity'] = df.apply(lambda x: doc2vec_similar(x), axis=1)
    

#    #doc2vec和word2vec分别得到相似度乘积
#    df['doc_word_list_similarity_product'] = df['word_list_similarity'] * df['doc_list_similarity']
    

#    #jaro_winkler和jaro的均值、乘积、差值
#    df['jaroWinkler_jaro_mean'] = (df['jaro'] + df['jaro_winkler'])/2
#    df['jaroWinkler_jaro_product'] = df['jaro'] * df['jaro_winkler']
#    df['jaroWinkler_jaro_subtraction'] = (df['jaro'] - df['jaro_winkler']).abs()
#    
#    time_elapsed = time.time() - since#######################################
#    print('{} {} complete in {:.0f}m {:.0f}s'.format(ite, df.columns[-1], time_elapsed // 60, time_elapsed % 60))
#    ite += 1
#      
#    
#    #Pehrson和cosine_similarity的均值、乘积
#    df['cosine_similarity_Pehrson_mean'] = (df['Pehrson'] + df['cosine_similarity'])/2
#    df['cosine_similarity_Pehrson_product'] = df['Pehrson'] * df['cosine_similarity']
#    df['cosine_similarity_Pehrson_subtraction'] = (df['Pehrson'] - df['cosine_similarity']).abs()
#    
#    time_elapsed = time.time() - since#######################################
#    print('{} {} complete in {:.0f}m {:.0f}s'.format(ite, df.columns[-1], time_elapsed // 60, time_elapsed % 60))
#    ite += 1
    
    
#    #TF系数
#    #max
    def tf(x):
        x = x.split(' ')
        x = pd.Series(x).value_counts().values
        x = x/x.sum()
        return x
#    df['query_tf_max'] = df.apply(lambda x: tf(x[1]).max(), axis=1)
    df['title_tf_max'] = df.apply(lambda x: tf(x[3]).max(), axis=1)
    df['query_title_tf_max'] = df.apply(lambda x: tf(x[1] + ' ' + x[3]).max(), axis=1)
    #mean
#    df['query_tf_mean'] = df.apply(lambda x: tf(x[1]).mean(), axis=1)
#    df['title_tf_mean'] = df.apply(lambda x: tf(x[3]).mean(), axis=1)
#    df['query_title_tf_mean'] = df.apply(lambda x: tf(x[1] + ' ' + x[3]).mean(), axis=1)
    
    time_elapsed = time.time() - since#######################################
    print('{} {} complete in {:.0f}m {:.0f}s'.format(ite, df.columns[-1], time_elapsed // 60, time_elapsed % 60))
    ite += 1
    
    #idf系数
    #max
    def idf(x):
        x = x.split(' ')
        l = len(x)
        x = pd.Series(x).value_counts().values
        x = np.log10(l/x)
        return x
#    df['query_idf_max'] = df.apply(lambda x: idf(x[1]).max(), axis=1)
#    df['title_idf_max'] = df.apply(lambda x: idf(x[3]).max(), axis=1)
#    df['query_title_idf_max'] = df.apply(lambda x: idf(x[1] + ' ' + x[3]).max(), axis=1)
    #mean
#    df['query_idf_mean'] = df.apply(lambda x: idf(x[1]).mean(), axis=1)
    df['title_idf_mean'] = df.apply(lambda x: idf(x[3]).mean(), axis=1)
    df['query_title_idf_mean'] = df.apply(lambda x: idf(x[1] + ' ' + x[3]).mean(), axis=1)
         
    
    time_elapsed = time.time() - since#######################################
    print('{} {} complete in {:.0f}m {:.0f}s'.format(ite, df.columns[-1], time_elapsed // 60, time_elapsed % 60))
    ite += 1
      
    #TF*IDF#####################################(40)
    def tfidf(x):
        t = tf(x)
        i = idf(x)
        ti = t*i
        return ti
    #max
#    df['query_tfidf_max'] = df.apply(lambda x: tfidf(x[1]).max(), axis=1)
#    df['title_tfidf_max'] = df.apply(lambda x: tfidf(x[3]).max(), axis=1)
#    df['query_title_tfidf_max'] = df.apply(lambda x: tfidf(x[1] + ' ' + x[3]).max(), axis=1)
    #mean
#    df['query_tfidf_mean'] = df.apply(lambda x: tfidf(x[1]).mean(), axis=1)
#    df['title_tfidf_mean'] = df.apply(lambda x: tfidf(x[3]).mean(), axis=1)
#    df['query_title_tfidf_mean'] = df.apply(lambda x: tfidf(x[1] + ' ' + x[3]).mean(), axis=1)
    
    time_elapsed = time.time() - since#######################################
    print('{} {} complete in {:.0f}m {:.0f}s'.format(ite, df.columns[-1], time_elapsed // 60, time_elapsed % 60))
    ite += 1
           

    
    #每句的tfidf最大值，平均值
    t = TfidfVectorizer(ngram_range=(1, 2), analyzer='char',binary=True)
    title = t.fit_transform(df[3]).toarray()
    df['title_tfidfvec_mean'] = title.mean(axis=1)
    df['title_tfidfvec_max'] = title.max(axis=1)
    del title
    q_t = t.fit_transform(df[1]+' '+df[3]).toarray()
    df['query_title_tfidfvec_mean'] = q_t.mean(axis=1)
    df['query_title_tfidfvec_max'] = q_t.max(axis=1)
    del q_t
    
    time_elapsed = time.time() - since#######################################
    print('{} {} complete in {:.0f}m {:.0f}s'.format(ite, df.columns[-1], time_elapsed // 60, time_elapsed % 60))
    ite += 1
    
    
###############################################################################    
#    skf = StratifiedKFold(n_splits=5, random_state=2019, shuffle=True)
#    #单维度转化率特征
#    single_fea_set = ['query', 'title']
#    df = get_single_dimension_rate_train_feature(df, single_fea_set)
#    
#    #交叉转化率特征
#    jiaoch_fea_set = ['query', 'title']
#    df = get_jiaocha_dimension_rate_train_feature(df, jiaoch_fea_set)
###############################################################################
    
    

#    try:
#        df.drop([0, 1, 2, 3, 4], axis=1, inplace=True)
#    except:
#        df.drop([0, 1, 2, 3], axis=1, inplace=True)
        
    return df




def getSimilarityFeature(df):
    
#    #Dice系数
#    def dice(x):
#        a1 = len(set(x[1].split(' ')) & set(x[3].split(' ')))
#        a2 = len(set(x[1].split(' '))) + len(set(x[3].split(' ')))
#        return np.float32(2*a1/a2)
#    df['dice'] = df.apply(dice, axis=1)
#    df['dice'] = df['dice'].astype(np.float32)
#    
#    #jaccord系数
#    def jaccord(x):
#        a1 = len(set(x[1].split(' ')) & set(x[3].split(' ')))
#        a2 = len(set(x[1].split(' ')) | set(x[3].split(' ')))
#        return np.float32(a1/a2)
#    df['jaccord'] = df.apply(jaccord, axis=1)
#    df['jaccord'] = df['jaccord'].astype(np.float32)
#    
#    
#    #lcseque_lens
#    df['lcseque_lens'] = df.apply(lambda x: lcseque_lens(x[1], x[3]), axis=1)
#    df['lcseque_lens'] = df['lcseque_lens'].astype(np.int32)
#    
#    
#    #lcsubstr_lens
#    df['lcsubstr_lens'] = df.apply(lambda x: lcsubstr_lens(x[1], x[3]), axis=1)
#    df['lcsubstr_lens'] = df['lcsubstr_lens'].astype(np.int32)
#    
#    #ratio
#    df['ratio'] = df.apply(lambda x: Levenshtein.ratio(x[1], x[3]), axis=1)
#    df['ratio'] = df['ratio'].astype(np.float32)
#    
#    #Jaro距离
#    df['jaro'] = df.apply(lambda x: Levenshtein.jaro(x[1], x[3]), axis=1)
#    df['jaro'] = df['jaro'].astype(np.float32)
#    
#    #jaro_winkler
#    df['jaro_winkler'] = df.apply(lambda x: Levenshtein.jaro_winkler(x[1], x[3]), axis=1)
#    df['jaro_winkler'] = df['jaro_winkler'].astype(np.float32)
#    
#    
#    #余弦相似度
#    #使用tfidf向量求相似度
#    ct = TfidfVectorizer()
#    def cosine_similarity_tfidf(x):
#        ct.fit([x['1'] + ' ' + x['3']])
#        vect1 = ct.transform([x['1']]).toarray()[0]
#        vect2 = ct.transform([x['3']]).toarray()[0]
#        sum = 0
#        sq1 = 0
#        sq2 = 0
#        for i in range(len(vect1)):
#            sum += vect1[i] * vect2[i]
#            sq1 += pow(vect1[i], 2)
#            sq2 += pow(vect2[i], 2)
#        try:
#            result = round(float(sum) / (math.sqrt(sq1) * math.sqrt(sq2)), 2)
#        except ZeroDivisionError:
#            result = 0.0
#        return result 
#    df['cosine_similarity_tfidf'] = df.apply(lambda x: cosine_similarity_tfidf(x), axis=1)
#    df['cosine_similarity_tfidf'] = df['cosine_similarity_tfidf'].astype(np.float32)
#    #使用词频向量求相似度
#    df['cosine_similarity'] = df.apply(lambda x: compute_cosine(x[1], x[3]), axis=1)
#    df['cosine_similarity'] = df['cosine_similarity'].astype(np.float32)


    #w2v向量求相似度
    model_query = Word2Vec.load('word2vec.model')
    model_title = Word2Vec.load('word2vec.model')
    
    def getVec(x, model):
        x = x.split(' ')
        a = 0
        for i in x:
            try:
                a += model.wv[i]
            except:
                pass
        a = np.array(a)
        a = (a - a.min())/(a.max() - a.min())
        return a+1
    
    def cosine_similarity_w2v(x):
        vect1 = getVec(x[1], model_query)
        vect2 = getVec(x[3], model_title)
        vect1mod = np.sqrt(vect1.dot(vect1))
        vect2mod = np.sqrt(vect2.dot(vect2))
        if vect1mod!=0 and vect2mod!=0:
            simla = (vect1.dot(vect2))/(vect1mod*vect2mod)
        else:
            simla = 0
        return simla

    df['cosine_similarity_w2v'] = df.apply(lambda x: cosine_similarity_w2v(x), axis=1)
    df['cosine_similarity_w2v'] = df['cosine_similarity_w2v'].astype(np.float32)
    
#    #计算两个字符串list的相似度
#    df['list_dis'] = df.apply(lambda x: Levenshtein.seqratio(x[1].split(' '), x[3].split(' ')), axis=1)
#    df['list_dis'] = df['list_dis'].astype(np.float32)
    
    
    #曼哈顿距离
    def manhattan_distance(p_vec, q_vec):
        """
        This method implements the manhattan distance metric
        :param p_vec: vector one
        :param q_vec: vector two
        :return: the manhattan distance between vector one and two
        """
        return np.sum(np.fabs(p_vec - q_vec))
    df['Manhattan_distance'] = df.apply(lambda x: manhattan_distance(getVec(x[1], model_query), getVec(x[3], model_title)), axis=1)
    df['Manhattan_distance'] = df['Manhattan_distance'].astype(np.float32)
    
    #欧式距离
    def Euclidean_distance(v1, v2):
        v1 = np.mat(v1)
        v2 = np.mat(v2)
        return float(np.array(np.sqrt((v1-v2)*((v1-v2).T))).squeeze())
    df['Euclidean_distance'] = df.apply(lambda x: Euclidean_distance(getVec(x[1], model_query), getVec(x[3], model_title)), axis=1)
    df['Euclidean_distance'] = df['Euclidean_distance'].astype(np.float32)

    #切比雪夫距离
    def ChebyshevDistance(v1, v2):
        v1 = np.mat(v1)
        v2 = np.mat(v2)
        return np.abs(v1-v2).max()
    df['ChebyshevDistance'] = df.apply(lambda x: ChebyshevDistance(getVec(x[1], model_query), getVec(x[3], model_title)), axis=1)
    df['ChebyshevDistance'] = df['ChebyshevDistance'].astype(np.float32)
    
    
    #闵可夫斯基距离测度家族
    #Euclidean
    def Euclidean(x):
        x_query = getVec(x[1], model_query)
        x_title = getVec(x[3], model_title)
        return np.sqrt(np.sum(np.power(x_query - x_title, 2)))
    df['Euclidean'] = df.apply(lambda x: Euclidean(x), axis=1)
    #City block 
    def CityBlock(x):
        x_query = getVec(x[1], model_query)
        x_title = getVec(x[3], model_title)
        return np.sum(np.abs(x_query - x_title))
    df['CityBlock'] = df.apply(lambda x: CityBlock(x), axis=1)
    #Chebyshev
    def Chebyshev(x):
        x_query = getVec(x[1], model_query)
        x_title = getVec(x[3], model_title)
        return np.max(np.abs(x_query - x_title))
    df['Chebyshev'] = df.apply(lambda x: Chebyshev(x), axis=1)
    
    #L1L1 范数测度家族
    #Sorensen
    def Sorensen(x):
        x_query = getVec(x[1], model_query)
        x_title = getVec(x[3], model_title)
        return np.sum(x_query - x_title)/np.sum(x_query + x_title)
    df['Sorensen'] = df.apply(lambda x: Sorensen(x), axis=1)
    #Gower
    def Gower(x):
        x_query = getVec(x[1], model_query)
        x_title = getVec(x[3], model_title)
        return np.sum(np.abs(x_query - x_title))/len(x_query)
    df['Gower'] = df.apply(lambda x: Gower(x), axis=1)
    #Soergel
    def Soergel(x):
        x_query = getVec(x[1], model_query)
        x_title = getVec(x[3], model_title)
        a = np.sum(np.array([np.max(i-x_title) for i in x_query]))
        return np.sum(np.abs(x_query - x_title))/a
    df['Soergel'] = df.apply(lambda x: Soergel(x), axis=1)
    
    
    #集合测度家族
    #Intersection
    def Intersection(x):
        x_query = getVec(x[1], model_query)
        x_title = getVec(x[3], model_title)
        return np.sum(np.abs(x_query-x_title))/2
    df['Intersection'] = df.apply(lambda x: Intersection(x), axis=1)
    #Wave Hedges
    def WaveHedges(x):
        x_query = getVec(x[1], model_query)
        x_title = getVec(x[3], model_title)
        return np.sum(np.array([np.abs(i-j)/np.max([i,j]) for i,j in zip(x_query, x_title)]))
    df['WaveHedges'] = df.apply(lambda x: WaveHedges(x), axis=1)
    #Czekanowski
    def Czekanowski(x):
        x_query = getVec(x[1], model_query)
        x_title = getVec(x[3], model_title)
        return np.sum(np.abs(x_query - x_title))/np.sum(x_query + x_title)
    df['Czekanowski'] = df.apply(lambda x: Czekanowski(x), axis=1)
    #Motyka
    def Motyka(x):
        x_query = getVec(x[1], model_query)
        x_title = getVec(x[3], model_title)
        return np.sum(np.max([x_query, x_title],axis=0))/np.sum(x_query + x_title)
    df['Motyka'] = df.apply(lambda x: Motyka(x), axis=1)
    #Kulczynski
    def Kulczynski(x):
        x_query = getVec(x[1], model_query)
        x_title = getVec(x[3], model_title)
        return np.sum(np.min([x_query, x_title],axis=0))/np.sum(x_query + x_title)
    df['Kulczynski'] = df.apply(lambda x: Kulczynski(x), axis=1)
    #Ruzicka
    def Ruzicka(x):
        x_query = getVec(x[1], model_query)
        x_title = getVec(x[3], model_title)
        return np.sum(np.min([x_query, x_title],axis=0))/np.sum(np.max([x_query, x_title], axis=0))
    df['Ruzicka'] = df.apply(Ruzicka, axis=1)
    
    
    #香农信息熵家族
    #Kullback-Leibler
    def KullbackLeibler(x):
        x_query = getVec(x[1], model_query)
        x_title = getVec(x[3], model_title)
        return np.sum(x_query*np.log(x_query/(x_title)))
    df['KullbackLeibler'] = df.apply(KullbackLeibler, axis=1)
    #Jeffreys
    def Jeffreys(x):
        x_query = getVec(x[1], model_query)
        x_title = getVec(x[3], model_title)
        return np.sum((x_query-x_title)*np.log(x_query/x_title))
    df['Jeffreys'] = df.apply(Jeffreys, axis=1)
    #KDivergence
    def KDivergence(x):
        x_query = getVec(x[1], model_query)
        x_title = getVec(x[3], model_title)
        return np.sum(x_query*np.log(2*x_query/(x_query+x_title)))
    df['KDivergence'] = df.apply(KDivergence, axis=1)
    #Topsoe
    def Topsoe(x):
        x_query = getVec(x[1], model_query)
        x_title = getVec(x[3], model_title)
        return np.sum(x_query*np.log(2*x_query/(x_query+x_title)) + x_title*np.log(2*x_title/(x_query+x_title)))
    df['Topsoe'] = df.apply(Topsoe, axis=1)
    #JensenShannon
    def JensenShannon(x):
        x_query = getVec(x[1], model_query)
        x_title = getVec(x[3], model_title)
        return (np.sum(x_query*np.log(2*x_query/(x_query+x_title))) + np.sum(x_title*np.log(2*x_title/(x_query+x_title))))/2
    df['JensenShannon'] = df.apply(JensenShannon, axis=1)
    #JensenDifference
    def JensenDifference(x):
        x_query = getVec(x[1], model_query)
        x_title = getVec(x[3], model_title)
        return np.sum((x_query*np.log(x_query)+x_title*np.log(x_title)) - ((x_query+x_title)/2)*np.log((x_query+x_title)/2))
    df['JensenDifference'] = df.apply(JensenDifference, axis=1)
    
    
    
    
    
    return df

def get(df):
#    
#        #余弦相似度
#    #使用tfidf向量求相似度
#    ct = TfidfVectorizer()
#    def cosine_similarity_tfidf(x):
#        ct.fit([x[1] + ' ' + x[3]])
#        vect1 = ct.transform([x[1]]).toarray()[0]
#        vect2 = ct.transform([x[3]]).toarray()[0]
#        sum = 0
#        sq1 = 0
#        sq2 = 0
#        for i in range(len(vect1)):
#            sum += vect1[i] * vect2[i]
#            sq1 += pow(vect1[i], 2)
#            sq2 += pow(vect2[i], 2)
#        try:
#            result = round(float(sum) / (math.sqrt(sq1) * math.sqrt(sq2)), 2)
#        except ZeroDivisionError:
#            result = 0.0
#        return result 
#    df['cosine_similarity_tfidf'] = df.apply(lambda x: cosine_similarity_tfidf(x), axis=1)
#    df['cosine_similarity_tfidf'] = df['cosine_similarity_tfidf'].astype(np.float32)
#    #使用词频向量求相似度
#    df['cosine_similarity'] = df.apply(lambda x: compute_cosine(x[1], x[3]), axis=1)
#    df['cosine_similarity'] = df['cosine_similarity'].astype(np.float32)
    
    
    
    
    
    #w2v向量求相似度
    model_query = Word2Vec.load('word2vec.model')
    model_title = Word2Vec.load('word2vec.model')
    
    def getVec(x, model):
        x = x.split(' ')
        a = 0
        for i in x:
            try:
                a += model.wv[i]
            except:
                pass
        a = np.array(a)
        a = (a - a.min())/(a.max() - a.min())
        return a+1
    
    def cosine_similarity_w2v(vect1, vect2):
        vect1mod = np.sqrt(vect1.dot(vect1))
        vect2mod = np.sqrt(vect2.dot(vect2))
        if vect1mod!=0 and vect2mod!=0:
            simla = (vect1.dot(vect2))/(vect1mod*vect2mod)
        else:
            simla = 0
        return simla

    
    
    #曼哈顿距离
    def manhattan_distance(p_vec, q_vec):
        """
        This method implements the manhattan distance metric
        :param p_vec: vector one
        :param q_vec: vector two
        :return: the manhattan distance between vector one and two
        """
        return np.sum(np.fabs(p_vec - q_vec))
    
    #欧式距离
    def Euclidean_distance(v1, v2):
        v1 = np.mat(v1)
        v2 = np.mat(v2)
        return float(np.array(np.sqrt((v1-v2)*((v1-v2).T))).squeeze())

    #切比雪夫距离
    def ChebyshevDistance(v1, v2):
        v1 = np.mat(v1)
        v2 = np.mat(v2)
        return np.abs(v1-v2).max()
    
    
    #闵可夫斯基距离测度家族
    #Euclidean
    def Euclidean(x_query, x_title):
        return np.sqrt(np.sum(np.power(x_query - x_title, 2)))
    #City block 
    def CityBlock(x_query, x_title):
        return np.sum(np.abs(x_query - x_title))
    #Chebyshev
    def Chebyshev(x_query, x_title):
        return np.max(np.abs(x_query - x_title))
    
    #L1L1 范数测度家族
    #Sorensen
    def Sorensen(x_query, x_title):
        return np.sum(x_query - x_title)/np.sum(x_query + x_title)
    #Gower
    def Gower(x_query, x_title):
        return np.sum(np.abs(x_query - x_title))/len(x_query)
    #Soergel
    def Soergel(x_query, x_title):
        a = np.sum(np.array([np.max(i-x_title) for i in x_query]))
        return np.sum(np.abs(x_query - x_title))/a
    
    
    #集合测度家族
    #Intersection
    def Intersection(x_query, x_title):
        return np.sum(np.abs(x_query-x_title))/2
    #Wave Hedges
    def WaveHedges(x_query, x_title):
        return np.sum(np.array([np.abs(i-j)/np.max([i,j]) for i,j in zip(x_query, x_title)]))
    #Czekanowski
    def Czekanowski(x_query, x_title):
        return np.sum(np.abs(x_query - x_title))/np.sum(x_query + x_title)
    #Motyka
    def Motyka(x_query, x_title):
        return np.sum(np.max([x_query, x_title],axis=0))/np.sum(x_query + x_title)
    #Kulczynski
    def Kulczynski(x_query, x_title):
        return np.sum(np.min([x_query, x_title],axis=0))/np.sum(x_query + x_title)
    #Ruzicka
    def Ruzicka(x_query, x_title):
        return np.sum(np.min([x_query, x_title],axis=0))/np.sum(np.max([x_query, x_title], axis=0))
    
    
    #香农信息熵家族
    #Kullback-Leibler
    def KullbackLeibler(x_query, x_title):
        return np.sum(x_query*np.log(x_query/(x_title)))
    #Jeffreys
    def Jeffreys(x_query, x_title):
        return np.sum((x_query-x_title)*np.log(x_query/x_title))
    #KDivergence
    def KDivergence(x_query, x_title):
        return np.sum(x_query*np.log(2*x_query/(x_query+x_title)))
    #Topsoe
    def Topsoe(x_query, x_title):
        return np.sum(x_query*np.log(2*x_query/(x_query+x_title)) + x_title*np.log(2*x_title/(x_query+x_title)))
    #JensenShannon
    def JensenShannon(x_query, x_title):
        return (np.sum(x_query*np.log(2*x_query/(x_query+x_title))) + np.sum(x_title*np.log(2*x_title/(x_query+x_title))))/2
    #JensenDifference
    def JensenDifference(x_query, x_title):
        return np.sum((x_query*np.log(x_query)+x_title*np.log(x_title)) - ((x_query+x_title)/2)*np.log((x_query+x_title)/2))

    func_names = [cosine_similarity_w2v, manhattan_distance,Euclidean_distance,
                 ChebyshevDistance, Euclidean, CityBlock, Chebyshev, Sorensen, 
                 Gower, Soergel, Intersection, WaveHedges, Czekanowski, Motyka, 
                 Kulczynski, Ruzicka, KullbackLeibler, Jeffreys, KDivergence, 
                 Topsoe, JensenShannon, JensenDifference]
    
    data = {}
    for i in range(len(df)):
        
        x_query = getVec(df.iloc[i, 1], model_query)
        x_title = getVec(df.iloc[i, 3], model_title)
        
        d = {}
        for func in func_names:
#            df[str(func)] = df.apply(func(x_query, x_title), axis=0)
            d[func.__name__] = func(x_query, x_title)
            
        data[i] = d
    
    data = pd.DataFrame(data).T
    data.index = df.index
    
    df = pd.concat((df, data), axis=1)
            
    return df



            
#test = pd.read_csv('test.csv', header=None)
#
#since =  time.time()
#t1, d = get(test)
#time_elapsed = time.time() - since
#print('complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
#
#since =  time.time()
#t2 = getSimilarityFeature(test)
#time_elapsed = time.time() - since
#print('complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))








def getfeature111(df):
    
    print(111111)
    
    word = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ' ']
    df[1] = df[1].apply(lambda x: ''.join([i for i in x if i in word]))
    df[3] = df[3].apply(lambda x: ''.join([i for i in x if i in word]))

    ite = 1
    since = time.time()
    
    #query 长度
    df['query_len'] = df.apply(lambda x: len(x[1].split(' ')), axis=1)
    df['query_len'] = df['query_len'].astype(np.int32)
    

    #query_unique 长度
    def get_query_unique(x):
        a = len(set(x[1].split(' '))-set(x[3].split(' ')))
        return a
    
    df['query_unique'] = df.apply(get_query_unique, axis=1)
    df['query_unique'] = df['query_unique'].astype(np.int32)
    

    #title 长度
    df['title_len'] = df.apply(lambda x: len(x[3].split(' ')), axis=1)
    df['title_len'] = df['title_len'].astype(np.int32)
    

    #title_unique 长度
    def get_query_unique(x):
        a = len(set(x[3].split(' '))-set(x[1].split(' ')))
        return a
    df['title_unique'] = df.apply(get_query_unique, axis=1)
    df['title_unique'] = df['title_unique'].astype(np.int32)
    

    #长度差、unique长度差
    df['dif'] = (df['query_len'] - df['title_len']).abs()
    df['dif'] = df['dif'].astype(np.int32)
    df['dif_unique'] = (df['query_unique'] - df['title_unique']).abs()
    df['dif_unique'] = df['dif_unique'].astype(np.int32)
    

    #query_title_same_len query和title相同的长度
    def query_title_same_len(x):
        a = len(set(x[3].split(' ')) & set(x[1].split(' ')))
        return a
    df['query_title_same_len'] = df.apply(query_title_same_len, axis=1)
    df['query_title_same_len'] = df['query_title_same_len'].astype(np.int32)
    

    #query和title相同的长度在query的比率
    def samelen_query_rate(x):
        a = len(set(x[1].split(' ')) & set(x[3].split(' ')))
        return a/len(x[1].split(' '))
    df['samelen_query_rate'] = df.apply(samelen_query_rate, axis=1)
    df['samelen_query_rate'] = df['samelen_query_rate'].astype(np.float32)
    

    #query和title相同的长度在title的比率
    def samelen_title_rate(x):
        a = len(set(x[1].split(' ')) & set(x[3].split(' ')))
        return a/len(x[3].split(' '))
    df['samelen_title_rate'] = df.apply(samelen_title_rate, axis=1)
    df['samelen_title_rate'] = df['samelen_title_rate'].astype(np.float32)
    

    
    #query和title总长度
    def q_t_all_len(x):
        a = len(x[1].split(' ')) + len(x[3].split(' '))
        return a
    df['q_t_all_len'] = df.apply(q_t_all_len, axis=1)
    df['q_t_all_len'] = df['q_t_all_len'].astype(np.int32)
    

    #query和title的unique长度
    def q_t_unique_len(x):
        a = len(set(x[1].split(' ') + x[3].split(' ')))
        return a
    df['q_t_unique_len'] = df.apply(q_t_unique_len, axis=1)
    df['q_t_unique_len'] = df['q_t_unique_len'].astype(np.int32)
    

    
    #query和title总词数
    def q_t_all_word_len(x):
        a = len(set(x[1].split(' ')) | set(x[3].split(' ')))
        return a
    df['q_t_all_word_len'] = df.apply(q_t_all_word_len, axis=1)
    df['q_t_all_word_len'] = df['q_t_all_word_len'].astype(np.int32)
    

    #query和title不同词数
    df['q_t_diff_len'] = df.apply(lambda x: len(set(x[1].split(' '))^set(x[3].split(' '))), axis=1)
    df['q_t_diff_len'] = df['q_t_diff_len'].astype(np.int32)
    

    #query和title不同词数在query的比率
    def q_t_diff_q_rate(x):
        a = len(set(x[1].split(' '))^set(x[3].split(' ')))
        return np.float32(a/len(x[1].split(' ')))
    df['q_t_diff_q_rate'] = df.apply(q_t_diff_q_rate, axis=1)
    df['q_t_diff_q_rate'] = df['q_t_diff_q_rate'].astype(np.float32)
    

    #query和title不同词数在title的比率
    def q_t_diff_t_rate(x):
        a = len(set(x[1].split(' '))^set(x[3].split(' ')))
        return np.float32(a/len(x[3].split(' ')))
    df['q_t_diff_t_rate'] = df.apply(q_t_diff_t_rate, axis=1)
    df['q_t_diff_t_rate'] = df['q_t_diff_t_rate'].astype(np.float32)
    

    
    #query在title中的不同词数
    def query_diff_title(x):
        a = len(set(x[1].split(' ')) - (set(x[1].split(' ')) & set(x[3].split(' '))))
        return np.int32(a)
    df['query_diff_title'] = df.apply(query_diff_title, axis=1)
    df['query_diff_title'] = df['query_diff_title'].astype(np.int32)
    

    #query在title中的不同词数在query的比率
    def query_diff_title_rate(x):
        a = len(set(x[1].split(' ')) - (set(x[1].split(' ')) & set(x[3].split(' '))))
        return np.float32(a/len(x[1].split(' ')))
    df['query_diff_title_rate'] = df.apply(query_diff_title_rate, axis=1)
    df['query_diff_title_rate'] = df['query_diff_title_rate'].astype(np.float32)
    

    #query在title中的不同词数在title的比率
    def query_diff_title_rate(x):
        a = len(set(x[1].split(' ')) - (set(x[1].split(' ')) & set(x[3].split(' '))))
        return np.float32(a/len(x[3].split(' ')))
    df['query_diff_title_rate'] = df.apply(query_diff_title_rate, axis=1)
    df['query_diff_title_rate'] = df['query_diff_title_rate'].astype(np.float32)
    

    #title在qurey中的不同词数
    def title_diff_query(x):
        a = len(set(x[3].split(' ')) - (set(x[1].split(' ')) & set(x[3].split(' '))))
        return np.int32(a)
    df['title_diff_query'] = df.apply(title_diff_query, axis=1)
    df['title_diff_query'] = df['title_diff_query'].astype(np.int32)
    

    #title在qurey中的不同词数在query的比率
    def title_diff_query_rate(x):
        a = len(set(x[3].split(' ')) - (set(x[1].split(' ')) & set(x[3].split(' '))))
        return np.float32(a/len(x[1].split(' ')))
    df['title_diff_query_rate'] = df.apply(title_diff_query_rate, axis=1)
    df['title_diff_query_rate'] = df['title_diff_query_rate'].astype(np.float32)
    

    #title在qurey中的不同词数在title的比率
    def title_diff_query_rate(x):
        a = len(set(x[3].split(' ')) - (set(x[1].split(' ')) & set(x[3].split(' '))))
        return np.float32(a/len(x[3].split(' ')))
    df['title_diff_query_rate'] = df.apply(title_diff_query_rate, axis=1)
    df['title_diff_query_rate'] = df['title_diff_query_rate'].astype(np.float32)
    
    time_elapsed = time.time() - since#######################################
    print('{} {} complete in {:.0f}m {:.0f}s'.format(ite, df.columns[-1], time_elapsed // 60, time_elapsed % 60))
    ite += 1
    
    #Dice系数
    def dice(x):
        a1 = len(set(x[1].split(' ')) & set(x[3].split(' ')))
        a2 = len(set(x[1].split(' '))) + len(set(x[3].split(' ')))
        return np.float32(2*a1/a2)
    df['dice'] = df.apply(dice, axis=1)
    df['dice'] = df['dice'].astype(np.float32)
    
    time_elapsed = time.time() - since#######################################
    print('{} {} complete in {:.0f}m {:.0f}s'.format(ite, df.columns[-1], time_elapsed // 60, time_elapsed % 60))
    ite += 1
    
    #jaccord系数
    def jaccord(x):
        a1 = len(set(x[1].split(' ')) & set(x[3].split(' ')))
        a2 = len(set(x[1].split(' ')) | set(x[3].split(' ')))
        return np.float32(a1/a2)
    df['jaccord'] = df.apply(jaccord, axis=1)
    df['jaccord'] = df['jaccord'].astype(np.float32)




    
    #lcseque_lens
    df['lcseque_lens'] = df.apply(lambda x: lcseque_lens(x[1], x[3]), axis=1)
    df['lcseque_lens'] = df['lcseque_lens'].astype(np.int32)
    
    time_elapsed = time.time() - since#######################################
    print('{} {} complete in {:.0f}m {:.0f}s'.format(ite, df.columns[-1], time_elapsed // 60, time_elapsed % 60))
    ite += 1
    
    #lcsubstr_lens
    df['lcsubstr_lens'] = df.apply(lambda x: lcsubstr_lens(x[1], x[3]), axis=1)
    df['lcsubstr_lens'] = df['lcsubstr_lens'].astype(np.int32)
    


    #ratio
    df['ratio'] = df.apply(lambda x: Levenshtein.ratio(x[1], x[3]), axis=1)
    df['ratio'] = df['ratio'].astype(np.float32)
    
    time_elapsed = time.time() - since#######################################
    print('{} {} complete in {:.0f}m {:.0f}s'.format(ite, df.columns[-1], time_elapsed // 60, time_elapsed % 60))
    ite += 1
    

    #Jaro距离
    df['jaro'] = df.apply(lambda x: Levenshtein.jaro(x[1], x[3]), axis=1)
    df['jaro'] = df['jaro'].astype(np.float32)
    

    
    #jaro_winkler
    df['jaro_winkler'] = df.apply(lambda x: Levenshtein.jaro_winkler(x[1], x[3]), axis=1)
    df['jaro_winkler'] = df['jaro_winkler'].astype(np.float32)
    
    time_elapsed = time.time() - since#######################################
    print('{} {} complete in {:.0f}m {:.0f}s'.format(ite, df.columns[-1], time_elapsed // 60, time_elapsed % 60))
    ite += 1


    #余弦相似度
    #使用词频向量求相似度
    df['cosine_similarity'] = df.apply(lambda x: compute_cosine(x[1], x[3]), axis=1)
    df['cosine_similarity'] = df['cosine_similarity'].astype(np.float32)
    

    
    
    #编辑距离
    df['levenshtein'] = df.apply(lambda x: Levenshtein.distance(x[1], x[3]), axis=1)
    df['levenshtein'] = df['levenshtein'].astype(np.int32)
    
    time_elapsed = time.time() - since#######################################
    print('{} {} complete in {:.0f}m {:.0f}s'.format(ite, df.columns[-1], time_elapsed // 60, time_elapsed % 60))
    ite += 1
    
    
    #皮尔逊相关系数Pehrson
    df['Pehrson'] = df.apply(lambda x:Pehrson(x[1], x[3]), axis=1)
    df['Pehrson'] = df['Pehrson'].fillna(df['Pehrson'].mean())
    df['Pehrson'] = df['Pehrson'].astype(np.float32)
    

    
    
    #计算两个字符串list的相似度
    df['list_dis'] = df.apply(lambda x: Levenshtein.seqratio(x[1].split(' '), x[3].split(' ')), axis=1)
    df['list_dis'] = df['list_dis'].astype(np.float32)
    
    time_elapsed = time.time() - since#######################################
    print('{} {} complete in {:.0f}m {:.0f}s'.format(ite, df.columns[-1], time_elapsed // 60, time_elapsed % 60))
    ite += 1
    


    #jaro_winkler和jaro的均值、乘积、差值
    df['jaroWinkler_jaro_mean'] = (df['jaro'] + df['jaro_winkler'])/2
#    df['jaroWinkler_jaro_product'] = df['jaro'] * df['jaro_winkler']
#    df['jaroWinkler_jaro_subtraction'] = (df['jaro'] - df['jaro_winkler']).abs()
#    
#
#    
#    #Pehrson和cosine_similarity的均值、乘积
#    df['cosine_similarity_Pehrson_mean'] = (df['Pehrson'] + df['cosine_similarity'])/2
#    df['cosine_similarity_Pehrson_product'] = df['Pehrson'] * df['cosine_similarity']
#    df['cosine_similarity_Pehrson_subtraction'] = (df['Pehrson'] - df['cosine_similarity']).abs()
#    
#    time_elapsed = time.time() - since#######################################
#    print('{} {} complete in {:.0f}m {:.0f}s'.format(ite, df.columns[-1], time_elapsed // 60, time_elapsed % 60))
#    ite += 1
        
    return df





def getSimilarityFeature111(df):
    
    since = time.time()
    

    model_query = Word2Vec.load('word2vec.model')
    model_title = Word2Vec.load('word2vec.model')
    
    def getVec(x, model):
        x = x.split(' ')
        a = 0
        for i in x:
            try:
                a += model.wv[i]
            except:
                pass
        a = np.array(a)
        a = (a - a.min())/(a.max() - a.min())
        return a+1
    
    def cosine_similarity_w2v(vect1, vect2):
        vect1mod = np.sqrt(vect1.dot(vect1))
        vect2mod = np.sqrt(vect2.dot(vect2))
        if vect1mod!=0 and vect2mod!=0:
            simla = (vect1.dot(vect2))/(vect1mod*vect2mod)
        else:
            simla = 0
        return simla

    
    
#    def manhattan_distance(p_vec, q_vec):
#        return np.sum(np.fabs(p_vec - q_vec))
#    
#    def Euclidean_distance(v1, v2):
#        v1 = np.mat(v1)
#        v2 = np.mat(v2)
#        return float(np.array(np.sqrt((v1-v2)*((v1-v2).T))).squeeze())
#
#    def ChebyshevDistance(v1, v2):
#        v1 = np.mat(v1)
#        v2 = np.mat(v2)
#        return np.abs(v1-v2).max()
#    
#    
#    def Euclidean(x_query, x_title):
#        return np.sqrt(np.sum(np.power(x_query - x_title, 2)))
#    def CityBlock(x_query, x_title):
#        return np.sum(np.abs(x_query - x_title))
#    def Chebyshev(x_query, x_title):
#        return np.max(np.abs(x_query - x_title))
#    
#    def Sorensen(x_query, x_title):
#        return np.sum(x_query - x_title)/np.sum(x_query + x_title)
#    def Gower(x_query, x_title):
#        return np.sum(np.abs(x_query - x_title))/len(x_query)
#    def Soergel(x_query, x_title):
#        a = np.sum(np.array([np.max(i-x_title) for i in x_query]))
#        return np.sum(np.abs(x_query - x_title))/a
#    
#    
#    def Intersection(x_query, x_title):
#        return np.sum(np.abs(x_query-x_title))/2
#    def WaveHedges(x_query, x_title):
#        return np.sum(np.array([np.abs(i-j)/np.max([i,j]) for i,j in zip(x_query, x_title)]))
#    def Czekanowski(x_query, x_title):
#        return np.sum(np.abs(x_query - x_title))/np.sum(x_query + x_title)
#    def Motyka(x_query, x_title):
#        return np.sum(np.max([x_query, x_title],axis=0))/np.sum(x_query + x_title)
#    def Kulczynski(x_query, x_title):
#        return np.sum(np.min([x_query, x_title],axis=0))/np.sum(x_query + x_title)
#    def Ruzicka(x_query, x_title):
#        return np.sum(np.min([x_query, x_title],axis=0))/np.sum(np.max([x_query, x_title], axis=0))
    
    
    def KullbackLeibler(x_query, x_title):
        return np.sum(x_query*np.log(x_query/(x_title)))
    def Jeffreys(x_query, x_title):
        return np.sum((x_query-x_title)*np.log(x_query/x_title))
    def KDivergence(x_query, x_title):
        return np.sum(x_query*np.log(2*x_query/(x_query+x_title)))
    def Topsoe(x_query, x_title):
        return np.sum(x_query*np.log(2*x_query/(x_query+x_title)) + x_title*np.log(2*x_title/(x_query+x_title)))
    def JensenShannon(x_query, x_title):
        return (np.sum(x_query*np.log(2*x_query/(x_query+x_title))) + np.sum(x_title*np.log(2*x_title/(x_query+x_title))))/2
    def JensenDifference(x_query, x_title):
        return np.sum((x_query*np.log(x_query)+x_title*np.log(x_title)) - ((x_query+x_title)/2)*np.log((x_query+x_title)/2))

#    func_names = [cosine_similarity_w2v, manhattan_distance,Euclidean_distance,
#                 ChebyshevDistance, Euclidean, CityBlock, Chebyshev, Sorensen, 
#                 Gower, Soergel, Intersection, WaveHedges, Czekanowski, Motyka, 
#                 Kulczynski, Ruzicka, KullbackLeibler, Jeffreys, KDivergence, 
#                 Topsoe, JensenShannon, JensenDifference]
#    
    func_names = [KullbackLeibler, Jeffreys, KDivergence, 
                 Topsoe, JensenShannon, JensenDifference]

    
    data = {}
    for i in tqdm.tqdm(range(len(df))):
        x_query = getVec(df.iloc[i, 1], model_query)
        x_title = getVec(df.iloc[i, 3], model_title)
        
        d = {}
        for func in func_names:
            d[func.__name__] = func(x_query, x_title)
            
        data[i] = d

    time_elapsed = time.time() - since#######################################
    print('complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    data = pd.DataFrame(data).T
    data.index = df.index
    
    df = pd.concat((df, data), axis=1)
            
    return df

