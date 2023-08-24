import numpy as np
import pandas as pd
import math
import time
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import metrics
from sklearn.model_selection import KFold
import lightgbm as lgb
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
#from gensim.models import Doc2Vec
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from sklearn import utils
from sklearn.naive_bayes import MultinomialNB
import Levenshtein
import tqdm
from sklearn import preprocessing
from sklearn.decomposition import PCA

#from gensim.test.test_doc2vec import ConcatenatedDoc2Vec







#y_train = train[4]
#train.drop([4], axis=1, inplace=True)

#ngram = 2
#tfidf1 = CountVectorizer(ngram_range=(1, ngram))
#tfidf1 = TfidfVectorizer(sublinear_tf=True,ngram_range=(1, ngram), max_df=0.2, analyzer='word', norm='l1')




#模型

#lr = SGDClassifier(penalty='l1', random_state=2, loss='log', alpha=0.001)
#lr  = MultinomialNB()
#lr = SVC(probability=True)





def augment(x,y,t=2):
    xs,xn = [],[]
    for i in range(t):
        mask = y>0
#        mask = mask.T[0]
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xs.append(x1)

    for i in range(t//2):
        mask = y==0
#        mask = mask.T[0]
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xn.append(x1)

    xs = np.vstack(xs)
    xn = np.vstack(xn)
    ys = np.ones(xs.shape[0])
#    ys = ys.reshape(ys.shape[0], 1)
    yn = np.zeros(xn.shape[0])
#    yn = yn.reshape(yn.shape[0], 1)
    print(xs.shape)
    print(xn.shape)
    x = np.vstack([x,xs,xn])
    y = np.concatenate([y,ys,yn])
    return x,y









####################################################################################3

##Doc2Vec

#class LabeledLineSentence(object):
#    def __init__(self, doc_list, labels_list):
#       self.labels_list = labels_list
#       self.doc_list = doc_list
#    def __iter__(self):
#        for idx, doc in enumerate(self.doc_list):
#            yield LabeledSentence(words=doc.split(),tags=[self.labels_list[idx]])
#
#it = LabeledLineSentence(train[1], train[4])
#
#model = Doc2Vec(size=300, window=10, min_count=5, workers=11,alpha=0.025, min_alpha=0.025
#
#model.build_vocab(it)
#
#for epoch in range(10):
#    model.train(it, total_examples=model.corpus_count)
#    model.alpha -= 0.002            # decrease the learning rate
#    model.min_alpha = model.alpha       # fix the learning rate, no deca
#    model.train(it,total_examples=model.corpus_count)

#####################################################################################









'''

l = []
x = 1
for df in train:
    ngram = 2
    tfidf1 = CountVectorizer(ngram_range=(1, ngram))

    
    #划分训练集和测试集
    n = np.random.choice(range(len(df)), 200, replace=False)
    test = df.iloc[n, :]
    df.index = range(len(df))
    df.drop(n, axis=0, inplace=True)
    
    
    #训练集
#    if x == 1:
#        y_train = df[[4]]
#        train_arr = getWordVec(df)
#    else:
#        y_train = df[[4]]
#        train_arr = getWordVecTest(df)
    
    y_train = df[[4]]
    train_arr = getWordVec(df)

#    title_oneh = np.array(pd.get_dummies(df[2]))
#    train_arr = np.hstack((train_arr, title_oneh))
#    del title_oneh
#    
    
    #测试集
    y_test = test[[4]]
    test_arr = getWordVecTest(test)
    
#    title_oneh = np.array(pd.get_dummies(test[2]))
#    test_arr = np.hstack((test_arr, title_oneh))
#    del title_oneh

    
    print("第%d部分训练开始"%x)
    
    y_pre_test = 0

    for tr_train, tr_test in kf.split(train_arr):
#        x_t1 = pd.DataFrame(train_arr[tr_train, :])
#        y_t1 = pd.DataFrame(y_train.iloc[tr_train])
#        x_t1, y_t1 = augment(x_t1.values, y_t1.values)
#        
#        x_t2 = pd.DataFrame(train_arr[tr_test, :])
#        y_t2 = pd.DataFrame(y_train.iloc[tr_test])
#        x_t2, y_t2 = augment(x_t2.values, y_t2.values)

        x_t1 = train_arr[tr_train, :]
        y_t1 = y_train.iloc[tr_train]
        
        x_t2 = train_arr[tr_test, :]
        y_t2 = y_train.iloc[tr_test]

#
        
#        lr.partial_fit(x_t1, y_t1, classes=[0,1])
        lr.fit(x_t1, y_t1)
        
#        y_pre_test = lr.predict_proba(test_arr)[:, 1]
        print('测试集：%f'%metrics.roc_auc_score(np.array(y_test), lr.predict_proba(test_arr)[:, 1]))
        
        l.append(metrics.roc_auc_score(y_test, lr.predict_proba(test_arr)[:, 1]))
        
#        print(metrics.roc_auc_score(y_t2, lr.predict_proba(x_t2)[:, 1]))
    
#    y_pre_test /= 3
#    print('测试集：%f'%metrics.roc_auc_score(list(y_test), list(y_pre_test)))
    
    
#    #lgb模型
#    lgb_params = {
#            'learning_rate': 0.0001,
#            'max_depth': -1,
#            'min_data_in_leaf': 2, 
#            'objective':'binary',
#            'bagging_fraction':0.999,
#            'feature_fraction':0.999,
#            "boosting": "rf",
#            "bagging_freq": 5,
#            "bagging_seed": 11,
#            "metric": 'auc',
#            "verbosity": -1,
#        }
#        
#        
#    for tr_train,tr_test in kf.split(train_arr):
#            
##        x_t1 = train_arr[tr_train, :]
##        y_t1 = y_train.iloc[tr_train]
##        
##        x_t2 = train_arr[tr_test, :]
##        y_t2 = y_train.iloc[tr_test]
#
#        x_t1 = pd.DataFrame(train_arr[tr_train, :])
#        y_t1 = pd.DataFrame(y_train.iloc[tr_train])
#        x_t1, y_t1 = augment(x_t1.values, y_t1.values)
#        
#        x_t2 = pd.DataFrame(train_arr[tr_test, :])
        y_t2 = pd.DataFrame(y_train.iloc[tr_test])
#        x_t2, y_t2 = augment(x_t2.values, y_t2.values)
#
#            
#        lgbtr1=lgb.Dataset(x_t1, y_t1.T[0])
#        lgbtr2=lgb.Dataset(x_t2, y_t2.T[0])
#            
#        gbm = lgb.train(lgb_params,
#                          lgbtr1,
#                          num_boost_round=50,
#                          valid_sets=[lgbtr1, lgbtr2],
#                          verbose_eval=100,
#                          early_stopping_rounds=20)
#        y_pre_test += gbm.predict(test_arr)
#    
#        print('测试集：%f'%metrics.roc_auc_score(np.array(y_test), gbm.predict(test_arr)))
#        l.append(metrics.roc_auc_score(y_test, gbm.predict(test_arr)))

        
        

        
    x += 1
print('mean score:%f'%np.mean(l))


'''


##################################################################################################

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



#
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



def getfeature(df):
    df.drop_duplicates(inplace=True)



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
        a = len(x[3].split(' ') and x[1].split(' '))
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

    #Dice系数
    def dice(x):
        a1 = len(set(x[1].split(' ')) & set(x[3].split(' ')))
        a2 = len(set(x[1].split(' '))) + len(set(x[3].split(' ')))
        return np.float32(2*a1/a2)
    df['dice'] = df.apply(dice, axis=1)
    df['dice'] = df['dice'].astype(np.float32)

    #jaccord系数
    def jaccord(x):
        a1 = len(set(x[1].split(' ')) & set(x[3].split(' ')))
        a2 = len(set(x[1].split(' ')) | set(x[3].split(' ')))
        return np.float32(a1/a2)
    df['jaccord'] = df.apply(jaccord, axis=1)
    df['jaccord'] = df['jaccord'].astype(np.float32)

    #同一个query出现次数
    df['query_count'] = df.groupby(1)[1].transform('count')
    df['query_count'] = df['query_count'].astype(np.int32)

    #同一个title出现的次数
    df['title_count'] = df.groupby(3)[3].transform('count')
    df['title_count'] = df['title_count'].astype(np.int32)

#    #query和labe的交叉点击
#    df['query_label_click'] = df.groupby([1, 4])[1].transform('count')
#    df['query_label_click'] = df['query_label_click'].astype(np.int32)
#
#    #title和labe的交叉点击
#    df['title_label_click'] = df.groupby([3, 4])[3].transform('count')
#    df['title_label_click'] = df['title_label_click'].astype(np.int32)


    #lcseque_lens
    df['lcseque_lens'] = df.apply(lambda x: lcseque_lens(x[1], x[3]), axis=1)
    df['lcseque_lens'] = df['lcseque_lens'].astype(np.int32)

    #lcsubstr_lens
    df['lcsubstr_lens'] = df.apply(lambda x: lcsubstr_lens(x[1], x[3]), axis=1)
    df['lcsubstr_lens'] = df['lcsubstr_lens'].astype(np.int32)


    #ratio
    df['ratio'] = df.apply(lambda x: Levenshtein.ratio(x[1], x[3]), axis=1)
    df['ratio'] = df['ratio'].astype(np.float32)


    #Jaro距离
    df['jaro'] = df.apply(lambda x: Levenshtein.jaro(x[1], x[3]), axis=1)
    df['jaro'] = df['jaro'].astype(np.float32)

    #jaro_winkler
    df['jaro_winkler'] = df.apply(lambda x: Levenshtein.jaro_winkler(x[1], x[3]), axis=1)
    df['jaro_winkler'] = df['jaro_winkler'].astype(np.float32)


    since = time.time()
    #余弦相似度
    ct = CountVectorizer()
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
    df['cosine_similarity'] = df.apply(lambda x: compute_cosine(x[1], x[3]), axis=1)
    df['cosine_similarity'] = df['cosine_similarity'].astype(np.float32)

    time_elapsed = time.time() - since#######################################
    print('complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    #编辑距离
    df['levenshtein'] = df.apply(lambda x: Levenshtein.distance(x[1], x[3]), axis=1)
    df['levenshtein'] = df['levenshtein'].astype(np.int32)


    #皮尔逊相关系数Pehrson
    df['Pehrson'] = df.apply(lambda x:Pehrson(x[1], x[3]), axis=1)
    df['Pehrson'] = df['Pehrson'].fillna(df['Pehrson'].mean())
    df['Pehrson'] = df['Pehrson'].astype(np.float32)

    #计算两个字符串list的相似度
    df['list_dis'] = df.apply(lambda x: Levenshtein.seqratio(x[1].split(' '), x[3].split(' ')), axis=1)
    df['list_dis'] = df['list_dis'].astype(np.float32)

#    #Word2vec计算两个list的相似度
#    df['word_list_similarity'] = df.apply(lambda x: model_word.n_similarity(x[1].split(' '), x[3].split(' ')), axis=1)

    #Doc2vec计算两个list的相似度
    df['doc_list_similarity'] = df.apply(lambda x: model_doc.n_similarity(x[1].split(' '), x[3].split(' ')), axis=1)
#
#    #doc2vec和word2vec分别得到相似度乘积
#    df['doc_word_list_similarity_product'] = df['word_list_similarity'] * df['doc_list_similarity']

    #jaro_winkler和jaro的均值、乘积、差值
    df['jaroWinkler_jaro_mean'] = (df['jaro'] + df['jaro_winkler'])/2
    df['jaroWinkler_jaro_product'] = df['jaro'] * df['jaro_winkler']
    df['jaroWinkler_jaro_subtraction'] = (df['jaro'] - df['jaro_winkler']).abs()

    #Pehrson和cosine_similarity的均值、乘积
    df['cosine_similarity_Pehrson_mean'] = (df['Pehrson'] + df['cosine_similarity'])/2
    df['cosine_similarity_Pehrson_product'] = df['Pehrson'] * df['cosine_similarity']
    df['cosine_similarity_Pehrson_subtraction'] = (df['Pehrson'] * df['cosine_similarity']).abs()

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
    df['title_tf_mean'] = df.apply(lambda x: tf(x[3]).mean(), axis=1)
    df['query_title_tf_mean'] = df.apply(lambda x: tf(x[1] + ' ' + x[3]).mean(), axis=1)

    #idf系数
    #max
    def idf(x):
        x = x.split(' ')
        l = len(x)
        x = pd.Series(x).value_counts().values
        x = np.log10(l/x)
        return x
#    df['query_idf_max'] = df.apply(lambda x: idf(x[1]).max(), axis=1)
    df['title_idf_max'] = df.apply(lambda x: idf(x[3]).max(), axis=1)
    df['query_title_idf_max'] = df.apply(lambda x: idf(x[1] + ' ' + x[3]).max(), axis=1)
    #mean
#    df['query_idf_mean'] = df.apply(lambda x: idf(x[1]).mean(), axis=1)
    df['title_idf_mean'] = df.apply(lambda x: idf(x[3]).mean(), axis=1)
    df['query_title_idf_mean'] = df.apply(lambda x: idf(x[1] + ' ' + x[3]).mean(), axis=1)

    #TF*IDF
    def tfidf(x):
        t = tf(x)
        i = idf(x)
        ti = t*i
        return ti
    #max
#    df['query_tfidf_max'] = df.apply(lambda x: tfidf(x[1]).max(), axis=1)
    df['title_tfidf_max'] = df.apply(lambda x: tfidf(x[3]).max(), axis=1)
    df['query_title_tfidf_max'] = df.apply(lambda x: tfidf(x[1] + ' ' + x[3]).max(), axis=1)
    #mean
#    df['query_tfidf_mean'] = df.apply(lambda x: tfidf(x[1]).mean(), axis=1)
    df['title_tfidf_mean'] = df.apply(lambda x: tfidf(x[3]).mean(), axis=1)
    df['query_title_tfidf_mean'] = df.apply(lambda x: tfidf(x[1] + ' ' + x[3]).mean(), axis=1)



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





    try:
        df.drop([0, 1, 2, 3, 4], axis=1, inplace=True)
    except:
        df.drop([0, 1, 2, 3], axis=1, inplace=True)

    return df






############################################################################
'''

#lr+doc2vec


def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=6)) for doc in sents])
    return targets, regressors


kf_n = 5
kf = KFold(n_splits=kf_n, shuffle=True, random_state=2)
pre = 0

random_state = 42

lgb_params = {
    'boosting_type': 'gbdt',
    'objective': "binary",
    'metric': ['binary_logloss',"auc"],
    'num_leaves': 32,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 1
}

#lgb_params = {
#        'learning_rate': 0.001,
#        'max_depth': -1,
#        'min_data_in_leaf': 2, 
#        'objective':'binary',
#        'bagging_fraction':0.999,
#        'feature_fraction':0.999,
#        "boosting": "gbdt",
#        "bagging_freq": 5,
#        "bagging_seed": 11,
#        "metric": 'auc',
#        "verbosity": -1,
#    }


# model = SGDClassifier(penalty='l1', random_state = 2, shuffle = True, loss='log', alpha=0.01)
lr = LogisticRegression(penalty='l1', C=0.14, n_jobs=-1)
x = 1
for df in train:
#    n = np.random.choice(range(len(df)), 10000, replace=False)
#    
#    df = df.iloc[n, :]
    
    
    y_train = df[4]
    df.drop(4, axis=1, inplace=True)
    x_train = getfeature(df)  
    x_train['Pehrson'] = x_train['Pehrson'].fillna(x_train['Pehrson'].mean())
#    x_train = x_train.values
#    y_train = y_train.values



    
##    df['all_review'] = pd.DataFrame(df[1] + ' ' +df[3])
#    d = pd.DataFrame(df[1] + ' ' +df[3])
#    d['label'] = df[4]
#    
#    
#    d['cut_review'] = d[0].apply(lambda x: [w for w in x.split(' ') if w != ' '])
##    df['cut_review'] = df['clean_review'].apply(lambda x: [w for w in list(jieba.cut(x)) if w != ' '])
#    
#    train_tagged = d.apply(lambda r: TaggedDocument(words=r['cut_review'], tags=[r['label']]), axis=1)
#    
#    model_dbow = Doc2Vec(dm=0,  negative=25, hs=0, min_count=50, sample = 0, workers=2)
#    model_dbow.vector_size = 256
#    model_dbow.build_vocab([x for x in tqdm(train_tagged.values)])
#    
##    model_dmm = Doc2Vec(dm=0,  negative=20, hs=0, min_count=2, sample = 0, workers=2)
##    model_dmm.build_vocab([x for x in tqdm(train_tagged.values)])
##    
#
#    for epoch in range(5):
#        model_dbow.train(utils.shuffle([x for x in tqdm(train_tagged.values)]), 
#                         total_examples=len(train_tagged.values), epochs=2)
#        model_dbow.alpha -= 0.002
#        model_dbow.min_alpha = model_dbow.alpha
#        
#    
#    print('第%d次训练开始'%x)
#    
#    y_train, x_train_doc = vec_for_learning(model_dbow, train_tagged)
#    x_train = np.array(x_train_doc)
#    y_train = np.array(y_train)
#    
#    d = getfeature(df)
#    d['Pehrson'] = d['Pehrson'].fillna(d['Pehrson'].mean())
#    del df
#    x_train = np.hstack((x_train, d))
#    
#    
#    print(x_train.shape)
#
#    model_dbow.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)



    loss_score_feature = ['query_title_same_len', 'query_diff_title', 'title_diff_query', 'title_diff_query_rate']
    predictors = [i for i in x_train.columns if i not in loss_score_feature]
    importance = pd.DataFrame({'columns':predictors})
        
    for x_tr, x_vail in kf.split(x_train):
#        x_t1 = x_train[x_tr]
#        y_t1 = y_train[x_tr]
##        x_t1, y_t1 = augment(x_t1, y_t1)
#        
#        x_t2 = x_train[x_vail]
#        y_t2 = y_train[x_vail]
#        

        x_t1 = x_train.ix[x_tr, predictors]
        y_t1 = y_train.ix[x_tr]
#        x_t1, y_t1 = augment(x_t1.values, y_t1.values)
        
        x_t2 = x_train.ix[x_vail, predictors]
        y_t2 = y_train.ix[x_vail]

        lr.fit(x_t1, y_t1)
        print(metrics.roc_auc_score(y_t2, lr.predict_proba(x_t2)[:, 1]))
        
    i = 1
    for x_tr, x_vail in kf.split(x_train):
        
        x_t1 = x_train.ix[x_tr, predictors]
        y_t1 = y_train.ix[x_tr]
#        x_t1, y_t1 = augment(x_t1.values, y_t1.values)
        
        x_t2 = x_train.ix[x_vail, predictors]
        y_t2 = y_train.ix[x_vail]
#        
        
#        x_t1 = x_train[x_tr]
#        y_t1 = y_train[x_tr]
##        x_t1, y_t1 = augment(x_t1, y_t1)
#        
#        x_t2 = x_train[x_vail]
#        y_t2 = y_train[x_vail]
#            
        lgbtr1=lgb.Dataset(x_t1, y_t1)
        lgbtr2=lgb.Dataset(x_t2, y_t2)
            
        evals_result = {}
        gbm = lgb.train(lgb_params,
                          lgbtr1,
                          num_boost_round=1000,
                          valid_sets=lgbtr2,
                          verbose_eval=100,
                          early_stopping_rounds=300,
                          evals_result=evals_result)
        
#        print(metrics.roc_auc_score(y_t2, gbm.predict(x_t2, num_iteration=gbm.best_iteration)))
        importance['importance_%d'%i] = gbm.feature_importance()
        i += 1
#        print(predictors)
#        predictors = [importance.ix[i, 'columns'] for i in range(len(importance)) if importance.ix[i, 'importance'] > 0]
                
    
    print('第%d次训练结束'%x)

#    del model_dbow
#    del d
#    del train_tagged
    x += 1
    
# pre1 = pre/(x-1)

'''

###################################################################################
'''
random_state = 42

lgb_params = {
    "objective" : "binary",
    "metric" : "auc",
    "boosting": 'gbdt',
    "max_depth" : -1,
    "num_leaves" : 13,
    "learning_rate" : 0.01,
    "bagging_freq": 5,
    "bagging_fraction" : 0.4,
    "feature_fraction" : 0.05,
    "min_data_in_leaf": 80,
    "min_sum_heassian_in_leaf": 10,
    "tree_learner": "serial",
    "boost_from_average": "false",
    #"lambda_l1" : 5,
    #"lambda_l2" : 5,
    "bagging_seed" : random_state,
    "verbosity" : 1,
    "seed": random_state
}
'''





####################################################################################

#组合同一query的title进行词向量训练

def vec_for_learning(model, tagged_docs):
    sents = tagged_docs.values
    targets, regressors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, regressors

train = pd.read_csv('train.csv', header=None, chunksize=20000)

def getvec(x):
    a = 0
    n = 0
    for i in x:
        try:
            a += model[i]
            n += 1
        except:
            pass
    a = np.array(a)/n
    a = a.astype(np.float32)
    return a

x = 1
for df in train:

#    y_train = df[4]

#    d = df.groupby([0])[1].unique().reset_index().rename(columns={0:'id', 1:'query'})
#    d['query'] = d['query'].apply(lambda x: x[0])
#    d['title_concat'] = df.groupby([0])[3].unique().reset_index().apply(lambda x: ' '.join(x[3]), axis=1)
#    d.drop(['id'], axis=1, inplace=True)
#
##    del df
#
#    d = pd.DataFrame(d['query'] + ' ' + d['title_concat'])
#    d['cut_review'] = d.apply(lambda x: x[0].split(' '), axis=1)
#
#
#    model = Word2Vec(d['cut_review'].values, sg=0, size=100,  window=10,
#                                min_count=2, sample=0.001, hs=1, workers=4, iter=10)
#
#    train_query_tagged = d.apply(lambda r: TaggedDocument(words=r['cut_review'], tags=[0]), axis=1)
#
#    model = Word2Vec(d['cut_review'].values, sg=0, size=256,  window=10,  min_count=0, sample=0.001, hs=1, workers=4, iter=10)
#    model.save('Word2vec_last1000W_groupby_1000W.model')
#
#
#    model_dbow = Doc2Vec(dm=0,  negative=5, hs=0, min_count=0, sample = 0, workers=2)
#    model_dbow.build_vocab([x for x in tqdm.tqdm(train_query_tagged.values)])
#
#    for epoch in range(5):
#        model_dbow.train(utils.shuffle([x for x in tqdm.tqdm(train_query_tagged.values)]),
#                                total_examples=len(train_query_tagged.values), epochs=2)
#        model_dbow.alpha -= 0.002
#        model_dbow.min_alpha = model_dbow.alpha
#    model.save('Doc2vec_last1000W_groupby_1000W.model')

    model_doc = Doc2Vec.load('Doc2vec_last1000W_groupby_1000W.model')
    model_word = Word2Vec.load('Word2vec_last1000W_groupby_1000W.model')


    y_train = df[4]
    x_train_df = getfeature(df)
    importance = pd.DataFrame({'columns':x_train_df.columns})
    x_train = preprocessing.PolynomialFeatures(include_bias=False).fit_transform(x_train_df)


    pca = PCA(n_components=50)
    x_train = pca.fit_transform(x_train)
    x_train = preprocessing.scale(x_train)
    y_train = y_train.values




##    _, x_train = vec_for_learning(model_dbow, train_query_tagged)
##    x_train = d.apply(lambda x: getvec(d['cut_review']), axis=1)
##    x_train = d['cut_review'].values
#
##    del d
#
##    x_train = [getvec(i) for i in tqdm.tqdm(x_train)]
#    x_train = np.array(x_train)
#    y_train = y_train.values

    kf_n = 5
    kf = KFold(n_splits=kf_n, shuffle=True, random_state=2)


    random_state = 42
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': "binary",
        'metric': ['binary_logloss',"auc"],
        'num_leaves': 64,
        'learning_rate': 0.05,
        'feature_fraction': 0.6,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 1
    }

    i = 1
    for x_tr, x_vail in kf.split(x_train):

        x_t1 = x_train[x_tr]
        y_t1 = y_train[x_tr]

        x_t2 = x_train[x_vail]
        y_t2 = y_train[x_vail]

        lgbtr1=lgb.Dataset(x_t1, y_t1)
        lgbtr2=lgb.Dataset(x_t2, y_t2)

        evals_result = {}
        gbm = lgb.train(lgb_params,
                              lgbtr1,
                              num_boost_round=1000,
                              valid_sets=lgbtr2,
                              verbose_eval=100,
                              early_stopping_rounds=300,
                              evals_result=evals_result)

#        importance['importance_%d'%i] = gbm.feature_importance()
        i += 1

#importance_mean = pd.DataFrame(importance.mean(axis=1).values, index=importance['columns'])


#    lr = LogisticRegression(penalty='l1', C=0.3)
#    for x_tr, x_vail in kf.split(x_train):
#
#        x_t1 = x_train[x_tr]
#        y_t1 = y_train[x_tr]
#
#        x_t2 = x_train[x_vail]
#        y_t2 = y_train[x_vail]
#
#        lr.fit(x_t1, y_t1)
#        print(metrics.roc_auc_score(y_t2, lr.predict_proba(x_t2)[:, 1]))
#        print(metrics.log_loss(y_t2, lr.predict_proba(x_t2)[:, 1]))







