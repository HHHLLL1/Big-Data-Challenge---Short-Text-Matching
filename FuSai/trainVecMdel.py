# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 08:50:35 2019

@author: Lenovo
"""

import numpy as np
import pandas as pd
import tqdm
from sklearn import utils
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument




#df = pd.read_csv('./query_titleConcat.txt')
#
##分词
##d.drop(['id'], axis=1, inplace=True)    
#df['fenci']  =df.apply(lambda x: (x['query']+x['title_concat']).split(' '), axis=1)
#


df = pd.read_csv('train.csv', header=None)
test = pd.read_csv('test.csv', header=None)

df = pd.concat((df, test))
df = df.rename(columns={4:'label'})

df['fenci'] = df[1] + ' ' +df[3]

df['fenci'] = df['fenci'].str.split(' ')

#word2vec
model_word = Word2Vec(tqdm.tqdm(df['fenci']), size=256, window=10, min_count=1, workers=4,iter=10) 
model_word.save('word2vec.model')



#doc2vec
#tagged = df.apply(lambda r: TaggedDocument(words=r['fenci'], tags=[r['query_id']]), axis=1)
#
#model_doc = Doc2Vec(dm=0,  negative=5, hs=0, min_count=1, sample = 0, workers=4, size=256)
#model_doc.build_vocab([x for x in tqdm.tqdm(tagged.values)])
#    
#for epoch in range(5):
#    model_doc.train(utils.shuffle([x for x in tqdm.tqdm(tagged.values)]), 
#                                total_examples=len(tagged.values), epochs=3)
#    model_doc.alpha -= 0.002
#    model_doc.min_alpha = model_doc.alpha
#model_doc.save('doc2vec.model')
