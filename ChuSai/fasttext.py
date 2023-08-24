# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 20:13:26 2019

@author: Lenovo
"""

import fastText
import csv


with open('train.txt', 'w') as f:
    with open('train.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            query = row[1]
            title = row[3]
            label = row[4]
            line_count+=1
            f.write("__label__{0} {1} {2}\n".format(label, query, title))
            if line_count > 20000: # 10000*10000 一亿条要一晚上
                break
        print(f'Processed {line_count} lines.')
        
model = fastText.FastText.train_supervised(input="train.txt", 
                            epoch=25, lr=0.1, wordNgrams=2, verbose=2, minCount=50)
classifier = fastText.FastText.fasttext.supervised('train.txt', 'model')
result = classifier.test('train.txt')
print(result)