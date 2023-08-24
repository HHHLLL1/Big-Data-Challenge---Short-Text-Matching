# Big-Data-Challenge---Short-Text-Matching

排名：28

本赛题是大数据文本匹配类题目。数据有10亿条，给出quary和title，预测这个quary是否会被点击。我的当时的方案是lightgbm+伪孪生网络，因为是第一次参加nlp类竞赛，了解的深度学习模型不多，采用了看起来还不错的伪孪生网络，不过当时前排用的都是ESIM。Lightgbm中，构造文本统计特征、距离特征等，进行模型训练；伪孪生网络中，用w2v构建词向量，把quary和title分别放入其中进行训练模型。使用多进程和数据抽样完成任务。
