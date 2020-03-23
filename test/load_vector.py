#!/usr/bin/env python3
#coding:utf-8

__author__ = 'xmxoxo<xmxoxo@qq.com>'

'''
向量使用示例
演示了如何加载向量字典，对文本进行编码

使用之前请确定已经生成了字典: test.pkl
或者使用下面来生成：
BERTVector.py --model_path=d:\\model\chinese_L-12_H-768_A-12 --in_file=test/ --out_file=test/test.pkl

'''

import pickle
import numpy as np

# 读取文件，加载向量
print('字向量加载'.center(40,'-'))

filename = './test.pkl'
dict_vector = dict()
with open(filename,'rb') as f:
    dict_vector = pickle.load(f)

key = list(dict_vector.keys())[0]
value = dict_vector[key]
print('字典大小：%d' % len(dict_vector.keys()))

emb_size = len(value)
print('向量维度:%d\n' % emb_size)

print('字典首个元素：')
print('字符:%s' % key)
print('向量(仅输出前20维):\n%s' % value[:20])

# 把字向量转化为句向量，简单
print('句向量示例'.center(40,'-'))
txt = '今日数据趣谈：阿杜比肩魔术师 热火中锋另类纪录新浪体育讯北京时间4月28日'
print('待编码句子：\n%s\n' % txt)

print('本示例中仅简单把各字的向量相加作为句向量...')
seg_vector = np.zeros(emb_size)
for w in txt:
    if w in dict_vector.keys():
        v = dict_vector[w]
        seg_vector += v

print('句向量(仅输出前20维):')
print(seg_vector[:20])



if __name__ == '__main__':
    pass

