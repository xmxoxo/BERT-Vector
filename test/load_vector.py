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

# 把字向量转化为句向量，简单相加
def seg_vector (txt):
    seg_v = np.zeros(emb_size)
    for w in txt:
        if w in dict_vector.keys():
            v = dict_vector[w]
            seg_v += v
    return seg_v

print('句向量示例'.center(40,'-'))
lst_txt = [
    '新浪体育纪录另类热火中锋', # 手工改写的句子
    '基金投资的热度近期有所回升', # 手工改写的句子
    '封闭式基金折价率近期有所上涨',
    '今日数据趣谈：阿杜比肩魔术师 热火中锋另类纪录新浪体育讯北京时间4月28日',
    '以下是今日比赛中诞生的一些有趣数据',
    '最年轻纪录属于“魔术师”约翰逊，他1980年总决赛对阵76人得到42分，',
    '首次有三人组合在季后赛做到这一点',
    '安东尼此役打了38分59秒没有任何运动战出手',
    '本周受权重股业绩超市场预期',
    '券商研究机构在本周密集发布二季度的基金投资策略报告',
    ]

print('本示例中仅简单把各字的向量相加作为句向量...')
   

print('句向量(仅输出前10维):')
lst_v = []
for txt in lst_txt:
    #print('待编码句子：%s\n' % txt)
    v = seg_vector(txt)
    lst_v.append(v)
    #print(v[:10])
print('句向量已生成')

print('余弦相似度计算'.center(40,'-'))

from scipy.spatial.distance import cosine
num = 1
for i in range(len(lst_v)):
    d = 1-cosine(lst_v[num], lst_v[i] )
    print('[%s] 与 [%s] 的相似度: %.4f' % (lst_txt[num], lst_txt[i],d))

#--- 词分布
#coding:utf-8
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
#有中文出现的情况，需要u'内容'

# 词空间分布图
def showWords (words, X):
    import numpy as np
    import numpy.linalg as la
    import matplotlib.pyplot as plt

    U,S,Vh=la.svd(X,full_matrices=False)
    #plt.axis([-0.8,0.2,-0.8,0.8])
    # 求出分布区域
    #x_min, x_max = min(U[:,0]), max(U[:,0])
    #y_min, y_max = min(U[:,1]), max(U[:,1])
    #A = list(np.array([x_min, x_max, y_min, y_max])*0.9)
    A = [-0.01, 0.021, -0.02, 0.1]
    plt.axis(A)
    for i in range(len(words)):
        plt.text(U[i,0],U[i,1],words[i])
    plt.savefig("../images/figure2d.png")
    plt.show()

# 3维分布图
def showWords3d (words, X):
    import numpy as np
    import numpy.linalg as la
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    U,S,Vh=la.svd(X,full_matrices=False)
    #plt.axis([-0.8,0.2,-0.8,0.8])
    # 求出分布区域
    #x_min, x_max = min(U[:,0]), max(U[:,0])
    #y_min, y_max = min(U[:,1]), max(U[:,1])
    #A = list(np.array([x_min, x_max, y_min, y_max])*0.9)

    fig = plt.figure()
    ax = Axes3D(fig)
    #ax = fig.add_subplot(111, projection='3d')
    #plt.axis(A)
    #print(S.shape)
    #print(Vh.shape)
    for i in range(len(words)):
        #plt.text(U[i,0],U[i,1],) # words[i] marker=words
        p=ax.scatter(U[i,0], U[i,1], S[i])

    p=ax.scatter(U[:,0], U[:,1], S)
    #fig.colorbar(p)
    plt.savefig("../images/figure3d.png")
    plt.show()
    

import jieba as jb
txt = open('./test.txt','r', encoding='UTF-8').read()
words = jb.lcut(txt)
words = list(set(words))
X = []
print('共有词汇%d个。'% len(words))
for word in words:
    X.append(seg_vector(word))

showWords (words, X)
showWords3d (words, X)

if __name__ == '__main__':
    pass

