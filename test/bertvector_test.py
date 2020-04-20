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
import os
import pickle
import numpy as np
import jieba as jb
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt

# 把字向量转化为句向量，简单相加
def seg_vector (txt, dict_vector, emb_size=768):
    seg_v = np.zeros(emb_size)
    for w in txt:
        if w in dict_vector.keys():
            v = dict_vector[w]
            seg_v += v
    return seg_v

# 词空间分布图
def showWords (words, X, path='./'):
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
    outpic = os.path.join(path,"figure2d.png")
    plt.savefig(outpic)
    plt.show()
    print('分布图已保存为: %s' % outpic )


# 3维分布图
def showWords3d (words, X, path='./'):
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
    outpic = os.path.join(path,"figure3d.png")
    plt.savefig(outpic)
    plt.show()
    print('分布图已保存为: %s' % outpic )
    

#-----------------------------------------

def bertvector_test ():

    current_work_dir = os.path.dirname(__file__)    

    print('BERTVector 示例'.center(40,'-'))
    print('本示例安装目录为：%s' % current_work_dir)
    # 读取文件，加载向量
    print('字向量加载'.center(40,'-'))

    filename = os.path.join(current_work_dir, 'test.pkl')

    dict_vector = dict()
    with open(filename,'rb') as f:
        dict_vector = pickle.load(f)

    # 获取第一个字符向量
    key = list(dict_vector.keys())[0]
    value = dict_vector[key]
    print('字典大小：%d' % len(dict_vector.keys()))

    emb_size = len(value)
    print('向量维度:%d\n' % emb_size)

    print('字典首个元素：')
    print('字符:%s' % key)
    print('向量(仅输出前20维):\n%s' % value[:20])


    print('句向量示例'.center(40,'-'))
    lst_txt = [
        '新浪体育纪录另类热火中锋', # 手工改写的句子
        '一个以工作换工作方式交易的维修工人社群', 
        '基金投资的热度近期有所回升', # 手工改写的句子
        '老解放区和五十年代初期曾经施行过的农业劳动互助的简单形式，是农民相互调剂劳动力的方法，有人工换人工、牛工换牛工、人工换牛工等。',
        '封闭式基金折价率近期有所上涨',
        '今日数据趣谈：阿杜比肩魔术师 热火中锋另类纪录新浪体育讯北京时间4月28日',
        '以下是今日比赛中诞生的一些有趣数据',
        '最年轻纪录属于“魔术师”约翰逊，他1980年总决赛对阵76人得到42分，',
        '首次有三人组合在季后赛做到这一点',
        '安东尼此役打了38分59秒没有任何运动战出手',
        '本周受权重股业绩超市场预期',
        '券商研究机构在本周密集发布二季度的基金投资策略报告',
        '唐代说唱艺术的一种。一般认为"转"是说唱，"变"是奇异，"转变"为说唱奇异故事之意。一说"变"即变易文体之意。以说唱故事为主，其说唱之底本称为"变文"﹑"变"。内容多为历史传说﹑民间故事和宗教故事。多数散韵交织，有说有唱，说唱时辅以图画。同后世之词话﹑鼓词﹑弹词等关系密切。变文作品于清光绪间始在敦煌石室中发现，是研究我国古代说唱文学和民间文学的重要资料。',
        '支援劳力，帮助干活',
        '农业生产单位之间或农户之间在自愿基础上互相换着干活。',
        ]

    print('本示例中仅简单把各字的向量相加作为句向量...')

    print('句向量(仅输出前10维):')
    lst_v = []
    for txt in lst_txt:
        #print('待编码句子：%s\n' % txt)
        v = seg_vector(txt, dict_vector=dict_vector)
        lst_v.append(v)
        #print(v[:10])
    print('句向量已生成')

    print('余弦相似度计算'.center(40,'-'))

    # 需要做对比的句子编号 
    num = 1
    for i in range(len(lst_v)):
        d = 1-cosine(lst_v[num], lst_v[i] )
        print('[%s] 与 [%s] 的相似度: %.4f' % (lst_txt[num], lst_txt[i],d))

    #--- 词分布 -----
    #coding:utf-8
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
    # 加载样例数据
    txt = open(os.path.join(current_work_dir, 'test.txt'),'r', encoding='UTF-8').read()
    words = jb.lcut(txt)
    words = list(set(words))
    X = []
    print('共有词汇%d个。'% len(words))
    for word in words:
        X.append(seg_vector(word, dict_vector=dict_vector))
    # 显示分布图
    showWords (words, X, path=current_work_dir)
    showWords3d (words, X, path=current_work_dir )

if __name__ == '__main__':
    pass
    bertvector_test()
