#!/usr/bin/env python
# coding: utf-8

__author__ = 'xmxoxo<xmxoxo@qq.com>'

'''
BERT预模型字向量提取工具
版本： v 0.2.2
更新:  2020/3/23

    本工具直接读取BERT预训练模型，从中提取样本文件中所有使用到字
向量，保存成向量文件，为后续模型提供embdding。
    本工具直接读取预训练模型，不需要其它的依赖，同时把样本中所有
出现的字符对应的字向量全部提取，后续的模型可以非常快速进行embdding，
不再需要庞大的预训练模型或者bert-as-service服务了。

向量文件格式：一行一个字符向量,中间使用空格分隔；
格式为：字符 768大小向量
例如：
! 0.06241071 -0.012108288 0.035899848....

参数说明：
--model_path    BERT预训练模型存放的目录，必填；
--in_file       待提取的样本文件名或者目录名，必填；
--out_file      提取后保存的向量文件,可选项，默认为'./bert_embedding.txt'
--ext           当指定为目录时，可选项，要处理的文件后缀，默认为['csv','txt']

命令行例子：

处理单个文件./data/train_interger.csv，保存到./data/need_bertembedding.txt
python BERTVector.py \
    --model_path=/mnt/sda1/models/chinese_L-12_H-768_A-12/ \
    --in_file=./data/train_interger.csv \
    --out_file=./data/need_bertembedding.txt

处理目录下的所有tsv,txt文件
python BERTVector.py \
    --model_path=/mnt/sda1/models/chinese_L-12_H-768_A-12/ \
    --in_file=./data/ \
    --ext tsv txt
'''

import argparse
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import numpy as np
import os
import sys
import traceback

gblVersion = '0.2.2'
# 如果模型的文件名不同，可修改此处
model_name = 'bert_model.ckpt'
vocab_name = 'vocab.txt'

# BERT embdding提取类 
class bert_embdding(): 
    def __init__(self, model_path=''):
        # 模型和词表的文件名
        ckpt_path = os.path.join(model_path, model_name)
        vocab_file = os.path.join(model_path, vocab_name)
        if not os.path.isfile(vocab_file):
            print('词表文件不存在，请检查...')
            #sys.exit()
            return 
        
        # 从模型读出指定层
        reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
        #param_dict = reader.get_variable_to_shape_map()
        self.emb = reader.get_tensor("bert/embeddings/word_embeddings")
        self.vocab = open(vocab_file,'r', encoding='utf-8').read().split("\n")
        print('embeddings size: %s' % str(self.emb.shape))
        print('词表大小：%d' % len(self.vocab))

    # 取出指定字符的embdding,返回向量
    def get_embdding (self, char):
        if char in self.vocab:
            index = self.vocab.index(char)
            return self.emb[index,:]
        else:
            return None

    # 根据字符串提取向量并保存到文件
    def export (self, txt_all, out_file=''):
        # 过滤重复，形成字典
        txt_lst = sorted(list(set(txt_all)))

        print('文本字典长度:%d, 正在提取字向量...' % len(txt_lst))
        count = 0
        with open(out_file, 'w', encoding='utf-8') as out:  
            for word in txt_lst:
                v = self.get_embdding(word)
                if not (v is None):
                    count += 1
                    out.write(word + " " + " ".join([str(i) for i in v])+"\n")

        print('字向量共提取:%d个' % count)

    # get all files and floders in a path
    # fileExt: ['png','jpg','jpeg']
    # return: 
    #    return a list ,include floders and files , like [['./aa'],['./aa/abc.txt']]
    @staticmethod
    def getFiles (workpath, fileExt = []):
        try:
            lstFiles = []
            lstFloders = []

            if os.path.isdir(workpath):
                for dirname in os.listdir(workpath) :
                    file_path = os.path.join(workpath, dirname)
                    if os.path.isfile(file_path):
                        if fileExt:
                            if dirname[dirname.rfind('.')+1:] in fileExt:
                               lstFiles.append (file_path)
                        else:
                            lstFiles.append (file_path)
                    if os.path.isdir( file_path ):
                        lstFloders.append (file_path)      

            elif os.path.isfile(workpath):
                lstFiles.append(workpath)
            else:
                return None
            
            lstRet = [lstFloders,lstFiles]
            return lstRet
        except Exception as e :
            return None

    # 增加批量处理目录下的某类文件 v 0.1.2  xmxoxo 2020/3/23
    def export_path (self, path, ext=['csv','txt'], out_file=''):
        try:
            files = self.getFiles(path,ext)
            # 合并数据
            txt_all = []
            tmp = ''
            for fn in files[1]:
                print('正在读取数据文件:%s' % fn)
                with open(fn, 'r', encoding='utf-8') as f: 
                    tmp = f.read()
                txt_all += list(set(tmp))
                txt_all = list(set(txt_all))
            self.export(txt_all, out_file=out_file)

        except Exception as e :
            print('批量处理出错:')
            print('Error in get_randstr: '+ traceback.format_exc())
            return None

# 命令行
def main_cli ():
    parser = argparse.ArgumentParser(description='BERT模型字向量提取工具')
    parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + gblVersion )
    parser.add_argument('--model_path', default='', required=True, type=str, help='BERT预训练模型的目录')
    parser.add_argument('--in_file', default='', required=True, type=str, help='待提取的文件名或者目录名')
    parser.add_argument('--out_file', default='./bert_embedding.txt', type=str,  help='输出文件名')
    parser.add_argument('--ext', default=['csv','txt'], type=str, nargs='+', help='指定目录时读取的数据文件扩展名')

    args = parser.parse_args()

    # 预训练模型的目录
    model_path = args.model_path
    # 输出文件名
    out_file = args.out_file
    # 包含所有文本的内容
    in_file = args.in_file
    # 指定的扩展名
    ext = args.ext
    
    if not os.path.isdir(model_path):
        print('模型目录不存在,请检查:%s' % model_path)
        sys.exit()

    if not (os.path.isfile(in_file) or os.path.isdir(in_file)):
        print('数据文件:不存在,请检查:%s')
        sys.exit()
    print('\nBERT 字向量提取工具 V' + gblVersion )
    print('-'*40)
   
    bertemb = bert_embdding(model_path=model_path)
    # 针对文件和目录分别处理 2020/3/23 by xmxoxo
    if os.path.isfile(in_file):
        txt_all = open(in_file).read()
        bertemb.export(txt_all, out_file=out_file)
    if os.path.isdir(in_file):
        bertemb.export_path(in_file, ext=ext, out_file=out_file)

if __name__ == '__main__':
    pass
    main_cli()