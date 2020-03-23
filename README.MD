# BERT预模型字向量提取工具


BERT预模型字向量提取工具
版本： v 0.2.2
更新:  2020/3/23

## 工具说明
本工具直接读取BERT预训练模型，从中提取样本文件中所有使用到字向量，保存成向量文件，为后续模型提供embdding。
本工具直接读取预训练模型，不需要其它的依赖，同时把样本中所有
出现的字符对应的字向量全部提取，后续的模型可以非常快速进行embdding，
不再需要庞大的预训练模型或者bert-as-service服务了。

## 向量文件格式 

一行一个字符向量,中间使用空格分隔；
格式为：字符 768大小向量
例如：
! 0.06241071 -0.012108288 0.035899848....

## 参数说明

参数说明：
--model_path    BERT预训练模型存放的目录，必填；
--in_file       待提取的样本文件名或者目录名，必填；
--out_file      提取后保存的向量文件,可选项，默认为'./bert_embedding.txt'
--ext           当指定为目录时，可选项，要处理的文件后缀，默认为['csv','txt']

## 使用例子

命令行例子：

处理单个文件`./data/train_interger.csv`，保存到`./data/need_bertembedding.txt`
```bash
python BERTVector.py \
    --model_path=/mnt/sda1/models/chinese_L-12_H-768_A-12/ \
    --in_file=./data/train_interger.csv \
    --out_file=./data/need_bertembedding.txt

```

处理目录下的所有tsv,txt文件,默认保存为:./need_bertembedding.txt

```bash
python BERTVector.py \
    --model_path=/mnt/sda1/models/chinese_L-12_H-768_A-12/ \
    --in_file=./data/ \
    --ext tsv txt
```
