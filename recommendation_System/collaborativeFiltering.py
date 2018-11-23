# -*- coding: utf-8 -*-
# @File  : collaborativeFiltering.py
# @Author: Panbo
# @Date  : 2018/11/21
# @Desc  : 基于项的协同过滤算法
import numpy as np


def cos_sim(x,y):
    '''

    :param x:  以行向量的形式存储可以使用户或者对应项
    :param y: 以列向量的形式存储，可以使用户或者对应项
    :return: x和y之间的余弦相似度
    '''
    numerator = x*y.T  #x和y之间的额内积
    denominator = np.sqrt(x * x.T) *np.sqrt(y*y.T)
    return (numerator / denominator)[0,0]

def similarity(data):
    '''
    计算矩阵中任意两行之间的相似度
    :param data: 任意矩阵
    :return: w ：任意两行之间的相似度
    '''


    m = np.shape(data)[0]  # 用户的数量
    #初始化相似度矩阵
    w = np.mat(np.zeros((m,m)))

    for i in range(m):
        for j in range(i,m):
            if j != i:
                #计算任意两行之间的相似度
                w[i,j] = cos_sim(data[i,], data[j,])
                w[j,i] = w[i,j]
            else:
                w[i,j] = 0

    return w

def item_based_recommend(data, w, user):
    '''
    基于商品相似度为用户user推荐商品
    :param data: 商品用户矩阵
    :param w: 商品与商品之间的相似性
    :param user: 用户编号
    :return: 推荐列表
    '''

    m, n = np.shape(data)  #m:商品数量  n：用户数量
    interaction = data [:, user].T # 用户user的互动商品信息

    #1 找到用户user没有互动的商品
    not_inter = []
    for i in range(n):
        if interaction[0,i] == 0: #用户user未打分
            not_inter.append(i)

    #2 度咩有互动过的商品进行预测
    predict = {}
    for x in not_inter:
        item = np.copy(interaction)  #获取用户user对商品的互动信息
        for j in range(m): #对每一个商品
            if item[0,j] !=0: #利用互动过的商品进行预测
                if x not in predict:
                    predict[x] =w[x,j] * item[0,j]
                else:
                    predict[x] = predict[x] + w[x,j]*item[0,j]


    #按照预测大小从大到小排序
    return sorted(predict.items(),key=lambda d:d[1],reverse=True)


def load_data(file_path):
    '''导入用户商品数据
    input:  file_path(string):用户商品数据存放的文件
    output: data(mat):用户商品矩阵
    '''
    f = open(file_path)
    data = []
    for line in f.readlines():
        lines = line.strip().split("\t")
        tmp = []
        for x in lines:
            if x != "-":
                tmp.append(float(x))  # 直接存储用户对商品的打分
            else:
                tmp.append(0)
        data.append(tmp)
    f.close()


def top_k(predict, k):
    '''为用户推荐前k个商品
    input:  predict(list):排好序的商品列表
            k(int):推荐的商品个数
    output: top_recom(list):top_k个商品
    '''
    top_recom = []
    len_result = len(predict)
    if k >= len_result:
        top_recom = predict
    else:
        for i in range(k):
            top_recom.append(predict[i])
    return top_recom

if __name__ == "__main__":
    # 1、导入用户商品数据
    print ("------------ 1. load data ------------")
    data = load_data("data.txt")
    # 将用户商品矩阵转置成商品用户矩阵
    #data = data.T
    data=np.transpose(data)
    # 2、计算商品之间的相似性
    print ("------------ 2. calculate similarity between items -------------")
    w = similarity(data)
    # 3、利用用户之间的相似性进行预测评分
    print ("------------ 3. predict ------------")
    predict = item_based_recommend(data, w, 0)
    # 4、进行Top-K推荐
    print("------------ 4. top_k recommendation ------------")
    top_recom = top_k(predict, 2)
    print (top_recom)