# -*- coding: utf-8 -*-
# @File  : stacked_denoising_autoencoder.py
# @Author: Panbo
# @Date  : 2018/10/22
# @Desc  : 利用TensorFlow框架来实现堆叠降噪自编码器 Stacked Denoising AutoEncoder


import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

class Denoisiong_AutoEncoder():
    def __init__(self, n_hidden, input_data, corruption_level=0.3):
        self.W = None # 输入层到隐含层的权重
        self.b = None # 输入层到隐含层的偏置
        self.encode_r = None #隐含层的输出
        self.layer_size = n_hidden #隐含层节点的个数
        self.input_data = input_data #输入样本
        self.keep_prob = 1 - corruption_level #特征保持不变的比例
        self.W_eval = None #权重W的值
        self.b_eval = None  #偏置b的值

def fit(self):
    #输入层节点的个数
    n_visible = (self.input_data).shape[1]


class Stacked_Denoising_AutoEncoder():
    def __init__(self, hidden_list, input_data_trainX, \
                 input_data_trainY, input_data_validX, \
                 input_data_validY, input_data_testX, \
                 input_data_testY, corruption_level=0.3):
        self.ecod_W = [] #保存网络中每一层的权重
        self.ecod_b = [] #保存网络中每一层的偏置
        self.hidden_list = hidden_list #每一个隐含层的节点个数
        self.input_data_trainX = input_data_trainX #训练样本的特征
        self.input_data_trainY = input_data_trainY #训练样本的标签
        self.input_data_validX =input_data_validX #验证样本的特征
        self.input_data_validY = input_data_validY #验证样本的标签
        self.input_data_testX = input_data_testX #测试样本的特征
        self.input_data_testY = input_data_testY #测试样本的标签


def fit(self):
    next_input_data = self.input_data_trainX
    # 1. 训练每一个降噪自编码器
    for i, hidden_size in enumerate(self.hidden_list):
        print("-------------------train the %s sda------------" %(i + 1))
        #dae = Denoising_AutoEncoder