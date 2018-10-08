#
import numpy as np
import pickle

class SVM:
    def __init__(self,dataSet, labels, C, toler,kernel_option):
        self.train_x =dataSet #训练特征
        self.train_y=labels #训练标签
        self.C=C  #惩罚因子
        self.toler=toler #迭代的终止条件之一
        self.n_samples =np.shape(dataSet)[0]#训练样本的个数
        self.alphas=np.mat(np.zeros((self.n_samples,1)))  #拉格朗日橙子
        self.b=0
        self.error_tmp=np.mat(np.zeros((self.n_samples,2)))  #保存E的缓存

        self.kernel_opt =kernel_option #选用的核函数及其参数
        self.kernel_mat=calc_kernel(self.train_x,self.kernel_opt)  #核函数的输出。

def calc_kernel(train_x, kernel_option):
    '''
    计算核函数的矩阵
    input: train_x(mat):训练样本的特征值
            train_option(tuple): 核函数的类型以及参数
    output: kernel_matrix(mat):样本的核函数的值
    :param train_x:
    :param kernel_option:
    :return:
    '''
    m=np.shape(train_x)[0]
    kernel_matrix =np.mat(np.zeros((m,m)))#初始化样本之间的核函数值
    for i in range(m):
        kernel_matrix[:i]=calc_kernel_value(train_x,train_x[i,:],kernel_option)

    return kernel_matrix

def calc_kernel_value(train_x,train_x_i,kernel_option):
    '''
    样本之间的核函数值
    :param train_x: 训练样本
    :param train_x_i: 第i个训练样本
    :param kernel_option:  核函数的类型及参数
    :return: 样本之间核函数的值
    '''
    kernel_type = kernel_option[0] #核函数的类型，分为rbf和其他
    m=np.shape(train_x)[0]  #样本个数

    kernel_value = np.mat(np.zeros((m,1)))

    if kernel_type=='rbf':
        sigma =kernel_option[1]
        if sigma ==0:
            sigma=1.0
        for i in range(m):
            diff= train_x[i,:]-train_x_i
            kernel_value[i]=np.exp(diff*diff.T/(-2.0*sigma**2))
    else:
            kernel_value=train_x*train_x_i.T
    return kernel_value

#利用SMO算法对SVM模型进行训练
def SVM_training(train_x,train_y,C,toler,max_iter,kernel_option=('rbf',0.431029)):
    '''
    SVM训练
    :param train_x: 特征
    :param train_y: 标签
    :param C: 惩罚系数
    :param toler: 迭代的终止条件之一
    :param max_iter: 最大迭代次数
    :param kernel_option: 核函数的类型及其参数
    :return: SVM模型
    '''
    #1.初始化SVM分类器
    svm=SVM_training(train_x,train_y,C,toler,kernel_option)
    #2 开始训练
    entireSet =True
    alpha_pairs_changed = 0
    iteration =0

    while(iteration<max_iter) and((alpha_pairs_changed>0 ) or entireSet ):
        print('\t iteration: ', iteration)
        alpha_pairs_changed=0

        if entireSet:
            #对所有算法
            for x in range(svm.n_samples):
                alpha_pairs_changed += choose_and_update(svm,x)
            iteration+=1
        else:
            #非边界点
            bound_samples=[]
            for i in range (svm.n_samples):
                if svm.alphas[i,0] >0 and svm.alphas[i,0]< svm.C:
                    bound_samples.append(i)
            for x in bound_samples:
                alpha_pairs_changed+=choose_and_update(svm,x)

        #在所有样本和非边界样本之间交替
        if entireSet:
            entireSet=False
        elif alpha_pairs_changed==0:
            entireSet=True

    return svm


