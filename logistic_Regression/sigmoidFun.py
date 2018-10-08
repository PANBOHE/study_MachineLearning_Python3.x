import numpy as np

def sig(x):
    '''Sigmoid 函数
    //阈值函数，将样本映射到不同的类别中
    //Sig有很多种，最简单的0-1，逻辑回归中常用的指数函数
    imput: x(mat):feature *w
    output: sigmoid(x)(mat):Sigmoid 值
    :param x:
    :return:
    '''
    return 1.0 / (1+np.exp(-x))


def lr_train_bgd(feature, label, maxCycle, alpha):
    '''
    使用梯度下降法训练逻辑回归模型
    input: feature (mat) 特征
            label(mat)标签
            maxCycle(int)最大迭代次数
            alpha(float) 学习率 也就是每次迭代的步长，固定和随机的，随机的比较好
    output: w(mat) 权重
    :param feature:
    :param label:
    :param maxCycle:
    :param alpha:
    :return:
    '''
    n=np.shape(feature)[1] #特征个数
    w=np.mat(np.ones((n,1))) #初始化权重
    i=0
    while i <= maxCycle:   #最大迭代次数内
        i=i+1  #当前的迭代次数
        h=sig(feature*w)   #计算Sigmoid值
        err =label-h
        if i%100==0:
            print ("\t---------iter="+str(i)+",train error rate= "+ str(error_rate(h,label)))
        w=w+alpha*feature.T*err  #权重修正
    return w



def error_rate(h,label):
    '''
    计算当前损失函数值
    input: h 预测值，
            label 实际值
    output: err/m(float)  错误率

    :param h:
    :param label:
    :return:
    '''
    m=np.shape(h)[0]
    sum_err=0.0
    for i in range(m):
        if h[i,0]>0 and (1-h[i,0])>0:
            sum_err-=(label[i,0]*np.log(h[i,0])+(1-label[i,0])*np.log(1-h[i,0]))
        else:
            sum_err-=0
    return sum_err / m




def load_data(file_name):
    '''
    input: file_name 数据集
    output: feature
            label
    :param file_name:
    :return:
    '''
    f=open(file_name)
    feature_data=[]
    label_data=[]
    for line in f.readlines():
        feature_tmp=[]
        label_tmp=[]
        lines=line.strip().split("\t")
        feature_tmp.append(1)
        for i in range(len(lines)-1):
            feature_tmp.append(float(lines[i]))
        label_tmp.append(float(lines[-1]))

        feature_data.append(feature_tmp)
        label_data.append(label_tmp)
    f.close()

    return np.mat(feature_data),np.mat(label_data)


def save_model(file_name,w):
    """
    保存模型
    input: 文件名
            权重
    :param file_name:
    :param w:
    :return:
    """
    m=np.shape(w)[0]
    f_w=open(file_name,"w")
    w_array=[]
    for i in range(m):
        w_array.append(str(w[i,0]))
    f_w.write("\t".join(w_array))
    f_w.close()

if __name__ == "__main__":
    #导入数据
    print("------- 1.load data --------")
    feature, label =load_data("traindata.txt")


    print("------- 2.training --------")
    w=lr_train_bgd(feature,label,1000,0.001)
    print("------- 3.save model --------")
    save_model("weights",w)
