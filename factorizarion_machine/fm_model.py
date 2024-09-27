import numpy as  np
from random import normalvariate

def load_data(data_file):
    data_mat = []
    label_mat = []
    with open(data_file, encoding='utf-8', mode='r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split('\t')
            arr = []
            for i in range(len(line) -1):
                arr.append(float(line[i]))
            data_mat.append(arr)
            label_mat.append(float(line[-1]) * 2 -1)  #转成{-1, 1}
            # label_mat.append(float(line[-1]) )  #{0, 1}

    return data_mat, label_mat



def init_v(n, k):
    """
    初始化交叉项
    """
    v = np.mat(np.zeros((n, k)))
    for i in range(n):
        for j in range(k):
            v[i, j] =normalvariate(0, 0.2)
    return v

def sigmoid(x):
    """
    sigmoid 函数
    """
    return 1/(1 + np.exp(-x))

def get_cost(predict, class_labels):
    """
    计算对数几率损失
    """
    m = len(predict)
    err = 0
    for i in range(m):
        err -= np.log(sigmoid(predict[i] * class_labels[i]))
    return err

def  get_accuracy(predict, class_labels):
    """
    计算预测准确率
    """
    m = len(predict)
    all = 0
    err = 0
    for i in range(m):
        all += 1
        if float(predict[i]) < 0.5 and class_labels[i] == 1.0:
            err += 1
        elif float(predict[i]) >= 0.5 and class_labels[i] == -1.0:
            err += 1
        else:
            continue
    return 1 - err / all

def get_predict(data_mat, w0, w, v):
    """
    获得模型预测值
    """
    m = np.shape(data_mat)[0]
    result = []
    for x in range(m):
        inter_1 = data_mat[x] * v
        inter_2 = np.multiply(data_mat[x], data_mat[x]) * np.multiply(v, v)
        interaction = np.sum(np.multiply(inter_1, inter_1) - inter_2) / 2
        p = w0 + data_mat[x] * w + interaction
        pre = sigmoid(p[0, 0])
        result.append(pre)
    return result



def stochastic_gradient_descent(data_mat, class_labels, k, max_iter, alpha):
    """
    随机梯度下降法训练FM模型
    """
    m, n = np.shape(data_mat)
    print('sgd: ', m, n)
    # 初始化参数
    w = np.zeros((n, 1))
    w0 = 0
    v = init_v(n, k)

    # 训练
    for it in range(max_iter):
        # print('iteration: ', it)
        for x in range(m):
            inter_1 = data_mat[x] * v
            inter_2 = np.multiply(data_mat[x], data_mat[x]) * np.multiply(v, v)
            # 完整交叉项
            interaction = np.sum(np.multiply(inter_1, inter_1) - inter_2) / 2.
            # 完整公式
            p = w0 + data_mat[x] * w + interaction

            loss = sigmoid(class_labels[x] * p[0, 0]) - 1

            w0 = w0 - alpha * loss * class_labels[x]

            for i in range(n):
                if data_mat[x, i] != 0:
                    w[i, 0] = w[i, 0] - alpha * loss * class_labels[x] * data_mat[x, i]

                    for j in range(k):
                        v[i, j] = v[i, j] - alpha * loss * class_labels[x] * \
                                   (data_mat[x, i] * inter_1[0, j] - v[i, j] * data_mat[x, i] * data_mat[x, i])

        # 计算损失函数的值
        if it%100 == 0:
            print('---- iter: ', it, 'cost: ', get_cost(get_predict(np.mat(data_mat), w0, w, v), class_labels))

    # 返回FM模型参数
    return w0, w, v


if __name__ == '__main__':
    data_train, label_train = load_data('train_data.txt')
    print(np.shape(data_train), np.shape(label_train))
    w0, w, v = stochastic_gradient_descent(np.mat(data_train), label_train, 3, 1000, 0.01)
    print(w0, '\n',  w, '\n', v)
    predict_result = get_predict(np.mat(data_train), w0, w, v)
    print('res: ', predict_result)
    print('labels: ', label_train)
    print('training accuracy : ', get_accuracy(predict_result, label_train))
    import matplotlib.pyplot as plt
    a = data_train[:100]
    b = data_train[100:]
    x, x2=[] ,[]
    y , y2= [], []
    for i in a:
        x.append(i[0])
        y.append(i[1])
    plt.scatter(x, y)

    for i in b:
        x2.append(i[0])
        y2.append(i[1])

    plt.scatter(x2, y2)
    plt.show()

# 10000次
# 5.5073130564521495
#  [[20.09722282]
#  [32.64412017]]
#  [[ 3.28399832 -3.69718642 -4.96946997]
#  [ 3.28417658 -3.69704478 -4.96941944]]
# training accuracy :  0.99