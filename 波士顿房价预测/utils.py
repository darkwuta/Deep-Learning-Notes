import numpy as np
def load_data(datafile):
    # 读入训练数据

    data = np.fromfile(datafile, sep=' ')# sep是区分每一个数据的标识，这里以空格为标记
    #print(data[1])

    # 读入之后的数据被转化成1维array，其中array的第0-13项是第一条数据，第14-27项是第二条数据，以此类推.... 
    # 这里对原始数据做reshape，变成N x 14的形式
    feature_names = [ 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE','DIS', 
                    'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV' ]#共14个
    feature_num = len(feature_names)
    data = data.reshape([data.shape[0] // feature_num, feature_num])# //是向下取整，这里是行列顺序

    # 查看数据
    # x = data[0]
    # print(x.shape)
    # print(x)

    # 数据集划分
    #print('data.rows=',data.shape[0],'   data.cols=',data.shape[1])
    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]
    test_data = data[offset:]

    # 数据归一化
    # 计算train数据集的最大值，最小值，平均值
    # axis指从那个维度取数据，0标识从列，1表示从行
    maximums, minimums, avgs = \
                        training_data.max(axis=0), \
                        training_data.min(axis=0), \
        training_data.sum(axis=0) / training_data.shape[0]
    # training_data.sum(axis=0)表示对每一列求和，平均值想要的是每列的平均值

    # 对数据进行归一化处理
    # maximums, minimums, avgs都是14列的行向量
    for i in range(feature_num):
        #print(maximums[i], minimums[i], avgs[i])
        data[:, i] = (data[:, i] - minimums[i]) / (maximums[i] - minimums[i])
    return training_data, test_data

