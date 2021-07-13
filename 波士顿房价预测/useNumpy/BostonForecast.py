# 波士顿房价预测

import json
import numpy as np
import matplotlib.pyplot as plt
import utils
from Network import Network


def main():
    datafile = '../DataBase/housing.data'
    train_data, test_data = utils.load_data(datafile)
    x = train_data[:, :-1]  # :-1取消最后一个
    y = train_data[:, -1:]  # -1:取最后一个
    # 查看数据
    #print(x[0])
    #print(y[0])

    #Y = WX + b
    net = Network(13)

    losses = net.train(train_data,test_data, num_epochs=100, batch_size=10, eta=0.1)

    # 画出损失函数的变化趋势
    plot_x = np.arange(len(losses))
    plot_y = np.array(losses)
    plt.plot(plot_x, plot_y)
    plt.show()

if __name__ == '__main__':
    main()