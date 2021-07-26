import paddle
from paddle.nn import Linear#组网相干的API
import paddle.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt

class MNIST(paddle.nn.Layer):
    def __init__(self):
        super(MNIST, self).__init__()
        
        # 定义一层全连接层，输出维度是1
        self.fc = paddle.nn.Linear(in_features=784, out_features=1)
        
    # 定义网络结构的前向计算过程
    def forward(self, inputs):
        outputs = self.fc(inputs)
        return outputs