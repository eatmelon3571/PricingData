import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

import params
from utils import get_net, get_optimizer


class DataProvider:
    def __init__(self, net, dataloader):
        """
        :param net: 客户端网络
        :param dataloader: 训练数据集
        """
        self.net = net
        self.dataloader = dataloader
        # self.lr = params.learning_rate  # maybe change

    def get_net_w(self):
        return self.net.get_w()

    def update_net_w(self, w):
        self.net.update_w(w)

    def enctypt(self):
        """加密模型"""

    def update_lr(self, lr):
        self.lr = lr

    def copy(self):
        w = self.get_net_w()
        net = get_net()
        net.load_state_dict(w)
        return DataProvider(net, self.dataloader)

    def train(self):
        self.net.train()
        """
        正常的训练过程  用客户端数据训练
        """
        # optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=params.momentum)
        optimizer = get_optimizer(self.net)

        criterion = nn.CrossEntropyLoss()  # 计算损失

        for epoch in range(params.epochnum):
            loss = 0
            for i, (images, labels) in enumerate(self.dataloader):
                # 判断cuda是否可用
                if torch.cuda.is_available():
                    images, labels = images.cuda(), labels.cuda()

                optimizer.zero_grad()
                outputs = self.net(images)

                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()  # 优化参数
                '''
                if i % 300 == 0:
                    print('current loss : %.5f' % loss.data.item())
                # '''
            print('current loss : %.5f' % loss.data.item())

    # 测试
    @torch.no_grad()
    def test(self, dataloader):
        """测试并返回精度"""
        self.net.eval()

        correct = 0
        total = 0
        for i, (images, labels) in enumerate(dataloader):
            # 判断cuda是否可用
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()

            outputs = self.net(images)
            predicts = torch.max(outputs.data, 1)[
                1]  # _输出的是最大概率的值，predicts输出的是最大概率值所在位置，max()函数中的1表示维度，意思是计算某一行的最大值
            correct += (predicts == labels).sum()
            total += len(images)
        print('total', total, 'correct', correct)
        p = 1.0 * correct / total
        print('Accuracy: %.8f' % p)
        return p

    def save_model(self, model_dir):
        np.save(model_dir, self.net.get_w())

    def load_model(self, model_dir):
        self.update_net_w(np.load(model_dir, allow_pickle=True).item())

    #
    @torch.no_grad()
    def get_outputs(self, images_test):
        """测试并返回所有outputs（类分数）"""
        self.net.eval()

        # 判断cuda是否可用
        if torch.cuda.is_available():
            images_test = images_test.cuda()

        outputs = self.net(images_test)
        return outputs.clone().detach()  # 切断梯度


