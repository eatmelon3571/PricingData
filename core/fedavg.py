import numpy as np
import torch

import params
from utils import get_net, get_test_dataloader


def fedavg(node_K_list, fed_train_time=params.fed_train_time):
    return get_net(), 0   # 测试用
    test_dataloader = get_test_dataloader()
    test_acc = 0

    server = Server()
    clients = []

    l = len(node_K_list)

    # 将所有叶节点加入训练
    for i in range(l):
        # clients.append(node_K_list[i].provider)
        add_tree_to_list(node_K_list[i], clients)

    num_client = len(clients)
    print('训练数量', num_client)

    # 训练
    for i in range(fed_train_time):
        print('第', i, '次训练')
        # 客户端训练
        for j in range(num_client):
            # print('开始训练的客户端编号:', j)
            clients[j].train()
        # 客户端传权值给服务器
        for j in range(num_client):
            cli_w = clients[j].get_net_w()
            server.add_w(cli_w)
        # 服务器做平均
        print('服务器平均')
        server.avg()

        # 服务器传模型、学习率给客户端
        ser_w = server.get_net_w()
        # ser_lr = server.get_lr()
        # print('本轮学习率:', ser_lr)
        for j in range(num_client):
            clients[j].update_net_w(ser_w)
            # clients[j].lr = ser_lr

        test_acc = server.test(test_dataloader)

    # 返回测试精度 作为v
    return server.net, test_acc


def add_tree_to_list(root, clients):
    # print('len(root.children)', len(root.children))
    if len(root.children) == 0:
        # 叶节点加入
        clients.append(root.provider)
        print('加入的节点编号', root.p_no)
        return
    else:
        for c in root.children:
            add_tree_to_list(c, clients)


class Server:
    def __init__(self):
        self.net = get_net()
        # self.lr = params.learning_rate           # 学习率
        self.w = {}             # 累计网络权重
        self.client_num = 0     # 累计网络权重时，客户端的数据量累计

    def get_net_w(self):
        return self.net.get_w()

    def update_net_w(self, w):
        self.net.update_w(w)

    def add_w(self, client_w):
        """
        增加一个客户端的网络权重（按客户端数据量大小比例）
        这么算增加得多了感觉会溢出
        :param client_w: 客户端权重
        """
        if not bool(self.w):   # 本次平均，累加第一个客户端（self.w为空）
            for key, values in client_w.items():
                self.w[key] = values
        else:
            for key, values in client_w.items():
                self.w[key] += values
        self.client_num += 1

    def clear_w(self):
        """
        清空累计的权重
        """
        self.w.clear()
        self.client_num = 0

    def avg(self):
        """
        平均之前先按权重累加（self.add）了很多客户端权重
        平均后更新服务器模型
        （清空累计权重的变量）
        """
        # 做平均
        for key, values in self.w.items():
            self.w[key] = torch.div(self.w[key], self.client_num)
        # 更新权重
        self.update_net_w(self.w)
        # 清空计算的    以便下一次计算
        self.clear_w()

    # 测试
    @torch.no_grad()
    def test(self, dataloader):
        self.net.eval()
        correct = 0
        test_loss = 0
        total = 0
        for i, (images, labels) in enumerate(dataloader):
            if torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()

            outputs = self.net(images)
            # test_loss += criterion(outputs, labels).data.item()  # .data.item()取得张量的第一个数?

            predicts = torch.max(outputs.data, 1)[
                1]  # _输出的是最大概率的值，predicts输出的是最大概率值所在位置，max()函数中的1表示维度，意思是计算某一行的最大值
            correct += (predicts == labels).sum()
            total += len(images)
        print('total', total)
        p = 1.0 * correct / total
        print('Accuracy: %.4f' % p)

        return p