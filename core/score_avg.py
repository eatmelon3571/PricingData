import numpy as np
import torch

import params
from utils import get_net, get_test_dataloader, get_test_dataset



def score_avg(node_K_list):
    """计算价值v：平均类分数（测试集）后的，测试集精确度"""

    # 获得测试集数据
    dataset_test = get_test_dataset()
    images_test = dataset_test.images
    labels_test = dataset_test.labels

    server = Server()
    clients = []

    l = len(node_K_list)

    # 将所有叶节点加入训练
    for i in range(l):
        add_tree_to_list(node_K_list[i], clients)

    num_client = len(clients)
    print('客户端数量', num_client)

    outputs_list = None     # 用于记录outputs
    flag = True

    # 客户端测试获得类分数,传给服务器
    for j in range(num_client):
        outputs_temp = clients[j].get_outputs(images_test)
        # 记录outputs---------------------------------------
        if params.flag2:
            if flag:
                outputs_list = outputs_temp
                flag = False
            else:
                outputs_list = torch.cat((outputs_list, outputs_temp), 0)
            print('outputs_list.shape', outputs_list.shape)
        # 传给服务器
        server.add_outputs(outputs_temp)

    # 保存papb
    outputs_papb_dir = params.dataset_division_testno + '/papb' + str(params.no_papa_pab) + '.npy'
    save_outputs(outputs_list, outputs_papb_dir)

    # 服务器做平均
    print('服务器平均')
    outputs_avg = server.avg_outputs()

    # 保存pab
    outputs_pab_dir = params.dataset_division_testno + '/pab' + str(params.no_papa_pab) + '.npy'
    save_outputs(outputs_avg, outputs_pab_dir)

    test_acc = server.test_with_outputs(outputs_avg, labels_test)

    # 返回测试精度 作为v
    return server.net, test_acc


def save_outputs(outputs_list, outputs_dir):
    torch.save(outputs_list, outputs_dir)


def show_papbpab():
    outputs_papb_dir = params.dataset_division_testno + '/papb63.npy'
    papb = torch.load(outputs_papb_dir)

    outputs_pab_dir = params.dataset_division_testno + '/pab63.npy'
    pab = torch.load(outputs_pab_dir)

    sum = 0
    for i in range(6):
        print('papb', papb[i * 10000])
        sum += papb[i * 10000]
    print('sum', sum)
    print(sum / 6)
    print('pab', pab[0])


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
        self.clients_outputs = None  # 累计网络输出outputs
        self.client_num = 0  # 累计网络输出outputs时，客户端的数据量累计，用作网络权重的权

    def get_net_w(self):
        return self.net.get_w()

    def update_net_w(self, w):
        self.net.update_w(w)

    def add_outputs(self, outputs):
        """
        增加一个客户端的网络输出outputs
        :param outputs: 客户端网络输出outputs
        :return:
        """
        if self.client_num == 0:  # 本次平均，累加第一个客户端（self.w为空）
            self.clients_outputs = outputs
        else:
            self.clients_outputs += outputs
        self.client_num += 1

    def clear_outputs(self):
        """
        清空累计的权重
        """
        self.clients_outputs = None
        self.client_num = 0

    def avg_outputs(self):
        """
        平均客户端outputs
        （清空累计outputs的变量）
        返回平均的outputs
        """
        # 做平均
        outputs = torch.div(self.clients_outputs, self.client_num)
        self.clear_outputs()
        # 清空计算的    以便下一次计算
        return outputs

    @torch.no_grad()
    def test_with_outputs(self, outputs_test, labels_test):
        """用平均的outputs测试"""
        # 判断cuda是否可用
        if torch.cuda.is_available():
            labels_test = labels_test.cuda()

        predicts = torch.max(outputs_test.data, 1)[1]

        print('torch.max(outputs_test.data, 1)', len(torch.max(outputs_test.data, 1)))

        correct = (predicts == labels_test).sum()
        total = len(labels_test)
        print('total', total, 'correct', correct)
        p = 1.0 * correct / total
        print('Accuracy: %.4f' % p)
        return p