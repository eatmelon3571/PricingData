import os
import torch
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import params
from module.net_mnist import NetMNIST
from module.net_cifar import NetCIFAR10
from datasets.mnist import get_provider_mnist, get_test_mnist, get_test_dataset_mnist
from datasets.cifar10 import get_provider_cifar10, get_test_cifar10, get_test_dataset_cifar10


def mkdir(dir):
    """创建文件夹"""
    isExists = os.path.exists(dir)
    if not isExists:
        os.makedirs(dir)


def create_dir():
    mkdir(params.root_dir)
    mkdir(params.dataset_dir)
    mkdir(params.dataset_npy_data)
    mkdir(params.dataset_division)
    mkdir(params.dataset_division_testno)
    mkdir(params.dataset_division_testno_save)


def get_net():
    """获得随机初始值网络"""
    net = None
    if params.dataset == 'mnist':
        net = NetMNIST()
    elif params.dataset == 'cifar10':
        net = NetCIFAR10()

    # 判断cuda是否可用
    if torch.cuda.is_available():
        # print('cuda可用')
        net.cuda()
    return net


def make_variable(tensor, volatile=False):
    """Convert Tensor to Variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor, volatile=volatile)


def get_data_loader(provider_no):
    """加载编号 provider_no 的 dataloader"""
    if params.dataset == "mnist":
        return get_provider_mnist(provider_no)
    elif params.dataset == "cifar10":
        return get_provider_cifar10(provider_no)


def get_test_dataloader():
    if params.dataset == "mnist":
        return get_test_mnist()
    elif params.dataset == "cifar10":
        return get_test_cifar10()


def get_test_dataset():
    if params.dataset == "mnist":
        return get_test_dataset_mnist()
    elif params.dataset == "cifar10":
        return get_test_dataset_cifar10()


def get_optimizer(net):
    if params.dataset == "mnist":
        return optim.SGD(net.parameters(),
                         lr=params.learning_rate_mnist,
                         momentum=params.momentum,
                         weight_decay=1e-4)
    elif params.dataset == "cifar10":
        return optim.Adam(net.parameters(),
                          lr=params.learning_rate_cifar10,
                          eps=1e-8,
                          betas=(params.beta1, params.beta2))


def load_provider_model(provider_no):
    """加载编号 provider_no 的 模型"""
    net = get_net()
    # if params.round_start > 0:  # 不为0说明不是第一次，有存储的模型，就加载一下
    provider_i_dir = params.dataset_division_testno + '/provider' + str(provider_no)
    # 根据 当前的轮数 命名 模型文件
    provider_i_model_dir = provider_i_dir + '/model.npy'
    net.load_state_dict(np.load(provider_i_model_dir, allow_pickle=True).item())
    # 判断cuda是否可用
    if torch.cuda.is_available():
        # print('cuda可用')
        net.cuda()
    return net


def save_provider_model(provider_no, net):
    """保存编号 provider_no 的 模型"""
    provider_i_dir = params.dataset_division_testno + '/provider' + str(provider_no)
    # 根据 当前的轮数 命名 模型文件
    provider_i_model_dir = provider_i_dir + '/model.npy'
    np.save(provider_i_model_dir, net.get_w())


# 对存储的outputs进行softmax
def softmax(in_dir, out_dir):
    outputs = torch.load(in_dir)
    # softmax
    temp = torch.nn.functional.softmax(outputs.data, dim=1)
    torch.save(temp, out_dir)


def all_softmax():
    for i in range(1, 64):
        in_dir = params.dataset_division_testno + '/pab' + str(i) + '.npy'
        out_dir = params.dataset_division_testno + '/pab' + str(i) + '_softmax.npy'
        softmax(in_dir, out_dir)

    for i in range(1, 64):
        in_dir = params.dataset_division_testno + '/papb' + str(i) + '.npy'
        out_dir = params.dataset_division_testno + '/papb' + str(i) + '_softmax.npy'
        softmax(in_dir, out_dir)

    for i in range(1, 64):
        in_dir = params.dataset_division_testno + '/fed_pab' + str(i) + '.npy'
        out_dir = params.dataset_division_testno + '/fed_pab' + str(i) + '_softmax.npy'
        softmax(in_dir, out_dir)


def cal_avg():
    for i in range(1, 64):
        outputs_papb_dir = params.dataset_division_testno + '/papb' + str(i) + '_softmax.npy'
        papb = torch.load(outputs_papb_dir)

        sum = papb[0:10000]
        l = int(len(papb) / 10000)
        for i in range(1, l):
            start = i * 10000
            sum += papb[start:start + 10000]
            print(sum.shape)
            print('sum[0]', sum[0])
        pab = sum / l

        print('sum', sum)
        print('avg', sum / l)
        print('pab', pab[0])

        outputs_pab_dir = params.dataset_division_testno + '/pab' + str(i) + '_softmax.npy'
        torch.save(pab, outputs_pab_dir)