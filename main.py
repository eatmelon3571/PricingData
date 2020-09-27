"""Main script for PricingData."""
import torch

import params
from data_pre_mnist.mnist_decode import decode_mnist_data_to_file
from data_pre_mnist.provider_data import mnist_allocation
from data_pre_cifar10.provider_data import cifar10_allocation
from data_pre_cifar10.cifar10_decode import decode_cifar10_data_to_file

from utils import create_dir, save_provider_model, get_net, load_provider_model
from utils import get_data_loader, softmax, all_softmax, cal_avg
from buyer_provider.data_provider import DataProvider

from core.collaborative_modelling import CollaborativeModelling, Original
from core.collaborative_modelling import ScoreAverage, show_pa_pb_pab
from core.score_avg import show_papbpab

# from excel import Excel


# 用于保证初始化模型一致
def creat_model():
    for i in range(params.provider_num):
        save_provider_model(i, get_net())


# 用于保证初始化模型一致
def load_model():
    dps = []
    for i in range(params.provider_num):
        net = load_provider_model(i)
        dataloader = get_data_loader(i)
        dps.append(DataProvider(net, dataloader))
    return dps


def show():

    txt_dir = params.dataset_division_testno + '/resualt.txt'

    no = 7
    outputs_papb_dir = params.dataset_division_testno + '/papb' + str(no) + '_softmax.npy'
    papb = torch.load(outputs_papb_dir)

    outputs_pab_dir = params.dataset_division_testno + '/pab' + str(no) + '_softmax.npy'
    pab = torch.load(outputs_pab_dir)

    outputs_fed_pab_dir = params.dataset_division_testno + '/fed_pab' + str(no) + '_softmax.npy'
    fed_pab = torch.load(outputs_fed_pab_dir)

    sum = 0
    l = int(len(papb) / 10000)

    for i in range(l):
        print('papb', papb[i * 10000])
        write_txt(papb[i * 10000], txt_dir)
        sum += papb[i * 10000]
    print('sum', sum)
    print('avg', sum / l)
    print('pab', pab[0])
    write_txt(pab[0], txt_dir)
    print('fed_pab', fed_pab[0])
    write_txt(fed_pab[0], txt_dir)



def write_txt(data, txt_dir):
    with open(txt_dir, "a") as f:  # 追加写
        for i in range(len(data)):
            f.write(str(data[i].item()) + " ")
        f.write("\n")


def write_xls():
    e = Excel()
    e.new_worksheet('mnistiid')

    no = 7

    outputs_papb_dir = params.dataset_division_testno + '/papb' + str(no) + '_softmax.npy'
    papb = torch.load(outputs_papb_dir)

    outputs_pab_dir = params.dataset_division_testno + '/pab' + str(no) + '_softmax.npy'
    pab = torch.load(outputs_pab_dir)

    outputs_fed_pab_dir = params.dataset_division_testno + '/fed_pab' + str(no) + '_softmax.npy'
    fed_pab = torch.load(outputs_fed_pab_dir)

    sum = 0
    l = int(len(papb) / 10000)

    for j in range(1):
        for i in range(l):
            e.add_data_papb(papb[i * 10000 + j])
            sum += papb[i * 10000 + j]
        e.add_data_pab(pab[0 + j])

        print('fed_pab', fed_pab.shape)
        e.add_data_fed_pab(fed_pab[0 + j])

    e.set_dir(params.dataset_division_testno + '/1.xls')
    e.save()




if __name__ == '__main__':
    # 创建所需文件夹
    create_dir()

    # cifar10数据文件（data_batch_1等）放在./data/cifar10/origin_data下
    # '''
    # params.test_no = 0     # 实验编号0
    if params.dataset == 'mnist':
        # 解码数据集
        # decode_mnist_data_to_file()
        # 分配数据集
        mnist_allocation()
        _ = 0
    elif params.dataset == 'cifar10':
        # decode_cifar10_data_to_file()
        cifar10_allocation()
        _ = 0
    # '''

    # cifar10 noniid 本地迭代次数要小一点

    # 先把一个网络初值存下来，然后每次都加载这个
    creat_model()



    # 原本的聚合方法：直接所有节点算SV  不用聚合树
    dps = load_model()
    Original(dps)


    # 协作建模
    dps = load_model()
    ScoreAverage(dps)

    # show()

    # cal_avg()

    # all_softmax()

    # write_xls()





