"""Main script for PricingData."""
import torch

import params
from data_pre_mnist.mnist_decode import decode_mnist_data_to_file
from data_pre_mnist.provider_data import mnist_allocation
from data_pre_cifar10.provider_data import cifar10_allocation
from data_pre_cifar10.cifar10_decode import decode_cifar10_data_to_file

from utils import create_dir, save_provider_model, get_net, load_provider_model
from utils import get_data_loader, softmax, all_softmax, cal_avg, get_test_dataset
from buyer_provider.data_provider import DataProvider

from core.collaborative_modelling import CollaborativeModelling, Original
from core.collaborative_modelling import ScoreAverage, show_pa_pb_pab
from core.score_avg import show_papbpab

# from excel import Excel
from show_resault import Show


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
    test_data = get_test_dataset()
    labels_test = test_data.labels

    e = Excel()
    e.new_worksheet('cifar10iid')

    no = 31

    outputs_papb_dir = params.dataset_division_testno + '/papb' + str(no) + '.npy'
    papb = torch.load(outputs_papb_dir)

    outputs_pab_dir = params.dataset_division_testno + '/pab' + str(no) + '.npy'
    pab = torch.load(outputs_pab_dir)

    outputs_fed_pab_dir = params.dataset_division_testno + '/fed_pab' + str(no) + '.npy'
    fed_pab = torch.load(outputs_fed_pab_dir)

    sum = 0
    l = int(len(papb) / 10000)

    for j in range(100):
        # 将标签数据构造好
        print('labels_test[0]', labels_test[j])
        label_write_data = torch.zeros(10)
        label_write_data[labels_test[j]] = 1
        e.add_data_papb(label_write_data)
        '''for i in range(l):
            e.add_data_papb(papb[i * 10000 + j])
            sum += papb[i * 10000 + j]'''
        e.add_data_pab(pab[0 + j])

        print('fed_pab', fed_pab.shape)
        e.add_data_fed_pab(fed_pab[0 + j])

    e.set_dir(params.dataset_division_testno + '/1.xls')
    e.save()


def print_acc(pab):
    test_data = get_test_dataset()
    labels_test = test_data.labels

    if torch.cuda.is_available():
        labels_test = labels_test.cuda()

    predicts = torch.max(pab.data, 1)[1]

    print('torch.max(outputs_test.data, 1)', len(torch.max(pab.data, 1)))

    correct = (predicts == labels_test).sum()
    total = len(labels_test)
    print('total', total, 'correct', correct)
    p = 1.0 * correct / total
    print('Accuracy: %.4f' % p.item())

    return p.item()


def check_acc(e: Excel, no):
    # no = 63

    outputs_papb_dir = params.dataset_division_testno + '/papb' + str(no) + '.npy'
    papb = torch.load(outputs_papb_dir)

    outputs_pab_dir = params.dataset_division_testno + '/pab' + str(no) + '.npy'
    pab = torch.load(outputs_pab_dir)

    outputs_fed_pab_dir = params.dataset_division_testno + '/fed_pab' + str(no) + '.npy'
    fed_pab = torch.load(outputs_fed_pab_dir)

    l = int(len(papb) / 10000)

    p_list = []
    for i in range(l):
        p = print_acc(papb[i * 10000: i * 10000 + 10000])
        p_list.append(p)


    acc_pab = print_acc(pab)

    acc_fedpab = print_acc(fed_pab)

    e.add_acc(p_list, acc_pab, acc_fedpab)



def acc_to_xls():
    e = Excel()
    e.new_worksheet('mnistiid')

    for no in range(1, 63):
        check_acc(e, no)

    e.save()


def show_plot():
    x_data = []
    pab_data = []
    fed_data = []

    s = Show()

    for no in range(1, 63):

        outputs_pab_dir = params.dataset_division_testno + '/pab' + str(no) + '.npy'
        pab = torch.load(outputs_pab_dir)

        outputs_fed_pab_dir = params.dataset_division_testno + '/fed_pab' + str(no) + '.npy'
        fed_pab = torch.load(outputs_fed_pab_dir)

        acc_pab = print_acc(pab)

        acc_fedpab = print_acc(fed_pab)

        x_data.append(no)
        pab_data.append(acc_pab)
        fed_data.append(acc_fedpab)

    s.add_data(x_data, pab_data, "pab", s.blue)
    s.add_data(x_data, fed_data, "fed", s.orange)

    s.set_title("cifar10 iid")
    s.show_plot()


def shishi():
    client_num = 3
    outputs_list = [torch.rand(10), torch.rand(10), torch.rand(10)]
    print("outputs")
    print(outputs_list)

    var_list = []
    m = 1  # 方差放大倍率
    for i in range(client_num):
        # 先计算softmax，要先计算softmax吗？
        temp = torch.nn.functional.softmax(outputs_list[i].data, dim=0)
        # 将方差进行放大
        temp *= m
        # 首先计算每个outputs的方差, 方差拼起来
        var_list.append(torch.var(temp))
    print("var_list")
    print(var_list)
    # 放大后的方差作为平均时的权重
    outputs_avg = None
    for i in range(client_num):
        if i == 0:
            outputs_avg = outputs_list[i] * var_list[i]
        else:
            outputs_avg += outputs_list[i] * var_list[i]
    print("avg")
    print(outputs_avg)
    return outputs_avg


if __name__ == '__main__':
    # 创建所需文件夹
    # create_dir()

    # cifar10数据文件（data_batch_1等）放在./data/cifar10/origin_data下
    # '''
    # params.test_no = 0     # 实验编号0
    if params.dataset == 'mnist':
        # 解码数据集
        # decode_mnist_data_to_file()
        # 分配数据集
        # mnist_allocation()
        _ = 0
    elif params.dataset == 'cifar10':
        # decode_cifar10_data_to_file()
        # cifar10_allocation()
        _ = 0
    # '''

    # cifar10 noniid 本地迭代次数要小一点

    # 先把一个网络初值存下来，然后每次都加载这个
    # creat_model()



    # 原本的聚合方法：直接所有节点算SV  不用聚合树
    # dps = load_model()
    # Original(dps)


    # 协作建模
    dps = load_model()
    ScoreAverage(dps)

    # show()

    # cal_avg()

    # all_softmax()

    # write_xls()

    # check_acc()

    # acc_to_xls()
    # show_plot()

    # shishi()



