import random
# import xlwt
import numpy as np
import torch

import params
from buyer_provider.data_buyer import DataBuyer
from buyer_provider.data_provider import DataProvider
from buyer_provider.third_party import ThirdParty
from core.tree import Tree
from utils import get_net, get_data_loader
from core.fedavg import fedavg
from core.shapley_value import ShapleyValue


def Original(dps):
    """原本聚合方式：FedAvg   +计算SV"""
    shapleyValue = ShapleyValue()

    db = DataBuyer(get_net())
    '''dps = []
    for i in range(params.provider_num):
        net = get_net()
        dataloader = get_data_loader(i)
        dps.append(DataProvider(net, dataloader))'''

    # 构成树节点 放入tree_list
    tree_list = []
    for i in range(params.provider_num):
        tree_list.append(Tree(i, dps[i]))


    # 预训练
    # _, p_fed = fedavg(tree_list)


    num_node = len(tree_list)

    # 先在本地数据集上训练至收敛----------------

    for i in range(params.provider_num):
        print("客户端", i, "预训练")
        for j in range(params.local_time):
            tree_list[i].provider.train()
    #
    print('开始计算SV')

    shapleyValue.v_way = 'fedavg'     # 计算v的方式fedavg和score_avg

    SVs = shapleyValue.cal_SV_all(tree_list)

    # 所有聚合后pab
    v_all = shapleyValue.v_all

    for i in range(num_node):
        tree_list[i].sv = SVs[i]
        print(SVs[i])

    # 最后剩一个节点为根
    root = tree_list[0]
    root.B = db.B
    # 根据树分配B
    all_B(root)

    # 根节点精确度
    p_root = shapleyValue.root_p

    # SV写入txt
    txt_dir = params.dataset_division_testno + '/21.txt'
    write_txt(tree_list, 0, p_root, txt_dir)
    # 把v_all写入txt
    v_all = shapleyValue.v_all
    print(v_all)
    npy_dir = params.dataset_division_testno + '/Original_v_all_2.npy'
    write_npy_v_all(v_all, npy_dir)
    # 第三方解密，发送结果给DP、DB
    return tree_list


def ScoreAverage(dps):
    """非树状聚合方式（即一起聚合）    计算SV方式：平均outputs在测试集上精度"""
    shapleyValue = ShapleyValue()
    tp = ThirdParty()

    db = DataBuyer(get_net())
    '''dps = []
    for i in range(params.provider_num):
        net = get_net()
        dataloader = get_data_loader(i)
        dps.append(DataProvider(net, dataloader))
    print('读取模型完成')'''

    # 随机聚合顺序
    '''order_rand = random_order(params.provider_num)
    print('聚合顺序', order_rand)
    '''

    # 构成树节点 放入tree_list
    tree_list = []

    for i in range(params.provider_num):
        tree_list.append(Tree(i, dps[i]))
    # 先在本地数据集上训练至收敛----------------

    # '''
    for i in range(params.provider_num):
        print("客户端", i, "预训练")
        for j in range(params.local_time):
            tree_list[i].provider.train()
    # '''
    # 计算SV-------------------
    print('开始计算SV')

    shapleyValue.v_way = 'score_avg'  # 计算v的方式fedavg和score_avg

    SVs = shapleyValue.cal_SV_all(tree_list)

    print("算得各个SV值：")
    for i in range(params.provider_num):
        tree_list[i].sv = SVs[i]
        print(SVs[i])

    # 找出SV>0的聚合-----------------
    positive_list = []
    for i in range(params.provider_num):
        if SVs[i] > 0:
            print(i, "SV>0并加入")
            positive_list.append(tree_list[i])

    net, acc = fedavg(positive_list, 100)

    print("聚合后精度", acc)

    # 写入txt
    txt_dir = params.dataset_division_testno + '/22.txt'
    write_txt(tree_list, 0, acc, txt_dir)
    # 把v_all写入txt
    v_all = shapleyValue.v_all
    print(v_all)
    npy_dir = params.dataset_division_testno + '/ScoreAverage_v_all_2.npy'
    write_npy_v_all(v_all, npy_dir)


def CollaborativeModelling(_tree_list=None):
    """树状聚合方式          +计算SV"""
    shapleyValue = ShapleyValue()
    tp = ThirdParty()

    db = DataBuyer(get_net())
    dps = []
    for i in range(params.provider_num):
        net = get_net()
        dataloader = get_data_loader(i)
        dps.append(DataProvider(net, dataloader))
    print('读取模型完成')

    # 随机聚合顺序
    '''order_rand = random_order(params.provider_num)
    print('聚合顺序', order_rand)
    '''

    # 构成树节点 放入tree_list
    tree_list = []

    if _tree_list is not None:
        for i in range(params.provider_num):
            tree_list.append(_tree_list[i])
    else:
        for i in range(params.provider_num):
            tree_list.append(Tree(i, dps[i]))

    """# 第三方生成密匙，传给DP、DB
    public_key, private_key = tp.generate_key()
    # DP加密model，发给DB
    for i in range(params.provider_num):
        dps[i].enctypt()
    # 聚合前先FedAvg   p_fed为fedavg的精度
    _, p_fed = fedavg(tree_list)
    """


    # 开始多次FedAvg、聚合
    last_node = tree_list[0]         # 上一次最优节点
    next_node_no = 1                 # 接下来要聚合的开始节点编号

    node_K_list = [last_node]

    while next_node_no < params.provider_num:
        # print('len(node_K_list[0].children)', len(node_K_list[0].children))
        # 要聚合的K个节点
        num = 0
        while num < params.K - 1 and next_node_no < params.provider_num:
            node_K_list.append(tree_list[next_node_no])
            next_node_no += 1
            num += 1

        # K个provider聚合  可能不足K个
        num_node = len(node_K_list)
        # DB计算特征函数v，发送给第三方
        print('开始计算SV')
        SVs = shapleyValue.cal_SV_all(node_K_list)

        for i in range(num_node):
            node_K_list[i].sv = SVs[i]
            print(SVs[i])

        # 判断是否聚合
        num_aggregation = 0
        for i in range(num_node):
            if node_K_list[i].if_aggregation():
                num_aggregation += 1
        if num_aggregation == num_node:   # 全部同意聚合
            # 用聚合的模型建树
            net = shapleyValue.root_net
            # 暂时用第一个孩子的dataloader做聚合节点的dataloader
            p = DataProvider(net, dataloader=get_data_loader(node_K_list[0].p_no))
            node = Tree(node_K_list[0].p_no, p)
            for i in range(num_node):
                node.children.append(node_K_list[i])
            # 记录上一次聚合的节点
            node_K_list = [node]
        else:
            # 选出SV最大的节点做根
            max_node = node_K_list[0]
            max_sv = node_K_list[0].sv
            for i in range(1, num_node):
                if node_K_list[i].sv > max_sv:
                    max_node = node_K_list[i]
                    max_sv = node_K_list[i].sv
            # 记录上一次最优的节点
            node_K_list = [max_node]



    # 最后剩一个节点为根
    root = node_K_list[0]
    root.B = db.B
    # 根据树分配B
    all_B(root)

    # 根节点精确度
    p_root = shapleyValue.root_p

    # 写入txt
    txt_dir = params.dataset_division_testno + '/10.txt'
    write_txt(tree_list, 0, p_root, txt_dir)
    # 第三方解密，发送结果给DP、DB


'''
def write_excel(tree_list, p_fed, p_root):
    workbook = xlwt.Workbook(encoding='ascii')
    worksheet = workbook.add_sheet('My Worksheet')

    worksheet.write(0, 0, 'id')  # 不带样式的写入
    worksheet.write(1, 0, 'sv')
    worksheet.write(2, 0, 'B')
    for i in range(params.provider_num):
        worksheet.write(0, i + 1, 'provider' + str(i))
        worksheet.write(1, i + 1, str(tree_list[i].sv))
        worksheet.write(2, i + 1, str(tree_list[i].B))

    worksheet.write(4, 0, 'fedavg下精确度')
    worksheet.write(5, 0, str(p_fed))
    worksheet.write(4, 1, '协作建模下精确度')
    worksheet.write(5, 1, str(p_root))

    workbook.save(params.excel_dir)  # 保存文件
'''


def write_txt(tree_list, p_fed, p_root, txt_dir=params.txt_dir):
    with open(txt_dir, "w") as f:
        f.write("id        sv        B\n")
        for i in range(params.provider_num):
            f.write('provider' + str(i) + "                       ")
            f.write(str(tree_list[i].sv) + "                          ")
            f.write(str(tree_list[i].B))
            f.write("\n")

        f.write('fedavg下精确度\n')
        f.write(str(p_fed) + "\n")
        f.write('协作建模下精确度\n')
        f.write(str(p_root) + "\n")


def write_npy_v_all(v_all, npy_dir):
    np.save(npy_dir, v_all)


def show_pa_pb_pab():
    npy1 = params.npy_original_v_all
    npy2 = params.npy_ScoreAverage_v_all
    arr1 = np.load(npy1)
    arr2 = np.load(npy2)

    n = params.provider_num

    print(arr2[1], arr2[2], arr1[3])


def all_B(root):
    """给子节点分配B"""
    l = len(root.children)
    if l == 0:
        return
    sum_SV = 0
    for c in root.children:
        print('c.sv', c.sv)
        sum_SV += c.sv

    if sum_SV != 0:
        for c in root.children:
            c.B = root.B * c.sv / sum_SV
            all_B(c)
    else:
        for c in root.children:   # 为避免除0  平均分
            c.B = root.B / l
            all_B(c)


def random_order(num):
    # 在[0,num)内生成num个不重复的整数 即随机顺序
    return random.sample(range(0, num), num)


def avg_w(node_K_list):
    l = len(node_K_list)
    w = {}
    for key, values in node_K_list[0].provider.get_net_w().items():
        w[key] = values
    for i in range(1, l):
        for key, values in node_K_list[i].provider.get_net_w().items():
            w[key] += values

    for key, values in w.items():
        w[key] = torch.div(w[key], l)
    return w



if __name__ == '__main__':
    CollaborativeModelling()