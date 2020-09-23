import math
from core.fedavg import fedavg
from core.score_avg import score_avg
import params


class ShapleyValue:

    def __init__(self):
        self.SV_all = None  # 保存SV计算结果
        self.v_all = None   # 保存子集的v值

        self.v_way = 'fedavg'   # 计算v的方式


        self.root_net = None    # 聚合后的根节点
        self.root_p = 0         # 根节点的精确度



    def cal_SV_all(self, node_K_list):
        """算provider_num 个SV"""
        num_M = len(node_K_list)
        self.SV_all = [0] * num_M

        trans_num_M = self.trans(num_M)
        self.v_all = [0] * (trans_num_M + 1)    # 子集共2^num_M-1个
        # print('trans_num_M, self.v_all', trans_num_M, self.v_all)
        j = trans_num_M

        # 遍历所有子集,计算子集的v   但是跳过单个节点？   节点要复制一份再训练
        while j != 0:
            #   j的二进制的每一位，1代表选，0代表不选
            print('当前子集')  # #
            self.print_bin(j, num_M)  # #
            # 计算一个子集S的v  存到v_all里
            net, v_S = self.cal_v_S(j, node_K_list)

            if j == trans_num_M:
                self.root_net = net
                self.root_p = v_S

            # 下次循环用，下一个子集
            j = (j - 1) & trans_num_M

        # 计算每个provider的SV
        for i in range(num_M):
            self.SV_all[i] = self.SV_i(i, num_M)

        return self.SV_all

    def SV_i(self, i, num_M):
        """
        """
        fai = 0

        trans_i = int(math.pow(2, i))

        M_reduce_i_index = self.generate_index(num_M, i)
        self.print_bin(M_reduce_i_index, num_M)  # #
        j = M_reduce_i_index

        # 遍历所有子集
        while j != 0:
            #   j的二进制的每一位，1代表选，0代表不选
            # print('当前子集')    # #
            # self.print_bin(j, num_M)    # #
            # 计算v(S∪{i})-v(S)
            # print('j , trans_i', j, trans_i)
            derta = self.v_all[j + trans_i] - self.v_all[j]

            num_S = self.num_j(j, num_M)
            fai += derta * self.factorial_of_derta(num_S, num_M)
            # 下次循环用，下一个子集
            j = (j - 1) & M_reduce_i_index
        print('fai', fai)
        return fai

    def factorial_of_derta(self, num_S, num_M):
        """计算系数"""
        a = math.factorial(num_S)
        b = math.factorial(num_M - num_S - 1)
        c = math.factorial(num_M)
        return a * b / c

    def cal_v_S(self, index, M):
        """
        计算v(S)
        S为由index标记的全集M
        """
        num_M = len(M)
        num_S = 0
        temp = index
        # 依次判断第i个对象是否在该集合中
        S = []
        for i in range(num_M):
            i_if_choose = (temp % 2 == 1)
            # print('index, i, i_if_choose', index, i, i_if_choose)    # #
            if i_if_choose:
                # print('add', i)
                S.append(M[i].copy())
                num_S += 1
            # 更新以判断下一个
            temp = int(temp / 2)

        # 根据index记录v(S)

        params.no_papa_pab = index
        print('index', index)

        net, v_S = self.choose_v(S)

        self.v_all[index] = v_S

        return net, v_S

    def set_v_choose(self, way):
        self.v_way = way

    def choose_v(self, S):
        if self.v_way == 'fedavg':
            return self.v_fedavg(S)
        elif self.v_way == 'score_avg':
            return self.v_score_avg(S)

    def v_fedavg(self, S):
        """计算价值v：fedavg聚合后的测试集精确度"""
        # 聚合时少训练点
        time = params.v_S_fed_train_time
        print('聚合时fedavg训练次数少一点', time)
        net, v_S = fedavg(S, fed_train_time=time)
        return net, v_S

    def v_score_avg(self,S):
        """计算价值v：平均类分数（测试集）后的，测试集精确度"""
        net, v_S = score_avg(S)
        return net, v_S

    def trans(self, n):
        return int(math.pow(2, n) - 1)

    def generate_index(self, sum, i):
        """一共sum个1，第i个为0   sum=3 i=1  M_='101'（二进制） """
        M_reduce_i_index = 0
        j = sum - 1             # 0到sum-1
        while j >= 0:
            M_reduce_i_index *= 2
            if j == i:
                M_reduce_i_index += 0
            else:
                M_reduce_i_index += 1
            j -= 1
        return M_reduce_i_index

    def num_j(self, j, num_M):
        num = 0
        for i in range(num_M):
            if j % 2 == 1:
                num += 1
            # 更新以判断下一个
            j = int(j / 2)
        return num

    def print_bin(self, i, num):
        """
        二进制输出i，右对齐，用0补足K位
        """
        # 右对齐 补0
        f = '{:0>' + str(num) + '}'
        print(f.format(format(i, 'b')))



if __name__ == '__main__':
    s = ShapleyValue()


