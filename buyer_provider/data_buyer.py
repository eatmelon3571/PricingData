import torch
import numpy as np

# 我的
import params
from module.net_mnist import NetMNIST
from core.shapley_value import ShapleyValue


class DataBuyer:
    def __init__(self, net):
        # 任务T
        self.v = None
        self.miu = params.miu
        self.B = params.B

        #
        self.net = net
        # self.lr = params.learning_rate  # 学习率
        self.provider_num = 0  # DataProvider的数据量累计
        self.dps_w = []       # 记录多个DataProvider的权重

        self.shapleyValue = ShapleyValue()

    def get_net_w(self):
        return self.net.get_w()

    def update_net_w(self, w):
        self.net.update_w(w)

    def calculate_characterisic_function(self, node_K_list):
        self.shapleyValue.cal_SV_all(node_K_list)
        return self.shapleyValue.SV_all


    def get_lr(self):
        return self.lr

    def learning_rate_decay(self):
        self.lr *= 0.99  # FedAvg  CIFAR-10



    def clear_w(self):
        """
        清空累计的权重
        """
        self.dps_w.clear()
        self.provider_num = 0

    def add_w(self, dp_w):
        self.dps_w.append(dp_w)
        self.provider_num += 1

    def save_model(self, model_dir):
        """
        保存模型
        """
        np.save(model_dir, self.net.get_w())

    def load_model(self, model_dir):
        """
        加载模型
        """
        self.update_net_w(np.load(model_dir, allow_pickle=True).item())

    def save_lr(self, lr_dir):
        """
        保存学习率
        """
        np.save(lr_dir, self.lr)

    def load_lr(self, lr_dir):
        """
        加载学习率
        """
        self.lr = np.load(lr_dir)

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
        print('Accuracy: %.2f' % (100 * correct / total))