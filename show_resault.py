import random

from sklearn import decomposition
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl


class Show:
    def __init__(self):
        self.legend = []      # 记录图例

        # 解决中文显示问题
        mpl.rcParams['font.sans-serif'] = [u'SimHei']
        mpl.rcParams['axes.unicode_minus'] = False

        self.orange = '#FFA54F'
        self.blue = '#00BFFF'
        self.green = '#32CD32'

    def add_data(self, x_data, y_data,
                 legend='l', color='#C0CED1', linestyle='--'):
        plt.plot(x_data, y_data, color=color, linewidth=2.0, linestyle=linestyle)
        self.legend.append(legend)

    def set_legend(self):
        plt.legend(np.array(self.legend))

    def set_title(self, title='title'):
        plt.title(title)

    def show_plot(self):
        plt.legend(np.array(self.legend))
        plt.show()

    def clear(self):
        plt.close('all')

    def randomcolor(self):
        """
        随机生成颜色
        :return: '#C0CED1'
        """
        colorArr = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                    'A', 'B', 'C', 'D', 'E', 'F']
        color = ""
        for i in range(6):
            color += colorArr[random.randint(0, 15)]
        return "#" + color



if __name__ == '__main__':

    x_data = ['2011', '2012', '2013', '2014', '2015', '2016', '2017']
    y_data = [58000, 60200, 63000, 71000, 84000, 90500, 107000]
    y_data2 = [52000, 54200, 51500, 58300, 56800, 59500, 62700]

    s = Show()

    s.add_data(x_data, y_data, 'a', s.randomcolor(), 'solid')
    s.add_data(x_data, y_data2, 'b', s.randomcolor(), '-')

    s.show_plot()
