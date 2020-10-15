import xlwt
import torch
import numpy as np


class Excel:
    def __init__(self):
        self.workbook = None
        self.worksheet = None

        self.name = 2        # 1.xls
        self.xls_dir = str(self.name) + '.xls'
        self.col = 1         # 0行
        self.row = 1

        self.new_workbook()
        # self.new_worksheet()

    def new_workbook(self):
        self.workbook = xlwt.Workbook(encoding='ascii')
        # 初始化
        self.name += 1
        self.xls_dir = str(self.name) + '.xls'
        self.col = 1
        # 设置行宽
        '''
        for i in range(1, 11):
            self.worksheet.col(i).width = 256 * 20
        # '''

    def new_worksheet(self, sheet_name='My Worksheet'):
        self.worksheet = self.workbook.add_sheet(sheet_name)
        for i in range(10):
            self.worksheet.write(0, i + 1, str(i))


    def set_dir(self, xls_dir):
        self.xls_dir = xls_dir

    def add_data(self, data,
                 max_style=xlwt.easyxf('pattern: pattern solid, fore_colour ice_blue'),
                 other_style=xlwt.easyxf('pattern: pattern solid, fore_colour ice_blue')):
        """10个数据，最大的用不同颜色标识"""
        # 找出最大的数
        max_index = 0
        for i in range(1, len(data)):
            if data[max_index] < data[i]:
                max_index = i
        # 写入数据
        for i in range(len(data)):
            s = '%.2f' % data[i].item()
            if i == max_index:
                self.worksheet.write(self.col, i + 1, s, max_style)
            else:
                self.worksheet.write(self.col, i + 1, s, other_style)

        # 更新行数
        self.col += 1

    # 下面这几个函数只是颜色不同而已
    def add_data_papb(self, data):
        """聚合前"""
        max_style = xlwt.easyxf('pattern: pattern solid, fore_colour yellow')
        other_style = xlwt.easyxf('pattern: pattern solid, fore_colour light_yellow')
        self.add_data(data, max_style, other_style)

    def add_data_pab(self, data):
        """平均outputs后"""
        max_style = xlwt.easyxf('pattern: pattern solid, fore_colour orange')
        other_style = xlwt.easyxf('pattern: pattern solid, fore_colour gold')
        self.worksheet.write(self.col, 0, 'avg_pab', other_style)
        self.add_data(data, max_style, other_style)

    def add_data_fed_pab(self, data):
        """fedavg聚合后"""
        max_style = xlwt.easyxf('pattern: pattern solid, fore_colour green')
        other_style = xlwt.easyxf('pattern: pattern solid, fore_colour light_green')
        self.worksheet.write(self.col, 0, 'fed_pab', other_style)
        self.add_data(data, max_style, other_style)

    def save(self):
        print("save ", self.xls_dir)
        self.workbook.save(self.xls_dir)

    def add_acc(self, acc_papb: list, acc_pab, acc_fedpab):
        max_style = xlwt.easyxf('pattern: pattern solid, fore_colour green')
        other_style = xlwt.easyxf('pattern: pattern solid, fore_colour light_green')
        self.worksheet.write(self.col, self.row, acc_fedpab, other_style)
        self.row += 1

        max_style = xlwt.easyxf('pattern: pattern solid, fore_colour orange')
        other_style = xlwt.easyxf('pattern: pattern solid, fore_colour gold')
        self.worksheet.write(self.col, self.row, acc_pab, other_style)
        self.row += 1

        max_style = xlwt.easyxf('pattern: pattern solid, fore_colour yellow')
        other_style = xlwt.easyxf('pattern: pattern solid, fore_colour light_yellow')
        l = len(acc_papb)
        for i in range(l):
            self.worksheet.write(self.col, self.row + i, acc_papb[i], other_style)

        self.col += 1
        self.row = 1