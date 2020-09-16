

class Tree:
    def __init__(self, p_no, provider):
        self.p_no = p_no
        self.provider = provider

        self.sibling = None
        self.children = []

        self.sv = 0
        self.B = 0

    def if_aggregation(self):
        # 判断是否聚合
        return self.sv > 0

    def copy(self):
        t = Tree(self.p_no, self.provider.copy())
        for c in self.children:
            t.children.append(c.copy())
        return t
