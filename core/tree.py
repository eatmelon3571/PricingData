


class Tree:
    def __init__(self, data=-1):
        self.sibling = None
        self.children = []
        self.data = data


def dfs_tree(root):
    if root is None:
        return

    dfs_tree(root.sibling)

    for node in root.children:
        dfs_tree(node)

    print(root.data)


def creat_tree_in_range(n=10):
    t0 = Tree(0)
    t1 = Tree(1)
    t2 = Tree(2)
    t3 = Tree(3)
    t4 = Tree(4)
    t5 = Tree(5)
    t6 = Tree(6)
    t7 = Tree(7)
    t8 = Tree(8)
    t9 = Tree(9)

    root = t0
    t0.sibling = t1
    t0.children.append(t2)
    t0.children.append(t3)
    t0.children.append(t4)

    t1.children.append(t5)
    t1.children.append(t6)
    t1.children.append(t7)

    t2.children.append(t8)
    t2.children.append(t9)

    dfs_tree(root)

    return root


if __name__ == '__main__':
    creat_tree_in_range()
