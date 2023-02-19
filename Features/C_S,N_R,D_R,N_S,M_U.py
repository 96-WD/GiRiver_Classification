class Node:
    def __init__(self, ID, name, start, end, length):
        self.ID = ID
        self.name = name
        self.start = start
        self.end = end
        self.length = length
        self.parent = None
        self.left = None
        self.right = None


class BinaryTree:
    def __init__(self, data):
        self.nodes = list()
        for i in data:
            self.nodes.append(Node(i[0], i[1], i[2], i[3], i[4]))
        self.create()
        for node in self.nodes:
            if node.parent == None:
                self.root = node
                break

    def create(self):
        for node in self.nodes:
            for i in self.nodes:
                if i.end == node.start:
                    if node.left == None:
                        node.left = i
                        i.parent = node
                    else:
                        node.right = i
                        i.parent = node
    def downstreamIsSame(self,ID):
        for node in self.nodes:
            if node.ID == ID:
                break
        if node.ID != ID:
            print("不存在此河段")
            return
        if node.parent == None:
            return 0
        elif node.parent.name == node.name:
            return 1
        else:
            return 0

    def upperstreamNums(self, ID):
        for node in self.nodes:
            if node.ID == ID:
                break
        if node.ID != ID:
            print("不存在此河段")
            return
        if node.left==None:
            return 0
        else:
            return 2
    def sourceNums(self, ID ):
        for node in self.nodes:
            if node.ID == ID:
                break
        if node.ID != ID:
            print("不存在此河段")
            return
        num=self.dfs(node)
        return num
    def dfs(self,node):
        if node.left == None:
            return 1
        else:
            l = self.dfs(node.left)
            r = self.dfs(node.right)
            return l+r
    def depth(self, ID):
        for node in self.nodes:
            if node.ID == ID:
                break
        if node.ID != ID:
            print("不存在此河段")
            return
        num = self.dfs1(node)
        return num
    def dfs1(self, node):
        if node.left == None:
            return 1
        else:
            l = self.dfs1(node.left)
            r = self.dfs1(node.right)
            return max(l, r)+1
    def maxLength(self, ID):
        for node in self.nodes:
            if node.ID == ID:
                break
        if node.ID != ID:
            print("不存在此河段")
            return
        num = self.dfs2(node)-node.length
        return num
    def dfs2(self, node):
        if node == None:
            return 0
        else:
            l = self.dfs2(node.left)
            r = self.dfs2(node.right)
            return max(l, r)+node.length

if __name__=='__main__':
    import pandas as pd
    path = "拓扑关系表.csv"
    data = pd.read_csv(path)
    print(data.head(10))
    data = data.values
    tree = BinaryTree(data)
    res = list()
    for i in tree.nodes:
        ID = i.ID
        name = i.name
        if i.parent:
            nextID = i.parent.ID
        # print("输入ID为：", ID)
        # 判断下游河段名称是否相同
        flag = tree.downstreamIsSame(ID)
        # print('下游河段名称是否相同:', flag)
        #上游河段数
        nums = tree.upperstreamNums(ID)
        # print('上游河段数:', nums)
        # 源头数
        source = tree.sourceNums(ID)
        # print('源头数:', source)
        # 河段深度
        dep = tree.depth(ID)
        # print('河段深度:', dep)
        # 该河段向上游延伸至上游河源中的最大长度
        len = tree.maxLength(ID)
        # print('该河段向上游延伸至上游河源中的最大长度:', len)
        m = [ID, name, nextID, flag, nums, source, dep, len]
        res.append(m)

    df = pd.DataFrame(data=res, columns=['ID', 'Name', 'Next reach ID', 'C_S', 'N_R', 'N_S', 'D_R', 'M_U'])
    df.to_csv('result.csv', index=False)


