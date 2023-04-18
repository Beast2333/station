import math
import random
import time
from copy import copy
import networkx as nx
import matplotlib.pyplot as plt
import numpy
from itertools import combinations
from random import choice
import xlwings


class TreeNode:
    def __init__(self, val):
        self.is_virtual = False
        self.value = val
        self.father = None
        self.left = None
        self.right = None
        self.direction = 1
        self.coordinate = [-1, -1]
        self.angle = 0
        self.length = 0


class Tree:
    def __init__(self, n, c):
        self.n = n
        self.choice = c
        self.order = []
        self.preorder = []

        self.root = None

        self.direction_index = []
        self.comb = []
        self.node_list = {}


        self.l_forbidden = 3
        self.total_length = 500
        self.throat_length = 300

        self.station_track_coordinate = {}
        self.station_track_node = []
        self.alpha = 0.1414

        self.edge = []
        self.coordinate_list = {}
        self.virtual_node_list = []
        self.counter = 2 * self.n + 1

        # statistic param
        self.pass_order = list()
        self.pass_order_list = {}
        self.pass_order_length_list = list()

        # changeable
        # self.inorder = [4,5,6,3,2,8,7,9,1,12,11,10,14,13,15,16]
        # self.direction_list = []
        # self.l = []
        # self.l_list = [25, 30, 50]
        # self.station_track_coordinate_startpoint = 80
        # self.station_track_space = 5
        # self.station_track_space_list = []

        # changeable
        self.inorder = [4, 5, 6, 3, 2, 8, 7, 9, 1, 12, 11, 10, 14, 13, 15, 16]
        self.direction_list = [1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1]
        self.l = [35.5, 35.5, 27.5, 73, 48, 150, 48, 35.5, 48, 73, 48, 27.5, 73, 35.5, 35.5, 99, 30, 50, 30, 30, 30, 30,
                  30, 30, 50, 50, 30, 30, 25, 25, 30, 99, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                  10, 10]
        self.l_list = [25, 30, 50]
        self.station_track_coordinate_startpoint = 80
        self.station_track_space = 5
        self.station_track_space_list = [4.8, 4.8, 4.8, 4.8, 4.8, 7.4, 4.8, 4.8, 4.8, 4.8, 4.8, 7.4, 4.8, 4.8, 4.8, 4.8]

        self.init()

    def pass_order_init(self, node):
        if node:
            if node.father:
                self.pass_order.append(node.value)
                if node.father.value == 1:
                    self.pass_order.append(1)
                    return
                self.pass_order_init(node.father)
        if node.value == 1:
            print('ERROR: INPUT NODE IS ROOT')
            return

    def init(self):
        # init order
        if self.n:
            for i in range(1, self.n+1):
                self.order.append(i)

        for i in range(self.n):
            if not i:
                continue
            self.direction_index.append(i)

        # init station track coordinate
        # a = (self.n + 1) // 2 * self.station_track_space
        if len(self.station_track_space_list) == 0:
            for i in range(self.n):
                self.station_track_space_list.append(self.station_track_space)

        space = 0
        for i in range(self.n + 1):
            id = i + 1 + self.n
            station_track_node = TreeNode(id)
            self.node_list[id] = station_track_node
            if i == 0:
                space = self.station_track_coordinate_startpoint
            else:
                space -= self.station_track_space_list[i-1]
            station_track_node.coordinate = (self.total_length, space)
            self.station_track_node.append(station_track_node)
            self.coordinate_list[id] = station_track_node.coordinate

        # init l list & virtual node list
        if len(self.l) == 0:
            for i in range(3 * self.n + 1):
                r = int(random.random() * 3)
                self.l.append(self.l_list[r])
        # for i in range(2 * self.n+2, self.n * 3 +3):
        #     self.virtual_node_list.append(i)

    def buildTree(self,preorder,inorder):
        if not inorder:
            return None
        root = TreeNode(preorder[0])
        self.node_list[preorder[0]] = root
        rootPots = inorder.index(preorder[0])
        root.left = self.buildTree(preorder[1:1+ rootPots],inorder[:rootPots])
        root.right = self.buildTree(preorder[rootPots++1:],inorder[rootPots+1:])
        if root.left:
            root.left.father = root
        if root.right:
            root.right.father = root
        return root

    def virtual_node_init(self, node):
        if not node:
            print('ERROR:node dose not exist')
            return

        if node.right is None:
            self.counter += 1
            node.right = TreeNode(self.counter)
            self.node_list[self.counter] = node.right
            self.virtual_node_list.append(node.right)
            node.right.is_virtual = True
            node.right.father = node
        else:
            self.virtual_node_init(node.right)

        if node.left is None:
            self.counter += 1
            node.left = TreeNode(self.counter)
            self.node_list[self.counter] = node.left
            self.virtual_node_list.append(node.left)
            node.left.is_virtual = True
            node.left.father = node
        else:
            self.virtual_node_init(node.left)

    # def virtual_node_list_init(self, node):
    #     if not node:
    #         print('node dose not exist')
    #         return
    #
    #     if node.right is None and node.left is None:
    #         self.virtual_node_list.append(node)
    #         return
    #
    #     if node.right:
    #         self.virtual_node_list_init(node.right)
    #     else:
    #         self.virtual_node_list_init(node.left)

    def random_order(self, order):
        return numpy.random.shuffle(order)

    def order_generator(self):
        # self.preorder = self.preorder[1:]
        # self.random_order(self.preorder)
        # self.preorder.insert(0, 1)
        self.random_order(self.inorder)

    def plan_generator(self, a):
        flag = True
        if len(self.preorder) == 0 and len(self.inorder) == 0:
            self.inorder = copy(self.order)
            self.order_generator()

        if a:
            self.order_generator()
        self.preorder = copy(self.order)

        while flag:
            print('inorder:' + str(self.inorder))
            print('preorder:' + str(self.preorder))
            try:
                self.root = self.buildTree(self.preorder, self.inorder)
                if self.root:
                    flag = False
            except:
                print('ERROR: inorder error, could not generate a plan')
                self.order_generator()
                continue

    def direction_generator(self):
        comb = []
        for i in range(self.n):
            for j in combinations(self.direction_index, i):
                comb.append(j)
        return comb

    def layer_detector(self, root):
        if not root:
            return 0
        ldepth = self.layer_detector(root.left)
        rdepth = self.layer_detector(root.right)
        return max(ldepth, rdepth) + 1

    def layer_filter(self):
        i = 1
        flag = True
        layer = []
        while flag:
            if self.n > 2**(i-1):
                if self.n <= 2**i:
                    layer.append(i)
                    layer.append(i+2)
                    flag = False
            i += 1
        l = self.layer_detector(self.root)
        if l <= layer[1]:
            print('符合层数要求')
            return 1
        else:
            print('不符合层数要求')
            return 0

    def preorder_reader(self, seed):
        if not seed:
            return
        # print(seed.value)

        self.preorder_reader(seed.left)
        self.preorder_reader(seed.right)

    def direction_filter(self):
        seed = self.root
        queue = [seed]
        for k in queue:
            if k.left and k.right:
                if k.left.direction == 1 and k.right.direction == -1:
                    print('左右节点发生冲突！')
                    return 1
            if k.left:
                queue.append(k.left)
            if k.right:
                queue.append(k.right)
        return 0

    def edge_init(self, node):
        if node:
            if node.left:
                self.edge.append((node.value, node.left.value))
                self.edge_init(node.left)
            if node.right:
                self.edge.append((node.value, node.right.value))
                self.edge_init(node.right)

    def coordinate_calculater(self, node):
        #     self.root.angle = self.alpha
        if node:
            if node.value == 1:
                node.angle = 0
                node.coordinate = (self.total_length - self.throat_length, 0)
            queue = [node]
            print('L列表：' + str(self.l))
            for k in queue:
                # print(k.value)
                if k.value != 1:
                    if k.father.is_virtual:
                        # init station track node length
                        cf = k.father.coordinate
                        ck = k.coordinate
                        k.length = pow((ck[0] - cf[0]) ** 2 + (ck[1] - cf[1]) ** 2, 0.5)
                        # print(str(k.value) + '号长度：' + str(k.length))
                        continue
                    if k.father.left:
                        if k.father.left.value == k.value:
                            k.angle = k.father.angle - self.alpha * (1 - k.father.direction) / 2
                    if k.father.right:
                        if k.father.right.value == k.value:
                            k.angle = k.father.angle + self.alpha * (1 + k.father.direction) / 2
                    # init node length
                    k.length = self.l[k.value - 2]
                    k.coordinate[0] = k.father.coordinate[0] + self.l[k.value - 2] * math.cos(k.angle)
                    k.coordinate[1] = k.father.coordinate[1] + self.l[k.value - 2] * math.sin(k.angle)
                    print(str(k.value) + '父节点：' + str(k.father.value))
                    print(str(k.value) + '方向：' + str(k.direction))
                    print(str(k.value) + '号角度：' + str(k.angle))
                    print(str(k.value) + '号坐标：' + str(k.coordinate))
                    print(str(k.value) + '号长度：' + str(k.length))

                if k.left:
                    queue.append(k.left)
                if k.right:
                    queue.append(k.right)
                self.coordinate_list[k.value] = k.coordinate

    def draw_track(self, name):
        g = nx.Graph()
        g, pos = self.graph_init(g, self.root)
        # graph.add_nodes_from(self.preorder)
        # self.edge_init(self.root)
        # graph.add_edges_from(self.edge)

        nx.draw_networkx(g, self.coordinate_list)
        plt.savefig('./pic/' + str(name) + '.jpg')
        plt.show()

    def graph_init(self, G, node, pos={}, layer=1):
        pos[node.value] = (node.coordinate[0], node.coordinate[1])
        if node.left:
            G.add_edge(node.value, node.left.value)
            l_layer = layer + 1
            self.graph_init(G, node.left, pos=pos, layer=l_layer)
        if node.right:
            G.add_edge(node.value, node.right.value)
            r_layer = layer + 1
            self.graph_init(G, node.right, pos=pos, layer=r_layer)
        return G, pos

    def node_direction_init(self,direction_list):
        # node direction init
        seed = self.root
        queue = [seed]
        for k in queue:
            if k.is_virtual:
                continue
            # print(k.value)
            k.direction = direction_list[k.value - 1]
            if k.left:
                queue.append(k.left)
            if k.right:
                queue.append(k.right)

    def total_length_init(self):
        queue = [self.root]
        length = 0
        for i in queue:
            # print(i.value)
            length += i.length
            if i.left:
                queue.append(i.left)
            if i.right:
                queue.append(i.right)
        return length

    def matrix_init(self):
        # a = numpy.array([0, [0], 0])
        m = x = numpy.empty([self.n + 1, self.n + 1], dtype = list)
        for i in range(self.n+1):
            for j in range(self.n+1):
                if i == j:
                    m[i][j] = [0]
                    continue
                # if i == 0:
                #     m[i][j] = j + self.n
                #     continue
                # if j == 0:
                #     m[i][j] = i + self.n
                #     continue

                x = i + self.n + 1
                y = j + self.n + 1
                node_x = self.node_list[x]
                node_y = self.node_list[y]
                path_x = self.pass_order_list[x]
                path_y = self.pass_order_list[y]
                # print(path_x, path_y)
                share_point_list = [i for i in path_x if i in path_y]
                share_point = max(share_point_list)
                index = path_y.index(share_point)
                path = path_y[0:index+1]
                length = 0
                for k in path[1:]:
                    node = self.node_list[k]
                    length += node.length
                m[i][j] = [share_point, path, length]
        return m

    def excel_output(self, m, name):
        workbook = xlwings.Book('./matrix.xlsx')
        sheet = workbook.sheets.add(str(name))
        # workbook.app(visible=False)
        for p in range(0, self.n + 1):

            for j in range(0, self.n + 1):
                jc = chr(j + 65)
                token = jc + str(p + 1)
                # print(token)
                # if p == 0:
                #     sheet[token].value = str(j)
                #     continue
                # if j == 0:
                #     sheet[token].value = str(i)
                #     continue
                sheet[token].value = str(m[p][j])
                time.sleep(0.05)
        # workbook.save('./matrix.xlsx')

    def main(self):
        flag = True
        # plan layer filter
        a = 0
        while flag:
            self.plan_generator(a)
            a = self.layer_filter()
            if a:
                flag = False
        # init virtual node
        self.virtual_node_init(self.root)

        # connect virtual_node to station_track_node
        for i in range(self.n +1):
            self.virtual_node_list[i].left = self.station_track_node[i]
            self.station_track_node[i].father = self.virtual_node_list[i]
        # direction generator & controller
        self.comb = self.direction_generator()
        name = 0
        # init node list

        # init direction
        if not self.direction_list:
            for i in range(self.n):
                self.direction_list.append(1)
                if not i:
                    continue
                self.direction_index.append(i)

            for i in range(self.choice):
                name += 1
                c = choice(self.comb)
                time.sleep(0.1)
                print('choice:'+str(c))
                direction_list = copy(self.direction_list)
                for j in c:
                    direction_list[j] = -1
                print('direction_list:'+str(direction_list))

                self.node_direction_init(direction_list)
            # direction filter
                if self.direction_filter():
                    continue
            # distance & angle & coordinate calculator
                self.coordinate_calculater(self.root)
                # draw track
                self.draw_track(name)
                print(self.station_track_coordinate)

                # pass order init & pass order length

                self.pass_order_list = {}
                for j in range(self.n + 1):
                    self.pass_order = []
                    self.pass_order_init(self.station_track_node[j])
                    self.pass_order_list[self.pass_order[0]] = self.pass_order
                    pass_order_length = 0
                    for k in self.pass_order:
                        k_node = self.node_list[k]
                        pass_order_length += k_node.length
                    self.pass_order_length_list.append(pass_order_length)
                print('出站路径' + str(self.pass_order_list))
                print('出站路径长度' + str(self.pass_order_length_list))

                # total length init
                print('轨道总长度:' + str(self.total_length_init()))
                # matrix init
                m = self.matrix_init()
                # print(m)
                self.excel_output(m, name)

        else:
            direction_list = copy(self.direction_list)
            print('direction_list:' + str(direction_list))
            self.node_direction_init(direction_list)
            # distance & angle & coordinate calculator
            self.coordinate_calculater(self.root)
            # draw track
            self.draw_track(name)
            print(self.station_track_coordinate)
            # pass order init & pass order length
            self.pass_order_list = {}
            for j in range(self.n + 1):
                self.pass_order = []
                self.pass_order_init(self.station_track_node[j])
                # print(self.pass_order_list)
                self.pass_order_list[self.pass_order[0]] = self.pass_order
                pass_order_length = 0
                for k in self.pass_order:
                    k_node = self.node_list[k]
                    pass_order_length += k_node.length
                self.pass_order_length_list.append(pass_order_length)
            print('出站路径' + str(self.pass_order_list))
            print('出站路径长度' + str(self.pass_order_length_list))
            # total length init
            print('轨道总长度:' + str(self.total_length_init()))
            # matrix init
            self.matrix_init()
            m = self.matrix_init()
            # print(m)
            self.excel_output(m, 1)


class draw:
    def __init__(self, node):
        self.node = node

    def create_graph(self, G, node, pos={}, x=0, y=0, layer=1):
        pos[node.value] = (x, y)
        if node.left:
            if not node.left.is_virtual:
                G.add_edge(node.value, node.left.value)
                l_x, l_y = x - 1 / 2 ** layer, y - 1
                l_layer = layer + 1
                self.create_graph(G, node.left, x=l_x, y=l_y, pos=pos, layer=l_layer)
        if node.right:
            if not node.right.is_virtual:
                G.add_edge(node.value, node.right.value)
                r_x, r_y = x + 1 / 2 ** layer, y - 1
                r_layer = layer + 1
                self.create_graph(G, node.right, x=r_x, y=r_y, pos=pos, layer=r_layer)
        return G, pos

    def draw_plan(self):  # 以某个节点为根画图
        graph = nx.DiGraph()
        graph, pos = self.create_graph(graph, self.node)
        fig, ax = plt.subplots(figsize=(8, 10))  # 比例可以根据树的深度适当调节
        nx.draw_networkx(graph, pos, ax=ax, node_size=300)

        plt.savefig('./pic/pic' + str('1') + '.jpg')
        plt.show()


if __name__ == "__main__":
    s = Tree(16, 5)
    s.main()
    # printTree(s.root)
    d = draw(s.root)
    d.draw_plan()
    # print(numpy.random.randint(1, 6, size=5))

    # inorder = [3, 2, 4, 1, 5]
    # preorder = [1, 2, 3, 4, 5]
    # solution = Solution()
    # root = solution.buildTree(preorder, inorder)
    # print("二叉树为：")
    # printTree(root)

    # draw(root)
