import math
import random
import time
from copy import copy
import networkx as nx
import matplotlib.pyplot as plt
import numpy
from itertools import combinations
from random import choice


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


class Tree:
    def __init__(self, n, c):
        self.n = n
        self.choice = c
        self.order = []
        self.preorder = [1, 2, 3, 4, 5, 6, 7]
        self.inorder = [4, 5, 3, 2, 6, 1, 7]
        self.root = None
        self.direction_list = [1, 1, -1, -1, 1, -1, 1]
        self.direction_index = []
        self.comb = []

        self.l_list = [25, 30, 50]
        self.l_forbidden = 3
        self.total_length = 500
        self.throat_length = 300
        self.station_track_space = 5
        self.station_track_coordinate = {}
        self.station_track_node = []
        self.alpha = 0.1414
        self.l = []

        self.edge = []
        self.coordinate_list = {}
        self.virtual_node_list = []
        self.counter = 2 * self.n + 1

        self.init()

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
        a = (self.n + 1) // 2 * self.station_track_space
        for i in range(self.n+1):
            id = i + 1 + self.n
            station_track_node = TreeNode(id)
            station_track_node.coordinate = (self.total_length, a - i * self.station_track_space)
            self.station_track_node.append(station_track_node)
            self.coordinate_list[id] = station_track_node.coordinate
            # self.station_track_coordinate[id] = (self.total_length, a + i * self.station_track_space)
        # init l list & virtual node list
        if len(self.l) == 0:
            for i in range(3 * self.n + 2):
                r = int(random.random() * 3)
                self.l.append(self.l_list[r])
        # for i in range(2 * self.n+2, self.n * 3 +3):
        #     self.virtual_node_list.append(i)

    def buildTree(self,preorder,inorder):
        if not inorder:
            return None
        root = TreeNode(preorder[0])
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
            self.virtual_node_list.append(node.right)
            node.right.is_virtual = True
            node.right.father = node
        else:
            self.virtual_node_init(node.right)

        if node.left is None:
            self.counter += 1
            node.left = TreeNode(self.counter)
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

    def plan_generator(self):
        flag = True
        if self.preorder is None and self.inorder is None:
            self.preorder = copy(self.order)
            self.inorder = copy(self.order)
            self.order_generator()

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
            for k in queue:
                # print(k.value)
                if k.value != 1:
                    if k.father.is_virtual:
                        continue
                    if k.father.left:
                        if k.father.left.value == k.value:
                            k.angle = k.father.angle - self.alpha * (1 - k.father.direction) / 2
                    if k.father.right:
                        if k.father.right.value == k.value:
                            k.angle = k.father.angle + self.alpha * (1 + k.father.direction) / 2
                    k.coordinate[0] = k.father.coordinate[0] + self.l[k.value - 1] * math.cos(k.angle)
                    k.coordinate[1] = k.father.coordinate[1] + self.l[k.value - 1] * math.sin(k.angle)
                    print(str(k.value) + '父节点：' + str(k.father.value))
                    print(str(k.value) + '方向：' + str(k.direction))
                    print(str(k.value) + '号角度：' + str(k.angle))
                    print(str(k.value) + '号坐标：' + str(k.coordinate))
                    print('L列表：' + str(self.l))
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

    def main(self):
        flag = True
        # plan layer filter
        while flag:
            self.plan_generator()
            if self.layer_filter():
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
        else:
            direction_list = copy(self.direction_list)
            print('direction_list:' + str(direction_list))
            self.node_direction_init(direction_list)
            # distance & angle & coordinate calculator
            self.coordinate_calculater(self.root)
            # draw track
            self.draw_track(name)
            print(self.station_track_coordinate)



def printTree(root):
    res =[]
    if root is None:
        print(res)
    queue = []
    queue.append(root)
    while len(queue) != 0:
        tmp=[]
        length = len(queue)
        for i in range(length):
            r = queue.pop(0)
            if r.left is not None:
                queue.append(r.left)
            if r.right is not None:
                queue.append(r.right)
            tmp.append(r.value)
        res.append(tmp)
    print(res)


class draw():
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
    s = Tree(7, 5)
    s.main()
    printTree(s.root)
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
