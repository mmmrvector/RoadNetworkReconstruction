import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import math
import csv
from scipy import spatial
import tools
import random
import matplotlib.pyplot as plt
from data_type import LinkedList
from data_type import Node
import pygal
from xml.dom.minidom import parse
import xml.dom.minidom
from reform_path import process_coincidence2
data_path = "E:\毕业论文\Truck\\new_data_without_static_truck.csv"


a = [1,2]
b = [1,2]
c =  [3,4]
d= [a,c]
print(d.index(a))
d.remove(a)
d.append(b)
print(d.index(a))
exit()
path_data = []

path1 = LinkedList()
for i in range(10):
    path1.append([i, i + 1])
path2 = LinkedList()
for i in range(9):
    path2.append([i+6, i + 7])
path3 = LinkedList()
for i in range(12):
    path3.append([i + 4, i + 5])
path4 = LinkedList()
for i in range(3):
    path4.append([i + 3, i + 4])
for i in range(12):
    path4.append([i+ 10, i + 11])
path_data.append(path1)
path_data.append(path2)
path_data.append(path3)
path_data.append(path4)
for path in path_data:
    print(path)

path_data = process_coincidence2(path_data)
print(len(path_data))
for path in path_data:
    print(path)





exit()
xy_chart = pygal.XY(stroke=True, truncate_legend=10, legend_box_size=5, legend_at_bottom_columns=10,
                    legend_at_bottom=True)
xy_chart.title = 'Correlation'

xy_chart.add('1', [[0,1],[2,3],[3,4]])
xy_chart.add('1', [[0,1],[2,3],[3,4]])
xy_chart.render_to_file("test.svg", height=2000, width=2000, disable_xml_declaration=True)
with open("test.svg", 'r+') as f:
    data1 = f.readlines()
    data = data1[0]
    index1 = data.find("svg")
    str1 = ' width = "2000"  height = "2000" '
    data = '<svg' + str1 + data.split('<svg')[1]
    data1[0] = data
    print(data1)
    f.seek(0,0)
    f.writelines(data1)
exit()
fig = plt.figure()
ax = fig.add_subplot(121)

def draw_arrow(A, B, _ax):
    len_param = math.sqrt(math.pow(B[0] - A[0], 2) + math.pow(B[1] - A[1], 2))
    _head_length = len_param / 2
    _head_width =  _head_length / 3
    _width = _head_width /3
    _ax.arrow(A[0], A[1], B[0] - A[0], B[1] - A[1], length_includes_head=True, head_width=_head_width, head_length=_head_length, fc= 'r', ec = 'r')
    _ax.set_xlim(0,5)
    _ax.set_ylim(0,5)


a = np.array([1,2])
b = np.array([3,4])
c = np.array([1,1])
d = np.array([1,2])
draw_arrow(a, b, ax)
draw_arrow(c, d, ax)
plt.show()
















exit()

data = []


with open(data_path) as f:
    csv_data = csv.reader(f)
    for row in csv_data:
        longitude = float(row[2]) - 116
        latitude = float(row[3]) - 39
        amuth = int(row[7])
        data.append([longitude, latitude])


eps = 0.00005
min_Pts = 5
k=-1
kd = spatial.KDTree(data)
NeighborPts=[]      #array,某点领域内的对象
Ner_NeighborPts=[]
fil=[]                                      #初始时已访问对象列表为空
gama=[x for x in range(len(data))]            #初始时将所有点标记为未访问
cluster=[-1 for y in range(len(data))]
points = []   #种子点集合
points_stats = [] #统计种子点频数
while len(gama)>0:

    j=random.choice(gama)
    gama.remove(j)  #未访问列表中移除
    fil.append(j)   #添加入访问列表
    #NeighborPts=findNeighbor(j,X,eps)
    NeighborPts = kd.query_ball_point(data[j], eps)
    if len(NeighborPts) < min_Pts:
        cluster[j]=-1   #标记为噪声点
    else:
        x, y = 0, 0
        for i in NeighborPts:
            x += data[i][0]
            y += data[i][1]
        x = x / len(NeighborPts)
        y = y / len(NeighborPts)
        points.append([x, y])
        points_stats.append(len(NeighborPts))
        for i in NeighborPts:
            if i not in fil:
                gama.remove(i)
                fil.append(i)

print(len(points))

