import csv
from data_type import LinkedList
from data_type import Node
import random
from scipy import spatial
from reconstruct import tools
import time
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import copy

data_path='../data/truck_data_to_reform_road_20170901_20170907_11.csv'
#data_path = "E:\毕业论文\Truck\\truck_data_to_reform_road_2.csv"
points = [] #点集合
amuths = [] #点方向集合
points_stats = [] #对应点所在道路数
points_visit = [] #对应点是否已被访问




'''
    函数用来获取路段，需要有一个起始点及起始点方向
    有几个超参数
'''


'''
返回值分一下三种情况
   1.  0, []
   2.  1, [(certain index)]
   3.  > 1 [(index arrays)] 数组的第一个要么就是道路延伸方向上最近的特征点，要么就是分叉点 
'''
def find_path_part(begin_point, begin_point_amuth, kd, _radius4, _cur_point_angle, _fork_angle):
    pp = []
    temp = []
    potential_points_index_set = kd.query_ball_point(begin_point, _radius4)
    nearest_point = -1
    nearest_distance = 9999
    nearest_angle = 9999
    if len(potential_points_index_set) == 0:
        return 0, []
    else:

        for i in potential_points_index_set:

            angle = tools.get_degree(begin_point[0], begin_point[1], points[i][0], points[i][1])

            if begin_point == [116.4415022, 39.72258294]:
                print(points[i], amuths[i])
                print(angle)
                print(begin_point_amuth)

            #选择距离最近的特征点延伸
            #首先寻找在道路延伸方向上的最近特征点
            '''
            if tools.angle_in_interval(angle, (begin_point_amuth - _cur_point_angle) % 360, (begin_point_amuth + _cur_point_angle) % 360) \
                    and tools.angle_in_interval(amuths[i], (angle - _cur_point_angle) % 360, (angle + _cur_point_angle) % 360):
                cur_distance = math.sqrt( math.pow((begin_point[0] - points[i][0]), 2) +  math.pow((begin_point[1] - points[i][1]), 2))
                if  cur_distance < nearest_distance and cur_distance != 0:
                    nearest_point = i
                    nearest_distance = cur_distance
            '''
            #选择夹角最近的特征点延伸
            if tools.angle_in_interval(angle, (begin_point_amuth - _cur_point_angle) % 360, (begin_point_amuth + _cur_point_angle) % 360) \
                    and tools.angle_in_interval(amuths[i], (angle - _cur_point_angle) % 360,(angle + _cur_point_angle) % 360):
                cur_angle = abs(angle - begin_point_amuth)
                if cur_angle > 180:
                    cur_angle = 360 - cur_angle
                if cur_angle < nearest_angle and i != begin_point:
                    nearest_point = i
                    nearest_angle = cur_angle

            #在一定角度范围内的特征点被认为是分叉路口
            elif (tools.angle_in_interval(angle, (begin_point_amuth - _fork_angle) % 360, (begin_point_amuth - _cur_point_angle) % 360) \
                    or tools.angle_in_interval(angle, (begin_point_amuth + _cur_point_angle)% 360, (begin_point_amuth + _fork_angle) % 360))\
                    and tools.angle_in_interval(amuths[i], (angle - _cur_point_angle) % 360, (angle + _cur_point_angle) % 360):
                pp.append(i)
                temp.append(points[i])

            else:
                pass
        if nearest_point != -1:
            pp.insert(0, nearest_point)
            temp.insert(0, points[nearest_point])
            #print(points[nearest_point], begin_point, amuths[nearest_point], begin_point_amuth)

        try:
            index = temp.index(begin_point)
            pp.pop(index)  #由于kd树查询会把 圆心点也返回，因此去除该点
        except Exception:
            pass
        return len(pp), pp


'''
合并道路段
'''
#
def merge_path(path_data, threshold):
    print("合并之前路段数量为", len(path_data))

    new_path_data = []
    flag = False

    while True:
        if flag:
            break
        flag = True
        while True:
            try:
                cur_path = path_data.pop(0)
            except Exception:
                break
            if cur_path._length < threshold:               #TODO 特征点太多时可以试着修改阈值
                new_path_data.append(cur_path)
                continue
            head_node = cur_path._head.getNext()
            head_point = head_node.getValue()
            tail_node = cur_path._head.getNext()
            while tail_node.getNext() is not None:
                tail_node = tail_node.getNext()
            tail_point = tail_node.getValue()

            tail_index = points.index(tail_point)
            # 优先且只合并最长的
            max_li1 = None
            max_li2 = None
            max_len1 = 0
            max_len2 = 0
            maxlen = 0
            for li in path_data:
                if li._length >= threshold and li._length > maxlen:

                    _head_node = li._head.getNext()
                    _head_point = _head_node.getValue()
                    li2 = LinkedList()
                    li2.deep_copy(li)
                    _tail_node = li2._head.getNext()
                    while _tail_node.getNext() is not None:
                        _tail_node = _tail_node.getNext()
                    _tail_point = _tail_node.getValue()

                    # 具有重合点则合并
                    if _head_point == tail_point:
                        max_li1 = li
                        maxlen1 = max_li1._length
                        maxlen = (max_len2 if max_len2 < max_len1 else max_len1)
                    if _tail_point == head_point:
                        max_li2 = li
                        maxlen2 = max_li2._length
                        maxlen = (max_len2 if max_len2 < max_len1 else max_len1)
            if max_li1 is not  None or max_li2 is not None:
                if max_li1 is not None:
                    path_data.remove(max_li1)
                    max_li1.remove(tail_point)
                    cur_path.union(max_li1)
                if max_li2 is not None:
                    path_data.remove(max_li2)
                    max_li2.remove(head_point)
                    max_li2.union(cur_path)
                    cur_path = max_li2

            new_path_data.append(cur_path)

        path_data = copy.deepcopy(new_path_data)
        new_path_data = []
    print("合并之后路段数量为", len(path_data))
    return path_data


'''
改进的合并道路段 -------弃之不用
'''
def merge_path2(path_data):
    print("合并之前路段数量为", len(path_data))
    #path_data中包含的点
    points_in_path_data = []
    #记录点所在的path
    paths_contain_point = [None for i in range(2100)] #TODO 临时设定的大小，可能存在bug
    #记录该点是否需要考虑合并其所在路段
    points_to_be_considered = []

    for li in path_data:

        temp_node = li._head.getNext()
        while temp_node != None:
            if temp_node.getValue() not in points_in_path_data:
                points_in_path_data.append(temp_node.getValue())
            i = points_in_path_data.index(temp_node.getValue())
            if paths_contain_point[i] == None:
                paths_contain_point[i] = []
            if li not in paths_contain_point[i]:
                (paths_contain_point[i]).append(li)
            temp_node = temp_node.getNext()

    points_to_be_considered =  [0 for i in range(len(points_in_path_data))]

    for i in range(len(points_in_path_data)):
        temp_list = paths_contain_point[i]
        if len(temp_list) > 1:
            points_to_be_considered[i] = 1
    #points_in_path_data = list(points_in_path_data)

    while sum(points_to_be_considered) != 0:
        for i in range(len(points_in_path_data)):
            if points_to_be_considered[i] == 0:
                continue
            index_list = []
            index_of_paths_list = []
            for li in paths_contain_point[i]:

                if li not in path_data:
                    #如果Li不在path_data中，则说明li已经被合并过了
                    #paths_contain_point[i].remove(li)
                    continue

                temp_index = li.index(points_in_path_data[i])
                #TODO 光取出索引仍然有问题，如何修改——不用索引，直接用链表操作
                #只取出路段起始点和终点索引
                if temp_index == 1 or temp_index == li._length:
                    index_list.append(temp_index)
                    index_of_paths_list.append(path_data.index(li))

            #开始合并
            while len(index_list) > 1:
                #不存在起始点或全都是起始点
                if 1 not in index_list or sum(index_list) == len(index_list):
                    points_to_be_considered[i] = 0
                    break
                temp_index_list = copy.deepcopy(index_list)
                for _index in index_list:
                    if _index not in temp_index_list:
                        continue
                    if _index == 1:
                        max_index = 0
                        for __index in index_list:
                            if __index not in temp_index_list:
                                continue
                            if __index != _index and __index > max_index:
                                max_index = __index
                        if max_index != 0 and max_index != 1:
                            path1 = path_data[index_of_paths_list[index_list.index(_index)]]
                            path2 = path_data[index_of_paths_list[index_list.index(max_index)]]
                            temp_p2 = LinkedList()
                            node = path2._head.getNext()
                            while node != None:
                                temp_p2.append(node.getValue())
                                node = node.getNext()
                            path_data.remove(path1)
                            path_data.remove(path2)

                            #paths_contain_point[i].remove(path1)
                            #paths_contain_point[i].remove(path2)
                            path2.union(path1)
                            path_data.append(path2)
                            temp_index_list.remove(_index)
                            temp_index_list.remove(max_index)
                            dis = path2.max_path_segment()
                            if dis > 0.002:
                                p1 = path1
                                p2 = path2
                                print(p1.max_path_segment())
                                print(temp_p2.max_path_segment())
                                p = temp_p2
                            # no more path1 or path2
                            for v in paths_contain_point:
                                if v != None and path1 in v:
                                    v.remove(path1)
                                    v.append(path2)
                                if v != None and temp_p2 in v:
                                    v.remove(path2)
                                    if path2 not in v:
                                        v.append(path2)

                            #paths_contain_point[i].append(path2)
                            #print("the length of path_data - 1")
                        else:
                            temp_index_list.remove(_index)
                index_list = copy.deepcopy(temp_index_list)
            points_to_be_considered[i] = 0
    print("合并之后路段数量为", len(path_data))
    return path_data


'''
改进的合并道路段2 -------弃之不用
'''
def merge_path3(path_data):
    print("合并之前路段数量为", len(path_data))
    # path_data中包含的点
    points_in_path_data = []
    # 记录点所在的path
    paths_contain_point = [None for i in range(2100)]  # TODO 临时设定的大小，可能存在bug
    # 记录该点是否需要考虑合并其所在路段
    points_to_be_considered = []

    for li in path_data:

        temp_node = li._head.getNext()
        while temp_node != None:
            if temp_node.getValue() not in points_in_path_data:
                points_in_path_data.append(temp_node.getValue())
            i = points_in_path_data.index(temp_node.getValue())
            if paths_contain_point[i] == None:
                paths_contain_point[i] = []
            if li not in paths_contain_point[i]:
                (paths_contain_point[i]).append(li)
            temp_node = temp_node.getNext()

    points_to_be_considered = [0 for i in range(len(points_in_path_data))]

    for i in range(len(points_in_path_data)):
        temp_list = paths_contain_point[i]
        if len(temp_list) > 1:
            points_to_be_considered[i] = 1
    # points_in_path_data = list(points_in_path_data)

    while sum(points_to_be_considered) != 0:
        for i in range(len(points_in_path_data)):
            if points_to_be_considered[i] == 0:
                continue
            index_list = []
            index_of_paths_list = []
            for li in paths_contain_point[i]:

                if li not in path_data:
                    # 如果Li不在path_data中，则说明li已经被合并过了
                    # paths_contain_point[i].remove(li)
                    continue

                temp_index = li.index(points_in_path_data[i])
                # TODO 光取出索引仍然有问题，如何修改——不用索引，直接用链表操作
                # 只取出路段起始点和终点索引
                if temp_index == 1 or temp_index == li._length:
                    index_list.append(temp_index)
                    index_of_paths_list.append(paths_contain_point[i].index(li))

            # 开始合并
            while len(index_list) > 1:
                # 不存在起始点或全都是起始点
                if 1 not in index_list or sum(index_list) == len(index_list):
                    points_to_be_considered[i] = 0
                    break
                temp_index_list = copy.deepcopy(index_list)
                temp_index_of_paths_list = copy.deepcopy(index_of_paths_list)
                for j in range(len(index_list)):
                    if index_list[j] == None or index_list[j] not in temp_index_list:
                        continue
                    if index_list[j] == 1:
                        max_index = -1
                        max_len = 0
                        for k in range(len(index_list)):
                            if index_list[k] == None or index_list[k] not in temp_index_list:
                                continue
                            if index_list[k] != index_list[j] and index_list[k] > max_len:
                                max_len = index_list[k]
                                max_index = k
                        if max_len != 0 and max_len != 1:
                            path1 = paths_contain_point[i][index_of_paths_list[j]]
                            path2 = paths_contain_point[i][index_of_paths_list[max_index]]
                            temp_p2 = LinkedList()
                            node = path2._head.getNext()
                            while node != None:
                                temp_p2.append(node.getValue())
                                node = node.getNext()
                            try:
                                path_data.remove(path1)
                                path_data.remove(path2)
                            except Exception:
                                debugg = 0
                                pass

                            # paths_contain_point[i].remove(path1)
                            # paths_contain_point[i].remove(path2)
                            path2.remove(points_in_path_data[i])
                            path2.union(path1)
                            path_data.append(path2)
                            temp_index_list[j] = None
                            temp_index_list[max_index] = None
                            temp_index_of_paths_list[j] = None
                            temp_index_of_paths_list[max_index] = None
                            dis = path2.max_path_segment()
                            if dis > 0.002:
                                p1 = path1
                                p2 = path2
                                print(p1.max_path_segment())
                                print(temp_p2.max_path_segment())
                                p = temp_p2
                            # no more path1 or path2
                            for v in paths_contain_point:
                                if v != None and path1 in v:
                                    v.remove(path1)
                                    v.append(path2)
                                if v != None and temp_p2 in v:
                                    v.remove(path2)
                                    if path2 not in v:
                                        v.append(path2)

                            # paths_contain_point[i].append(path2)
                            # print("the length of path_data - 1")
                        else:
                            temp_index_list[j] = None
                            temp_index_of_paths_list[j] = None
                for v1,v2 in zip(temp_index_list, temp_index_of_paths_list):
                    if v1 == None:
                        temp_index_list.remove(v1)
                        temp_index_of_paths_list.remove(v2)
                index_list = copy.deepcopy(temp_index_list)
                index_of_paths_list = copy.deepcopy(temp_index_of_paths_list)
            points_to_be_considered[i] = 0
    print("合并之后路段数量为", len(path_data))
    return path_data

'''
处理长度为阈值的路段
'''
def process_threshold(path_data, threshold):
    print("处理阈值路段之前路段数量为", len(path_data))
    new_path_data = []
    for li in path_data:
        if li._length == threshold:
            # index1, index2 分别用来记录li首、尾两个节点在其他道路段的索引
            index1 = -1
            index2 = -1
            head_node = li._head.getNext()
            head_point = head_node.getValue()
            tail_node = head_node
            while tail_node.getNext() is not None:
                tail_node = tail_node.getNext()
            tail_point = tail_node.getValue()
            for li2 in path_data:
                if li2 == li:
                    continue
                if li2.search(head_point):
                    index1 = path_data.index(li2)
                if li2.search(tail_point):
                    index2 = path_data.index(li2)
            # 首尾在相同道路段上，则舍弃
            if index1 == index2 and index1 != -1:
                continue
            # 首尾在不同道路段上，则保留
            if index1 != index2 and index1 != -1 and index2 != -1:
                new_path_data.append(li)
            if index1 == index2 and index1 == -1:
                new_path_data.append(li)
        else:
            new_path_data.append(li)
    path_data = new_path_data
    print("处理阈值路段之后路段数量为", len(path_data))
    return path_data


'''
处理有重合的路段——弃之不用
'''
def process_coincidence(path_data):
    print("去重前路段数量为", len(path_data))
    #标记是否还存在路段需要去重
    flag = False
    while True:
        print("循环一次")
        print("路段个数为", len(path_data))
        if flag:
            break
        flag = True
        new_path_data = []

        while True:
            tt1 = time.perf_counter()
            if len(path_data) <= 1:
                break
            try:
                cur_path = path_data.pop(0)
            except Exception:
                print(Exception.with_traceback())
                break
            if cur_path._length < 5:
                continue
            flag2 = True
            flag3 = False
            for path in path_data:
                coincidence_points = cur_path.get_same_segment(path)
                if len(coincidence_points) == 0:
                    continue
                if cur_path._length < path._length:
                    temp_path = LinkedList()
                    temp_path.deep_copy(cur_path)
                    cur_path = LinkedList()
                    cur_path.deep_copy(path)
                    path = LinkedList()
                    path.deep_copy(temp_path)
                    flag3 = True

                print(cur_path)
                #point在coincidence_points中是按其在path中的顺序排列的，因此剩下的point必然在path分裂的右半部分
                #TODO 去重后，长度不到阈值的路段尝试去除
                for point in coincidence_points:
                    print(coincidence_points.index(point), len(coincidence_points))
                    print(point)
                    left_part, right_part = path.split(point)
                    if left_part is not None and left_part._length >= 4:
                        path_data.append(left_part)
                        new_path_data.append(left_part)
                    print("1")
                    if right_part is not None and right_part._length >= 4:
                        path_data.append(right_part)
                    print("2")
                    if path in path_data:
                        path_data.remove(path)
                    print("3")
                    if right_part is not None:
                        path = right_part
                print("ok1")
                if path not in new_path_data and path._length >= 4:
                    new_path_data.append(path)
                flag = False
                flag2 = False
                print("ok")
                if flag3:
                    break

            if flag2:
                new_path_data.append(cur_path)
        path_data = copy.deepcopy(new_path_data)
    print("去重后路段数量为", len(path_data))
    return path_data


'''
处理有重合的路段2
'''
def process_coincidence2(path_data):
    print("去重前路段数量为", len(path_data))
    path_exist_list = [1 for i in range(len(path_data))]
    flag_end = False
    while True:
        if flag_end:
            break
        flag_end = True
        for index1, path in enumerate(path_data):
            if path_exist_list[index1] == 0:
                continue
            for index2, path2 in enumerate(path_data):
                if path_exist_list[index2] == 0 or index2 == index1:
                    continue
                flag = False
                coincidence_points = path.get_same_segment(path2)
                if len(coincidence_points) == 0:
                    continue
                cur_index = index2
                if path._length < path2._length:
                    temp_path = LinkedList()
                    temp_path.deep_copy(path)
                    path = LinkedList()
                    path.deep_copy(path2)
                    path2 = LinkedList()
                    path2.deep_copy(temp_path)
                    cur_index = index1
                    flag = True
                for point in coincidence_points:
                    left_part, right_part = path2.split(point)
                    if left_part is not None and left_part._length >= 4:
                        path_data.append(left_part)
                        path_exist_list.append(1)
                    if right_part is not None and right_part._length >= 4:
                        path_data.append(right_part)
                        path_exist_list.append(1)
                    #去除path2

                    path_exist_list[cur_index] = 0
                    if right_part is not None:
                        path2 = right_part
                flag_end = False
                if flag:
                    break
    new_path_data = []
    for index, path in enumerate(path_data):
        if path_exist_list[index] == 0:
            continue
        new_path_data.append(path)
    path_data.clear()
    path_data = new_path_data
    print("去重后路段数量为 ", len(path_data))
    return path_data






def reform_path(_radius4, _cur_point_angle, _fork_angle):
    radius4 = _radius4  # 最大范围
    cur_point_angle = _cur_point_angle  # 当前特征点延伸方向最大角度范围
    fork_angle = _fork_angle

    #设置图像分辨率
    plt.rcParams['figure.dpi'] = 600
    #plt.rcParams['savefig.dpi'] = 1000
    fig = plt.figure(num=1)
    fig2 = plt.figure(num=2)
    fig3 = plt.figure(num=3)
    ax = fig.add_subplot(111)
    ax2 = fig2.add_subplot(111)
    ax3 = fig3.add_subplot(111)

    with open(data_path) as f:
        data = csv.reader(f)
        for row in data:
            longitude = float(row[0])
            latitude = float(row[1])
            amuth = float(row[2])
            points.append([longitude, latitude])
            amuths.append(amuth)

    kd = spatial.KDTree(points)


    t0 = time.perf_counter()

    #点个数
    points_num = len(amuths)

    #初始化
    points_stats = [0 for i in range(points_num)]
    points_visit = [0 for i in range(points_num)]

    #路段起始库
    path_begin_pool = []
    path_data = []

    #随机选取一个

    begin_index = random.choice(range(points_num))
    l, p = find_path_part(points[begin_index], amuths[begin_index], kd, radius4, cur_point_angle ,fork_angle)
    while l == 0:
        begin_index = random.choice(range(points_num))
        l, p = find_path_part(points[begin_index], amuths[begin_index], kd, radius4, cur_point_angle ,fork_angle)
    for i in p:
        li = LinkedList()
        li.append(points[begin_index])
        li.append(points[i])
        path_begin_pool.append(li)


    #开始循环，结束条件为所有点都遍历到了
    while sum(points_visit) != points_num:
        while len(path_begin_pool) == 0:
            begin_index = random.choice(range(points_num))
            #如果该点已经被访问过，则循环找到一个未访问过的点
            while points_visit[begin_index] != 0:
                begin_index += 1
                begin_index %= points_num
            l, p = find_path_part(points[begin_index], amuths[begin_index], kd, radius4, cur_point_angle ,fork_angle)
            if l == 0:
                points_visit[begin_index] = 1
                if sum(points_visit) == points_num:
                    break
                continue
            for i in p:
                li = LinkedList()
                li.append(points[begin_index])
                li.append(points[i])
                path_begin_pool.append(li)
        if sum(points_visit) == points_num:
            break
        #从路段起始库中随机取出一段路
        li = path_begin_pool.pop(0)
        try:
            point_x = points.index(li._head.getNext().getValue())
            point_y = points.index(li._head.getNext().getNext().getValue())
        except Exception:
            print(li._head.getNext().getValue())
            print(points.index(li._head.getNext().getValue()))
        #标记已访问点
        points_visit[point_x] = 1
        points_visit[point_y] = 1
        #沿路段方向延伸
        while True:
            l,p = find_path_part(points[point_y], amuths[point_y], kd, radius4, cur_point_angle ,fork_angle)
            cur_begin_point = point_y
            if l == 0:
                break
            elif l == 1:
                point_y = p[0]
                li.append(points[point_y])
                if points_visit[point_y] == 1:
                    break
                points_visit[point_y] = 1
            else:
                point_y = p[0]
                li.append(points[point_y])
                if points_visit[point_y] == 1:
                    break
                points_visit[point_y] = 1
                for i in p:
                    if i == point_y:
                        continue
                    li2 = LinkedList()
                    li2.append(points[cur_begin_point])
                    li2.append(points[i])
                    path_begin_pool.append(li2)

        path_data.append(li)



    print("共的得到",len(path_data),"个路段    共有", points_num, "个特征点")


    for loop in range(2):
        '''
        形成路段集合
        '''
        visited_path = []
        t2 = time.perf_counter()
        for li in path_data:
            if li._length >= 4:
                temp_node = li._head.getNext()
                while temp_node.getNext() is not None:
                    cur_path = [temp_node.getValue(), temp_node.getNext().getValue()]
                    if cur_path not in visited_path:
                        visited_path.append(cur_path)
                    temp_node = temp_node.getNext()
        '''
        修正路段
        + 接续两个在同一方向的路段
        '''
        new_path_data = []
        #首先延长道路段，由于部分地方数据缺失，因此采用延长道路段的方式尽可能连接上
        for li in path_data:
            while True:
                tail = li._head.getNext()
                while tail.getNext() is not None:
                    tail = tail.getNext()
                tail_index = points.index(tail.getValue())
                tail_point = points[tail_index]
                tail_amuth = amuths[tail_index]
                tail_neighbor_index_set = kd.query_ball_point(tail_point, 0.0008)
                nearest_angle = 9999
                nearest_dis = 9999
                nearest_index = -1
                #TODO 改为最近方向优先
                for i in tail_neighbor_index_set:
                    angle = tools.get_degree(tail_point[0], tail_point[1], points[i][0], points[i][1])
                    if tools.angle_in_interval(angle, (tail_amuth - 7)%360, (tail_amuth + 7) % 360) \
                        and tools.angle_in_interval(amuths[i], (angle - 5) % 360, (angle + 5) % 360):
                        #cur_dis = math.sqrt(math.pow(tail_point[0] - points[i][0], 2) + math.pow(tail_point[1] - points[i][1], 2))
                        cur_angle = abs(angle - tail_amuth)
                        if cur_angle > 180:
                            cur_angle = 360 - cur_angle
                        if cur_angle < nearest_angle and i != tail_index:
                            nearest_angle = cur_angle
                            nearest_index = i
                #延长时需保证该路段未被访问过
                if nearest_index != -1 and [tail_point, points[nearest_index]] not in visited_path:
                    visited_path.append([tail_point, points[nearest_index]])
                    li.append(points[nearest_index])
                else:
                    break
        t3 = time.perf_counter()
        print("接续两个在同一方向的路段耗时", t3-t2, "s")
        '''
        #对两个路段首尾相连
        for li in path_data:
            tail = li._head.getNext()
            while tail.getNext() != None:
                tail = tail.getNext()
            tail_index = points.index(tail.getValue())
            tail_point = points[tail_index]
            tail_amuth = amuths[tail_index]
            for li2 in path_data:
                if li != li2:
                    head = li2._head.getNext()
                    head_index = points.index(head.getValue())
                    head_point = head.getValue()
                    head_amuth = amuths[head_index]
                    dis = math.sqrt(math.pow(tail_point[0] - head_point[0], 2) + math.pow(tail_point[1] - head_point[1], 2))
                    angle = tools.get_degree(tail_point[0], tail_point[1], head_point[0], head_point[1])
                    if dis < 0.001 and tools.angle_in_interval(angle, (tail_amuth - 15)%360, (tail_amuth+15)%360)\
                            and tools.angle_in_interval(head_amuth, (angle - 10)%360, (angle + 10) % 360)\
                            and [tail_point, head_point] not in visited_path:
                        visited_path.append([tail_point ,head_point])
                        li.append(head_point)
        '''
        print(len(path_data))
        '''
            形成路段集合
        '''
        visited_path = []
        stats = []
        for li in path_data:
            if li._length >= 4:
                temp_node = li._head.getNext()
                while temp_node.getNext() is not None:
                    cur_path = [temp_node.getValue(), temp_node.getNext().getValue()]
                    num = 1
                    for li2 in path_data:
                        if li != li2 and li2._length >= 4:
                            temp_node2 = li2._head.getNext()
                            while temp_node2.getNext() is not None:
                                temp_path = [temp_node2.getValue(), temp_node2.getNext().getValue()]
                                if temp_path == cur_path:
                                    num += 1
                                temp_node2 = temp_node2.getNext()
                    stats.append(num)
                    visited_path.append(cur_path)
                    temp_node = temp_node.getNext()

        t4 = time.perf_counter()
        print("路段首尾相接耗时", t4-t3, "s")
        '''
        合并道路段
        '''
        path_data = merge_path(path_data, 4)
        '''
        处理长度为阈值的路段
        '''
        path_data = process_threshold(path_data, 4)
        path_data = merge_path(path_data, 4)
    path_data = process_coincidence2(path_data)
    path_data = merge_path(path_data, 4)

    '''
    测试是否仍存在首尾相连的路段没有合并
    
    for li in path_data:
        if li._length < 4:
            continue
        tail_node = li._head.getNext()
        while tail_node.getNext() != None:
            tail_node = tail_node.getNext()
        tail_point = tail_node.getValue()
        for li2 in path_data:
            if li2._length < 4:
                continue
            head_node = li2._head.getNext()
            head_point = head_node.getValue()
            if tail_point == head_point:
                print("exist")
    '''
    '''
    统计每个点所在路段数
    
    for li in path_data:
        if li._length > 3:
            tempNode = li._head.getNext()
            while  tempNode != None:
                temp_point = tempNode.getValue()
                temp_index = points.index(temp_point)
                points_stats[temp_index] += 1
                tempNode = tempNode.getNext()
    tt = [0 for i in range(20)]
    for i in range(len(points_stats)):
        tt[points_stats[i]] += 1
    print(tt)
    '''

    #路段起始点
    path_head_x = []
    path_head_y = []
    path_head_points = []

    '''
    此处单独提取出长度大于3的路段，并将其画出来
    '''
    length_array = [0 for i in range(300)]
    path_array = [] #每个元素是一个数组，数组内为路段的点序列
    maxlen= 0
    for li in path_data:
        length_array[li._length] += 1
        if li._length > maxlen:
            maxlen = li._length
        if li._length >= 4:
            temp = []
            temp_Node = li._head.getNext()
            path_head_x.append(temp_Node.getValue()[0])
            path_head_y.append(temp_Node.getValue()[1])
            path_head_points.append(temp_Node.getValue())
            while temp_Node is not None:
                temp.append(temp_Node.getValue())
                temp_Node = temp_Node.getNext()
            path_array.append(temp)

    print("路段长度分布频数如下:\n",length_array)
    print("其中最长路段长度为 ",maxlen)
    print("路段长度超过3的路段个数为 ", len(path_array))
    #print(path_array)

    t1 = time.perf_counter()
    print("路段延伸共耗时",t1 - t0,"s")


    total_x = []
    total_y = []
    for point in points:
        total_x.append(point[0])
        total_y.append(point[1])

    x = []
    y = []
    #for i in range(len(points)):
       # x.append(points[i][0])
       # y.append(points[i][1])
    x1 = None
    y1 = None


    x2 = []
    y2 = []
    for i in range(len(points_stats)):
        if points_stats[i] ==3:
            x2.append(points[i][0])
            y2.append(points[i][1])


    for v in path_array:

        i = path_array.index(v)
        dd = np.mat(v)
        x1 = dd[:, 0]
        x1 = x1.tolist()
        y1 = dd[:, 1]
        y1= y1.tolist()
        x = x + x1
        y = y + y1
        ax.plot(x1, y1, linewidth = 0.5, color = 'r')

        if len(x1) >= 4:
            ax2.plot(x1, y1, linewidth=0.5, color='r')
            ax3.plot(x1, y1, linewidth=0.5, color='r')
        else:
            ax2.plot(x1, y1, linewidth=0.5, color='b')




    ax2.scatter(total_x, total_y, s =0.1)
    ax3.scatter(x, y, s=0.1)
    ax2.scatter(x2, y2, s=1, marker='o', c='yellow')
    #ax2.scatter(path_head_x, path_head_y, s=1, marker='*', c='green')
    #ax3.scatter(path_head_x, path_head_y, s=1, marker='*', c='green')
    print(x2, y2)
    for i in range(len(points)):
        A = points[i]
        B = [A[0] + 0.0001*math.sin( (amuths[i] / 180) * math.pi ), A[1] + 0.0001 * math.cos( (amuths[i] / 180) * math.pi )]
        tools.draw_arrow(A, B, ax)
        if A in path_head_points:
            tools.draw_arrow(A, B, ax2)
            #tools.draw_arrow(A, B, ax3)
    #ax.savefig('plot1.png', dpi=600)
    plt.savefig('E:\毕业论文\Figures\\plot.png', dpi=600)
    plt.show()

    #绘制svg图
    tools.draw_svg(path_array, "xy-scatter-plot.svg")
    new_path_array= []
    for path in path_array:
        if (len(path)) >=4:
           new_path_array.append(path)
        else:
            new_path_array.append(None)
    tools.draw_svg(new_path_array, "E:\毕业论文\Figures\\plot.svg")

    for point in points:
        if point == [116.4415022, 39.72258294]:
            print(amuths[points.index(point)])
if __name__ == "__main__":
    reform_path(0.0004, 15, 70)


'''
test
'''
'''
l,p2 = find_path_part(points[11], amuths[11])
print(points[11], amuths[11])
print(l)
for i in range(l):
    print(points[p2[i]], amuths[p2[i]])

'''



'''剔除孤立点，以max_eps为半径'''
'''
kd = spatial.KDTree(points)
gama = [i for i in range(len(amuths))]
fill = []
isolated_points = []
print(len(amuths))
while(len(gama) > 0):
    j = random.choice(gama)
    fill.append(j)
    gama.remove(j)
    neighbor_set = kd.query_ball_point(points[j], 0.0002)
    if  len(neighbor_set) != 1:
        for i in neighbor_set:

            if i not in fill:
                fill.append(i)
                gama.remove(i)
    else:
        isolated_points.append(j)
temp_p = []
temp_a = []
for i in isolated_points:
    temp_p.append(points[i])
    temp_a.append(amuths[i])
for i in range(len(temp_p)):
    points.remove(temp_p[i])
    amuths.remove(temp_a[i])

'''





