from math import radians, cos, sin, asin, sqrt, degrees, atan2
import numpy as np
import random
from scipy import spatial
import math
import pygal

#MAX_NUMBER
MAX_NUMBER = 99999999


#判断角度是否落在某一区间内
def angle_in_interval(angle, left, right):
    if left > right:
        if angle < left :
            return  angle <= right
        else:
            return True
    else:
        return angle >= left and angle <= right


#公式计算两点间距离（m）
def get_distance(lng1,lat1,lng2,lat2):
    #lng1,lat1,lng2,lat2 = (120.12802999999997,30.28708,115.86572000000001,28.7427)
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)]) # 经纬度转换成弧度
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance=2*asin(sqrt(a))*6371*1000 # 地球平均半径，6371km
    distance=round(distance/1000,3)
    return distance
#计算两点之间的方位角


def get_degree(lonA, latA, lonB, latB):
    """
    Args:
        point p1(latA, lonA)
        point p2(latB, lonB)
    Returns:
        bearing between the two GPS points,
        default: the basis of heading direction is north
    """
    radLatA = radians(latA)
    radLonA = radians(lonA)
    radLatB = radians(latB)
    radLonB = radians(lonB)
    dLon = radLonB - radLonA
    y = sin(dLon) * cos(radLatB)
    x = cos(radLatA) * sin(radLatB) - sin(radLatA) * cos(radLatB) * cos(dLon)
    brng = degrees(atan2(y, x))
    brng = (brng + 360) % 360
    return brng


#DBSCAN算法
def findNeighbor(j,X,eps):
    N=[]
    for p in range(X.shape[0]):   #找到所有领域内对象
        temp=np.sqrt(np.sum(np.square(X[j]-X[p])))   #欧氏距离
        if(temp<=eps):
            N.append(p)
    return N

def dbscan(X, eps,min_Pts):
    k=-1
    kd = spatial.KDTree(X)
    NeighborPts=[]      #array,某点领域内的对象
    Ner_NeighborPts=[]
    fil=[]                                      #初始时已访问对象列表为空
    gama=[x for x in range(len(X))]            #初始时将所有点标记为未访问
    cluster=[-1 for y in range(len(X))]
    points = []   #特征点集合
    while len(gama)>0:

        j=random.choice(gama)
        gama.remove(j)  #未访问列表中移除
        fil.append(j)   #添加入访问列表
        #NeighborPts=findNeighbor(j,X,eps)
        NeighborPts = kd.query_ball_point(X[j], eps)
        if len(NeighborPts) < min_Pts:
            cluster[j]=-1   #标记为噪声点
        else:
            x, y = 0, 0
            for i in NeighborPts:
                x += X[i][0]
                y += X[i][1]
            x = x / len(NeighborPts)
            y = y / len(NeighborPts)
            points.append([x, y])
            for i in NeighborPts:
                if i not in fil:
                    gama.remove(i)
                    fil.append(i)

    return points


def update_points(X, amuth_data, eps, sd_angle, kd):
    #kd = spatial.KDTree(X)

    fil = []  # 初始时已访问对象列表为空
    gama = [x for x in range(len(X))]  # 初始时将所有点标记为未访问
    points = []  # 特征点集合
    amuth_data2 = []# 特征点方向集合
    while len(gama) > 0:

        j = random.choice(gama)
        gama.remove(j)  # 未访问列表中移除
        fil.append(j)  # 添加入访问列表
        NeighborPts = kd.query_ball_point(X[j], eps)
        x, y, amuth, sum = 0, 0, 0, 0
        for i in NeighborPts:
            if angle_in_interval(amuth_data[i], (amuth_data[j] - sd_angle) % 360, (amuth_data[j] + sd_angle) % 360):
                x += X[i][0]
                y += X[i][1]
                amuth += amuth_data[i]
                sum += 1

        x = x / sum
        y = y / sum
        amuth = amuth / sum
        points.append([x, y])
        amuth_data2.append(amuth)


    return points, amuth_data2

def judge_angle(angle_x, angle_y):
    #if angle_x == 0:
        #return True
    #TODO 此处有问题
    if angle_x >= (angle_y - 3) % 360 and angle_x <= (angle_y + 3) % 360:
        return True
    #非单向行驶道路，需考虑对向行驶车辆
    #if (angle_x + 180) % 360 >= (angle_y - 5) % 360 and (angle_x + 180) % 360 <= (angle_y + 5) % 360:
        #return True
    return False


#dbscan算法，考虑点的方向
def dbscan2(X, amuth_data, eps, min_Pts, sd_angle, kd):
    k = -1
    #kd = spatial.KDTree(X)
    NeighborPts = []  # array,某点领域内的对象
    Ner_NeighborPts = []
    fil = []  # 初始时已访问对象列表为空
    gama = [x for x in range(len(X))]  # 初始时将所有点标记为未访问
    cluster = [-1 for y in range(len(X))]
    points = []  # 特征点集合
    amuth_data2 = [] #特征点方向角集合
    while len(gama) > 0:

        j = random.choice(gama)
        gama.remove(j)  # 未访问列表中移除
        fil.append(j)  # 添加入访问列表
        # NeighborPts=findNeighbor(j,X,eps)
        NeighborPts = kd.query_ball_point(X[j], eps)
        actual_neigbor_pts = []
        if len(NeighborPts) < min_Pts:
            cluster[j] = -1  # 标记为噪声点
        else:
            x, y, amuth, sum = 0, 0, 0,  0
            for i in NeighborPts:
                if angle_in_interval(amuth_data[i], (amuth_data[j] - sd_angle) % 360, (amuth_data[j] + sd_angle) % 360):
                    x += X[i][0]
                    y += X[i][1]
                    amuth += amuth_data[i]
                    actual_neigbor_pts.append(i)
                    sum += 1
            if sum >= min_Pts :
                x = x / sum
                y = y / sum
                amuth = amuth / sum
                points.append([x, y])
                amuth_data2.append(amuth)

                #TODO 以下代码是否影响了特征点的精确度
                #改进？

                for i in actual_neigbor_pts:
                    if i not in fil:
                        gama.remove(i)
                        fil.append(i)


    return points, amuth_data2

#绘制箭头
def draw_arrow(A, B, _ax):
    len_param = math.sqrt(math.pow(B[0] - A[0], 2) + math.pow(B[1] - A[1], 2))
    _head_length = len_param / 2
    _head_width =  _head_length / 3
    _width = _head_width / 3
    _ax.arrow(A[0], A[1], B[0] - A[0], B[1] - A[1], length_includes_head=True, head_width=_head_width, head_length=_head_length, width=_width, fc= 'black', ec = 'black')
    #_ax.set_xlim(0,5)
    #_ax.set_ylim(0,5)


def draw_svg(path_array, file_name):
    # stroke参数是指是否禁用连线
    xy_chart = pygal.XY(stroke=True,truncate_legend=10, legend_box_size=5, legend_at_bottom_columns=10, legend_at_bottom=True)
    xy_chart.title = 'Correlation'
    for i in range(len(path_array)):
        xy_chart.add(str(i), path_array[i])
    xy_chart.render_to_file(file_name, height=2000, width=2000, disable_xml_declaration=True)
    with open(file_name, 'r+') as f:
        data1 = f.readlines()
        data = data1[0]
        index1 = data.find("svg")
        str1 =   ' width = "2000"  height = "2000"  '
        data = '<svg' + str1 + data.split('<svg')[1]
        data1[0] = data
        f.seek(0,0)
        f.writelines(data1)


'''
计算点到线段距离，且只计算当点的投影在线段上时的距离，否则返回MAX
'''


def cal_point_2_line(point, A, B):
    if math.sqrt( abs(pow(point[0] - A[0], 2) + pow(point[1] - A[1], 2))) < 0.00000001\
            or math.sqrt( abs(pow(point[0] - B[0], 2) + pow(point[1] - B[1], 2))) < 0.00000001:
        return 0

    # 首先判断∠CAB是否为钝角
    if (point[0] - A[0]) * (B[0] - A[0]) + (point[1] - A[1]) * (B[1] - A[1]) < 0:
        return MAX_NUMBER
    # 判断∠CBA是否为钝角
    if (point[0] - B[0]) * (A[0] - B[0]) + (point[1] - B[1]) * (A[1] - B[1]) < 0:
        return MAX_NUMBER
    para_A = A[1] - B[1]
    para_B = -A[0] + B[0]
    para_C = -(A[1] - B[1]) * B[0] + (A[0] - B[0]) * B[1]
    try:
        ans = abs(para_A * point[0] + para_B * point[1] + para_C) / sqrt(para_A * para_A + para_B * para_B)
    except Exception:
        print("Exception", point, A, B, para_A, para_B)
    return ans


'''
判断路段相似点是否连续，且从路段起点开始，或到路段终点结束
return 0    不符合要求
return 1    从起点开始
return 2    到终点结束
return 3    从起点到终点
'''


def judge_similar(similar_point_index, path_len):
    flag = 0
    similar_point_index.sort()
    for index1, index2 in zip(similar_point_index[:-1], similar_point_index[1:]):
        if index1 != index2 - 1:
            return 0

    if 0 in similar_point_index and path_len - 1 in similar_point_index:
        return 3
    if 0 in similar_point_index:
        return 1
    if path_len - 1 in similar_point_index:
        return 2

    return 0


def data_2_array(path_data, threshold):
    path_array = []
    for li in path_data:
        if li._length >= threshold:
            temp = []
            temp_Node = li._head.getNext()
            while temp_Node is not None:
                temp.append(temp_Node.getValue())
                temp_Node = temp_Node.getNext()
            path_array.append(temp)
    return path_array


def same_point(point1, point2):
    if math.sqrt( abs(pow(point1[0] - point2[0], 2) + pow(point1[1] - point2[1], 2))) < 0.00000001:
        return True
    else:
        return False

cnames = {
'aliceblue':            '#F0F8FF',
'antiquewhite':         '#FAEBD7',
'aqua':                 '#00FFFF',
'aquamarine':           '#7FFFD4',
'azure':                '#F0FFFF',
'beige':                '#F5F5DC',
'bisque':               '#FFE4C4',
'black':                '#000000',
'blanchedalmond':       '#FFEBCD',
'blue':                 '#0000FF',
'blueviolet':           '#8A2BE2',
'brown':                '#A52A2A',
'burlywood':            '#DEB887',
'cadetblue':            '#5F9EA0',
'chartreuse':           '#7FFF00',
'chocolate':            '#D2691E',
'coral':                '#FF7F50',
'cornflowerblue':       '#6495ED',
'cornsilk':             '#FFF8DC',
'crimson':              '#DC143C',
'cyan':                 '#00FFFF',
'darkblue':             '#00008B',
'darkcyan':             '#008B8B',
'darkgoldenrod':        '#B8860B',
'darkgray':             '#000000',
'darkgreen':            '#006400',
'darkkhaki':            '#BDB76B',
'darkmagenta':          '#8B008B',
'darkolivegreen':       '#556B2F',
'darkorange':           '#FF8C00',
'darkorchid':           '#9932CC',
'darkred':              '#8B0000',
'darksalmon':           '#E9967A',
'darkseagreen':         '#8FBC8F',
'darkslateblue':        '#483D8B',
'darkslategray':        '#000000',
'darkturquoise':        '#00CED1',
'darkviolet':           '#9400D3',
'deeppink':             '#FF1493',
'deepskyblue':          '#00BFFF',
'dimgray':              '#000000',
'dodgerblue':           '#1E90FF',
'firebrick':            '#B22222',
'floralwhite':          '#FFFAF0',
'forestgreen':          '#228B22',
'fuchsia':              '#FF00FF',
'gainsboro':            '#DCDCDC',
'ghostwhite':           '#F8F8FF',
'gold':                 '#FFD700',
'goldenrod':            '#DAA520',
'gray':                 '#000000',
'green':                '#008000',
'greenyellow':          '#ADFF2F',
'honeydew':             '#F0FFF0',
'hotpink':              '#FF69B4',
'indianred':            '#CD5C5C',
'indigo':               '#4B0082',
'ivory':                '#FFFFF0',
'khaki':                '#F0E68C',
'lavender':             '#E6E6FA',
'lavenderblush':        '#FFF0F5',
'lawngreen':            '#7CFC00',
'lemonchiffon':         '#FFFACD',
'lightblue':            '#ADD8E6',
'lightcoral':           '#F08080',
'lightcyan':            '#E0FFFF',
'lightgoldenrodyellow': '#FAFAD2',
'lightgreen':           '#90EE90',
'lightgray':            '#000000',
'lightpink':            '#FFB6C1',
'lightsalmon':          '#FFA07A',
'lightseagreen':        '#20B2AA',
'lightskyblue':         '#87CEFA',
'lightslategray':       '#000000',
'lightsteelblue':       '#B0C4DE',
'lightyellow':          '#FFFFE0',
'lime':                 '#00FF00',
'limegreen':            '#32CD32',
'linen':                '#FAF0E6',
'magenta':              '#FF00FF',
'maroon':               '#800000',
'mediumaquamarine':     '#66CDAA',
'mediumblue':           '#0000CD',
'mediumorchid':         '#BA55D3',
'mediumpurple':         '#9370DB',
'mediumseagreen':       '#3CB371',
'mediumslateblue':      '#7B68EE',
'mediumspringgreen':    '#00FA9A',
'mediumturquoise':      '#48D1CC',
'mediumvioletred':      '#C71585',
'midnightblue':         '#191970',
'mintcream':            '#F5FFFA',
'mistyrose':            '#FFE4E1',
'moccasin':             '#FFE4B5',
'navajowhite':          '#FFDEAD',
'navy':                 '#000080',
'oldlace':              '#FDF5E6',
'olive':                '#808000',
'olivedrab':            '#6B8E23',
'orange':               '#FFA500',
'orangered':            '#FF4500',
'orchid':               '#DA70D6',
'palegoldenrod':        '#EEE8AA',
'palegreen':            '#98FB98',
'paleturquoise':        '#AFEEEE',
'palevioletred':        '#DB7093',
'papayawhip':           '#FFEFD5',
'peachpuff':            '#FFDAB9',
'peru':                 '#CD853F',
'pink':                 '#FFC0CB',
'plum':                 '#DDA0DD',
'powderblue':           '#B0E0E6',
'purple':               '#800080',
'red':                  '#FF0000',
'rosybrown':            '#BC8F8F',
'royalblue':            '#4169E1',
'saddlebrown':          '#8B4513',
'salmon':               '#FA8072',
'sandybrown':           '#FAA460',
'seagreen':             '#2E8B57',
'seashell':             '#FFF5EE',
'sienna':               '#A0522D',
'silver':               '#C0C0C0',
'skyblue':              '#87CEEB',
'slateblue':            '#6A5ACD',
'slategray':            '#000000',
'snow':                 '#FFFAFA',
'springgreen':          '#00FF7F',
'steelblue':            '#4682B4',
'tan':                  '#D2B48C',
'teal':                 '#008080',
'thistle':              '#D8BFD8',
'tomato':               '#FF6347',
'turquoise':            '#40E0D0',
'violet':               '#EE82EE',
'wheat':                '#F5DEB3',
'white':                '#FFFFFF',
'whitesmoke':           '#F5F5F5',
'yellow':               '#FFFF00',
'yellowgreen':          '#9ACD32'}
