import matplotlib.pyplot as plt
import csv
import random
from sklearn.cluster import DBSCAN
import numpy as np
from reconstruct import tools
from reconstruct.pre_process.get_distribution_of_trucks import point_in_area
import time
from scipy import spatial



#data_path = "../data/new_data_without_static_truck_20170901_20170907.csv"
data_path = "../data/new_data_without_static_truck_20170901_20170907.csv"
x = []
y = []


coordinate_data = []
amuth_data = []

def get_points(sd_angle, radius1, radius2, radius3, min_pts1):
    _sd_angle = sd_angle
    _radius1 = radius1
    _radius2 = radius2
    _radius3 = radius3
    _min_pts1 = min_pts1
    t0 = time.perf_counter()

    with open(data_path) as f:
        data = csv.reader(f)
        for row in data:
            longitude = float(row[2])
            latitude = float(row[3])
            amuth = int(row[7])
            coordinate_data.append([longitude, latitude])
            amuth_data.append(amuth)

    #plt.scatter(x, y, s=0.1)
    #plt.show()
    data = np.array(coordinate_data)
    kd = spatial.KDTree(coordinate_data)
    points, amuth = tools.dbscan2(coordinate_data, amuth_data, eps=_radius1, min_Pts=min_pts1, sd_angle=_sd_angle, kd=kd)

    kd = spatial.KDTree(points)
    points, amuth = tools.update_points(points, amuth, eps=_radius2, sd_angle=_sd_angle, kd=kd)
    kd = spatial.KDTree(points)
    points, amuth = tools.update_points(points, amuth, eps=_radius2, sd_angle=_sd_angle, kd=kd)
    kd = spatial.KDTree(points)
    points, amuth = tools.update_points(points, amuth, eps=_radius2, sd_angle=_sd_angle, kd=kd)
    kd = spatial.KDTree(points)
    points, amuth = tools.update_points(points, amuth, eps=_radius2, sd_angle=_sd_angle, kd=kd)
    kd = spatial.KDTree(points)
    points, amuth = tools.update_points(points, amuth, eps=_radius2, sd_angle=_sd_angle, kd=kd)
    kd = spatial.KDTree(points)
    points, amuth = tools.update_points(points, amuth, eps=_radius2, sd_angle=_sd_angle, kd=kd)
    kd = spatial.KDTree(points)
    points, amuth = tools.update_points(points, amuth, eps=_radius2, sd_angle=_sd_angle, kd=kd)
    kd = spatial.KDTree(points)
    points, amuth = tools.update_points(points, amuth, eps=_radius2, sd_angle=_sd_angle, kd=kd)
    kd = spatial.KDTree(points)
    points, amuth = tools.update_points(points, amuth, eps=_radius2, sd_angle=_sd_angle, kd=kd)

    kd = spatial.KDTree(points)
    points, amuth = tools.dbscan2(points, amuth, eps=_radius3, min_Pts=0, sd_angle=_sd_angle, kd=kd)



    with open("E:\毕业论文\Truck\\truck_data_to_reform_road_20170901_20170907_14.csv", 'w',newline='') as csv_f:
        csv_writer = csv.writer(csv_f)
        for coord, angle in zip(points, amuth):
            temp = [coord[0], coord[1], angle]
            csv_writer.writerow(temp)


    print(len(points))


    for i in range(len(points)):
        x.append(points[i][0])
        y.append(points[i][1])
    plt.scatter(x, y, s=0.1)
    plt.show()

    t1 = time.perf_counter()
    print("得到特征点共", len(points), "个, 花费时间", t1 - t0, "s")


'''
根据不同区域点的数量不同，选取不同的参数进行DBSCAN
'''


def get_points_according_to_truck_distribution(sd_angle, radius1, radius2, radius3, min_pts1):
    _sd_angle = sd_angle
    _radius1 = radius1
    _radius2 = radius2
    _radius3 = radius3
    _min_pts1 = min_pts1
    t0 = time.perf_counter()

    min_lon, min_lat = 99999, 99999
    max_lon, max_lat = 0, 0
    with open(data_path) as f:
        data = csv.reader(f)
        for row in data:
            longitude = float(row[2])
            latitude = float(row[3])
            amuth = int(row[7])
            coordinate_data.append([longitude, latitude])
            amuth_data.append(amuth)
            if longitude > max_lon:
                max_lon = longitude
            if longitude < min_lon:
                min_lon = longitude
            if latitude > max_lat:
                max_lat = latitude
            if latitude < min_lat:
                min_lat = latitude

    lon_step = (max_lon - min_lon) / 15
    lat_step = (max_lat - min_lat) / 15

    point_distribution_num = [0 for i in range(15 * 15)]
    points_distribution = [[] for i in range(15 * 15)]
    amuths = [[]for i in range(15*15)]
    for index, point in enumerate(coordinate_data):
        for i in range(15):
            for j in range(15):
                if point_in_area(i, j, point, lon_step, lat_step, min_lon, min_lat):
                    point_distribution_num[i* 15 + j] += 1
                    points_distribution[i*15+j].append(point)
                    amuths[i*15+j].append(amuth_data[index])

    data = np.array(coordinate_data)
    kd = spatial.KDTree(coordinate_data)
    t1 = time.perf_counter()
    print("得到卡车分布，共耗时",t1 - t0, 's')
    for i in range(15):
        for j in range(15):
            t5 = time.perf_counter()
            if point_distribution_num[i*15+j] < 20:
                continue
            kd = spatial.KDTree(points_distribution[i * 15 + j])
            if point_distribution_num[i*15+j] > 3000:
                points_distribution[i*15+j], amuths[i*15+j] = tools.dbscan2(points_distribution[i*15+j],amuths[i*15+j], eps=0.00008, min_Pts=10, sd_angle=2,kd=kd)
            else:
                points_distribution[i*15+j], amuths[i*15+j] = tools.dbscan2(points_distribution[i*15+j],amuths[i*15+j], eps=0.00005, min_Pts=7, sd_angle=3,kd=kd)
            t6 = time.perf_counter()
            print('完成第', i, '列，第', j, '行数据的聚类,共有数据',point_distribution_num[i*15+j],'个，耗时', t6-t5, 's ', len(points_distribution[i*15+j]), len(amuths[i*15+j]))
    points = []
    amuth = []
    for i in range(15*15):
        if point_distribution_num[i]<20:
            continue
        points = points + points_distribution[i]
        amuth = amuth + amuths[i]

    t2 = time.perf_counter()
    print("完成dbscan算法，共耗时", t2 - t1, 's')

    t7 = time.perf_counter()
    kd = spatial.KDTree(points)
    points, amuth = tools.update_points(points, amuth, eps=_radius2, sd_angle=_sd_angle, kd=kd)
    t8 = time.perf_counter()
    print("完成一次点更新，共 剩下", len(points), '个点，耗时',t8-t7, 's')
    kd = spatial.KDTree(points)
    points, amuth = tools.update_points(points, amuth, eps=_radius2, sd_angle=2, kd=kd)
    kd = spatial.KDTree(points)
    points, amuth = tools.update_points(points, amuth, eps=_radius2, sd_angle=2, kd=kd)
    kd = spatial.KDTree(points)
    points, amuth = tools.update_points(points, amuth, eps=_radius2, sd_angle=2, kd=kd)
    kd = spatial.KDTree(points)
    points, amuth = tools.update_points(points, amuth, eps=_radius2, sd_angle=2, kd=kd)
    kd = spatial.KDTree(points)
    points, amuth = tools.update_points(points, amuth, eps=_radius2, sd_angle=2, kd=kd)


    kd = spatial.KDTree(points)
    points, amuth = tools.dbscan2(points, amuth, eps=_radius3, min_Pts=0, sd_angle=_sd_angle, kd=kd)

    with open("../data/truck_data_to_reform_road_20170901_20170907_19.csv", 'w', newline='') as csv_f:
        csv_writer = csv.writer(csv_f)
        for coord, angle in zip(points, amuth):
            temp = [coord[0], coord[1], angle]
            csv_writer.writerow(temp)

    print(len(points))

    for i in range(len(points)):
        x.append(points[i][0])
        y.append(points[i][1])
    plt.rcParams['figure.dpi'] = 600
    plt.scatter(x, y, s=0.1)
    plt.show()

    t3 = time.perf_counter()
    print("得到特征点共", len(points), "个, 花费时间", t3 - t0, "s")


if __name__ == "__main__":
    get_points_according_to_truck_distribution(5, 0.0001, 0.0001, 0.00002, 10)

'''
y_pred = DBSCAN(eps=0.0001, min_samples=7).fit_predict(data)

plt.scatter(data[:,0], data[:,1], c= y_pred, s= 0.1)
plt.show()

x = []
y = []
for i in range(len(data)):
    if y_pred[i] != -1:
        x.append(data[i][0])
        y.append(data[i][1])
plt.scatter(x, y, s=0.1)
plt.show()

t2 = time.perf_counter()

print(t2 - t0)
'''