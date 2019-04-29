import matplotlib.pyplot as plt
import csv
import random
from sklearn.cluster import DBSCAN
import numpy as np
import tools
import time
from scipy import spatial



#data_path = "E:\毕业论文\Truck\\new_data_without_static_truck.csv"
data_path = "E:\毕业论文\Truck\\new_data_without_static_truck_20170901_20170907.csv"

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

if __name__ == "__main__":
    get_points(5, 0.0001, 0.0001, 0.00005, 10)

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