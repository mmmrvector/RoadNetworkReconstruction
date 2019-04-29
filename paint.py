import matplotlib.pyplot as plt
import csv
import random
from sklearn.cluster import DBSCAN
import numpy as np
import tools
import time



data_path = "E:\毕业论文\Truck\\new_data_without_static_truck.csv"
x = []
y = []
dict = {}
amuth_dict = {}



t0 = time.perf_counter()

with open(data_path) as f:
    data = csv.reader(f)
    for row in data:
        longitude = float(row[2]) - 116
        latitude = float(row[3]) - 39
        amuth = int(row[7])
        truck_no = row[0]
        if truck_no not in dict:
            dict[truck_no] = []
            dict[truck_no].append([longitude, latitude])
        else:
            dict[truck_no].append([longitude, latitude])
l = len(dict)

data = [i for i in range(l)]
slice = random.sample(data, int(l))
data = []
amuth_data = []


for key in dict:
    tt = dict[key]
    data.append(tt)

data2 = []

for i in slice:
    data2.append(data[i])
data = []

for v in data2:
    for v2 in v:
        data.append([v2[0], v2[1]])


#plt.scatter(x, y, s=0.1)
#plt.show()
data = np.array(data)
points = tools.dbscan(data, eps=0.0001, min_Pts=7)
points = tools.update_points(points, eps=0.0001)
points = tools.update_points(points, eps=0.0001)
points = tools.update_points(points, eps=0.0001)


points = tools.dbscan(points, eps=0.00005, min_Pts=0)
print(len(points))

for i in range(len(points)):
    x.append(points[i][0])
    y.append(points[i][1])
plt.scatter(x, y, s=0.1)
plt.show()

t1 = time.perf_counter()
print(t1 - t0)

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