import csv
import time
import math
import Areas

data_path = "E:\毕业论文\Truck\\20170901.csv"

#北京市经纬度坐标
'''
max_east = 117.5
min_east = 115.4166666667
max_north = 41.05
min_north = 39.4333333333
'''
#
max_east = 116.60714285715714
min_east = 116.30952380954285
max_north = 39.89523809521428
min_north = 39.66428571425714

#划分区域
num_of_areas = 49
per_longitude = (max_east - min_east) / math.sqrt(num_of_areas)
per_latitude = (max_north - min_north) / math.sqrt(num_of_areas)
areas = []

for i in range(int(math.sqrt(num_of_areas))):
    for j in range(int(math.sqrt(num_of_areas))):
        e1 = min_east + per_longitude * j
        n1 = min_north + per_latitude * i
        e2 = min_east + per_longitude * (j + 1)
        n2 = min_north + per_latitude * (i + 1)
        tt = Areas.Area(e1, n1, e2, n2)
        areas.append(tt)



t0 = time.perf_counter()

with open(data_path) as f:
    data = csv.reader(f)
    ans = 0
    for row in data:
        east = float(row[2])
        north = float(row[3])
        if east < min_east or east > max_east:
            continue
        if north < min_north or north > max_north:
            continue
        for area in areas:
            area.in_area(east, north)

for area in areas:
    print(area.east1, area.north1, area.east2, area.north2, area.truck_num)


t1 = time.perf_counter()

print(t1 - t0)