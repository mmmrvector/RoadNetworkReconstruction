import matplotlib.pyplot as plt
import csv

points = []
amuths = []



def point_in_area(i, j, point, lon_step, lat_step, min_lon, min_lat):
    if point[0] >= min_lon + i * lon_step and point[0] < min_lon + (i + 1) * lon_step:
        if point[1] >= min_lat + j * lat_step and point[1] < min_lat + (j + 1) * lat_step:
            return True
    return False




def get_ditribution():
    min_lon, min_lat = 99999, 99999
    max_lon, max_lat = 0, 0
    with open("../data/new_data_without_static_truck_20170901_20170907.csv") as csv_f:
        data = csv.reader(csv_f)
        for row in data:
            longitude = float(row[2])
            latitude = float(row[3])
            amuth = int(row[7])
            points.append([longitude, latitude])
            amuths.append(amuth)
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

    point_distribute = [0 for i in range(15*15)]

    for point in points:
        for i in range(15):
            for j in range(15):
                if point_in_area(i, j, point, lon_step, lat_step, min_lon, min_lat):
                    point_distribute[i* 15 + j] += 1


    temp = []
    for index, ele in enumerate(point_distribute):
        temp.append(point_distribute[index])
        if index % 15 == 14:
            print(temp)
            temp = []



    x = [116.44]
    y = [39.72]
    x_ticks = []
    y_ticks = []
    for i in range(16):
        x_ticks.append(min_lon + i *lon_step)
        y_ticks.append(min_lat + i * lat_step)

    return x_ticks, y_ticks, point_distribute, lon_step, lat_step, min_lon, min_lat

'''
    fig = plt.figure()
    plt.rcParams['figure.dpi'] = 600
    plt.scatter(x, y, s= 1)
    plt.grid(True)
    plt.tick_params(labelsize=6)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    for i in range(15):
        for j in range(15):
            plt.text(min_lon + (i + 0.2) * lon_step, min_lat + (j + 0.2) * lat_step, point_distribute[i*15 +j], fontsize=6)

    plt.show()
'''
if __name__ =="__main__":
    get_ditribution()