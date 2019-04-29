import csv
import time
data_path = "E:\毕业论文\Truck\\new_data.csv"
#new_data_path = "E:\毕业论文\Truck\\new_data_without_static_truck.csv"
new_data_path = "E:\毕业论文\Truck\\new_data_without_static_truck_20170901_20170907.csv"
total = 0
new_data = []

t0 = time.perf_counter()
dict = {}
with open(data_path) as f:
    data = csv.reader(f)
    for row in data:
        longitude = float(row[2])
        latitude = float(row[3])
        truck_no = row[0]
        if truck_no not in dict:
            dict[truck_no] = []
            dict[truck_no].append([truck_no, row[1], longitude, latitude, row[4], row[5], row[6], row[7]])
        else:
            dict[truck_no].append([truck_no, row[1], longitude, latitude, row[4], row[5], row[6], row[7]])
l = len(dict)
for key in dict:
    tt = dict[key]
    #删去静止的卡车数据
    new_tt = []
    for dd in tt:
        flag = True
        for dd2 in new_tt:
            if dd2[2] == dd[2] and dd2[3] == dd[3]:
                flag = False
                break
        if flag:
            new_tt.append(dd)
    for dd in new_tt:
        new_data.append(dd)





index = 0
with open(new_data_path, 'w', newline ='') as csv_f:
    csv_writer = csv.writer(csv_f)
    for row in new_data:
        index += 1
        total += 1
        if index % 100 == 0:
            print('...')
        csv_writer.writerow(row)

print(total)


t1 = time.perf_counter()

print(t1 - t0)