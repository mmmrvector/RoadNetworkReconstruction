import csv
import time
data_path = "E:\毕业论文\Truck\\20170901.csv"
data_path2 = "E:\毕业论文\Truck\\20170902.csv"
data_path3 = "E:\毕业论文\Truck\\20170903.csv"
new_data_path = "E:\毕业论文\Truck\\new_data.csv"
total = 0
new_data = []

t0 = time.perf_counter()

with open(data_path) as f:
    data = csv.reader(f)
    for row in data:
        try:
            longitude = float(row[2])
            latitude = float(row[3])
        except Exception as e:
            continue
        if longitude >= 116.4282989502 and longitude <= 116.4508295059 and latitude >= 39.7150109122 and latitude <= 39.7293365731:
            total += 1
            new_data.append(row)

with open(data_path2) as f:
    data = csv.reader(f)
    for row in data:
        try:
            longitude = float(row[2])
            latitude = float(row[3])
        except Exception as e:
            continue
        if longitude >= 116.4282989502 and longitude <= 116.4508295059 and latitude >= 39.7150109122 and latitude <= 39.7293365731:
            total += 1
            new_data.append(row)

with open(data_path3) as f:
    data = csv.reader(f)
    for row in data:
        try:
            longitude = float(row[2])
            latitude = float(row[3])
        except Exception as e:
            continue
        if longitude >= 116.4282989502 and longitude <= 116.4508295059 and latitude >= 39.7150109122 and latitude <= 39.7293365731:
            total += 1
            new_data.append(row)
for i in range(4):
    with open("E:\毕业论文\Truck\\2017090"+str(i+4) + ".csv") as f:
        data = csv.reader(f)
        for row in data:
            try:
                longitude = float(row[2])
                latitude = float(row[3])
            except Exception as e:
                continue
            if longitude >= 116.4282989502 and longitude <= 116.4508295059 and latitude >= 39.7150109122 and latitude <= 39.7293365731:
                total += 1
                new_data.append(row)

index = 0

with open(new_data_path, 'w', newline ='') as csv_f:
    csv_writer = csv.writer(csv_f)
    for row in new_data:
        index += 1
        if index % 100 == 0:
            print('...')
        csv_writer.writerow(row)


print(total)


t1 = time.perf_counter()

print(t1 - t0)