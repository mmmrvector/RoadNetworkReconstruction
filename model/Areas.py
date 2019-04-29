
class Area:
    east1 = 0
    north1 = 0
    east2 = 0
    north2 = 0
    truck_num = 0

    def __init__(self, e1, n1, e2, n2):
        self.east1 = e1
        self.north1 = n1
        self.east2 = e2
        self.north2 = n2
        self.truck_num = 0

    def in_area(self, e, n):
        if (e >= self.east1 and e <= self.east2) and (n >= self.north1 and n <= self.north2):
            self.truck_num += 1
