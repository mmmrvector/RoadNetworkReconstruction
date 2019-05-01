from reconstruct import reform_path
from model import data_type
from reconstruct import tools



#这个是python 会java，但是不会.net
A = [116.4322102, 39.71634492]
B = [116.4322715, 39.71567964]
C = [116.4322588, 39.71606775]
print(tools.cal_point_2_line(C, A, B))