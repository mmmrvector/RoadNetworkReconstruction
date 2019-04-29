import csv
import time
from reconstruct import get_feature_points
from reconstruct import reform_path

sd_angle = 3
radius1 = 0.0001
min_pts1 = 7
radius2 = 0.0001
radius3 = 0.00005
cur_point_angle = 15
fork_angle = 70
radius4 = 0.0004



#paint2.get_points(sd_angle, radius1, radius2, radius3, min_pts1)
reform_path.reform_path(radius4, cur_point_angle, fork_angle)