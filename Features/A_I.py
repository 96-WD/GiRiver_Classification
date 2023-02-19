from geographiclib.geodesic import Geodesic

def calc_angle(x1,y1,x2,y2,y3,x3):
    x1, y1, x2, y2, y3, x3 = map(float, [x1, y1, x2, y2, y3, x3])
    angle = Geodesic.WGS84.Inverse(y2, x2, y1, x1)['azi1'] - Geodesic.WGS84.Inverse(y2, x2, y3, x3)['azi1']
    angle = angle % 360
    return angle if angle < 180 else 360 - angle

file_name = '经纬度坐标.txt'
with open(file_name, 'r', encoding='utf-8') as f:
    contents = f.readlines()

my_lists = [xy.strip().split(',')[9:13]+xy.strip().split(',')[-2:] for xy in contents[1:]]

all_result = [calc_angle(*data) for data in my_lists]
print(all_result)