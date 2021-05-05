import cv2
from utils.camera_calibration import CameraCalibration
from utils.distance_calculation import CalculateDistance
from utils.pose_detector import PoseDetector
from utils.google_mediapipe import MediaPipeBase
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
from gluoncv import model_zoo, data, utils

camera = CameraCalibration()
distance_calc = CalculateDistance()
pose_detector = PoseDetector()

PATH1 = '/home/howran/PycharmProjects/BodyMeasurment/2_1.jpg'
# PATH1 = '/home/howran/PycharmProjects/BodyMeasurment/2_1.jpg'
PATH2 = '/home/howran/PycharmProjects/BodyMeasurment/2_2.jpg'
# PATH2 = '//home/howran/PycharmProjects/BodyMeasurment/2.jpg'
PATH3 = '/home/howran/PycharmProjects/BodyMeasurment/2_3.png'
PATH4 = '/home/howran/PycharmProjects/BodyMeasurment/2_4.png'

#############################################################

# image = cv2.imread(PATH1)
# arm_spread_image = cv2.imread(PATH2)
# waist_image = cv2.imread(PATH3)
# metric_x, metric_y = camera.get_metrics(image)
# print(metric_x, metric_y)
# distance = distance_calc.calculate_distance_between_points(arm_spread_image, metric_x, metric_y)
# # distance = distance_calc.calculate_distance_between_preset_points(metric_x, metric_y, point1, point2)
# print(distance)

################################################################

_, calibration_image = data.transforms.presets.ssd.load_test(PATH1, short=512)
values, image = data.transforms.presets.ssd.load_test(PATH2, short=512)

metric_x, metric_y = camera.get_metrics(calibration_image)
pose_coord = pose_detector.extract_pose_from_image(image,values)

####
pose_coord = pose_detector.convert_coords(pose_coord)

for coor in pose_coord:
    x, y = coor
    cv2.circle(image, center=(int(x), int(y)), radius=2, color=(0,255,0),  thickness=-1)

distance_calc.calculate_pose_distances(metric_x, metric_y, pose_coord)
#
# cv2.imshow('1', image)
# cv2.waitKey()

####
# MP
###
mp = MediaPipeBase()

pose_coord, im = mp.analyze_image(PATH2)
# im = cv2.imread(PATH2)

f = {13: 55.6, 14: 56.2, 25: 86.9, 26:86.7}
f2 = {13: (20, 60), 14: (30, 45), 25: (-20, 0), 26:(80, 0)}
for coor in pose_coord:
    if coor in [11, 13, 15, 12, 14, 16, 23, 25, 27, 24, 26, 28]:
        x, y = pose_coord[coor]
        cv2.circle(im, center=(int(x), int(y)), radius=2, color=(0,255,0),  thickness=-1)
        if coor in [13, 14, 25, 26]:
            r, t = f2[coor]
            cv2.putText(im, str(f[coor]), (x - r , y - t), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

x1, y1 = pose_coord[11]
x2, y2 = pose_coord[12]
cv2.putText(im, '36.4', (int((x1 + x2)/2) - 20 , int((y1 + y2) /2) - 15 ), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
x1, y1 = pose_coord[12]
x2, y2 = pose_coord[24]
cv2.putText(im, '53.2', (int((x1 + x2)/2) - 80 , int((y1 + y2) /2) ), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)

cv2.imshow('2', im)
cv2.waitKey()
cv2.imwrite('im_numb.jpg', im)

distance_calc.calculate_mediapipe_pose_distances(metric_x, metric_y, pose_coord)



# distance = distance_calc.calculate_distance_between_points(image, metric_x, metric_y)
# print('DISTANCE : ', distance)
# cv2.waitKey()
# cv2.destroyAllWindows()
# print(pose_coord)
# print( distance_calc.calculate_distance_between_preset_points(metric_x, metric_y, pose_coord[5], pose_coord[7]) +
# distance_calc.calculate_distance_between_preset_points(metric_x, metric_y, pose_coord[7], pose_coord[9]))

