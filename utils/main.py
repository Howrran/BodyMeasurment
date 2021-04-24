import cv2
from utils.camera_calibration import CameraCalibration
from utils.distance_calculation import CalculateDistance
from utils.pose_detector import PoseDetector
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
from gluoncv import model_zoo, data, utils

camera = CameraCalibration()
distance_calc = CalculateDistance()
pose_detector = PoseDetector()

PATH1 = '/home/howran/PycharmProjects/BodyMeasurment/1.jpg'
# PATH1 = '/home/howran/PycharmProjects/BodyMeasurment/2_1.jpg'
PATH2 = '/home/howran/PycharmProjects/BodyMeasurment/2_2.jpg'
# PATH2 = '//home/howran/PycharmProjects/BodyMeasurment/2.jpg'
PATH3 = '/home/howran/PycharmProjects/BodyMeasurment/3.jpg'

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


# calculate left hand

# distance_calc.calculate_pose_distances(pose_coord)
pose_coord = pose_detector.convert_coords(pose_coord)

for coor in pose_coord:
    x, y = coor
    cv2.circle(image, center=(int(x), int(y)), radius=2, color=(0,255,0),  thickness=-1)

distance_calc.calculate_pose_distances(metric_x, metric_y, pose_coord)

cv2.imshow('1', image)

cv2.waitKey()
# distance = distance_calc.calculate_distance_between_points(image, metric_x, metric_y)
# print('DISTANCE : ', distance)
# cv2.waitKey()
# cv2.destroyAllWindows()
# print(pose_coord)
# print( distance_calc.calculate_distance_between_preset_points(metric_x, metric_y, pose_coord[5], pose_coord[7]) +
# distance_calc.calculate_distance_between_preset_points(metric_x, metric_y, pose_coord[7], pose_coord[9]))

