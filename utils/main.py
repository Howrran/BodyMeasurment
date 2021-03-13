import cv2
from utils.camera_calibration import CameraCalibration
from utils.distance_calculation import CalculateDistance

camera = CameraCalibration()
distance_calc = CalculateDistance()
PATH1 = '/home/howran/PycharmProjects/BodyMeasurment/1.jpg'
PATH2 = '//home/howran/PycharmProjects/BodyMeasurment/2.jpg'
PATH3 = '/home/howran/PycharmProjects/BodyMeasurment/3.jpg'

image = cv2.imread(PATH1)
arm_spread_image = cv2.imread(PATH2)
waist_image = cv2.imread(PATH3)
metric_x, metric_y = camera.get_metrics(image)
print(metric_x, metric_y)
distance = distance_calc.calculate_distance_between_points(arm_spread_image, metric_x, metric_y)
# distance = distance_calc.calculate_distance_between_preset_points(metric_x, metric_y, point1, point2)
print(distance)
