import cv2
from utils.camera_calibration import CameraCalibration
from utils.distance_calculation import CalculateDistance
from utils.pose_detector import PoseDetector
from utils.google_mediapipe import MediaPipeBase
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
from gluoncv import model_zoo, data, utils

import argparse

my_parser = argparse.ArgumentParser(description='List the content of a folder')

# Add the arguments
my_parser.add_argument('Calibration_Path',
                       metavar='Calibration_Path',
                       type=str,
                       help='the path to calibration photo')

my_parser.add_argument('Body_Path',
                       metavar='Body_Path',
                       type=str,
                       help='the path to body photo')

# args = my_parser.parse_args()

camera = CameraCalibration()
distance_calc = CalculateDistance()
pose_detector = PoseDetector()


PATH1 = '/home/howran/PycharmProjects/BodyMeasurment/11.jpg'
PATH2 = '/home/howran/PycharmProjects/BodyMeasurment/2_2.jpg'


_, calibration_image = data.transforms.presets.ssd.load_test(PATH1, short=512)
values, image = data.transforms.presets.ssd.load_test(PATH2, short=512)

metric_x, metric_y = camera.get_metrics(calibration_image)
pose_coord = pose_detector.extract_pose_from_image(image,values)

####
pose_coord = pose_detector.convert_coords(pose_coord)

distance_calc.calculate_pose_distances(metric_x, metric_y, pose_coord)

####
# MP
###
mp = MediaPipeBase()

pose_coord, im = mp.analyze_image(PATH2)

distance_calc.calculate_mediapipe_pose_distances(im, metric_x, metric_y, pose_coord)
