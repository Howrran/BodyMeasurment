from matplotlib import pyplot as plt
from gluoncv import model_zoo, data, utils
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
import cv2


class PoseDetector:

    detector = model_zoo.get_model('yolo3_mobilenet1.0_coco', pretrained=True)
    pose_net = model_zoo.get_model('simple_pose_resnet18_v1b', pretrained=True)

    def __init__(self):
        # Note that we can reset the classes of the detector to only include
        # human, so that the NMS process is faster.
        self.detector.reset_class(["person"], reuse_weights=['person'])
        self.PATH1 = '../1.jpg'
        self.PATH2 = '../2.jpg'
        self.PATH3 = '../3.jpg'

    def extract_pose_from_image(self, img, values):
        class_IDs, scores, bounding_boxs = self.detector(values)

        pose_input, upscale_bbox = detector_to_simple_pose(img, class_IDs, scores, bounding_boxs)
        predicted_heatmap = self.pose_net(pose_input)
        # left hand 5 7 9
        # right hand 6 8 10
        # left leg 11 13 15
        # right leg 12 14 16
        pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)

        ax = utils.viz.plot_keypoints(
            img, pred_coords, confidence,
            class_IDs, bounding_boxs, scores,
            box_thresh=0.5, keypoint_thresh=0.2
        )

        return pred_coords[0]

    def convert_coords(self, pose_coords):
        coords = []

        for coor in pose_coords:
            coords.append(coor.asnumpy().tolist())

        return coords
# a = PoseDetector()
# a.extract_pose_from_image('../2.jpg')
# # image = cv2.imread(PATH1)
# # arm_spread_image = cv2.imread(PATH2)
# # waist_image = cv2.imread(PATH3)
#
# x2, img2 = data.transforms.presets.ssd.load_test(PATH2, short=512)
#
#
# distance_calc = CalculateDistance()
# camera = CameraCalibration()
#
# metric_x, metric_y = camera.get_metrics(img)
# print(metric_x, metric_y)
#
# print('Shape of pre-processed image:', x.shape)
#
#
#
# print(pred_coords)
# for coor in pred_coords[0]:
#     # print(coor)
#     x, y = coor.asnumpy().tolist()
#     print(x, y)
#     # cv2.circle(img2, center=(int(x), int(y)), radius=2, color=(0,255,0),  thickness=-1)
#
#     # cv2.circle(img2, center=(243, 232), radius=2, color=(0, 255, 0), thickness=-1)
#

#
# for coor in pred_coords[0][5:13:6]:
#     # print(coor)
#     x, y = coor.asnumpy().tolist()
#     # print(x, y)
#     cv2.circle(img2, center=(int(x), int(y)), radius=2, color=(0,255,0),  thickness=-1)
#
#     # print(type(x), type(y)))
# # ax = utils.viz.plot_keypoints(img2, pred_coords, confidence,
# #                               class_IDs, bounding_boxs, scores,
# #                               box_thresh=0.5, keypoint_thresh=0.2)
#
# x1, y1 = pred_coords[0][5].asnumpy().tolist()
# x2, y2 = pred_coords[0][11].asnumpy().tolist()
#
# print(x1, x2)
#
# # distance = distance_calc.calculate_distance_between_points(img2, metric_x, metric_y)
# # print('DISTANCE : ', distance)
# distance = distance_calc.calculate_distance_between_preset_points(metric_x, metric_y, [x1, y1], [x2, y2])
# print('DISTANCE : ', distance)
#
# cv2.imshow('1', img2)
#
# cv2.waitKey()
# cv2.destroyAllWindows()