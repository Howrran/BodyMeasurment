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
