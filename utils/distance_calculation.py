import cv2
import math

class CalculateDistance:
    draw_radius = 5

    def calculate_distance_between_points(self, image, metric_x, metric_y):
        points = self.get_points(image)
        distance = self.getDistance(points[0], points[1])
        distance = self.pixel_to_distance(distance, metric_x, metric_y)

        return distance

    def calculate_distance_between_preset_points(self, metric_x, metric_y, point1, point2):
        # points = self.get_points(image)
        distance = self.getDistance(point1, point2)
        distance = self.pixel_to_distance(distance, metric_x, metric_y)

        return distance

    def calculate_pose_distances(self, metric_x, metric_y, pose_coords):
        # left hand 5 7 9
        # right hand 6 8 10
        # left leg 11 13 15
        # right leg 12 14 16
        left_hand = (
                self.calculate_distance_between_preset_points(metric_x, metric_y, pose_coords[5], pose_coords[7])
                +
                self.calculate_distance_between_preset_points(metric_x, metric_y, pose_coords[7], pose_coords[9])
        )
        left_leg = (
                self.calculate_distance_between_preset_points(metric_x, metric_y, pose_coords[11], pose_coords[13])
                +
                self.calculate_distance_between_preset_points(metric_x, metric_y, pose_coords[13], pose_coords[15])
        )
        right_hand = (
                self.calculate_distance_between_preset_points(metric_x, metric_y, pose_coords[6], pose_coords[8])
                +
                self.calculate_distance_between_preset_points(metric_x, metric_y, pose_coords[8], pose_coords[10])
        )
        right_leg = (
                self.calculate_distance_between_preset_points(metric_x, metric_y, pose_coords[12], pose_coords[14])
                +
                self.calculate_distance_between_preset_points(metric_x, metric_y, pose_coords[14], pose_coords[16])
        )
        print('PoseNet: ')

        print('left hand: ', left_hand)
        print('left leg: ', left_leg)
        print('right hand: ', right_hand)
        print('right leg: ', right_leg)


    def calculate_mediapipe_pose_distances(self, image, metric_x, metric_y, pose_coords):
        # left hand 11 13 15
        # right hand 12 14 16
        # left leg 23 25 27
        # right leg 24 26 28
        left_hand = (
                self.calculate_distance_between_preset_points(metric_x, metric_y, pose_coords[11], pose_coords[13])
                +
                self.calculate_distance_between_preset_points(metric_x, metric_y, pose_coords[13], pose_coords[15])
        )/2
        left_leg = (
                self.calculate_distance_between_preset_points(metric_x, metric_y, pose_coords[24], pose_coords[26])
                +
                self.calculate_distance_between_preset_points(metric_x, metric_y, pose_coords[26], pose_coords[28])
        )/2
        right_hand = (
                self.calculate_distance_between_preset_points(metric_x, metric_y, pose_coords[12], pose_coords[14])
                +
                self.calculate_distance_between_preset_points(metric_x, metric_y, pose_coords[14], pose_coords[16])
        )/2
        right_leg = (
                self.calculate_distance_between_preset_points(metric_x, metric_y, pose_coords[24], pose_coords[26])
                +
                self.calculate_distance_between_preset_points(metric_x, metric_y, pose_coords[26], pose_coords[28])
        )/2

        shoulder = self.calculate_distance_between_preset_points(metric_x, metric_y, pose_coords[11], pose_coords[12]) /2
        torso = self.calculate_distance_between_preset_points(metric_x, metric_y, pose_coords[12], pose_coords[24]) /2


        print('MediaPipe: ')
        print('left hand: ', left_hand)
        print('left leg: ',  left_leg)
        print('right hand: ', right_hand)
        print('right leg: ', right_leg)
        print('shoulder: ', shoulder)
        print('torso: ', torso)

        distance_keypoint = {}
        distance_keypoint[13] = left_hand
        distance_keypoint[14] = right_hand
        distance_keypoint[25] = left_leg
        distance_keypoint[26] = right_leg
        distance_position = {13: (20, 60), 14: (30, 45), 25: (-20, 0), 26: (80, 0)}

        for coor in pose_coords:
            if coor in [11, 13, 15, 12, 14, 16, 23, 25, 27, 24, 26, 28]:
                x, y = pose_coords[coor]
                cv2.circle(image, center=(int(x), int(y)), radius=2, color=(0, 255, 0), thickness=-1)
                if coor in [13, 14, 25, 26]:
                    r, t = distance_position[coor]
                    cv2.putText(image, str(round(distance_keypoint[coor], 1)), (x - r, y - t), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                                cv2.LINE_AA)

        x1, y1 = pose_coords[11]
        x2, y2 = pose_coords[12]
        cv2.putText(image, str(round(shoulder, 1)), (int((x1 + x2) / 2) - 20, int((y1 + y2) / 2) - 15), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2, cv2.LINE_AA)
        x1, y1 = pose_coords[12]
        x2, y2 = pose_coords[24]
        cv2.putText(image, str(round(torso, 1)), (int((x1 + x2) / 2) - 80, int((y1 + y2) / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Calculation Results', image)
        cv2.waitKey()
        cv2.imwrite('im_numb.jpg', image)

    def get_points(self, img):
        points = []
        img_to_show = img.copy()

        def draw_circle(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                cv2.circle(img_to_show, (x, y), self.draw_radius, (255, 0, 0), -1)
                points.append([x, y])

        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', img.shape[0], img.shape[1])
        cv2.setMouseCallback('image', draw_circle)

        while (1):
            cv2.imshow('image', img_to_show)
            k = cv2.waitKey(20) & 0xFF
            if k == 27:
                break
        cv2.destroyAllWindows()
        return points

    @staticmethod
    def getDistance(p1, p2):
        return (p1[0] - p2[0], p1[1] - p2[1])

    @staticmethod
    def pixel_to_distance(p1, mx, my):
        return math.sqrt((p1[0] * mx) ** 2 + (p1[1] * my) ** 2)

