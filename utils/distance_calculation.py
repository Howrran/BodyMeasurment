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

        print('left hand: ', left_hand)
        print('left leg: ', left_leg)
        print('right hand: ', right_hand)
        print('right leg: ', right_leg)

    def calculate_mediapipe_pose_distances(self, metric_x, metric_y, pose_coords):
        # left hand 11 13 15
        # right hand 12 14 16
        # left leg 23 25 27
        # right leg 24 26 28
        left_hand = (
                self.calculate_distance_between_preset_points(metric_x, metric_y, pose_coords[11], pose_coords[13])
                +
                self.calculate_distance_between_preset_points(metric_x, metric_y, pose_coords[13], pose_coords[15])
        )
        left_leg = (
                self.calculate_distance_between_preset_points(metric_x, metric_y, pose_coords[24], pose_coords[26])
                +
                self.calculate_distance_between_preset_points(metric_x, metric_y, pose_coords[26], pose_coords[28])
        )
        right_hand = (
                self.calculate_distance_between_preset_points(metric_x, metric_y, pose_coords[12], pose_coords[14])
                +
                self.calculate_distance_between_preset_points(metric_x, metric_y, pose_coords[14], pose_coords[16])
        )
        right_leg = (
                self.calculate_distance_between_preset_points(metric_x, metric_y, pose_coords[24], pose_coords[26])
                +
                self.calculate_distance_between_preset_points(metric_x, metric_y, pose_coords[26], pose_coords[28])
        )

        print('left hand: ', left_hand)
        print('left leg: ', left_leg)
        print('right hand: ', right_hand)
        print('right leg: ', right_leg)

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

