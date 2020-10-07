import cv2
import math

class CalculateDistance:
    draw_radius = 5

    def calculate_distance_between_points(self, image, metric_x, metric_y):
        points = self.get_points(image)
        distance = self.getDistance(points[0], points[1])
        distance = self.pixel_to_distance(distance, metric_x, metric_y)

        return distance

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

