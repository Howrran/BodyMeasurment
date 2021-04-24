import cv2
import mediapipe as mp
import numpy as np


class MediaPipeBase:
    """
    Class for using Google MediaPipe library

    for face features numbers check mesh_map.jpeg or
    https://github.com/tensorflow/tfjs-models/blob/master/facemesh/mesh_map.jpg
    """
    pose_features = {
        'indexes': [i for i in range(33)],
        'coordinates': None,
        'lines': [[16, 14], [12, 14], [12, 11], [12, 24], [11, 13], [13, 15], [11, 23], [24, 23],
                  [24, 26], [26, 28], [23, 25], [25, 27],
                  # [28, 30], [28, 32], [32, 30],
                  # [27, 29], [27, 31], [31, 29]
                  ]
        }

    @staticmethod
    def _process_image_by_analyzer(image, analyzer):
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = analyzer.process(image)
        # Draw the face mesh annotations on the image.
        image.flags.writeable = True

        return results, image

    @staticmethod
    def _denormalize_feature(image_height, image_width, x, y):
        """
        In google mesh library coordinates are normalized to image percentage
        This method denormalize them to get actual pixel indexes
        """
        denormalized_x = int(round(image_width * (x * 100) / 100))
        denormalized_y = int(round(image_height * (y * 100) / 100))

        return denormalized_x, denormalized_y

    def _get_feature_coordinates_from_landmarks(self, image_height, image_width, face_landmarks, feature_indexes):
        """
        Return np.array of specific feature(eg. lips, forehead) pixels coordinates in the image
        """
        return np.array(
            [
                self._denormalize_feature(
                    image_height,
                    image_width,
                    face_landmarks.landmark[feature_index].x,
                    face_landmarks.landmark[feature_index].y
                )
                for feature_index in feature_indexes
            ]
        )

    @staticmethod
    def _get_feature_pixels_from_coordinates(image, feature_coordinates):
        try:
            return np.array([image[x, y] for y, x in feature_coordinates])
        except IndexError:
            return None

    @staticmethod
    def _draw_line(image, x, y, r=255, g=255, b=255):
        cv2.line(image, x, y, (b, g, r), lineType=cv2.LINE_AA)

    def draw_pose_features(self, image, pose_landmarks):
        height, width, _ = image.shape
        features_coordinates = {}

        pose_coordinates = self._get_feature_coordinates_from_landmarks(
            image_height=height,
            image_width=width,
            face_landmarks=pose_landmarks,
            feature_indexes=self.pose_features['indexes']
        )

        features_coordinates.update(dict(zip(self.pose_features['indexes'], pose_coordinates)))

        for line_feature in self.pose_features['lines']:
            feature1, feature2 = line_feature
            x1, y1 = features_coordinates[feature1]
            x2, y2 = features_coordinates[feature2]
            self._draw_line(image, (x1, y1), (x2, y2), r=0, g=255, b=0)

        return image, features_coordinates

    def analyze_image(self, image):
        pose =  mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

        image = cv2.imread(image)
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        img, pose_coordinates = self.draw_pose_features(image, results.pose_landmarks)
        # cv2.imshow('1', img)
        # cv2.waitKey()

        return  pose_coordinates
# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose
#
# # For static images:
# pose =  mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
# image = cv2.imread('/home/howran/PycharmProjects/BodyMeasurment/2_2.jpg')

# a = MediaPipeBase()
# a.analyze_image('/home/howran/PycharmProjects/BodyMeasurment/2_2.jpg')
# image_height, image_width, _ = image.shape
# # Convert the BGR image to RGB before processing.
# results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#
# if not results.pose_landmarks:
#   pass
# print(
#     f'Nose coordinates: ('
#     # f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width}, '
#     # f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_height})'
# )
# # Draw pose landmarks on the image.
# annotated_image = image.copy()
# # Use mp_pose.UPPER_BODY_POSE_CONNECTIONS for drawing below when
# # upper_body_only is set to True.
# mp_drawing.draw_landmarks(
#     annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
#
# cv2.imshow('1', annotated_image)
# cv2.waitKey()