import argparse
import cv2
import numpy as np
# import matplotlib.pyplot as plt
import math


class CameraCalibration:
    refPt = []
    r1 = 5  # for affine correction
    r2 = 2  # for measurement
    # ref_ht = 2.84
    # ref_ht = 3.45
    ref_ht = 1.62
    # ref_ht = 0.82  # mediapipe
    rectangle_row = 9
    rectangle_col = 6
    # square_size=int(r+1)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    metre_pixel_x = 0
    metre_pixel_y = 0
    window_name1 = "image"
    draw_radius = 10

    def get_metrics(self, image):
        metre_pixel_x, metre_pixel_y = self.analyze_chessboard(image)
        metre_pixel_x, metre_pixel_y = metre_pixel_x, metre_pixel_x

        return metre_pixel_x, metre_pixel_y

    def analyze_chessboard(self, image):
        dst = np.copy(image)  # created to ease affine_correct mode

        gray2 = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        temp = self.chess_board_corners(dst, gray2)

        ret, corners = cv2.findChessboardCorners(dst, (self.rectangle_row, self.rectangle_col), None)

        cv2.cornerSubPix(gray2, corners, (11, 11), (-1, -1), self.criteria)
        cv2.drawChessboardCorners(dst, (9, 6), corners, ret)

        metre_pixel_x = (self.r2 * self.ref_ht) / (abs(temp[0][0] - temp[1][0]))
        metre_pixel_y = (self.r2 * self.ref_ht) / (abs(temp[0][1] - temp[2][1]))


        return metre_pixel_x, metre_pixel_y

    def chess_board_corners(self, image, gray):
        """
        Return 4 points at square_size of checkboard
        """
        square_size = int(self.r2 + 1)
        ret, corners = cv2.findChessboardCorners(image, (self.rectangle_row, self.rectangle_col), None)

        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)
        corners2 = corners

        coordinates = []
        coordinates.append(
            (
                corners2[0, 0, 0],
                corners2[0, 0, 1]
            )
        )
        coordinates.append(
            (
                corners2[square_size - 1, 0, 0],
                corners2[square_size - 1, 0, 1]
            )
        )
        coordinates.append(
            (
                corners2[self.rectangle_row * (square_size - 1), 0, 0],
                corners2[self.rectangle_row * (square_size - 1), 0, 1]
            )
        )
        coordinates.append(
            (
                corners2[self.rectangle_row * (square_size - 1) + square_size - 1, 0, 0],
                corners2[self.rectangle_row * (square_size - 1) + square_size - 1, 0, 1]
            )
        )
        return coordinates
