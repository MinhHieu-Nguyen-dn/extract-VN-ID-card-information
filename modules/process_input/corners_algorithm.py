from scipy.spatial import distance as dist
import numpy as np
import matplotlib.pyplot as plt
import itertools
import math
import cv2
from pylsd.lsd import lsd
import copy


def filter_corners(corners, min_dist=20):
    """
    Remove corners (points) that are too closed to other points.
    :param corners: All potential corners to filter.
    :param min_dist: Minimum distance between 2 potential corners (points).
    Default = 20; assign a value if you want to modify.
    :return: Corners that are clearly separated from each other.
    """
    filtered_corners = []

    for c in corners:
        if all(d > min_dist for d in [dist.euclidean(representative, c) for representative in filtered_corners]):
            filtered_corners.append(c)

    return filtered_corners


def sort_points(points):
    """
    Sort points: top-left -> top-right -> bottom-right -> bottom-left
    :param points: array of (4, 2) shape with each row has (x, y) coordinates.
    :return: A new list of sorted points.
    """
    xSorted = points[np.argsort(points[:, 0]), :]
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # top-left and bottom-left
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # bottom-right = farthest from top-left
    D = dist.cdist(tl[np.newaxis], rightMost, 'euclidean')[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :] # sort by D, reverse order and get the first one

    points = np.array([tl, tr, br, bl], dtype='float32')

    return points


def get_angle(point1, point2, point3):
    """
    Calculate angle at point 2 created by 2 lines: point2-point1 and point2-point3.
    :param point1: coordinates of point 1 with [x1, y1] format.
    :param point2: coordinates of point 2 with [x2, y2] format.
    :param point3: coordinates of point 3 with [x3, y3] format.
    :return: Angle value of the angle created from 2 lines from 3 points.
    """
    a = np.array(point1)
    b = np.array(point2)
    c = np.array(point3)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)


def angle_range(quad):
    """
    The range between maximum and minimum interior angles of a quadrilateral.
    :param quad: numpy array with shape (4, 1, 2) of 4 corners in clockwise order, starting with the top-left one.
    :return: Range of interior angles.
    """
    # format of each point: [a list of 1 element [x, y]]
    tl, tr, br, bl = quad

    upper_right = get_angle(tl[0], tr[0], br[0])
    upper_left = get_angle(bl[0], tl[0], tr[0])
    lower_right = get_angle(tr[0], br[0], bl[0])
    lower_left = get_angle(br[0], bl[0], tl[0])

    angles = [upper_right, upper_left, lower_right, lower_left]

    return np.ptp(angles)


def get_corners(edged,
                MIN_QUAD_AREA_RATIO=0.25, MAX_QUAD_ANGLE_RANGE=40):
    """
    Find 4 most potential corners of the ID card from an edged (rescaled) image.
    :param edged: Edged image (gray-scale and edged by Canny)
    :param MIN_QUAD_AREA_RATIO: modify if needed
    :param MAX_QUAD_ANGLE_RANGE: modify if needed
    :return: Coordinates of 4 most potential corners (points) in the RESCALED image.
    """

    img_height, img_width = edged.shape

    # initiate list of potential corners with format: [ list of (x, y)]
    corners = []
    # get lines's tuple (x1, y1, x2, y2, line's width) that represent edges in the edged image
    lines = lsd(edged)
    if lines.shape[0] > 1:
        lines = lines.squeeze().astype(np.int32).tolist()
    else:
        lines = lines.astype(np.int32).tolist()

    # initiate horizontal and vertical canvas to draw representative lines
    horizontal_lines_canvas = np.zeros(edged.shape, dtype=np.uint8)
    vertical_lines_canvas = np.zeros(edged.shape, dtype=np.uint8)

    for line in lines:
        x1, y1, x2, y2, _ = line

        # if width > height -> the representative line (of an edge) is in the horizontal canvas
        # if width < height -> representative line is in the vertical canvas
        if abs(x2 - x1) > abs(y2 - y1):
            (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[0]) # sorted by x coordinate
            cv2.line(horizontal_lines_canvas,
                     (max(x1 - 5, 0), y1), (min(x2+5, edged.shape[1] - 1), y2),
                     255, 2)

        else:
            (x1, y1), (x2, y2) = sorted(((x1, y1), (x2, y2)), key=lambda pt: pt[1]) # sorted by y coordinate
            cv2.line(vertical_lines_canvas,
                     (x1, max(y1 - 5, 0)), (x2, min(y2 + 5, edged.shape[0] - 1)),
                     255, 2)

    lines = []

    # find the horizontal lines from the horizontal canvas by getting top 3 contours in perimeter
    (contours, _) = cv2.findContours(horizontal_lines_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:2]
    horizontal_lines_canvas = np.zeros(edged.shape, dtype=np.uint8)

    for contour in contours:
        contour = contour.reshape((contour.shape[0], contour.shape[2]))

        min_x = np.amin(contour[:, 0], axis=0) + 2
        max_x = np.amax(contour[:, 0], axis=0) - 2
        left_y = int(np.average(contour[contour[:, 0] == min_x][:, 1]))
        right_y = int(np.average(contour[contour[:, 0] == max_x][:, 1]))

        lines.append((min_x, left_y, max_x, right_y))
        cv2.line(horizontal_lines_canvas, (min_x, left_y), (max_x, right_y), 1, 1)

        corners.append((min_x, left_y))
        corners.append((max_x, right_y))

    # vertical lines
    (contours, _) = cv2.findContours(vertical_lines_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)[:2]
    vertical_lines_canvas = np.zeros(edged.shape, dtype=np.uint8)

    for contour in contours:
        contour = contour.reshape((contour.shape[0], contour.shape[2]))
        min_y = np.amin(contour[:, 1], axis=0) + 2
        max_y = np.amax(contour[:, 1], axis=0) - 2
        top_x = int(np.average(contour[contour[:, 1] == min_y][:, 0]))
        bottom_x = int(np.average(contour[contour[:, 1] == max_y][:, 0]))

        lines.append((top_x, min_y, bottom_x, max_y))
        cv2.line(vertical_lines_canvas, (top_x, min_y), (bottom_x, max_y), 1, 1)

        corners.append((top_x, min_y))
        corners.append((bottom_x, max_y))

    # find intersections of horizontal and vertical canvases
    # y goes first because y coordinate = the number of row; x coordinate = the number of column
    # return: array of y(s) and array of x(s)
    corners_y, corners_x = np.where(horizontal_lines_canvas + vertical_lines_canvas == 2)
    corners += zip(corners_x, corners_y)

    # filter to keep points that are clearly separated from each other
    # remain the same format: [ list of (x, y)]
    corners = filter_corners(corners)

    # find contours approximately from potential corners
    # format: [ list of ]
    approx_contours = []

    def valid_contour(cnt):
        return len(cnt) == 4 and \
                cv2.contourArea(cnt) > img_width * img_height * MIN_QUAD_AREA_RATIO and \
                angle_range(cnt) < MAX_QUAD_ANGLE_RANGE

    if len(corners) >= 4:
        quads = []

        for quad in itertools.combinations(corners, 4):
            # itertools.combinations = [ list of lists [(x1, y1), (x2, y2)]]
            # format of quad: quad = [(x1, y1), (x2, y2)]
            # create an array of points with format: 1 row = 1 point; column 1 = x; column 2 = y
            points = np.array(quad)
            # sort points: top-left -> top-right -> bottom-right -> bottom-left
            # format of sorted points array: 1 row = 1 point; column 1 = x; column 2 = y
            points = sort_points(points) # shape = (4, 2)
            # change format of points to contour's format
            # format of points below: 1 row = 1 point = 1 list; 2 elements x, y in the list
            points = np.array([[p] for p in points], dtype='int32') # shape = (4, 1, 2)
            quads.append(points)

        # sort and get top 5 quads by area
        quads = sorted(quads, key=cv2.contourArea, reverse=True)[:5]
        # sort quads by angle range
        quads = sorted(quads, key=angle_range)

        # get the quad with the smallest angle range
        # since the more "rectangle" it is, the closer angles' value are (all close to 90 degree)
        approx = quads[0]

        if valid_contour(approx):
            approx_contours.append(approx)

    # check over contours from cv2 built-in function
    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    for c in cnts:
        approx = cv2.approxPolyDP(c, 80, True)  # approximate the connected curve through the contour
        if valid_contour(approx):
            approx_contours.append(approx)
            break

    # if no valid contours -> use the whole img
    if not approx_contours:
        TOP_RIGHT = (img_width, 0)
        BOTTOM_RIGHT = (img_width, img_height)
        BOTTOM_LEFT = (0, img_height)
        TOP_LEFT = (0, 0)
        return np.array([[TOP_RIGHT], [BOTTOM_RIGHT], [BOTTOM_LEFT], [TOP_LEFT]]).reshape(4, 2), False
    else:
        return max(approx_contours, key=cv2.contourArea).reshape(4, 2), True
