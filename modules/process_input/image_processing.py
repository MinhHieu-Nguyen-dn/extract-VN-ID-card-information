import cv2
import numpy as np


def get_constants(height, width,
                  MORPH=9, CANNY=84, HOUGH=25):
    """
    Initiate constants that are used in the image pre-processing process.
    Modify these constants by assigning them while calling the function.
    :param height: Height of the image.
    :param width: Width of the image.
    :return: A dictionary of constants: MORPH, CANNY, HOUGH.
    """

    return {
        'MORPH': MORPH,
        'CANNY': CANNY,
        'HOUGH': HOUGH,
    }


def to_gray(img):
    """
    :param img: Input image.
    :return: Gray-scale image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7,7), 0)

    return gray


def to_dilated(img, MORPH):
    """
    :param img: Gray-scale image.
    :param MORPH: A constant of the (closing) dilation process.
    :return: Dilated image.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH, MORPH))
    dilated = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    return dilated


def to_edged(img, CANNY):
    """
    :param img: Dilated image.
    :param CANNY: A constant to find edges of the input image with Canny.
    :return: Edged image.
    """
    edged = cv2.Canny(img, 0, CANNY)

    return edged


def to_bird_eye_view(orig, corners):
    """
    :param orig: Original image.
    :param corners: 4 potential corners of the object.
    :return: A new image that is cropped around the object and bird-eye-view transformed.
    """
    (tl, tr, br, bl) = corners

    # bird-eye view's width = max width of (tl - tr) or (bl - br)
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # bird-eye view's height = max height of (tl - bl) or (tr - br)
    heightA = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    heightB = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # destination points
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype='float32')

    # perspective transform
    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

    return warped

