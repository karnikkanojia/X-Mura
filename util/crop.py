# Python 2/3 compatibility
import numpy as np
import cv2
import imutils
import os
import sys
import pathlib
from os.path import basename, dirname
import configparser

config = configparser.ConfigParser()
config.read('config.ini')
DATA_PATH = config.get('training-parameters', 'root')


PY3 = sys.version_info[0] == 3

if PY3:
    xrange = range

SHOW = True


def angle_cos(p0, p1, p2):
    d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
    return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))


def find_squares(img, min_area=100000, max_skew=0.45):
    """
    A method to find inner square images on bigger images
    :param min_area: specifies minimal square area in pixels
    :param max_skew: specifies maximum skewness of squares
    :param img: numpy array representation of an image
    :return: list of found squares
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    squares = []
    for gray in cv2.split(img):
        for thrs in xrange(0, 255, 26):
            if thrs == 0:
                bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                bin = cv2.dilate(bin, None)
            else:
                _retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
            bin = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(bin)
            for cnt in contours:
                cnt_len = cv2.arcLength(cnt, True)
                cnt = cv2.approxPolyDP(cnt, 0.02 * cnt_len, True)
                if len(cnt) >= 4 and cv2.contourArea(cnt) >= min_area \
                        and cv2.isContourConvex(cnt):
                    cnt = cnt.reshape(-1, 2)
                    max_cos = np.max([angle_cos(cnt[i], cnt[(i + 1) % 4],
                                                cnt[(i + 2) % 4])
                                      for i in xrange(4)])
                    if max_cos < max_skew:
                        squares.append(cnt)
    return squares


def crop_squares(squares, img):
    rect = cv2.minAreaRect(squares[0])
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    if SHOW:
        cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

    width = int(rect[1][0])
    height = int(rect[1][1])

    src_pts = box.astype("float32")
    dst_pts = np.array([[0, height - 1],
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1]],
                       dtype="float32")

    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (width, height))

    if width > height:
        warped = imutils.rotate_bound(warped, 270)

    if SHOW:
        cv2.imshow("crop_img.jpg", warped)

    return warped


def main():
    print("Started...")
    """
    Runs script to find and crop squares from the data folder with predefined
    folder structure /patient_folder/inner_folder/xxx.png
    """
    patients_cnt = 0
    path_to_data = DATA_PATH+'/'
    paths = list(pathlib.Path(path_to_data).rglob('*.png'))
    patients_cnt = 0
    for path in paths:
        print(f'Converting image: {path.as_posix()}')
        patients_cnt += 1
        fn = path.as_posix()
        img = cv2.imread(fn)
        squares = find_squares(img)

        if SHOW:
            cv2.drawContours(img, squares, 0, (0, 255, 0), 3)
            cv2.imshow('squares', img)
        if squares:
            warped = crop_squares(squares, img)
            cv2.imwrite(fn, warped)
        else:
            cv2.imwrite(fn, img)


    print("Processed patients: " + str(patients_cnt))
    print('Done')


if __name__ == '__main__':
    # main()
    input_1 = cv2.imread('MURA-v1.1/train/XR_ELBOW/patient01321/study1_positive/image5.png')
    input_2 = cv2.imread('MURA-v1.1/train/XR_HUMERUS/patient02751/study1_positive/image2.png')

    squares_1 = find_squares(input_1)
    squares_2 = find_squares(input_2)

    if SHOW:
        cv2.drawContours(input_1, squares_1, 0, (0, 255, 0), 3)
        cv2.imshow('squares', input_1)
        cv2.waitKey(0)

        cv2.drawContours(input_2, squares_2, 0, (0, 255, 0), 3)
        cv2.imshow('squares', input_2)

        cv2.waitKey(0)
    if squares_1:
        warped = crop_squares(squares_1, input_1)
        cv2.imwrite('temp/out1.png', warped)

    else:
        cv2.imwrite('temp/out1.png', input_1)

    if squares_2:
        warped = crop_squares(squares_2, input_2)
        cv2.imwrite('temp/out2.png', warped)

    else:
        cv2.imwrite('temp/out2.png', input_2)

    cv2.destroyAllWindows()
