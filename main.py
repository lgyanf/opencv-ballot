# -*- coding: utf-8 -*-
from __future__ import print_function

import cv2
import numpy as np


def angle_cos(p0, p1, p2):
    d1, d2 = (p0-p1).astype('float'), (p2-p1).astype('float')
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )


def max_area(img):
    height, width = img.shape
    return (width / 20.)**2


def is_new(centers, candidate):
    min_distance = np.inf
    for cx, cy in centers:
        d = np.sqrt((cx - candidate[0])**2 + (cy - candidate[1])**2)
        min_distance = min(d, min_distance)
    return min_distance is None or min_distance > 100


def find_squares(bin):
    # img = cv2.GaussianBlur(img, (5, 5), 0)
    squares = []
    centers = []
    bin, contours, hierarchy = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cnt_len = cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
        area = cv2.contourArea(cnt)
        if len(cnt) == 4 and 1000 < area < max_area(bin) and cv2.isContourConvex(cnt):
            cnt = cnt.reshape(-1, 2)
            max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in range(4)])
            moments = cv2.moments(cnt)
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            if max_cos < 0.1 and is_new(centers, (cx, cy)):
                centers.append((cx, cy))
                squares.append(cnt)
    zipped = list(zip(centers, squares))
    zipped.sort(key=lambda e: e[0][1])
    return [square for center, square in zipped]


def resize(img):
    height, width = img.shape
    ratio = 1000. / width
    return cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)


def filter_marked_squares(bin, squares, threshold=0.1):
    marked = []
    for square in squares:
        xs = []
        ys = []
        for y, x in square:
            xs.append(x)
            ys.append(y)
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        roi = bin[min_x: max_x, min_y: max_y]
        height, width = roi.shape
        n_pixels = height * width
        nonzero = np.count_nonzero(roi)
        if (1. - float(nonzero) / n_pixels) > threshold:
            marked.append(square)
    return marked


def main():
    img = cv2.imread('scan.jpg', cv2.IMREAD_GRAYSCALE)
    img = resize(img)
    retval, bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    squares = find_squares(bin)
    marked = filter_marked_squares(bin, squares)
    print('marked', len(marked))
    cv2.drawContours(img, marked, -1, (0, 255, 0), 3)
    cv2.imwrite('squares.jpg', img, cv2.RGB)


if __name__ == '__main__':
    main()
