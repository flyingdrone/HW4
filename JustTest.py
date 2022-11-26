import cv2
import numpy as np


def main():

    img = cv2.imread('mask.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
    contours, heriachy = cv2.findContours(binary, 2, 1)
    # cv2.imshow('fig_source', img)
    # cv2.waitKey(0)

    # 简单的想法是将每个凸包手动画线连接成一整个连通区域，然后重新寻找一个大凸包
    hulls = []
    lines = []
    for cnt in contours:
        hull = cv2.convexHull(cnt)
        lines.append(tuple(hull[0][0]))

    for j in range(len(lines)):
        if j + 1 < len(lines):
            cv2.line(img, lines[j], lines[j + 1], (255, 255, 255), 2)

    # 在连线完的图片上重新寻找最外层轮廓
    gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret2, binary2 = cv2.threshold(gray2, 235, 255, cv2.THRESH_BINARY)
    contours2, heriachy2 = cv2.findContours(binary2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt2 in contours2:
        hull2 = cv2.convexHull(cnt2)
        hulls.append(hull2)

    draw_hulls = cv2.drawContours(img, hulls, -1, (0, 0, 255), 2)  # 最后一个参数-1表示填充

    cv2.imwrite('fig_hull.png', draw_hulls)
    cv2.imshow('fig_lines', img)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()