import cv2


def SobelCal(img):
    cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)  # 获取水平方向边缘梯度，第二个参数表示获取所有边缘信息不要遗漏
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)  # 获取垂直方向边缘梯度，第二个参数表示获取所有边缘信息不要遗漏

    sobelx = cv2.convertScaleAbs(sobelx)  # 取绝对值
    sobely = cv2.convertScaleAbs(sobely)  # 取绝对值

    img_edge = cv2.addWeighted(sobelx, 1, sobely, 1, 0)  # 将两个方向的梯度结合成新的一个完整图像的梯度

    return img_edge


def main():
    IMG1 = cv2.imread('HW4-1rh.png')
    IMG2 = cv2.imread('HW4-2rh.png')

    # # Sobel算子
    # IMG1Sobel = SobelCal(IMG1)
    # IMG2Sobel = SobelCal(IMG2)
    # cv2.imwrite("IMG1Sobel.png", IMG1Sobel)
    # cv2.imwrite("IMG2Sobel.png", IMG2Sobel)

    # Canny算子
    IMG1Canny = cv2.Canny(IMG1, 60, 280, (3, 3))
    IMG2Canny = cv2.Canny(IMG2, 60, 280, (3, 3))
    cv2.imwrite("IMG1Canny.png", IMG1Canny)
    cv2.imwrite("IMG2Canny.png", IMG2Canny)

    img3 = cv2.subtract(IMG2Canny, IMG1Canny)

    cv2.imshow("win_name", img3)
    cv2.waitKey(0)

    cv2.imwrite('JustSubstract.png', img3)


if __name__ == "__main__":
    main()
