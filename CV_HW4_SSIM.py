from skimage.metrics import structural_similarity
import cv2
import numpy as np


first = cv2.imread('HW4-1.png')
second = cv2.imread('HW4-2.png')

# 转为灰度图片
first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
second_gray = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)

# 计算SSIM值
score, diff = structural_similarity(first_gray, second_gray, full=True)
print("Similarity Score: {:.3f}%".format(score * 100))

# 转为8bit以便于openCV使用
diff = (diff * 255).astype("uint8")

# 将照片进行二值化处理(cv2.threshold)，进行边界标定
thresh = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

# 标明不同点
mask = np.zeros(first.shape, dtype='uint8')
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
filled = second.copy()

for c in contours:
    area = cv2.contourArea(c)
    if area > 100:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(first, (x, y), (x + w, y + h), (36,255,12), 2)
        cv2.rectangle(second, (x, y), (x + w, y + h), (36,255,12), 2)
        cv2.drawContours(mask, [c], 0, 255, -1)
        cv2.drawContours(filled, [c], 0, (0,255,0), -1)

# cv2.imshow('first', first)
# cv2.imshow('second', second)
# cv2.imshow('diff', diff)
# cv2.imshow('mask', mask)
# cv2.imshow('filled', filled)
# cv2.waitKey()

cv2.imwrite('diff.jpg', diff)
cv2.imwrite('mask.jpg', mask)
cv2.imwrite('filled.jpg', filled)