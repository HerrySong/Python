#coding:utf-8
import cv2
import numpy as np
from matplotlib import pyplot as plt

image = plt.imread("lena.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.subplot(131), plt.imshow(image, "gray")
plt.title("source image"), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.hist(image.ravel(), 256)
plt.title("Histogram"), plt.xticks([]), plt.yticks([])
ret1, th1 = cv2.threshold(gray, 152, 255, cv2.THRESH_BINARY)
plt.subplot(133), plt.imshow(th1, "gray")
plt.title("2-Mode Method"), plt.xticks([]), plt.yticks([])
plt.show()