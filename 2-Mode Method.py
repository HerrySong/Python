#coding:utf-8
import cv2
import numpy as np
from matplotlib import pyplot as plt

image = plt.imread("lena.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.subplot(221), plt.imshow(image, "gray")
plt.title("source image"), plt.xticks([]), plt.yticks([])

plt.subplot(222), plt.hist(image.ravel(), 256)
plt.title("Histogram")

plt.show()