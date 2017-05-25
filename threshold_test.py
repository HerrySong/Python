#coding:utf-8
import cv2
import numpy as np
from matplotlib import pyplot as plt

Image = plt.imread('lena.jpg')#cv2的读取图片会导致通道错误
GrayImage= cv2.cvtColor(Image,cv2.COLOR_BGR2GRAY)
ret1,thresh1=cv2.threshold(GrayImage,0,255,cv2.THRESH_OTSU)
thresh2=cv2.adaptiveThreshold(GrayImage,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
ret3,thresh3=cv2.threshold(GrayImage,0,255,cv2.THRESH_TOZERO)
ret4,thresh4=cv2.threshold(GrayImage,0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)


plt.subplot(331), plt.imshow(Image)
plt.title("source image"), plt.xticks([]), plt.yticks([])
plt.subplot(332), plt.imshow(GrayImage, "gray")
plt.title("gray image"), plt.xticks([]), plt.yticks([])
plt.subplot(334), plt.imshow(thresh1, "gray")
plt.title("OTSU,threshold is " + str(ret1)), plt.xticks([]), plt.yticks([])
plt.subplot(335), plt.imshow(thresh2, "gray")
plt.title("5,threshold is " ), plt.xticks([]), plt.yticks([])
plt.subplot(336), plt.imshow(thresh4, "gray")
plt.title("OTSU,threshold is " + str(ret4)), plt.xticks([]), plt.yticks([])