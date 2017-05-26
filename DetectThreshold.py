#coding:utf-8
import cv2
import numpy as np
from matplotlib import pyplot as plt

Image = plt.imread('lena.jpg') #cv2的读取图片会导致通道错误,采取pyplot的读取
GrayImage= cv2.cvtColor(Image,cv2.COLOR_BGR2GRAY) #灰度化图像

def detect_way(img_gray):
    max_g = 0
    suitable_th = 0

    threshold = (img_gray.min()+img_gray.max())/2

    while True:
        bin_img = img_gray > threshold
        bin_img_inv = img_gray <= threshold
        fore_pix = np.sum(bin_img)
        back_pix = np.sum(bin_img_inv)
        if 0== fore_pix:
            break
        if 0==back_pix:
            continue

        u0 = float(np.sum(img_gray*bin_img))/fore_pix
        u1 = float(np.sum(img_gray*bin_img_inv))/back_pix
        suitable_th = (u0 + u1)/2
        if suitable_th == threshold:
            break
        else:
            threshold = suitable_th
    return suitable_th


ret1,thresh1=cv2.threshold(GrayImage,detect_way(GrayImage),255,cv2.THRESH_BINARY)
plt.subplot(111), plt.imshow(thresh1,"gray")
plt.title("detect image,the threshold ="+str(ret1)), plt.xticks([]), plt.yticks([])
plt.show()
