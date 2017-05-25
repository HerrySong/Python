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

'''otsu实现
下面定义类间方差的计算公式： 
假设我们使用阈值T将灰度图像分割为前景和背景
size：图像总像素个数 
u：图像的平均灰度 
w0：前景像素点占整幅图像大小的比例
u0：前景像素点的平均值 
w1：背景像素点占整幅图像大小的比例 
u0：背景像素点的平均值 
g：类间方差 

u = w0 * u0 + w1 * u1  (1) 
g = w0*(u - u0)^2 + w1*(u - u1)^2 (2) 
将(1)代入(2)得： 
g = w0 * w1 * (u0 - u1)^2 
'''
def otsu_way(img_gray,th_begin=0,th_end=256,th_step=1):
   max_g = 0
   suitable_th = 0
   for threshold in xrange(th_begin,th_end,th_step):
      bin_img = img_gray > threshold
      bin_img_inv = img_gray <= threshold
      fore_pix = np.sum(bin_img)
      back_pix = np.sum(bin_img_inv)
      if 0== fore_pix:
         break
      if 0==back_pix:
         continue

      w0 = float(fore_pix)/img_gray.size
      u0 = float(np.sum(img_gray*bin_img))/fore_pix
      w1 = float(back_pix)/img_gray.size
      u1 = float(np.sum(img_gray*bin_img_inv))/back_pix
      g = w0*w1*(u0-u1)*(u0-u1)

      if g > max_g:
          max_g = g
          suitable_th = threshold
   return suitable_th


print otsu_way(GrayImage)

'''
调用opcv的库
'''
#(r, g, b)=cv2.split(Image)
#Image=cv2.merge([b,g,r])
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





plt.show()




