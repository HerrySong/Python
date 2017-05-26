#coding:utf-8
import cv2
import numpy as np
from matplotlib import pyplot as plt

Image = plt.imread('lena.jpg') #cv2的读取图片会导致通道错误,采取pyplot的读取
GrayImage= cv2.cvtColor(Image,cv2.COLOR_BGR2GRAY) #灰度化图像
blurred = cv2.GaussianBlur(GrayImage, (5, 5), 0)

# 采取opencv库中的otsu算法,ret1接收返回阈值,thresh1接收处理后图像
ret1,thresh1=cv2.threshold(GrayImage,0,255,cv2.THRESH_OTSU)
ret2,thresh2=cv2.threshold(blurred,0,255,cv2.THRESH_OTSU)
'''
显示图像
'''
plt.subplot(231), plt.imshow(Image)
plt.title("source image"), plt.xticks([]), plt.yticks([])
plt.subplot(232), plt.imshow(GrayImage, "gray")
plt.title("gray image"), plt.xticks([]), plt.yticks([])
plt.subplot(233), plt.imshow(blurred, "gray")
plt.title("gsblur image"), plt.xticks([]), plt.yticks([])
plt.subplot(234), plt.imshow(thresh1, "gray")
plt.title("otsu image"), plt.xticks([]), plt.yticks([])
plt.subplot(235), plt.imshow(thresh2, "gray")
plt.title("blurred otsu image"), plt.xticks([]), plt.yticks([])

plt.show()



'''
自己编写的单阈值otsu
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

print "opencv otsu得到 阈值为："+str(ret1)
print "自己编写 otsu阈值为："+str(otsu_way(GrayImage))






