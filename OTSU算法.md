* 相关程序及实验截图所在远程仓库：https://github.com/RaphaelZheng/OTSU
# 学习相关准备
* 图像二值化
    * 图像的二值化，就是将图像上的像素点的灰度值设置为0或255，也就是将整个图像呈现出明显的只有黑和白的视觉效果。、
    * 一幅图像包括目标物体、背景还有噪声，要想从多值的数字图像中直接提取出目标物体，常用的方法就是设定一个阈值T，用T将图像的数据分成两部分：大于T的像素群和小于T的像素群。这是研究灰度变换的最特殊的方法，称为图像的二值化（Binarization）。
* OTSU介绍
    * 使用的是聚类的思想，把图像的灰度数按灰度级分成2个部分，使得两个部分之间的灰度值差异最大，每个部分之间的灰度差异最小，通过方差的计算来寻找一个合适的灰度级别 来划分。 所以 可以在二值化的时候 采用otsu算法来自动选取阈值进行二值化。otsu算法被认为是图像分割中阈值选取的最佳算法，计算简单，不受图像亮度和对比度的影响。因此,使类间方差最大的分割意味着错分概率最小。
* 语言选择：python
    * 由于自己一直使用python，加上python相关库也比较多，所以选择python
* 测试图像选择:lena
    * ![lena](https://raw.githubusercontent.com/RaphaelZheng/OTSU/master/lena.jpg) 
        * 选择原因：该图适度的混合了细节、平滑区域、阴影和纹理，从而能很好的测试各种图像处理算法。 
# 二值化算法选择阈值
* 直方图双峰法
    * 算法原理
        * 直方图双峰法（2-Mode method）即如果图像灰度直方图呈明显的双峰状，则选取双峰间的最低谷出作为图像分割的阈值所在。
    * 得到灰度直方图
        * ![image](https://raw.githubusercontent.com/RaphaelZheng/OTSU/master/SCREEN/2-Mode%20Method.png) 
    * 评价
        * 从这张测试图片可以看出，并没有明显的双峰现象，也就不能够直接简单地根据低谷阈值二值化
    * 代码实现 
        * 参加代码清单 1
* 迭代阈值法
    * 原理：迭代选择法是首先猜测一个初始阈值，然后再通过对图像的多趟计算对阈值进行改进的过程。重复地对图像进行阈值操作，将图像分割为对象类和背景类，然后来利用每一个类中的灰阶级别对阈值进行改进。
    * 算法
        1. 为全局阈值选择一个初始估计值T(图像的平均灰度)
        2. 用T分割图像。产生两组像素：G1有灰度值大于T的像素组成，G2有小于等于T像素组成
        3. 计算G1和G2像素的平均灰度值m1和m2
        4. 计算一个新的阈值:T = (m1 + m2) / 2
        5. 重复步骤b和d,直到连续迭代中的T值间的差小于一个预定义参数为止
    * 得到图像结果：
        * ![image](https://raw.githubusercontent.com/RaphaelZheng/OTSU/master/SCREEN/Detect.png)
    * 代码实现
        *  * 参加代码清单 2
* 单阈值OSTU
    * 算法
        * 假设我们使用阈值T将灰度图像分割为前景和背景
        * size：图像总像素个数 
        * u：图像的平均灰度 
        * w0：前景像素点占整幅图像大小的比例
        * u0：前景像素点的平均值 
        * w1：背景像素点占整幅图像大小的比例 
        * u0：背景像素点的平均值 
        * g：类间方差 
        * u = w0 * u0 + w1 * u1  (1) 
        * g = w0*(u - u0)^2 + w1*(u - u1)^2 (2) 
        * 将(1)代入(2)得： 
        * g = w0 * w1 * (u0 - u1)^2
    * 单阈值otsu处理后图像
        * ![image](https://raw.githubusercontent.com/RaphaelZheng/OTSU/master/SCREEN/1-otsu.png)
    * 调用库得到的阈值和自己编程的otsu得到阈值的区别
        * ![image](https://raw.githubusercontent.com/RaphaelZheng/OTSU/master/SCREEN/2-otsu.png)
    * 改进--对于图像高斯处理，不过对于这张图进行高斯模糊处理后，看起来图像更加平滑
    * 评价
        * 根据实验结果，可以看到人还是还是能够比较好的从背景中分离出来 
    * 代码实现 
        * 参加代码清单 2

# 参考资料
* 维基百科
* 常见的二值化方法
    * http://blog.csdn.net/xdhywj/article/details/7734117

# 代码清单
* 1-双峰法
```
#coding:utf-8
import cv2
import numpy as np
from matplotlib import pyplot as plt

image = plt.imread("lena.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.subplot(221), plt.imshow(image, "gray")
plt.title("source image"), plt.xticks([]), plt.yticks([])

plt.subplot(222), plt.hist(image.ravel(), 256)
plt.title("Histogram"), plt.xticks([]), plt.yticks([])

plt.show()
```
* 2-迭代阈值法
```
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
```
* 3-单阈值OSTU
```
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
```
