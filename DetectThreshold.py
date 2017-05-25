#coding:utf-8
import cv2
import numpy as np
from matplotlib import pyplot as plt

Image = cv2.imread('lena.jpg', 0)
T = np.average(Image)
while True:
    g= Image>=
