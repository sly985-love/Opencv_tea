import cv2
import numpy as np
from matplotlib import pyplot as plt

# 一、图像二值化
# 灰度图读入


# #5种阈值法图像分割
# ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
# ret, thresh3 = cv2.threshold(img, 127, 255,cv2.THRESH_TRUNC)
# ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
# ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
#
# images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
# #使用for循环进行遍历，matplotlib进行显示
# for i in range(6):
#     plt.subplot(2,3, i+1)
#     plt.imshow(images[i],cmap='gray')
#     plt.xticks([])
#     plt.yticks([])
#
# plt.suptitle('fixed threshold')
# plt.show()

# # 固定阈值
# ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# # 自适应阈值
# th2 = cv2.adaptiveThreshold(
#     img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11, 4)
# th3 = cv2.adaptiveThreshold(
#     img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 4)
# #全局阈值，均值自适应，高斯加权自适应对比
# titles = ['Original', 'Global(v = 127)', 'Adaptive Mean', 'Adaptive Gaussian']
# images = [img, th1, th2, th3]
# for i in range(4):
#     plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
#     plt.title(titles[i], fontsize=8)
#     plt.xticks([]), plt.yticks([])
# plt.show()

# img=cv2.imread('img.png',0)
# threshold=160
# # 阈值分割
# ret,th=cv2.threshold(img,threshold,255,cv2.THRESH_BINARY)
# print(ret)
# # cv2.imshow('thresh',th)
# # 中值滤波
# median = cv2.medianBlur(th,3)
# # cv2.imshow('median',median)
# # 开操作
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
# opening = cv2.morphologyEx(median,cv2.MORPH_OPEN,kernel)
# # cv2.imshow('open',opening)
# # 闭操作
# kernel = np.ones((3,3),np.uint8)
# closing = cv2.morphologyEx(opening,cv2.MORPH_CLOSE,kernel)
# # cv2.imshow('clos',closing)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# titles = ['gray','thresh','median','open','clos']
# images = [img, th, median, opening, closing]
# #使用for循环进行遍历，matplotlib进行显示
# for i in range(5):
#     plt.subplot(2,3, i+1)
#     plt.imshow(images[i],cmap='gray')
#     plt.title(titles[i], fontsize=8)
#     plt.xticks([])
#     plt.yticks([])
# plt.show()





# Step1. 加载图像
img = cv2.imread('img.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("gray", gray)
# Step2.阈值分割，将图像分为黑白两部分
ret,thresh=cv2.threshold(gray,150,255,cv2.THRESH_BINARY)
#ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("thresh", thresh)

median = cv2.medianBlur(thresh,3)
cv2.imshow('median',median)

# Step3. 对图像进行“开运算”，先腐蚀再膨胀
kernel = np.ones((4, 4), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
cv2.imshow("opening", opening)

# Step4. 对“开运算”的结果进行膨胀，得到大部分都是背景的区域
sure_bg = cv2.dilate(opening, kernel, iterations=3)
cv2.imshow("sure_bg", sure_bg)

# Step5.通过distanceTransform获取前景区域
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)  # DIST_L1 DIST_C只能 对应掩膜为3    DIST_L2 可以为3或者5
ret, sure_fg = cv2.threshold(dist_transform, 0.1* dist_transform.max(), 255, 0)

cv2.imshow("sure_fg", sure_fg)

# Step6. sure_bg与sure_fg相减,得到既有前景又有背景的重合区域   #此区域和轮廓区域的关系未知
sure_fg = np.uint8(sure_fg)
unknow = cv2.subtract(sure_bg, sure_fg)
cv2.imshow("unknow", unknow)

# Step7. 连通区域处理
ret, markers = cv2.connectedComponents(sure_fg, connectivity=8)  # 对连通区域进行标号  序号为 0 - N-1
markers = markers + 1  # OpenCV 分水岭算法对物体做的标注必须都 大于1 ，背景为标号 为0  因此对所有markers 加1  变成了  1  -  N
# 去掉属于背景区域的部分（即让其变为0，成为背景）
# 此语句的Python语法 类似于if ，“unknow==255” 返回的是图像矩阵的真值表。
markers[unknow == 255] = 0

# Step8.分水岭算法
markers = cv2.watershed(img, markers)  # 分水岭算法后，所有轮廓的像素点被标注为  -1
print(markers)

img[markers == -1] = [0, 0, 255]  # 标注为-1 的像素点标 红
cv2.imshow("dst", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
