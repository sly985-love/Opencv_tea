# 基于HSV色彩空间的嫩芽ROI提取算法
# 原理先通过cv.cvtColor()函数，将原RGB彩色图像转换为hsv色彩空间的图像，
# 然后通过cv.inRange()函数获得ROI区域的Mask，
# 最后利用cv.bitwise()函数提取得到ROI区域。
import cv2 as cv
import numpy as np


src = cv.imread('img_6.png')
cv.imshow('src', src)
hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)       # 转换成hsv色彩风格
lower_hsv=np.array([25,95,46])
upper_hsv=np.array([41,255,255])
# 通过设置不同的H、S、V的min和max阈值可以获取不同色彩的一个二值的mask图像
mask = cv.inRange(hsv, lower_hsv, upper_hsv)        # 利用inRange产生mask
cv.imshow('mask1', mask)
cv.imwrite('mask1.jpg', mask)

# 获取mask
# mask = cv.bitwise_not(mask)
# cv.imshow('mask2', mask)
# cv.imwrite('mask2.jpg', mask)
# 将原始图像与原始图像在mask区域进行逻辑与操作，即可获取（分割后的图像）
timg1 = cv.bitwise_and(src, src, mask=mask)
cv.imshow('timg1', timg1)
cv.imwrite('timg1.jpg', timg1)

# # 拓展：将提取出来的图片贴到其他背景上
# # 生成背景（得到一张蓝色背景图）
# background = np.zeros(src.shape, src.dtype)
# background[:,:,0] = 255
#
# # 将人物贴到背景中
# mask = cv.bitwise_not(mask)
# dst = cv.bitwise_or(timg1, background, mask=mask)
# cv.imshow('dst1', dst)
# cv.imwrite('dst1.jpg', dst)
#
# dst = cv.add(dst, timg1)
# cv.imshow('dst2', dst)
# cv.imwrite('dst2.jpg', dst)


# 将其转化为二值图像(省略此步骤直接调用mask二值图像)
bgr= cv.cvtColor(timg1, cv.COLOR_HSV2BGR)
# cv.imshow('1',bgr)
gray=cv.cvtColor(bgr,cv.COLOR_BGR2GRAY)
# cv.imshow('2',gray)
ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
cv.imshow("thresh", thresh)
# 中值滤波：进行嫩芽降噪
median = cv.medianBlur(thresh,3)
cv.imshow('median',median)
# 开操作：去掉目标特征外的孤立点
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
opening = cv.morphologyEx(median,cv.MORPH_OPEN,kernel)
cv.imshow('open',opening)
# 闭操作：去掉目标特征内的孔
kernel = np.ones((5,5),np.uint8)
closing = cv.morphologyEx(opening,cv.MORPH_CLOSE,kernel)
cv.imshow('clos',closing)
cv.imwrite('closing.jpg', closing)
# 求出最大连通区域

# 在第一次分割嫩芽图像上框选出最大连通区域的嫩芽

# 最终在原图像上框选出每个嫩芽，并标注中心点，输出矩阵框左上角和右下角的坐标，描绘嫩芽轮廓

cv.waitKey(0)
cv.destroyAllWindows()
