# 基于HSV色彩空间的嫩芽ROI提取算法
# 原理先通过cv.cvtColor()函数，将原RGB彩色图像转换为hsv色彩空间的图像，
# 然后通过cv.inRange()函数获得ROI区域的Mask，
# 最后利用cv.bitwise()函数提取得到ROI区域。
import cv2 as cv
import numpy as np
from skimage import measure

src = cv.imread('1 (1).jpg')
cv.imshow('src', src)
hsv = cv.cvtColor(src, cv.COLOR_BGR2HSV)       # 转换成hsv色彩风格
lower_hsv=np.array([25,95,46])
upper_hsv=np.array([41,255,255])
# 通过设置不同的H、S、V的min和max阈值可以获取不同色彩的一个二值的mask图像
mask = cv.inRange(hsv, lower_hsv, upper_hsv)        # 利用inRange产生mask
cv.imshow('mask1', mask)
cv.imwrite('mask1.jpg', mask)

# 获取mask 将原始图像与原始图像在mask区域进行逻辑与操作，即可获取（分割后的图像）
timg1 = cv.bitwise_and(src, src, mask=mask)
cv.imshow('timg1', timg1)
cv.imwrite('timg1.jpg', timg1)

# 将其转化为二值图像(省略此步骤直接调用mask二值图像)
bgr= cv.cvtColor(timg1, cv.COLOR_HSV2BGR)
# cv.imshow('1',bgr)
gray=cv.cvtColor(bgr,cv.COLOR_BGR2GRAY)
# cv.imshow('2',gray)
ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
# cv.imshow("thresh", thresh)
# 中值滤波：进行嫩芽降噪
median = cv.medianBlur(thresh,9)
cv.imshow('median',median)
# 开操作：去掉目标特征外的孤立点
kernel = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
opening = cv.morphologyEx(median,cv.MORPH_OPEN,kernel)
cv.imshow('open',opening)
# 闭操作：去掉目标特征内的孔
kernel = np.ones((5,5),np.uint8)
closing = cv.morphologyEx(opening,cv.MORPH_CLOSE,kernel)
cv.imshow('clos',closing)
cv.imwrite('closing.jpg', closing)

# 求出最大连通区域
# image:二值图像  threshold_point:符合面积条件大小的阈值

def remove_small_points(image, threshold_point):
    img = cv.imread(image, 0)
    img_label, num = measure.label(img, connectivity=2, return_num=True)  # 输出二值图像中所有的连通域
    props = measure.regionprops(img_label)  # 输出连通域的属性，包括面积等

    resMatrix = np.zeros(img_label.shape)
    for i in range(1, len(props)):
        if props[i].area > threshold_point:
            tmp = (img_label == i + 1).astype(np.uint8)
            resMatrix += tmp  # 组合所有符合条件的连通域
    resMatrix *= 255
    return resMatrix

threshold_point=400
maxconnect = remove_small_points('closing.jpg', threshold_point)
cv.imshow("maxconnect", maxconnect)
# 再闭操作一次：去掉目标特征内的孔
kernel = np.ones((5,5),np.uint8)
closing1 = cv.morphologyEx(maxconnect,cv.MORPH_CLOSE,kernel)
cv.imshow('clos1',closing1)
cv.imwrite('closing1.jpg', closing1)
# 图像类型不符合要求就没有办法送到canny算子里面计算
# 正常图像（closing）一个点的类型 type(closing[0][0]) 为：<class 'numpy.uint8'>
# 而我的图像（closing1）一个点的类型 type(closing1[0][0]) 是这样的：<class 'numpy.float64'>
# 故对图像类型进行了改正：array.dtype=np.uint8 或是 array=array.astype( np.uint8 )
# print(type(closing[0][0]))
# print(type(closing1[0][0]))
closing1=closing1.astype( np.uint8 )
print(type(closing1[0][0]))
# 在第一次分割嫩芽图像上框选出最大连通区域的嫩芽
# canny边缘检测
def canny_demo(image):
    t = 100
    print(type(image[0][0]))
    canny_output = cv.Canny(image, t, t * 2)
    cv.imshow("canny_output", canny_output)
    cv.imwrite("canny_output.png", canny_output)
    return canny_output
# 调用
binary = canny_demo(closing1)
k = np.ones((3, 3), dtype=np.uint8)
binary = cv.morphologyEx(binary, cv.MORPH_DILATE, k)

# 轮廓发现
contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
for c in range(len(contours)):
    rect = cv.minAreaRect(contours[c])
    cx, cy = rect[0]
    box = cv.boxPoints(rect)
    box = np.int0(box)
    cv.drawContours(src,[box],0,(0,255,0),2)
    cv.circle(src, (np.int32(cx), np.int32(cy)), 2, (255, 0, 0), 2, 8, 0)
    cv.drawContours(src, contours, c, (0, 0, 255), 1, 8)
    print(cx,cy)

# 图像显示
cv.imshow("contours_analysis", src)
cv.imwrite("contours_analysis.png", src)
# 最终在原图像上框选出每个嫩芽，并标注中心点，输出矩阵框左上角和右下角的坐标，描绘嫩芽轮廓

cv.waitKey(0)
cv.destroyAllWindows()






