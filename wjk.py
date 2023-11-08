import cv2 as cv
import numpy as np
# 图像二值图像的每个轮廓，OpenCV都提供了API可以求取轮廓的外接矩形，其中求取轮廓外接矩形API解释如下：
# atedRect cv::minAreaRect(
# InputArray points
# )
#
# 输入参数points可以一系列点的集合，对轮廓来说就是该轮廓的点集
# 返回结果是一个旋转矩形，包含下面的信息：
#
# 矩形中心位置
# 矩形的宽高
# 旋转角度
# canny边缘检测
def canny_demo(image):
    t = 100
    canny_output = cv.Canny(image, t, t * 2)
    cv.imshow("canny_output", canny_output)
    cv.imwrite("canny_output.png", canny_output)
    return canny_output

# 读取图像
src = cv.imread("img_6.png")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)

# median = cv.GaussianBlur(src,(9, 9), 0)
# cv.imshow('median',median)


# 调用
binary = canny_demo(src)
k = np.ones((3, 3), dtype=np.uint8)
binary = cv.morphologyEx(binary, cv.MORPH_DILATE, k)

# 轮廓发现
contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
for c in range(len(contours)):
    rect = cv.minAreaRect(contours[c])
    cx, cy = rect[0]
    box = cv.boxPoints(rect)
    box = np.int0(box)
    cv.drawContours(src,[box],0,(0,255,0),1)
    cv.circle(src, (np.int32(cx), np.int32(cy)), 2, (255, 0, 0), 2, 8, 0)
    cv.drawContours(src, contours, c, (0, 0, 255), 1, 8)
    print(cx,cy)

# 图像显示
cv.imshow("contours_analysis", src)
cv.imwrite("contours_analysis.png", src)
cv.waitKey(0)
cv.destroyAllWindows()