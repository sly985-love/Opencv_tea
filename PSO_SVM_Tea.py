try:
    from PIL import Image
except ImportError:
    import Image

from PIL import Image
import numpy as np
import cv2 as cv
from skimage import measure
from sklearn.svm import SVC # 非线性 分类 SVM
tea = cv.imread('img_4.png')
pic = 'img_4.png' # 茶叶图片
img = Image.open(pic)
# img.show() # 显示原始图像
img_arr = np.asarray(img,np.float64)
# 选取茶叶嫩芽背景图像上的关键点RGB值(10个)
bg_RGB = np.array(
    [[143,150,133],[103, 86, 76,],[90, 133, 118],[166, 186 ,152,],[29, 51, 32],
     [65, 56, 34],[74 ,111, 95],[119 ,102, 67],[75 ,80, 4],[149, 153, 134]]
)
# 选取茶叶嫩芽上的关键点RGB值(10个)
tea_RGB = np.array(
    [[110, 139, 43],[79, 130, 0],[146, 171, 36],[196, 218, 106],[107, 145, 31],
     [164, 197, 28],[76 ,108, 0],[132, 170, 10],[103, 129, 5],[104 ,130, 29]]
)
RGB_arr = np.concatenate((bg_RGB,tea_RGB),axis=0) # 按列拼接
# 背景用0标记，茶叶嫩芽用1标记
label = np.append(np.zeros(bg_RGB.shape[0]),np.ones(tea_RGB.shape[0]))
# 原本 img_arr 形状为(m,n,k),现在转化为(m*n,k)
img_reshape = img_arr.reshape([img_arr.shape[0]*img_arr.shape[1],img_arr.shape[2]])
svc = SVC(kernel='poly',degree=3) # 使用多项式核，次数为3
svc.fit(RGB_arr,label) # SVM 训练样本
predict = svc.predict(img_reshape) # 预测测试点
bg_bool = predict == 0. # 为背景的序号(bool)
bg_bool = bg_bool[:,np.newaxis] # 增加一列(一维变二维)
bg_bool_3col = np.concatenate((bg_bool,bg_bool,bg_bool),axis=1) # 变为三列
bg_bool_3d = bg_bool_3col.reshape((img_arr.shape[0],img_arr.shape[1],img_arr.shape[2])) # 变回三维数组(逻辑数组)
# img_arr[bg_bool_3d] = 255. # 将背景像素点变为白色
img_arr[bg_bool_3d] = 0. # 将背景像素点变为黑色
img_split = Image.fromarray(img_arr.astype('uint8')) # 数组转image
# img_split.show() # 显示分割之后的图像
img_split.save('split_tea.jpg') # 保存
# 再依次进行二值化、中值滤波、腐蚀、空洞填充，求最大连通区域等形态学操作
# 最后利用二值化图像求出最大连通区域
# 再原嫩芽图像上画出茶叶嫩芽的最小外包矩形和中心点坐标；
# 将其转化为二值图像(省略此步骤直接调用mask二值图像)
tea_s = cv.imread('split_tea.jpg')
bgr= cv.cvtColor(tea_s, cv.COLOR_HSV2BGR)
# cv.imshow('1',bgr)
gray=cv.cvtColor(bgr,cv.COLOR_BGR2GRAY)
# cv.imshow('2',gray)
ret, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
# cv.imshow("thresh", thresh)
# 中值滤波：进行嫩芽降噪
median = cv.medianBlur(thresh,3)
cv.imshow('median',median)
# 开操作：去掉目标特征外的孤立点
kernel = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
opening = cv.morphologyEx(median,cv.MORPH_OPEN,kernel)
cv.imshow('open',opening)
# 闭操作：去掉目标特征内的孔
kernel = np.ones((3,3),np.uint8)
closing = cv.morphologyEx(opening,cv.MORPH_CLOSE,kernel)
cv.imshow('clos',closing)
cv.imwrite('SVM_clos.jpg', closing)

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
maxconnect = remove_small_points('SVM_clos.jpg', threshold_point)
cv.imshow("maxconnect", maxconnect)

# 再闭操作一次：去掉目标特征内的孔
kernel = np.ones((5,5),np.uint8)
closing1 = cv.morphologyEx(maxconnect,cv.MORPH_CLOSE,kernel)
cv.imshow('clos1',closing1)

closing1=closing1.astype( np.uint8 )

# 在第一次分割嫩芽图像上框选出最大连通区域的嫩芽
# canny边缘检测
def canny_demo(image):
    t = 100
    print(type(image[0][0]))
    canny_output = cv.Canny(image, t, t * 2)
    cv.imshow("canny_output", canny_output)
    # cv.imwrite("svm_canny_output.png", canny_output)
    return canny_output
# 调用
binary = canny_demo(closing1)
k = np.ones((3, 3), dtype=np.uint8)
binary = cv.morphologyEx(binary, cv.MORPH_DILATE, k)
cv.imshow('binary',binary)

# 轮廓发现
contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
for c in range(len(contours)):
    rect = cv.minAreaRect(contours[c])
    cx, cy = rect[0]
    box = cv.boxPoints(rect)
    box = np.int0(box)
    cv.drawContours(tea,[box],0,(0,255,0),2)
    cv.circle(tea, (np.int32(cx), np.int32(cy)), 2, (255, 0, 0), 2, 8, 0)
    cv.drawContours(tea, contours, c, (0, 0, 255), 1, 8)
    print(cx,cy)

# 图像显示
cv.imshow("contours_analysis", tea)
# cv.imwrite("svm_contours_analysis.png", pic)
# # 最终在原图像上框选出每个嫩芽，并标注中心点，输出矩阵框左上角和右下角的坐标，描绘嫩芽轮廓

cv.waitKey(0)
cv.destroyAllWindows()


