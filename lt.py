import cv2
import numpy as np
import matplotlib.pyplot as plt
# 首先通过findContours函数找到二值图像中的所有边界(这块看需要调节里面的参数)
# 然后通过contourArea函数计算每个边界内的面积
# 最后通过fillConvexPoly函数将面积最大的边界内部涂成背景
if __name__ == '__main__':
    img = cv2.imread('img_3.png',0)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray',gray)
    # ret, th = cv2.threshold(img, 170, 255, cv2.THRESH_BINARY)
    # print(ret)
    # cv2.imshow('thresh',th)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #find contours of all the components and holes
    # 找到所有部件和孔的轮廓
    # 复制灰色图像，因为该功能
    # findContours会将输入的图像更改为另一个图像
    gray_temp = gray.copy() #copy the gray image because function
                            #findContours will change the imput image into another
    contours, hierarchy = cv2.findContours(gray_temp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    #show the contours of the imput image
    # 显示输入图像的轮廓
    cv2.drawContours(img, contours, -1, (255,0,0), 2)
    plt.figure('original image with contours'), plt.imshow(img, cmap = 'gray')
    # 找到所有轮廓的最大面积并用0填充
    #find the max area of all the contours and fill it with 0
    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))
    max_idx = np.argmax(area)
    cv2.fillConvexPoly(gray, contours[max_idx], 0)
    #show image without max connect components
    # 显示没有最大连接组件的图像
    plt.figure('remove max connect com'), plt.imshow(gray, cmap = 'gray')

    plt.show()

