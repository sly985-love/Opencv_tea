import cv2
import numpy as np
from skimage import measure


def remove_small_points(image, threshold_point):
    img = cv2.imread(image, 0)
    img_label, num = measure.label(img, connectivity=2, return_num=True)  # 输出二值图像中所有的连通域
    props = measure.regionprops(img_label)  # 输出连通域的属性，包括面积等

    resMatrix = np.zeros(img_label.shape)
    for i in range(1, len(props)):
        if props[i].area > threshold_point:
            tmp = (img_label == i + 1).astype(np.uint8)
            resMatrix += tmp  # 组合所有符合条件的连通域
    resMatrix *= 255
    return resMatrix

threshold_point=800
res = remove_small_points('closing.jpg', threshold_point)
cv2.imshow("h", res)
cv2.waitKey(0)
cv2.destroyAllWindows()

