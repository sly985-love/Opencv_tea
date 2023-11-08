import cv2
import numpy as np
from matplotlib import pyplot as plt
# 步骤：
# 1.扫描整幅图像，判断当前像素点是前景点同时还未打上标签，将该像素点入栈。
# （1）将当前栈中top元素出栈，通过4邻域判断(也需要同时满足是前景点未打上标签)进行元素入栈，同时对入栈的元素打上标签。
# (2) 重复(1)中的操作，直到栈表为空，退出当前循环，至此一个连通域的标签打完。
# 2.重复1中的所有操作。

def Connected_Separation(image):
    image_shape = image.shape
    rows = image_shape[0]
    cols = image_shape[1]
    index_map = np.zeros((rows, cols))
    label = 1
    for row in range(rows):
        for col in range(cols):
            # 扫描当前像素为前景且没有被访问过，将其入栈
            if image[row][col] == 1 and index_map[row][col] == 0:
                # 创建新站
                s = []
                # 入栈
                s.append((row, col))
                # 我们将(row,col)赋予一个label值
                index_map[row][col] = label
                # 循环判断4连通域是否与这个（row，col）相连，如果相连进行入栈操作，如果这个栈不为空
                while (len(s) != 0):
                    # 出栈,判断出栈元素的4领域
                    a = s.pop()
                    # 这边出栈的元素是不是还得赋予一个值
                    # 判断，可以入栈的元素，但是我还需要判断这个栈里面是否有这个元素，如果存在这个元素，那么就不能入栈
                    p=[]
                    p.append((a[0],a[1]-1 if a[1]-1>0 else 0))
                    p.append((a[0],a[1]+1 if a[1]+1<cols-1 else cols-1))
                    p.append((a[0]-1 if a[0]-1>0 else 0,a[1]))
                    p.append((a[0]+1 if a[0]+1<rows-1 else rows-1,a[1]))
                    # 判断栈里面是否以已经存在需要入栈的元素
                    for i in range(4):
                        if  image[p[i][0],p[i][1]] == 1 and index_map[p[i][0],p[i][1]] == 0:
                            s.append(p[i])
                            index_map[p[i][0], p[i][1]] = label
                    # 当栈里面的元素全部出去之后，我们的while就结束
                    if len(s) == 0:
                        label += 1
                        break
    return index_map

if __name__ == '__main__':
    image = cv2.imread('closing.jpg', 0)
    # 进行二值化
    # cv2.imshow('1',image)
    ret1, th1 = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)  # 也就是大于0的都写出255
    # cv2.imshow('2',th1)
    # 翻转
    image = 1 - th1 / 255
    kernel = np.ones((11, 11))
    # 孔洞填补(闭运算)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel=kernel)
    index_map =Connected_Separation(image)
    plt.imshow(image, plt.cm.gray)
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
