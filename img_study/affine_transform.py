import cv2
from matplotlib import pyplot as plt
import numpy as np

# 仿射变换 = 平移（translation） + 缩放（scale） + 反转（flip） + 旋转（rotation） + 斜切（shear）

img = cv2.imread("img/palace.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def Affine(img_ori, src, dst):
    # 该方法就是通过计算参数src到dst的对应仿射变换的矩阵，其中参数src和dst分别代表原坐标和变换后的坐标，并且均为3行2列的二维ndarray, 数据必须为浮点型
    M = cv2.getAffineTransform(src, dst)
    new_img = cv2.warpAffine(img_ori, M, (img_ori.shape[1], img_ori.shape[0]))
    return new_img


src = np.float32([[50, 50], [200, 50], [50, 200]])
dst = np.float32([[10, 100], [200, 50], [100, 250]])

new_img = Affine(img_rgb, src, dst)

plt.imshow(new_img)
plt.show()
