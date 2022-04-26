import cv2
from matplotlib import pyplot as plt
import numpy as np

# 投影变换 = 旋转 + 平移 + 缩放 + 切变 + 射影
# 读取图片
img = cv2.imread("img/Lane_line.jpg", 1)
# 读取图片大小信息
rows, cols = img.shape[:2]
# 选取原图中需要被转换物体的四个顶点
src = np.float32([[0, 600], [1277, 600], [500, 400], [800, 400]])
# 设置在新图像中原图像的四个顶点的位置
dst = np.float32([[50, 800], [1200, 800], [50, 150], [1200, 150]])
# 计算转换M矩阵
M = cv2.getPerspectiveTransform(src, dst)
new_img = cv2.warpPerspective(img, M, (cols, rows))
# 返回一个包含figure和axes对象的元组  建立一个fig对象，建立一个axis对象
plt.subplot(121), plt.imshow(img), plt.title('input')
plt.subplot(122), plt.imshow(new_img), plt.title('output')
plt.show()
