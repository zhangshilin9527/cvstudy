import cv2
from matplotlib import pyplot as plt


# 相似变换（similarly transform） = 旋转（rotation） + 平移（translation） + 缩放（scale）

def rotate(img_ori, aug_value, scale):
    # img_ori：图像
    # aug_value：旋转角度
    # scale：缩放值
    M = cv2.getRotationMatrix2D((0.5 * img_ori.shape[1], 0.5 * img_ori.shape[0]), aug_value, scale)  # 获得旋转矩阵
    new_img = cv2.warpAffine(img_ori, M, (img_ori.shape[1], img_ori.shape[0]))  # 简单的重映射
    return new_img


img = cv2.imread("img/palace.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

new_img = rotate(img_rgb, 45, 0.8)
plt.imshow(new_img)
plt.show()
