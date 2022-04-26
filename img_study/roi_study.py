## 图像 ROI
## 图像 ROI

import cv2
import numpy as np
from matplotlib import pyplot as plt
import random

img = cv2.imread("img/palace.jpg", 1)
img_new = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
person_bgr = cv2.imread('img/person.jpeg')
# person = cv2.cvtColor(person_bgr, cv2.COLOR_BGR2RGB)
# plt.imshow(person)
# plt.show()

# print(person.shape)  # (390, 398, 3)

# HEIGHT, WIDTH, CHANNEL = person_bgr.shape  # 获取长款、宽、通道
#
# person_bgr = person_bgr[0:HEIGHT, :, :]
# random_substract = random.randint(0, HEIGHT)   #在0和height（390）之间随机取一个数字
# person_bgr = person_bgr[0:HEIGHT - random_substract, :, :]  #截取height图片
# print('You have removed {} pixel(s)'.format(random_substract))
person = cv2.cvtColor(person_bgr, cv2.COLOR_BGR2RGB)
# plt.imshow(person)
# plt.show()
#
# print(person.shape)

# cv2.inRange函数通过设置不同的h、s、v的min和max阈值可以获取不同色彩的一个二值的mask图
# cv2.inRange函数通过设置不同的h、s、v的min和max阈值可以获取不同色彩的一个二值的mask图
MAX_VALUE = 2 ** 8 - 1

YELLOW_HMIX, YELLOW_SMIN, YELLOW_VMIN = 26, 43, 46
YELLOW_HMAX = 34
person_hsv = cv2.cvtColor(person_bgr, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(person_hsv, (YELLOW_HMIX, YELLOW_SMIN, YELLOW_VMIN), (YELLOW_HMAX, MAX_VALUE, MAX_VALUE))
# print(mask.shape)  # (390, 398)
#
# plt.imshow(mask, cmap='gray')  # 背景灰色，灰白色·
# plt.show()

# cv2.bitwise_not 按位取反操作函数，将img在R,G,B三个分量分别进行按位取反操
mask1 = cv2.bitwise_not(mask)
# plt.imshow(mask1, cmap='gray')
# plt.show()


# cv.bitwise_and(img1, img2, mask) 按位与操作函数，将img1和img2在mask的区域内，R,G,B三个分量分别进行按位与操作
# cv2.bitwise_or(img1, img2, mask) 按位或操作函数，将img1和img2在mask的区域内，R,G,B三个分量分别进行按位或操作
person_and_mask1 = cv2.bitwise_and(person, person, mask=mask1)
# plt.imshow(person_and_mask1)
# plt.show()

# print(person_and_mask1.shape)  # (390, 398, 3)

array = np.uint8(np.full((331, 102), MAX_VALUE))  # 需要补充的大小
array1 = np.uint8(np.full((331, 102, 3), 0))  # 需要补充的大小
# print(array.shape)  # (331, 102)

mask2 = mask[0:331, :]
# print(mask2.shape)  # (331, 398)
new_mask = np.hstack((mask2, array))  # 拼接mask2和array
# print(new_mask.shape)  # (331, 500)
# cv2.imshow('new_mask', new_mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

img_and_new_mask = cv2.bitwise_and(img_new, img_new, mask=new_mask)
# plt.imshow(img_and_new_mask)
# plt.show()

print(person_and_mask1.shape)  # (390, 398, 3)
person_and_mask2 = np.hstack((person_and_mask1[0:331, :, :], array1))
print(person_and_mask2.shape)  # (331, 500, 3)
person_and_img = person_and_mask2 + img_and_new_mask

plt.imshow(person_and_img)
# plt.show()

person_and_img1 = cv2.addWeighted(person_and_mask2, 0.8, img_new, 1, 1)
plt.imshow(person_and_img1)
plt.show()



