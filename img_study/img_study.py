import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("img/palace.jpg", 1)

# 图像基本属性: 图像大小、维度、数据类型、通道
print('type of image: ', type(img))  # <class 'numpy.ndarray'>
print('dtype of image: ', img.dtype)  # 数据类型 unint8
print('dim of image :', img.ndim)  # 维度
print('shape of image :', img.shape)  # 通道 参数-1为按原通道读入，不写的话默认读入三通道图片，
print('size of image :', img.size)  # 大小
print('331*500*3 = ', 331 * 500 * 3)

# cv2显示
cv2.imshow('palace', img)
key = cv2.waitKey()
if key == 27:  # escape
    cv2.destroyAllWindows()

# 打印通道，请三个维度中都有什么内容
print(img[:, :, 0])  # img[:,:,0]  img[:,:,1]  img[:,:,2]表示图像单个通道的像素
print(img[:, 0])  # img[:, 0]  img[:,1]  img[:,2]表示图像列的三通道像素
print("shape of img[:,:,0] : ", img[:, 0].shape)
print("shape of img[:,:,0] : ", img[:, :, 0].shape)

# img的最大值和最小值
print('max value in img : ', img.min())  # min value in img :  0
print('min value in imge : ', img.max())  # max value in imge :  255

# 图像基本属性: 颜色空间
# 图像基本属性: 颜色空间
# 图像基本属性: 颜色空间
all_0_array = np.zeros((331, 550, 3))  # 利用numpy创建全是0的331*500*3的矩阵
all_255_array = np.full((331, 550, 3), 255)  # 利用numpy创建全是255的331*500*3的矩阵
all_180_array = np.full((331, 550, 3), 180)  # 利用numpy创建全是180的331*500*3的矩阵

plt.title("all 0")
plt.imshow(all_0_array)
plt.show()  # 黑色
plt.title("all 180")
plt.imshow(all_180_array)
plt.show()  # 灰色
plt.title("all 255")
plt.imshow(all_255_array)
plt.show()  # 白色


# 定义方法
def img_show(x, title_caption):
    plt.title(title_caption)
    plt.imshow(x)
    plt.show()


img_show(all_0_array, 'all 0')
img_show(all_180_array, 'all 180')
img_show(all_255_array, 'all 255')

# 把第一列的红，绿，蓝图片可视化。如（255，0，0）表示，三通道的图片，第一个通道全255，第二个通道全0,第三个通道全0。
# 把第一列的红，绿，蓝图片可视化。如（255，0，0）表示，三通道的图片，第一个通道全255，第二个通道全0,第三个通道全0。

# 红色
red_img_array = np.full((400, 400, 3), 0)  # 用numpy创建全是0的400*400*3的矩阵
red_img_array[:, :, 0] = np.full((400, 400), 255)  # 用numpy创建全是0的400*400*3的第一通道全部设置为255
img_show(red_img_array, 'red_img')

# 绿色
green_img_array = np.full((400, 400, 3), 0)
green_img_array[:, :, 1] = np.full((400, 400), 255)
img_show(green_img_array, 'green_img')

# 蓝色
blue_img_array = np.full((400, 400, 3), 0)
blue_img_array[:, :, 2] = np.full((400, 400), 255)
img_show(blue_img_array, 'blue_img')

plt.imshow(img)
plt.show()  # 图片异常, 由于通道问题：cv2这个包读取图片是BGR通道，平常的图片的三通道，是RGB。

# 通道转换 bgr2rgb
img_b = img[:, :, 0]
img_g = img[:, :, 1]
img_r = img[:, :, 2]
Img = np.full((img.shape), 0)
Img[:, :, 0] = img_r
Img[:, :, 1] = img_g
Img[:, :, 2] = img_b
plt.imshow(Img)
plt.show()

# 利用cv2.cvtColor转换图像的通道
img_new = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img_new)
plt.show()

B, G, R = cv2.split(img)
cv2.imshow('B', B)
cv2.imshow('G', G)
cv2.imshow('R', R)
key = cv2.waitKey()
if key == 27:
    cv2.destroyAllWindows()


def RGB2HSI(rgb_img):
    """
    这是将RGB彩色图像转化为HSI图像的函数
    :param rgm_img: RGB彩色图像
    :return: HSI图像
    """
    # 保存原始图像的行列数
    row = np.shape(rgb_img)[0]  # 331
    col = np.shape(rgb_img)[1]  # 500
    # 对原始图像进行复制
    hsi_img = rgb_img.copy()
    # 对图像进行通道拆分
    B, G, R = cv2.split(rgb_img)
    # 把通道归一化到[0,1]区间上
    [B, G, R] = [i / 255.0 for i in ([B, G, R])]

    H = np.zeros((row, col))  # 定义H通道 (331, 500)
    S = np.zeros((row, col))  # 定义S通道 (331, 500)

    # 计算I通道
    I = (R + G + B) / 3.0  # (331, 500)

    # 计算H通道
    for i in range(row):
        ## den=(R-G)²+（R-B）*（G-B）的数字平方根
        den = np.sqrt((R[i] - G[i]) ** 2 + (R[i] - B[i]) * (G[i] - B[i]))
        ## 计算夹角   0.5*（R-B+R-G）/den的反余弦值
        thetha = np.arccos(0.5 * (R[i] - B[i] + R[i] - G[i]) / den)
        h = np.zeros(col)  # 定义临时数组
        # den>0且G>=B的元素h赋值为thetha
        h[B[i] <= G[i]] = thetha[B[i] <= G[i]]
        # den>0且G<=B的元素h赋值为2pi - thetha
        h[G[i] < B[i]] = 2 * np.pi - thetha[G[i] < B[i]]
        # den<0的元素h赋值为0
        h[den == 0] = 0
        H[i] = h / (2 * np.pi)  # 弧度归一化后赋值给H通道

    # 计算S通道
    for i in range(row):
        min_ = []
        # 找出每组RGB值的最小值
        for j in range(col):
            arr = [B[i][j], G[i][j], R[i][j]]
            min_.append(np.min(arr))
        min_ = np.array(min_)
        # 计算S通道
        S[i] = 1 - min_ * 3 / (R[i] + B[i] + G[i])
        # I为0的值直接赋值0
        S[i][R[i] + B[i] + G[i] == 0] = 0

    # 扩充到255以方便显示，一般H分量在[0,2pi]之间，S和I在[0,1]之间
    hsi_img[:, :, 0] = H * 255
    hsi_img[:, :, 1] = S * 255
    hsi_img[:, :, 2] = I * 255
    return hsi_img


img_hsv = RGB2HSI(img_new)
plt.imshow(img_hsv)
plt.show()

cv2.imshow('img_hsv', img_hsv)

img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow('img_HSV', img_HSV)

cv2.waitKey(0)
cv2.destroyAllWindows()

img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow('img_HSV', img_HSV)
plt.imshow(img_HSV)
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()



