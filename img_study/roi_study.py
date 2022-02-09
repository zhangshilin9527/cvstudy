## 图像 ROI
## 图像 ROI

import cv2
from matplotlib import pyplot as plt

person_bgr = cv2.imread('img/person.jpeg')
person = cv2.cvtColor(person_bgr, cv2.COLOR_BGR2RGB)
plt.imshow(person)
plt.show()

print(person.shape)  # (390, 398, 3)
print(person.shape)  # (390, 398, 3)
