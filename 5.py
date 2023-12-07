import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
import cv2

sample_image = cv2.imread('cv.jpg')

img = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)

plt.subplot(121)
plt.axis('off')
plt.imshow(img)
plt.title('Original Image')

low = np.array([0, 0, 0])
high = np.array([215, 51, 51])

mask = cv2.inRange(img, low, high)

plt.subplot(122)
plt.axis('off')
plt.imshow(mask, cmap='gray')
plt.title('Binary Mask')

result = cv2.bitwise_and(img, img, mask=mask)

plt.figure()
plt.axis('off')
plt.imshow(result)
plt.title('Result')

plt.show()