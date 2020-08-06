import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img1 = cv.imread('long-focus.jpg',0)  # queryImage
img2 = cv.imread('wide-angle.jpg',0) # trainImage
plt.subplot(121),plt.imshow(img1,'gray')
plt.subplot(122),plt.imshow(img2,'gray')
plt.show()