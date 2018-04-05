import numpy as np
import cv2 as cv
import math

a = np.array([[1., 3., 5., 8.],
              [4., 7., 6., 4.],
              [1., 1., 1., 1.],
              [1., 2., 3., 5.]])
xx = cv.Sobel(a, cv.CV_16S, 1, 0)
yy = cv.Sobel(a, cv.CV_16S, 0, 1)
xy = (xx**2 + yy**2)
ind = np.unravel_index(np.argmax(xy, axis=None), xy.shape)
print(xx)
print(yy)
print(xy)
print(xx[ind])
print(yy[ind])
# print(a[ind[0], ind[1]])
# print(np.diff(a, axis=0))
# print(np.diff(a, axis=1))
# print(np.gradient(a))
