import numpy as np
import scipy.ndimage

a = np.zeros((5,5))
a[:2,:2] = 1

b = np.zeros((5,5))
b[3:, 3:] = 1

c = a

acb = np.stack((a,c,b))

s = scipy.ndimage.generate_binary_structure(3, 0)

print(s)

l, n = scipy.ndimage.label(acb, structure=s)

print(l)