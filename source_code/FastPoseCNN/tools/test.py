import numpy as np
import matplotlib.pyplot as plt

#https://stackoverflow.com/questions/42510161/colormap-by-vector-direction-in-python-using-quiver

w = h = 10
x_coord = np.mod(np.arange(w*h, dtype=np.float32), w).reshape((h,w))
y_coord = np.mod(np.arange(w*h, dtype=np.float32), h).reshape((h,w)).transpose()
coord = np.dstack([y_coord,x_coord])

center = np.array([4,4], dtype=np.float32)

diff = np.divide((center - coord), np.expand_dims(np.linalg.norm(center - coord, axis=-1), axis=-1))

#angle_diff = np.arctan2(diff[:,:,0], diff[:,:,1])

#plt.imshow(angle_diff, cmap=plt.get_cmap('hsv')); plt.show()

#print(angle_diff.shape)
#print(angle_diff)

print(diff.shape)