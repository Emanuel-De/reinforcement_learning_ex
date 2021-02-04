from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

from numpy import load

#select whether z-value is action or Q-value
z_axis = 'action'
#z_axis = 'qvalue'

partition = load('partition.npy')

mean_th = np.mean(partition[:, 1:3], axis = 1)
mean_thdot = np.mean(partition[:, 3:5], axis = 1)
mean_a = np.mean(partition[:, 5:7], axis = 1)

partition = np.append(partition, mean_th.reshape((partition.shape[0],1)), axis = 1)
partition = np.append(partition, mean_thdot.reshape((partition.shape[0],1)), axis = 1)
partition = np.append(partition, mean_a.reshape((partition.shape[0],1)), axis = 1)

ax = plt.axes(projection='3d')

xdata = partition[:,10]
ydata = partition[:,11]

ax.set_xlabel('theta')
ax.set_ylabel('theta_dot')

markersize = partition[:,7]

if z_axis == 'action':
    zdata = partition[:,12]
    data_colour = partition[:,8]
    ax.set_zlabel('action')
    ax.set_title('Colour refers to Q-value of partition (green for low value, red for high value)')
    cmap = 'RdYlGn'
elif z_axis == 'qvalue':
    zdata = partition[:,8]
    data_colour = partition[:,12]
    ax.set_zlabel('Q-value')
    ax.set_title('Colour refers to action of partition (blue for clockwise acceleration, red for counterclockwise acceleration)')
    cmap = 'coolwarm'

ax.scatter(xdata, ydata, zdata, c=data_colour, cmap=cmap,
           edgecolors = None, alpha = 0.75, marker='.' , s=markersize)

plt.show()
