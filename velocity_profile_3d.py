import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import cm

T = np.arange(0,2.1,0.1)
X = np.arange(0,2,0.02)

x,t = np.meshgrid(X,T)
U = np.load('roilqr_burgers.npy')
# U = np.load('Vel_profile.npy')
fig = plt.figure()
ax = Axes3D(fig)
Z = np.zeros((21,100))
Z[0] = np.round(np.sin(np.linspace(-np.pi, np.pi, 100)),2).flatten()
Z[1:]= U.reshape((20,100))
# Z= Z.reshape(np.shape(t))
ax.plot_surface(t,x, Z,cmap=cm.viridis,linewidth=3, antialiased=False)

ax.set_xlabel('Time')
ax.set_ylabel('X')
ax.set_zlabel('Velocity Profile U(t,x)')

# fig = plt.figure()
# plt.plot(np.arange(0,2,0.02),Z[0])
# plt.xlabel('X')
# plt.ylabel('U(0,x)')

# fig = plt.figure()
# plt.plot(np.arange(0,2,0.02),-Z[0])
# plt.xlabel('X')
# plt.ylabel('U(T,x)')

plt.show()