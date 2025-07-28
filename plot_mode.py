import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import PFM
import sys

final_state=np.loadtxt('Final.txt')
state_dimension=50*50
sdim = 50
xdim = 50
initial_state = np.zeros((sdim*sdim))
initial_state = initial_state.reshape((state_dimension, 1))
X = np.zeros((11,state_dimension))
x = np.zeros((11,xdim,xdim))
X[0] = initial_state.flatten()
x[0] = X[0].reshape((xdim,xdim))
y = np.zeros((12,xdim,xdim))

u = np.load('control_ac_imp.npy')
print(np.shape(u))

p_mask = (final_state.reshape((xdim, xdim))==1)*1
m_mask = (final_state.reshape((xdim,xdim))==-1)*1

ctrl_T = np.zeros((10,50,50))
ctrl_h = np.zeros((10,50,50))

for t in range(10):
	k=0
	ctrl_T[t] = p_mask*u[k,0]+m_mask*u[k,1]
	ctrl_h[t] = p_mask*u[k,2]+m_mask*u[k,3]

for t in range(10):
	X[t+1] = PFM.simulation(3, 0.001, 0.001, ctrl_T[t], ctrl_h[t], X[t].reshape((xdim, xdim))).flatten()
	x[t+1] = X[t+1].reshape((xdim,xdim))
	for p in range(50):
		y[t+1,:,p]=x[t+1,:,50-p-1]			




for t in range(10):
	k=t
	ctrl_T[t] = p_mask*u[k,0]+m_mask*u[k,1]
	ctrl_h[t] = p_mask*u[k,2]+m_mask*u[k,3]
	X[t+1] = PFM.simulation(30, 0.001, 0.001, ctrl_T[t], ctrl_h[t], X[t].reshape((xdim, xdim))).flatten()
	x[t+1] = X[t+1].reshape((xdim,xdim))
	for p in range(50):
		y[11,:,p]=x[t+1,:,50-p-1]


X = 0.1*np.arange(50)
Y = 0.1*np.arange(50)#np.linspace(0, 2, 100)
X, Y = np.meshgrid(X, Y)
for t in range(12):
	Z = y[t]

	fig = plt.figure()#figsize =(14, 9))
	ax = plt.axes(projection ='3d')
	my_cmap = plt.get_cmap('RdBu')
 
	# Creating plot
	surf = ax.plot_surface(X, Y, Z,
                       cmap = my_cmap,
                       edgecolor ='none', vmin=-1.0, vmax=1.0)
	plt.colorbar(surf, ticks = np.linspace(-1,1,9))
	ax.set_zlim3d(-1, 1)
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('$\\phi$')
	ax.view_init(elev=45, azim=120)
	if t<9:
		plt.savefig('ATM_0'+str(t+1)+'.png', bbox='tight')
	if t>=9:
		plt.savefig('ATM_'+str(t+1)+'.png', bbox='tight')
# plt.show()
###############################################3

'''x0 = np.load('Init_traj_bg.npy')
xf = np.load('Fin_traj_bg.npy')

phi0 = np.load('Init_traj_bg.npy')
phif = np.load('Fin_traj_bg.npy')

state_dimension = 100
initial_state = np.round(np.sin(np.linspace(-np.pi, np.pi, state_dimension)),2).reshape(state_dimension, 1)

X0 = np.zeros((21,100,1))
Xf = np.zeros((21,100,1))

X0[0] = initial_state
Xf[0] = initial_state
X0[1:] = x0
Xf[1:] = xf



# Make data.
X = 0.1*np.arange(21)
Y = 0.02*np.arange(100)#np.linspace(0, 2, 100)
X, Y = np.meshgrid(X, Y)

Z = X0.reshape((21,100)).T

fig = plt.figure()#figsize =(14, 9))
ax = plt.axes(projection ='3d')
my_cmap = plt.get_cmap('viridis')
 
# Creating plot
surf = ax.plot_surface(X, Y, Z,
                       cmap = my_cmap,
                       edgecolor ='none')
ax.set_xlabel('Time')
ax.set_ylabel('x')
ax.set_zlabel('U(x,t)',rotation=90)
ax.set_title('Initial trajectory')

Z = Xf.reshape((21,100)).T

fig = plt.figure()#figsize =(14, 9))
ax = plt.axes(projection ='3d')

 
# Creating plot
surf = ax.plot_surface(X, Y, Z,
                       cmap = my_cmap,
                       edgecolor ='none')
ax.set_xlabel('Time')
ax.set_ylabel('x')
ax.set_zlabel('U(x,t)',rotation=90)
ax.set_title('Final trajectory')

plt.show()
sys.exit()



print(np.shape(phi0))
eps = np.linspace(0,2,100)



for t in range(10):
	plt.figure()
	plt.plot(eps,phi0[t].flatten())
	plt.plot(eps,phif[t].flatten())
	plt.legend(['Mode w.r.t. initial trajectory', 'Mode w.r.t. final trajectory'])
	plt.xlabel('x')
	plt.ylabel('$\\phi(x)$')

plt.show()'''