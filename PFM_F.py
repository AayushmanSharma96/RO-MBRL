import numpy as np
# from model_free_DDP import DDP
# import time
# from ltv_sys_id import ltv_sys_id_class
# from material_params import *
import os
import math
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import math
# import PFM_CH
# import PFM
# import modred as mr
import sys

# traj = np.load('trajectories.npy')
control = np.load('control_tr.npy')
traj = np.load('ch_traj.npy')
x = np.load('x.npy')
x = np.zeros((50,50))
for i in range(50):
	x[i]=0.02*np.arange(50)
y = x.T
x = x.flatten()
y=y.flatten()


# print(x)
# print(np.shape(x))
# x = np.linspace(0,1,2500)
# print(x)
# print(np.shape(x))
# sys.exit()



final_state=np.loadtxt('Final.txt')#[15:35, 2:22]
print(np.mean(final_state))
# final_state[:4,:]=-1
# p_mask = (final_state==1)*1
# m_mask = (final_state==-1)*1

initial_state = np.zeros((50,50))
X_p = np.zeros((2,50,50))
X_p_fin = np.zeros((11,50,50))
# x = np.load('ch_traj.npy').reshape((10,20,20))
# print(np.shape(x))
X_p[0] = initial_state
X_p_fin[0] = initial_state
X_p[1] = final_state

'''N = 20
ctrl = np.load('control_ch_imp.npy')
for t in range(10):
	# ctrl = control[0,:,1].flatten()
	ctrl_in = ctrl[0].flatten()

	# ctrl_T = p_mask*ctrl[0]+m_mask*ctrl[1]
	# ctrl_h = p_mask*ctrl[2]+m_mask*ctrl[3]

	ctrl_Tin = p_mask*ctrl_in[0]+m_mask*ctrl_in[1]
	ctrl_hin = p_mask*ctrl_in[2]+m_mask*ctrl_in[3]

	# X_p[t+1] = PFM_CH.simulation(250, 0.001, dt_ch, ctrl_T, ctrl_h, X_p[t])
	X_p_fin[t+1] = PFM_CH.simulation(int(n_ch/10), 0.001, dt_ch, ctrl_Tin, ctrl_hin, X_p_fin[t])'''

for t in range(2):
	z = X_p[t].flatten()
	print(z)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.invert_yaxis()
	mat = ax.plot_trisurf(x, y, z, cmap=plt.cm.get_cmap('RdBu'),vmin=-1, vmax=1)#, alpha=0.5)
	# ax.scatter(x,y,z, c='red')
	ax.view_init(30,120)
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('$\phi$')
	ax.set_zlim(-1,1)
	plt.colorbar(mat)#, ax=[ax], location='left')
	# plt.clim(mat, -1,1)
	print(str(t))
	savefile='/home/karthikeya/D2C-2.0/burgers_ro/Images/Upload_'+str(t)+'.png'
	plt.savefig(savefile)#, dpi=300, bbox_inches='tight')

sys.exit()

# POD_res0 = mr.compute_POD_arrays_snaps_method(
# 		    traj0.T)

# POD_res1 = mr.compute_POD_arrays_snaps_method(
# 		    traj1.T)

# POD_res20 = mr.compute_POD_arrays_snaps_method(
# 		    traj20.T)

# modes0 = POD_res0.modes[:,:4].reshape((20,20,4))
# modes1 = POD_res1.modes[:,:4].reshape((20,20,4))
# modes20 = POD_res20.modes[:,:4].reshape((20,20,4))



# plt.matshow(modes0[:,:,0], cmap=plt.cm.get_cmap('RdBu'))
# savefile='/home/karthikeya/D2C-2.0/burgers_ro/Images'+'/Basis01.png'
# plt.savefig(savefile, dpi=300, bbox_inches='tight')
# plt.matshow(modes0[:,:,1], cmap=plt.cm.get_cmap('RdBu'))
# savefile='/home/karthikeya/D2C-2.0/burgers_ro/Images'+'/Basis02.png'
# plt.savefig(savefile, dpi=300, bbox_inches='tight')
# plt.matshow(modes0[:,:,2], cmap=plt.cm.get_cmap('RdBu'))
# savefile='/home/karthikeya/D2C-2.0/burgers_ro/Images'+'/Basis03.png'
# plt.savefig(savefile, dpi=300, bbox_inches='tight')
# plt.matshow(modes0[:,:,3], cmap=plt.cm.get_cmap('RdBu'))
# savefile='/home/karthikeya/D2C-2.0/burgers_ro/Images'+'/Basis04.png'
# plt.savefig(savefile, dpi=300, bbox_inches='tight')




plt.matshow(X_p[0,:,:], cmap=plt.cm.get_cmap('RdBu'))
plt.colorbar()
plt.clim(-1,1)
savefile='/home/karthikeya/D2C-2.0/burgers_ro/Images'+'/Traj11.png'
plt.savefig(savefile, dpi=300, bbox_inches='tight')
plt.matshow(X_p[1,:,:], cmap=plt.cm.get_cmap('RdBu'))
plt.colorbar()
plt.clim(-1,1)
savefile='/home/karthikeya/D2C-2.0/burgers_ro/Images'+'/Traj12.png'
plt.savefig(savefile, dpi=300, bbox_inches='tight')
plt.matshow(X_p[3,:,:], cmap=plt.cm.get_cmap('RdBu'))
plt.colorbar()
plt.clim(-1,1)
savefile='/home/karthikeya/D2C-2.0/burgers_ro/Images'+'/Traj13.png'
plt.savefig(savefile, dpi=300, bbox_inches='tight')
plt.matshow(X_p[9,:,:], cmap=plt.cm.get_cmap('RdBu'))
plt.colorbar()
plt.clim(-1,1)
savefile='/home/karthikeya/D2C-2.0/burgers_ro/Images'+'/Traj14.png'
plt.savefig(savefile, dpi=300, bbox_inches='tight')

plt.matshow(X_p_fin[0,:,:], cmap=plt.cm.get_cmap('RdBu'))
plt.colorbar()
plt.clim(-1,1)
savefile='/home/karthikeya/D2C-2.0/burgers_ro/Images'+'/Traj201.png'
plt.savefig(savefile, dpi=300, bbox_inches='tight')
plt.matshow(X_p_fin[1,:,:], cmap=plt.cm.get_cmap('RdBu'))
plt.colorbar()
plt.clim(-1,1)
savefile='/home/karthikeya/D2C-2.0/burgers_ro/Images'+'/Traj202.png'
plt.savefig(savefile, dpi=300, bbox_inches='tight')
plt.matshow(X_p_fin[3,:,:], cmap=plt.cm.get_cmap('RdBu'))
plt.colorbar()
plt.clim(-1,1)
savefile='/home/karthikeya/D2C-2.0/burgers_ro/Images'+'/Traj203.png'
plt.savefig(savefile, dpi=300, bbox_inches='tight')
plt.matshow(X_p_fin[9,:,:], cmap=plt.cm.get_cmap('RdBu'))
plt.colorbar()
plt.clim(-1,1)
savefile='/home/karthikeya/D2C-2.0/burgers_ro/Images'+'/Traj204.png'
plt.savefig(savefile, dpi=300, bbox_inches='tight')

sys.exit()
final_state=np.loadtxt('Final.txt')[15:35, 2:22]
final_state[:4,:]=-1

sdim = 50
initial_state = np.zeros((50,50))#np.mean(final_state)*np.ones((sdim, sdim))
u = np.load('control_ch.npy')
u_init = np.load('U_INIT_TEST.npy')

# plt.plot(0.025*np.arange(10), u[:,0])
# plt.plot(0.025*np.arange(10), u[:,1])
# plt.plot(0.025*np.arange(10), u[:,2])
# plt.plot(0.025*np.arange(10), u[:,3])
# plt.xlabel('Time (s)')
# plt.ylabel('Control inputs')
# plt.legend(['U_1','U_2','U_3','U_3'])
# plt.title('Optimal Trajectory')

# plt.figure()
# plt.plot(0.025*np.arange(10), u_init[:,0])
# plt.plot(0.025*np.arange(10), u_init[:,1])
# plt.plot(0.025*np.arange(10), u_init[:,2])
# plt.plot(0.025*np.arange(10), u_init[:,3])
# plt.xlabel('Time (s)')
# plt.ylabel('Control inputs')
# plt.legend(['U_1','U_2','U_3','U_3'])
# plt.title('Initial Trajectory')
# plt.show()

# sys.exit()
p_mask = (final_state==1)*1
m_mask = (final_state==-1)*1

X_p = np.zeros((11,50,50))
X_p_in = np.zeros((11,50,50))
# x = np.load('ch_traj.npy').reshape((10,20,20))
# print(np.shape(x))
# X_p[0] = initial_state

N = 50
for t in range(10):
	ctrl = u[t].flatten()
	ctrl_in = u_init[t].flatten()

	ctrl_T = p_mask*ctrl[0]+m_mask*ctrl[1]
	ctrl_h = p_mask*ctrl[2]+m_mask*ctrl[3]

	ctrl_Tin = p_mask*ctrl_in[0]+m_mask*ctrl_in[1]
	ctrl_hin = p_mask*ctrl_in[2]+m_mask*ctrl_in[3]

	X_p[t+1] = PFM.simulation(n_ac, 0.001, 0.001, ctrl_T, ctrl_h, X_p[t])
	X_p_in[t+1] = PFM.simulation(n_ac, 0.001, 0.001, ctrl_Tin, ctrl_hin, X_p_in[t])


state_dimension = 50*50
x = X_p.reshape((11,state_dimension))
xin = X_p_in.reshape((11,state_dimension))

POD_res = mr.compute_POD_arrays_snaps_method(
		    x.T)

POD_res_in = mr.compute_POD_arrays_snaps_method(
		    xin.T)

modes = POD_res.modes
modes_in = POD_res_in.modes

# print(np.shape(modes))
fig = plt.figure()
ax = fig.add_subplot(121.5)
mat = ax.matshow(modes[:,0].reshape((50,50)), cmap=plt.cm.get_cmap('RdBu'))
savefile='/home/karthikeya/D2C-2.0/burgers_ro/Images'+'/fin_basis.png'
cbar = fig.colorbar(mat)
cbar.draw_all()
plt.savefig(savefile, dpi=300, bbox_inches='tight')
fig = plt.figure()
ax = fig.add_subplot(121.5)
mat = ax.matshow(modes_in[:,0].reshape((50,50)), cmap=plt.cm.get_cmap('RdBu'))
savefile='/home/karthikeya/D2C-2.0/burgers_ro/Images'+'/init_basis.png'
cbar = fig.colorbar(mat)
cbar.draw_all()
plt.savefig(savefile, dpi=300, bbox_inches='tight')
plt.show()
'''for t_time in range(11):#horizon):
		phi=X_p_in[t_time]
		ims = []
		x=(np.arange(int(math.sqrt(state_dimension))))*0.1
		y=(np.arange(int(math.sqrt(state_dimension))))*0.1
		X,Y = np.meshgrid(x,y)
		fig = plt.figure()#figsize=(10, 6.5))

		ax = fig.add_subplot(121.5)
		CS = ax.contourf(X, -Y, phi, cmap=plt.cm.get_cmap('RdBu'), levels = np.linspace(-1.5,1.5))#, vmax=1.0, vmin=-1.0)# v, cmap=plt.cm.get_cmap('RdBu'))
		cbar = fig.colorbar(CS)
		cbar.set_clim(vmax=1.0, vmin=-1.0)
		cbar.draw_all()
		ax.set_title('phase field evolution')
		ax.set_xlabel('x')
		ax.set_ylabel('y')
		ax.set_aspect(1.0)
		axs=[ax]
		# fig.suptitle('Explicit | 50x50', position=(0.2, 0.94), fontsize=20, fontweight='bold')
		#if t_time <= 5:
		#	fig.text(0.15, 0.84, 'T = -2, h = 0', fontsize=16)
		#else:
		#	fig.text(0.02, 0.84, 'T = rand(-2,2), h = rand(-1,0)', fontsize=16)
			
		# fig.text(0.35, 0.915, 'Step = %f' % t_time, fontsize=16, bbox={'facecolor': 'yellow', 'alpha': 0.3})
				
		ims.append([CS.collections,])

		idx=int(t_time)
		savefile='/home/karthikeya/D2C-2.0/burgers_ro/Images'+'/file_ac_init_' + str(idx) + '.png'
		
		# plt.show()
		plt.savefig(savefile, dpi=300, bbox_inches='tight')
		plt.draw()
		plt.pause(0.002)
		plt.close()'''