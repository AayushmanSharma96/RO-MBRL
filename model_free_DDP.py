'''
copyright @ Karthikeya S Parunandi - karthikeyasharma91@gmail.com
Python class for model free DDP method.

Date: July 6, 2019

ASSUMPTIONS :

1) Costs are quadratic functions
2) Default is set to ILQR - by dropping the second order terms of dynamics.

'''
#!/usr/bin/env python

from __future__ import division

# Numerics
import numpy as np
import time
# Parameters
#import params
import matplotlib.pyplot as plt
from ltv_sys_id import ltv_sys_id_class
import json
#from PFM import *
import PFM
import PFM_CH
# import burgers_test as sim
import math
import burgers as sim
from progressbar import *
import modred as mr
from material_params import dt_ch, n_ch, n_ac, save_buffer_traj, save_buffer_control
from testnlse import NLSE
from scipy.linalg import fractional_matrix_power as frac
import scipy


class DDP(object):

	def __init__(self, MODEL_XML, n_x, n_u, alpha, horizon, initial_state, final_state):

		self.X_p_0 = initial_state
		self.X_g   = final_state

		widgets = [Percentage(), '   ', ETA(), ' (', Timer(), ')']
		self.pbar = ProgressBar(widgets=widgets)

		self.n_x = n_x
		self.n_u = n_u
		self.N = horizon
		self.n_a = self.N#3

		self.alpha = alpha

		
		# Define nominal state trajectory
		self.X_p = np.zeros((self.N, self.n_x, 1))
		self.X_p_temp = np.zeros((self.N, self.n_x, 1))
		self.X_t = np.zeros((100, self.n_x, 1))  #ADDED FOR COST COMP

		# Define POD reduced order parameters
		self.modes = np.zeros((self.n_x, self.n_a))
		self.proj = np.zeros((self.n_a, self.N))

		# Define nominal control trajectory
		self.U_p  = np.zeros((self.N, self.n_u, 1))
		self.U_p_temp = np.zeros((self.N, self.n_u, 1))
		self.U_t = np.zeros((100, self.n_u, 1))  #ADDED FOR COST COMP

		# Define sensitivity matrices
		# self.K = np.zeros((self.N, self.n_u, self.n_x))
		self.K = np.zeros((self.N, self.n_u, self.n_a))
		self.k = np.zeros((self.N, self.n_u, 1))
		
		self.V_xx = np.zeros((self.N, self.n_x, self.n_x))
		self.V_x = np.zeros((self.N, self.n_x, 1))

		self.V_aa = np.zeros((self.N, self.n_a, self.n_a))
		self.V_a = np.zeros((self.N, self.n_a, 1))

		
		# regularization parameter
		self.mu_min = 1e-3
		self.mu = 1e-3	#10**(-6)
		self.mu_max = 10**(8)
		self.delta_0 = 2
		self.delta = self.delta_0
		
		self.c_1 = 0#-6e-1
		self.count = 0
		self.episodic_cost_history = []
		self.control_cost_history = []

		__x_dim = int(math.sqrt(self.n_x)) 
		self.p_mask = (self.X_g.reshape((__x_dim, __x_dim))==1)*1
		self.m_mask = (self.X_g.reshape((__x_dim, __x_dim))==-1)*1

		self.print_flag = 0##
		self.it_count = 0

		self.ctr = 1


	def iterate_ddp(self, n_iterations, finite_difference_gradients_flag=False, u_init=None): #ADD OPTION TO INPUT INITIAL GUESS IN INIT TRAJ()
		
		'''
			Main function that carries out the algorithm at higher level

		'''
		# Initialize the trajectory with the desired initial guess
		self.initialize_traj(init=u_init)
		self.episodic_cost_history.append(self.calculate_total_cost(self.X_p_0, self.X_p, self.U_p, self.N)[0][0])
		
		for j in self.pbar(range(n_iterations)):	
			c1 = self.calculate_total_cost(self.X_p_0, self.X_p, self.U_p, self.N)[0][0]			

			self.it_count = j
			b_pass_success_flag, del_J_alpha = self.backward_pass_ro(finite_difference_gradients_flag, activate_second_order_dynamics=0)

			if b_pass_success_flag == 1:

				self.regularization_dec_mu()
				f_pass_success_flag = self.forward_pass_ro(del_J_alpha, n_iter=j)

				if not f_pass_success_flag:

					#print("Forward pass doomed")
					i = 2

					while not f_pass_success_flag:
 
						#print("Forward pass-trying %{}th time".format(i))
						self.alpha = self.alpha*0.99	#simulated annealing
						i += 1
						f_pass_success_flag = self.forward_pass_ro(del_J_alpha, n_iter=j)

						#print("alpha = ", self.alpha)
				print('test = ',self.mu_min)
				print('Relevant modes = ', self.n_a)

			else:

				self.regularization_inc_mu()
				print("This iteration %{} is doomed".format(j))

			if j<5:
				self.alpha = self.alpha*0.9
			else:
				self.alpha = self.alpha*0.999
			
			self.episodic_cost_history.append(self.calculate_total_cost(self.X_p_0, self.X_p, self.U_p, self.N)[0][0])
			print('Cost at this iteration = ',self.calculate_total_cost(self.X_p_0, self.X_p, self.U_p, self.N)[0][0])
			# c2 = self.calculate_total_cost(self.X_p_0, self.X_p, self.U_p, self.N)[0][0]
			# if abs(c1-c2)<0.1:
			# 	break
			#self.episodic_cost_history.append(self.calculate_total_cost(self.X_p_0, self.X_t, np.repeat(self.U_p, int(100/self.N), axis=0), 100)[0][0])

			#self.control_cost_history.append(self.calculate_total_cost(self.X_p_0, self.X_t, np.repeat(self.U_p, int(100/self.N), axis=0), 100)[0][0])	



	def backward_pass(self, finite_difference_gradients_flag=False, activate_second_order_dynamics=0):

		################## defining local functions & variables for faster access ################

		partials_list = self.partials_list
		k = np.copy(self.k)
		K = np.copy(self.K)
		V_x = np.copy(self.V_x)
		V_xx = np.copy(self.V_xx)

		##########################################################################################
		
		V_x[self.N-1] = self.l_x_f(self.X_p[self.N-1])	

		np.copyto(V_xx[self.N-1], 2*self.Q_final)

		# Initialize before forward pass
		del_J_alpha = 0

		for t in range(self.N-1, -1, -1):
			
			if t>0:

				Q_x, Q_u, Q_xx, Q_uu, Q_ux = partials_list(self.X_p[t-1], self.U_p[t], V_x[t], V_xx[t], activate_second_order_dynamics, finite_difference_gradients_flag)

			elif t==0:

				Q_x, Q_u, Q_xx, Q_uu, Q_ux = partials_list(self.X_p_0, self.U_p[0], V_x[0], V_xx[0], activate_second_order_dynamics, finite_difference_gradients_flag)

			try:
				# If a matrix cannot be positive-definite, that means it cannot be cholesky decomposed
				np.linalg.cholesky(Q_uu)

			except np.linalg.LinAlgError:
				
				print("FAILED! Q_uu is not Positive definite at t=",t)

				b_pass_success_flag = 0

				# If Q_uu is not positive definite, revert to the earlier values 
				np.copyto(k, self.k)
				np.copyto(K, self.K)
				np.copyto(V_x, self.V_x)
				np.copyto(V_xx, self.V_xx)
				
				break

			else:

				b_pass_success_flag = 1
				
				# update gains as follows
				Q_uu_inv = np.linalg.inv(Q_uu)
				k[t] = -(Q_uu_inv @ Q_u)
				K[t] = -(Q_uu_inv @ Q_ux)

				del_J_alpha += -self.alpha*((k[t].T) @ Q_u) - 0.5*self.alpha**2 * ((k[t].T) @ (Q_uu @ k[t]))
				
				if t>0:

					V_x[t-1] = Q_x + (K[t].T) @ (Q_uu @ k[t]) + ((K[t].T) @ Q_u) + ((Q_ux.T) @ k[t])
					V_xx[t-1] = Q_xx + ((K[t].T) @ (Q_uu @ K[t])) + ((K[t].T) @ Q_ux) + ((Q_ux.T) @ K[t])


		######################### Update the new gains ##############################################

		np.copyto(self.k, k)
		np.copyto(self.K, K)
		np.copyto(self.V_x, V_x)
		np.copyto(self.V_xx, V_xx)
		
		#############################################################################################

		self.count += 1

		return b_pass_success_flag, del_J_alpha




	def backward_pass_ro(self, finite_difference_gradients_flag=False, activate_second_order_dynamics=0):

		################## defining local functions & variables for faster access ################

		partials_list = self.partials_list_ro
		k = np.copy(self.k)
		K = np.copy(self.K)
		V_a = np.copy(self.V_a)
		V_aa = np.copy(self.V_aa)
		modes = np.copy(self.modes)

		##########################################################################################
		
		V_a[self.N-1] = self.l_a_f(self.X_p[self.N-1])	

		np.copyto(V_aa[self.N-1], 2*modes.T @ (self.Q_final @ modes))

		# Initialize before forward pass
		del_J_alpha = 0
		A_T = np.zeros((self.n_a, self.n_a, self.N))
		B_T = np.zeros((self.n_a, self.n_u, self.N))
		Quu_T = np.zeros((self.n_u, self.n_u, self.N))
		Qu_T = np.zeros((self.n_u, 1, self.N))
		Qux_T = np.zeros((self.n_u, self.n_a, self.N))


		for t in range(self.N-1, -1, -1):
			
			if t>0:

				Q_a, Q_u, Q_aa, Q_uu, Q_ua, A_T[:,:,t], B_T[:,:,t] = partials_list(self.X_p[t-1], self.U_p[t], V_a[t], V_aa[t], modes, modes.T @ self.X_p[t-1], activate_second_order_dynamics, finite_difference_gradients_flag)
				Quu_T[:,:,t] = Q_uu
				Qu_T[:,:,t] = Q_u
				Qux_T[:,:,t] = Q_ua

			elif t==0:

				Q_a, Q_u, Q_aa, Q_uu, Q_ua, A_T[:,:,0], B_T[:,:,0] = partials_list(self.X_p_0, self.U_p[0], V_a[0], V_aa[0], modes, modes.T @ self.X_p_0, activate_second_order_dynamics, finite_difference_gradients_flag)
				Quu_T[:,:,0] = Q_uu
				Qu_T[:,:,0] = Q_u
				Qux_T[:,:,0] = Q_ua
			try:
				# If a matrix cannot be positive-definite, that means it cannot be cholesky decomposed
				np.linalg.cholesky(Q_uu)

			except np.linalg.LinAlgError:
				
				print("FAILED! Q_uu is not Positive definite at t=",t)

				b_pass_success_flag = 0

				# If Q_uu is not positive definite, revert to the earlier values 
				np.copyto(k, self.k)
				np.copyto(K, self.K)
				np.copyto(V_a, self.V_a)
				np.copyto(V_aa, self.V_aa)
				
				break

			else:

				b_pass_success_flag = 1
				
				# update gains as follows
				Q_uu_inv = np.linalg.inv(Q_uu)
				k[t] = -(Q_uu_inv @ Q_u)
				K[t] = -(Q_uu_inv @ Q_ua)

				del_J_alpha += -self.alpha*((k[t].T) @ Q_u) - 0.5*self.alpha**2 * ((k[t].T) @ (Q_uu @ k[t]))
				
				if t>0:

					V_a[t-1] = Q_a + (K[t].T) @ (Q_uu @ k[t]) + ((K[t].T) @ Q_u) + ((Q_ua.T) @ k[t])
					V_aa[t-1] = Q_aa + ((K[t].T) @ (Q_uu @ K[t])) + ((K[t].T) @ Q_ua) + ((Q_ua.T) @ K[t])


		######################### Update the new gains ##############################################

		np.copyto(self.k, k)
		np.copyto(self.K, K)
		np.copyto(self.V_a, V_a)
		np.copyto(self.V_aa, V_aa)

		# np.save('A_1.npy', A_T)
		# np.save('B_1.npy', B_T)

		# A_a, B_a, V_a_F_XU_XU = self.sys_id_ro_arma(self.X_p_0, self.U_p, modes, central_diff=1, activate_second_order=activate_second_order_dynamics, V_a_=None, p_mask=self.p_mask, m_mask=self.m_mask, n_a = self.n_a)

		
		# np.save('A_arma.npy', A_a)
		# np.save('B_arma.npy', B_a)		
		# np.save('Qu_ro.npy', Qu_T)
		# np.save('Quu_ro.npy', Quu_T)
		# np.save('Qux_ro.npy', Qux_T)
		# np.save('PHI.npy', self.modes)
		# np.save('k_fo.npy', self.k)
		# np.save('K_fo.npy', self.K)
		#############################################################################################

		self.count += 1

		return b_pass_success_flag, del_J_alpha




	def forward_pass_ro(self, del_J_alpha, n_iter):

		# Cost before forward pass
		J_1 = self.calculate_total_cost(self.X_p_0, self.X_p, self.U_p, self.N)
		
		np.copyto(self.X_p_temp, self.X_p)
		np.copyto(self.U_p_temp, self.U_p)

		self.forward_pass_sim()
		self.print_flag = 1##
		# Update POD basis
		self.pod(self.X_p_0, self.X_p, it=n_iter) 

		# Cost after forward pass
		J_2 = self.calculate_total_cost(self.X_p_0, self.X_p, self.U_p, self.N)
		
		z = (J_1 - J_2 )/del_J_alpha

		if z < self.c_1:

			np.copyto(self.X_p, self.X_p_temp)
			np.copyto(self.U_p, self.U_p_temp)
	
			f_pass_success_flag = 0
			#print("f",z, del_J_alpha, J_1, J_2)

		else:

			f_pass_success_flag = 1
			

		return f_pass_success_flag


	def pod(self, x_init, x_nom, it):

		x = np.zeros((self.n_x, self.N+1))
		x[:,0] = x_init.flatten()
		x_nom = x_nom.reshape((self.N, self.n_x))
		x[:,1:] = x_nom.T#x_nom[:,:,0].T#x_nom.reshape((self.n_x, self.N))

		# print(x[0,:])	
		# print('ctrl = ', self.U_p)

		# Compute POD
		# num_modes = 5
		# print(np.shape(x.T @ x))
		# V, S, Vh = scipy.linalg.svd(x_nom @ x_nom.T,lapack_driver="gesvd")
		
		# s2 = np.linalg.inv(np.diag(np.sqrt(S)))
		# print(self.U_p)
		# print(x)
		# print('Control break = ',self.U_p)
		POD_res = mr.compute_POD_arrays_snaps_method(
		    x)#, list(mr.range(num_modes)))


		modes = POD_res.modes
		# print('MODES = ', np.shape(modes)[1])
		proj = POD_res.proj_coeffs
		eigvals = POD_res.eigvals



		# print(np.shape(modes))
		# U = x_nom.T @(Vh @ np.sqrt(np.linalg.inv(np.diag(S))))
		# print(np.shape(modes))
		# print(U-modes.reshape(np.shape(U)))
		# sys.exit()
		# print(eigvals)
		# print(np.linalg.norm(modes-modes1,2))
		# sys.exit()
		sum_eigvals=np.sum(eigvals)
		rel_energy=0
		# print(x[:,0])
		for i in range(np.shape(eigvals)[0]):
			rel_energy+=eigvals[i]/sum_eigvals
			if rel_energy>0.99999:
				# print('sum = ', sum_eigvals)
				# print(eigvals)
				
				self.n_a = np.shape(eigvals)[0]#i+1
				# self.n_a = i+1
				# print('Rel mod = ', i+1)
				
				# if self.print_flag==1:
				# 	print(np.shape(eigvals))
				# 	print('Relevant modes = ', i+1)
				# 	self.print_flag=0

				break
				# sys.exit()

		# Update modes and projection coefficients
		if self.it_count>=-1000000:
			self.n_a = self.n_x

		self.modes = np.zeros((self.n_x, self.n_a))
		self.proj = np.zeros((self.n_a, self.N+1))
		self.V_aa = np.zeros((self.N, self.n_a, self.n_a))
		self.V_a = np.zeros((self.N, self.n_a, 1))
		self.K = np.zeros((self.N, self.n_u, self.n_a))


		if self.it_count<-100000:
			np.copyto(self.modes, modes[:, :self.n_a])
			np.copyto(self.proj, proj[:self.n_a])
		else:
			np.copyto(self.modes, np.eye(self.n_a))

		# if it==-1:

		# 	np.save('Init_traj_bg.npy', self.X_p)
		# 	np.save('Init_modes_bg.npy', self.modes)
		# if it>0:
		# 	np.save('Fin_traj_bg.npy', self.X_p)
		# 	np.save('Fin_modes_bg.npy', self.modes)
		# np.copyto(self.proj, x)
		# np.copyto(self.modes, modes[:, :self.n_a])
		# np.save('Final_mode.npy',self.modes)
		# np.copyto(self.proj, proj[:self.n_a,1:])

		# x1 = self.modes @ proj[:self.n_a,:]
		
		# print('phi.T x phi = ',self.modes.T @ self.modes)
		
		# sys.exit()


		# print(np.shape(self.modes))
		# sys.exit()


	def partials_list_ro(self, x, u, V_a_next, V_aa_next, modes, proj, activate_second_order_dynamics, finite_difference_gradients_flag=False, ):	

		################## defining local functions / variables for faster access ################

		n_x = self.n_x
		n_u = self.n_u
		n_a = self.n_a
		modes = self.modes
		##########################################################################################
		if finite_difference_gradients_flag:

			AB, V_x_F_XU_XU = self.sys_id_FD(x, u, central_diff=1, activate_second_order=activate_second_order_dynamics, V_x_=V_x_next)

		else:

			AB_a, V_a_F_XU_XU = self.sys_id_ro(x, u, modes, central_diff=1, activate_second_order=activate_second_order_dynamics, V_a_=V_a_next, p_mask=self.p_mask, m_mask=self.m_mask, n_a = self.n_a)
		
		A = np.copy(AB_a[:, 0:n_a])
		B = np.copy(AB_a[:, n_a:])


		Q_a = self.l_a(x) + ((A.T) @ V_a_next)
		Q_u = self.l_u(u) + ((B.T) @ V_a_next)
		Q_aa = 2*modes.T @ (self.Q @ modes) + ((A.T) @ ((V_aa_next)  @ A)) 
		Q_ua = (B.T) @ ((V_aa_next + self.mu*np.eye(V_aa_next.shape[0])) @ A)
		Q_uu = 2*self.R + (B.T) @ ((V_aa_next + self.mu*np.eye(V_aa_next.shape[0])) @ B)
		
		
		if(activate_second_order_dynamics):

			Q_aa +=  V_a_F_XU_XU[:n_x, :n_x]  
			Q_ua +=  0.5*(V_a_F_XU_XU[n_x:n_x + n_u, :n_x ] + V_a_F_XU_XU[:n_x, n_x: n_x + n_u].T)
			Q_uu +=  V_a_F_XU_XU[n_x:n_x + n_u, n_x:n_x + n_u]

		return Q_a, Q_u, Q_aa, Q_uu, Q_ua, A, B# REmove A,B later

	def forward_pass_sim(self, render=0, std=0, pr=False):
		
		################## defining local functions & variables for faster access ################

		#sim = self.sim
		
		##########################################################################################

		ctrl = np.zeros(self.n_u)
		
		# AC and CH Params##############
		xdim = int(math.sqrt(self.n_x))
		N = xdim
		################################

		# Burgers Params ###############################################

		# N = self.n_x
		# dx = (2)/self.n_x
		# dt = 1e-4
		# nu = 0.1#0.01/np.pi
		# tperiod = 1000

		################################################################
		# ctrl_nlse = np.zeros(np.shape(self.X_p_0[:int(self.n_x/2)]))

		# print('Control check = ', self.U_p)
		# print('k = ',self.k)
		# print('K = ', self.K)
		for t in range(0, self.N):
			

			if t==0:

				self.U_p[t] = self.U_p_temp[t] + self.alpha*self.k[t]# + np.random.normal(np.zeros(np.shape(self.U_p_temp[t])), std)
			
			else:

				self.U_p[t] = self.U_p_temp[t] + self.alpha*self.k[t] + (self.K[t] @ self.modes.T @ (self.X_p[t-1] - self.X_p_temp[t-1]))# + np.random.normal(np.zeros(np.shape(self.U_p_temp[t])), std)
			
			
			ctrl[:] = self.U_p[t].flatten()#+np.random.normal(np.zeros(np.shape(ctrl[:])), std)
			
			##GSO ONLY##############################################
			ctrl_T = self.p_mask*ctrl[0]+self.m_mask*ctrl[1]
			ctrl_h = self.p_mask*ctrl[2]+self.m_mask*ctrl[3]
			
			# ctrl_nlse[0] = ctrl[0]
			# ctrl_nlse[-1] = ctrl[1]
			##GSO ONLY##############################################
			
			if t is 0:	
				# self.X_p[t] = sim.simulation(tperiod, nu, dt, ctrl[:int(len(ctrl)/2)], ctrl[int(len(ctrl)/2):], self.X_p_0)#sim.burgers_1d(self.X_p_0, dx, dt, nu, ctrl[:int(len(ctrl)/2)], ctrl[int(len(ctrl)/2):], tperiod)
				# self.X_p[t] = PFM.simulation(n_ac, 0.001, 0.001, ctrl_T, ctrl_h, self.X_p_0.reshape((xdim, xdim))).reshape((xdim*xdim, 1))
				# self.X_p[t] = PFM.simulation(n_ac, 0.001, 0.001, ctrl[:int(len(ctrl)/2)].reshape((xdim, xdim)), ctrl[int(len(ctrl)/2):].reshape((xdim, xdim)), self.X_p_0.reshape((xdim, xdim))).reshape((xdim*xdim, 1))
				self.X_p[t] = PFM_CH.simulation(n_ch, 0.001, dt_ch, ctrl_T, ctrl_h, self.X_p_0.reshape((xdim, xdim))).reshape((xdim*xdim, 1))
				# psi0 = self.X_p_0[:int(self.n_x/2)]+1j*self.X_p_0[int(self.n_x/2):]
				# self.X_p[t] = self.simulator.simulate(dt=0.05, psi0=psi0.flatten(), ctrl=ctrl_nlse).reshape((self.n_x,1))
			else:
				# self.X_p[t] = sim.simulation(tperiod, nu, dt, ctrl[:int(len(ctrl)/2)], ctrl[int(len(ctrl)/2):], self.X_p[t-1])#sim.burgers_1d(self.X_p[t-1], dx, dt, nu, ctrl[:int(len(ctrl)/2)], ctrl[int(len(ctrl)/2):], tperiod)
				# self.X_p[t] = PFM.simulation(n_ac, 0.001, 0.001, ctrl_T, ctrl_h, self.X_p[t-1].reshape((xdim, xdim))).reshape((xdim*xdim, 1))
				# self.X_p[t] = PFM.simulation(n_ac, 0.001, 0.001, ctrl[:int(len(ctrl)/2)].reshape((xdim, xdim)), ctrl[int(len(ctrl)/2):].reshape((xdim, xdim)), self.X_p[t-1].reshape((xdim, xdim))).reshape((xdim*xdim, 1))
				self.X_p[t] = PFM_CH.simulation(n_ch, 0.001, dt_ch, ctrl_T, ctrl_h, self.X_p[t-1].reshape((xdim, xdim))).reshape((xdim*xdim, 1))
				# psi0 = self.X_p[t-1,:int(self.n_x/2)]+1j*self.X_p[t-1,int(self.n_x/2):]
				# self.X_p[t] = self.simulator.simulate(dt=0.05, psi0=psi0.flatten(), ctrl=ctrl_nlse).reshape((self.n_x,1))

		# print('Control here = ', self.U_p)

	def cost(self, state, control):

		raise NotImplementedError()



	def initialize_traj(self):
		# initial guess for the trajectory
		pass



	def test_episode(self, render=0, path=None, noise_stddev=0, printch=False, check_plan=True, init=None, version=0, cst=0):
		
		'''
			Test the episode using the current policy if no path is passed. If a path is mentioned, it simulates the controls from that path
		'''
		ctrl = np.zeros(self.n_u)
		u_net = np.zeros((self.n_u,1))
		u1 = np.zeros((self.n_x,1))
		u2 = np.zeros((self.n_x,1))
		ctrl_seq = np.zeros((self.N, 800))
		dx = 1/self.n_x
		dt = 0.1
		nu = 0.01/np.pi
		#ctrl_seq_temp = np.zeros((self.N, self.n_u))

		xdim = int(math.sqrt(self.n_x))
		N=xdim
		#cst=np.zeros(self.N)
		
		if path is None:
		
			self.forward_pass_sim(render=1, std=noise_stddev, pr=printch)
			
		
		else:
		

			
			#self.X_p[-1] = np.zeros((self.n_x, 1))#+np.random.normal(np.zeros(np.shape(self.X_p[-1])), noise_stddev)
			CTR = 0
			cost = cst
			costZ=0
			control_cost=np.zeros((self.N,1))
			#print(self.X_p[-1].reshape((10,10)))
			# if init is not None:
			# 	self.X_p[-1] = init
			# 	#self.X_p[0] = init
				#CTR=1

			with open(path) as f:

				Pi = json.load(f)

			for i in range(CTR, self.N):
				
				#self.sim.forward()

				if i is not -1:#==0:
					#ctrl_seq[i, :] = np.array(Pi['U'][str(i)]).reshape(np.shape(ctrl_seq[i]))
					# print(np.shape(u))
					# u = np.array(Pi['U'][str(i)])
					# u1 = u[0:4].reshape((2, 2))
					# u1r = np.repeat(u1, 5, axis=0)
					# u1 = np.repeat(u1r, 5, axis=1).reshape((100,))
					# u2 = u[4:].reshape((2, 2))
					# u2r = np.repeat(u2, 5, axis=0)
					# u2 = np.repeat(u2r, 5, axis=1).reshape((100,))

					# ctrl_seq[i,:100]=u1
					# ctrl_seq[i, 100:]=u2
					ctrl[:] = ((np.array(Pi['U'][str(i)])) + np.random.normal(np.zeros(np.shape(Pi['U'][str(i)])), noise_stddev)).flatten()										
					# ctrl_seq[i,:] = ctrl
					'''elif i==3:#To compensate for reduced horizon, h_orig= 10
					
						ctrl[:] = (np.array(Pi['U'][str(i)]) + \
												np.random.normal(np.zeros(np.shape(Pi['U'][str(i)])), noise_stddev) + \
												np.array(Pi['K'][str(i-1)]) @ (self.X_p[i-1] - np.array(Pi['X'][str(i-1)]))).flatten()'''

				else:
					
					#ctrl_seq[i] = np.array(Pi['U'][str(i)]).reshape(np.shape(ctrl_seq[i]))
					# u1 = (np.array(Pi['U'][str(i)])[0:4]).reshape((2, 2))
					# u1r = np.repeat(u1, 5, axis=0)
					# u1 = np.repeat(u1r, 5, axis=1).flatten()
					# u2 = (np.array(Pi['U'][str(i)]))[4:].reshape((2, 2))
					# u2r = np.repeat(u2, 5, axis=0)
					# u2 = np.repeat(u2r, 5, axis=1).flatten()
					# # k1 = np.array(Pi['K'][str(i-1)])
					# # print('K1 = '+str(np.shape(k1)))
					# k1r = np.repeat(k1, int(self.n_u/8), axis=0)
					# # print('K1r = '+str(np.shape(k1r)))
					# # print(self.n_u)
					# k1 = np.repeat(k1r, int((self.n_x/4)), axis=1)
					# # print('K1 = '+str(np.shape(k1)))
					# x1 = np.array(Pi['X'][str(i-1)])
					# x1=x1.reshape((2,2))
					
					# x1r = np.repeat(x1, int(math.sqrt(self.n_x/4)), axis=0)
					# x1 = np.repeat(x1r, int(math.sqrt(self.n_x/4)), axis=1).reshape(self.n_x,1)
					# ctrl_seq[i, :100] = u1
					# ctrl_seq[i, 100:]= u2
					# #print(np.shape(u_net))
					
					# print(np.shape(k1@(self.X_p[i-1]-x1)))
					# ctrl[:] = (u_net + (k1 @ (self.X_p[i-1]-x1))).flatten()
					ctrl[:] = (np.array(Pi['U'][str(i)]) + \
											np.random.normal(np.zeros(np.shape(Pi['U'][str(i)])), noise_stddev) + \
											np.array(Pi['K'][str(i-1)]) @ (self.X_p[i-1] - np.array(Pi['X'][str(i)]))).flatten()#self.state_output(self.sim.get_state()) - np.array(Pi['X'][str(i-1)]))).flatten()'''
					#ctrl_seq[i,:] = ctrl
				#self.X_p[i] = self.sim.step(self.X_p[i-1].reshape((xdim, xdim)), ctrl[0:int(len(ctrl)/2)], ctrl[int(len(ctrl)/2):], 25).reshape([xdim*xdim, 1])
				
				'''for p in range(int(100/self.N)):
					if p==0:
						self.X_p_temp[i] = PFM.simulation(int(n_ac/int(100/self.N)), 0.001, 0.001, ctrl[:int(len(ctrl)/2)].reshape((xdim, xdim)), ctrl[int(len(ctrl)/2):].reshape((xdim, xdim)), self.X_p[i-1].reshape((xdim, xdim))).reshape((xdim*xdim, 1))
						cst[i]+=(self.cost(self.X_p_temp[i], ctrl))
					if p==int(100/self.N)-1:
						self.X_p[i] = PFM.simulation(int(n_ac/int(100/self.N)), 0.001, 0.001, ctrl[:int(len(ctrl)/2)].reshape((xdim, xdim)), ctrl[int(len(ctrl)/2):].reshape((xdim, xdim)), self.X_p_temp[i].reshape((xdim, xdim))).reshape((xdim*xdim, 1))
						cst[i]+=(self.cost(self.X_p[i], ctrl))
					else:
						self.X_p_temp[i] = PFM.simulation(int(n_ac/int(100/self.N)), 0.001, 0.001, ctrl[:int(len(ctrl)/2)].reshape((xdim, xdim)), ctrl[int(len(ctrl)/2):].reshape((xdim, xdim)), self.X_p_temp[i].reshape((xdim, xdim))).reshape((xdim*xdim, 1))
						cst[i]+=(self.cost(self.X_p_temp[i], ctrl))'''

				# print('PREV FUCKIN VALUE = ', self.X_p[i-1])
				# u1 = ctrl[0:self.n_x].reshape((xdim, xdim))
				# u2 = ctrl[self.n_x:].reshape((xdim, xdim))
				# u1_temp = np.repeat(u1, int(20/xdim), axis=0)
				# u2_temp = np.repeat(u2, int(20/xdim), axis=0)
				# u1 = np.repeat(u1_temp, int(20/xdim), axis=1)
				# u2 = np.repeat(u2_temp, int(20/xdim), axis=1)

				# ctrl_seq[i, 0:400] = u1.reshape(np.shape(ctrl_seq[i, 0:400]))
				# ctrl_seq[i, 400:] = u2.reshape(np.shape(ctrl_seq[i, 400:]))
				
				if i==0:
					self.X_p[i] = PFM.simulation(n_ac, 0.001, 0.001, ctrl[:int(len(ctrl)/2)].reshape((xdim, xdim)), ctrl[int(len(ctrl)/2):].reshape((xdim, xdim)), init.reshape((xdim, xdim))).reshape((xdim*xdim, 1))
					# self.X_p[i] = sim.burgers_1d(init, dx, dt, nu, ctrl[:int(len(ctrl)/2)], ctrl[int(len(ctrl)/2):])
				else:
					# self.X_p[i] = sim.burgers_1d(self.X_p[i-1], dx, dt, nu, ctrl[:int(len(ctrl)/2)], ctrl[int(len(ctrl)/2):])
					self.X_p[i] = PFM.simulation(n_ac, 0.001, 0.001, ctrl[:int(len(ctrl)/2)].reshape((xdim, xdim)), ctrl[int(len(ctrl)/2):].reshape((xdim, xdim)), self.X_p[i-1].reshape((xdim, xdim))).reshape((xdim*xdim, 1))
				
				#print(self.X_p[i].reshape((10,10)))
				#self.X_p[i] = PFM.simulation(int(n_ac/int(100/self.N)), 0.001, 0.001, ctrl[:int(len(ctrl)/2)].reshape((xdim, xdim)), ctrl[int(len(ctrl)/2):].reshape((xdim, xdim)), self.X_p[i-1].reshape((xdim, xdim))).reshape((xdim*xdim, 1))
				#self.X_p[i] = PFM.simulation(25, 0.001, 0.001, np.zeros((xdim, xdim)), ctrl[:].reshape((xdim, xdim)), self.X_p[i-1].reshape((xdim, xdim))).reshape((xdim*xdim, 1))
				#self.X_p[i] = PFM_CH.simulation(n_ch, 0.001, dt_ch, ctrl[:int(len(ctrl)/2)].reshape((xdim, xdim)), ctrl[int(len(ctrl)/2):].reshape((xdim, xdim)), self.X_p[i-1].reshape((xdim, xdim))).reshape((xdim*xdim, 1))
				#self.sim.plotCMAP(self.X_p[i].reshape((xdim, xdim)), i)
				#print(np.linalg.norm(self.X_p[i-1]-np.array(Pi['X'][str(i)])))
				#print('Simulated = '+str(ctrl[:]))
				# print('PREV FUCKIN VALUE = ', self.X_p[i-1])
				# print('NOMINAL COST = ',np.array(Pi['U'][str(i)]))
				# print('X_nom = ',np.array(Pi['X'][str(i)]))
				# print('Delta Friggin X = ',self.X_p[i-1] - np.array(Pi['X'][str(i)]))
				# print('X_act = ',self.X_p[i-1])
				# print('X_upd = ',self.X_p[i])
				

				
					#print('X_non_re_term = ',self.X_p[i])

				
				if i==0:
					costZ+=self.cost(init, ctrl)
					print('Cost = ',self.cost(init, ctrl))
				else:
					costZ+=self.cost(self.X_p[i-1], ctrl)
					print('Cost = ',self.cost(self.X_p[i-1], ctrl))
				control_cost[i]+=(((ctrl_seq[i].T) @ (.05*2*np.eye(800))) @ ctrl_seq[i])
				#print('X_non_re = ',self.X_p[i])
				#print('U_non_re = ',ctrl)
				if check_plan is True:
					if i==0:
						#np.linalg.norm(self.X_p[i]-np.array(Pi['X'][str(i)]))>3:#say
						#print("The divergence occurs at t = "+str(i))
						#print(np.linalg.norm(self.X_p[i]-np.array(Pi['X'][str(i)])))
						cost += self.cost(init, ctrl)
						# print('X here = ', init.reshape((2,2)))
						print('replanning con cost = ', cost)
						print('replanning inc Cost = ',self.cost(init, ctrl))
						#print('X=',self.X_p[i])
						#print('Control = ',ctrl)
						self.replan(self.X_p[i], i+1, self.N, version+1, cost)
						break
				
				#if render:
					#self.sim.render(mode='window')
			#print(self.X_p[-1])
			#print(np.sum(cst))

		if check_plan is not True:
			costZ+=self.cost_final(self.X_p[self.N-1])
			print('Cost_term = ',self.cost_final(self.X_p[self.N-1]))
			print('Total z cost = '+str(costZ))
			print("Total re cost 1 = "+str(cost+self.cost_final(self.X_p[self.N-1])))#self.cost_final(self.X_p[i])))
			# print("X Final = ", self.X_p[self.N-1].reshape((10,10)))
			#temp = np.load('control_cost_totals_disc.npy')
			print("Control cost = ", control_cost)
			np.shape(control_cost)
			# print('Max Control = ', np.max(ctrl_seq))
			#np.save('control_cost_totals_disc.npy', np.append(temp, control_cost))
			#print('Term rep Cost = ',self.cost_final(self.X_p[i]))
			#print('X_re_term = ',self.X_p[i])
			print(i)
			feedback_costs = np.load('std8_noise_nfb.npy')
			np.save("std8_noise_nfb.npy", np.append(feedback_costs, costZ))

			# replan_costs = np.load('replan_costs_std8_noise.npy')
			# np.save("replan_costs_std8_noise.npy", np.append(replan_costs, cost+self.cost_final(self.X_p[self.N-1])))
			# print('Post-save check')

	
			

		#print('Total cost = '+str(self.calculate_total_cost(self.X_p[-1], self.X_p, ctrl_seq, self.N)))

		'''if check_plan is False:
			print(ctrl_seq)
		else:
			print(ctrl_seq[i])'''
		return self.X_p[self.N-1]#self.state_output(self.sim.get_state())
			


	def feedback(self, W_x_LQR, W_u_LQR, W_x_LQR_f, finite_difference_gradients_flag=False):
		'''
		AB matrix comprises of A and B as [A | B] stacked at every ascending time-step, where,
		A - f_x
		B - f_u
		'''	

		P = W_x_LQR_f

		for t in range(self.N-1, 0, -1):

			if finite_difference_gradients_flag:

				AB, V_x_F_XU_XU = self.sys_id_FD(self.X_p[t-1], self.U_p[t], central_diff=1)

			else:

				AB, V_x_F_XU_XU = self.sys_id(self.X_p[t-1], self.U_p[t], central_diff=1)

			A = AB[:, 0:self.n_x]
			B = AB[:, self.n_x:]

			S = W_u_LQR + ( (np.transpose(B) @ P) @ B)

			# LQR gain 
			self.K[t] = -np.linalg.inv(S) @ ( (np.transpose(B) @ P) @ A)
			
			# second order equation
			P = W_x_LQR  +  ((np.transpose(A) @ P) @ A) - ((np.transpose(self.K[t]) @ S) @ self.K[t]) 



	def calculate_total_cost(self, initial_state, state_traj, control_traj, horizon):

		# assign the function to a local function variable
		incremental_cost = self.cost

		#initialize total cost
		cost_total = incremental_cost(initial_state, control_traj[0])
		cost_total += sum(incremental_cost(state_traj[t], control_traj[t+1]) for t in range(0, horizon-1)) #CHANGE HERE FOR COMP/ change here
		cost_total += self.cost_final(state_traj[horizon-1])

		return cost_total



	def regularization_inc_mu(self):

		# increase mu - regularization 

		self.delta = np.maximum(self.delta_0, self.delta_0*self.delta)

		self.mu = np.maximum(self.mu_min, self.mu*self.delta)

		if self.mu > self.mu_max:

			self.mu = self.mu_max


		#print(self.mu)



	def regularization_dec_mu(self):

		# decrease mu - regularization 

		self.delta = np.minimum(1/self.delta_0, self.delta/self.delta_0)

		if self.mu*self.delta > self.mu_min:

			self.mu = self.mu*self.delta

		else:
			self.mu = self.mu_min



	def plot_(self, y, save_to_path=None, x=None, show=1):

		if x==None:
			
			plt.figure(figsize=(7, 5))
			plt.plot(y, linewidth=2)
			plt.xlabel('Training iteration count', fontweight="bold", fontsize=12)
			plt.ylabel('Episodic cost', fontweight="bold", fontsize=12)
			#plt.grid(linestyle='-.', linewidth=1)
			plt.grid(color='.910', linewidth=1.5)
			plt.title('Episodic cost vs No. of training iterations')
			if save_to_path is not None:
				plt.savefig(save_to_path, format='png')#, dpi=1000)
			plt.tight_layout()
			plt.show()
		
		else:

			plt.plot(y, x)
			plt.show()



	def plot_episodic_cost_history(self, save_to_path=None):

		try:
			self.plot_(np.asarray(self.episodic_cost_history).flatten(), save_to_path=save_to_path, x=None, show=1)

		except:

			print("Plotting failed")
			pass


	def save_policy(self, path_to_file):

		Pi = {}
		# Open-loop part of the policy
		Pi['U'] = {}
		# Closed loop part of the policy - linear feedback gains
		Pi['K'] = {}
		Pi['X'] = {}

		for t in range(0, self.N):
			
			Pi['U'][t] = np.ndarray.tolist(self.U_p[t])
			Pi['K'][t] = np.ndarray.tolist(self.K[t])
			Pi['X'][t] = np.ndarray.tolist(self.X_p[t])
			
		with open(path_to_file, 'w') as outfile:  

			json.dump(Pi, outfile)



	def l_x(self, x):

		return 2*self.Q @ (x - self.X_g)


	def l_x_f(self, x):

		return 2*self.Q_final @ (x - self.X_g)

	def l_a(self, x):

		return 2*(self.modes.T @ (self.Q @ self.modes)) @ (self.modes.T @ (x-self.X_g))#x - self.modes.T @ self.X_g)


	def l_a_f(self, x):


		return 2*(self.modes.T @ (self.Q_final @ self.modes)) @ (self.modes.T @ (x-self.X_g))

	def l_u(self, u):

		return 2*self.R @ u


	def replan(self):
		#Replan when exceeding a threshold from nominal

		pass


	# def compute_snapshots(self, n_s=100, render=0, std=0, pr=False):
		
	# 	################## defining local functions & variables for faster access ################

	# 	#sim = self.sim
		
	# 	##########################################################################################

	# 	ctrl = np.zeros(self.n_u)
	# 	xdim = int(math.sqrt(self.n_x))
	# 	N = xdim
	# 	# xdim = int(math.sqrt(self.n_x))
	# 	# N = sdim#self.n_x
	# 	dx = (2*np.pi)/self.n_x
	# 	dt = 1e-4
	# 	nu = 0.1#0.01/np.pi
	# 	tperiod = 1000
	# 	k = int(n_s/self.N)


	# 	for i in range(0, n_s):#self.N):
			
	# 		t = int(i/10)
	# 		if i==0:

	# 			self.U_p[t] = self.U_p_temp[t] + self.alpha*self.k[t]# + np.random.normal(np.zeros(np.shape(self.U_p_temp[t])), std)
			
	# 		else:

	# 			self.U_p[t] = self.U_p_temp[t] + self.alpha*self.k[t] + (self.K[t] @ self.modes.T @ (self.X_p[t-1] - self.X_p_temp[t-1]))# + np.random.normal(np.zeros(np.shape(self.U_p_temp[t])), std)
			
			
	# 		ctrl[:] = self.U_p[t].flatten()#+np.random.normal(np.zeros(np.shape(ctrl[:])), std)
			
	# 		##GSO ONLY##############################################
	# 		ctrl_T = self.p_mask*ctrl[0]+self.m_mask*ctrl[1]
	# 		ctrl_h = self.p_mask*ctrl[2]+self.m_mask*ctrl[3]
	# 		##GSO ONLY##############################################
	# 		if t is 0:	
	# 			# self.X_p[t] = sim.simulation(tperiod, nu, dt, ctrl[:int(len(ctrl)/2)], ctrl[int(len(ctrl)/2):], self.X_p_0)#sim.burgers_1d(self.X_p_0, dx, dt, nu, ctrl[:int(len(ctrl)/2)], ctrl[int(len(ctrl)/2):], tperiod)
	# 			self.X_p[t] = PFM.simulation(n_ac, 0.001, 0.001, ctrl_T, ctrl_h, self.X_p_0.reshape((xdim, xdim))).reshape((xdim*xdim, 1))
	# 			# self.X_p[t] = PFM.simulation(n_ac, 0.001, 0.001, ctrl[:int(len(ctrl)/2)].reshape((xdim, xdim)), ctrl[int(len(ctrl)/2):].reshape((xdim, xdim)), self.X_p_0.reshape((xdim, xdim))).reshape((xdim*xdim, 1))
	# 			# self.X_p[t] = PFM_CH.simulation(n_ch, 0.001, dt_ch, ctrl_T, ctrl_h, self.X_p_0.reshape((xdim, xdim))).reshape((xdim*xdim, 1))
	# 		else:
	# 			# self.X_p[t] = sim.simulation(tperiod, nu, dt, ctrl[:int(len(ctrl)/2)], ctrl[int(len(ctrl)/2):], self.X_p[t-1])#sim.burgers_1d(self.X_p[t-1], dx, dt, nu, ctrl[:int(len(ctrl)/2)], ctrl[int(len(ctrl)/2):], tperiod)
	# 			self.X_p[t] = PFM.simulation(n_ac, 0.001, 0.001, ctrl_T, ctrl_h, self.X_p[t-1].reshape((xdim, xdim))).reshape((xdim*xdim, 1))
	# 			# self.X_p[t] = PFM.simulation(n_ac, 0.001, 0.001, ctrl[:int(len(ctrl)/2)].reshape((xdim, xdim)), ctrl[int(len(ctrl)/2):].reshape((xdim, xdim)), self.X_p[t-1].reshape((xdim, xdim))).reshape((xdim*xdim, 1))
	# 			# self.X_p[t] = PFM_CH.simulation(n_ch, 0.001, dt_ch, ctrl_T, ctrl_h, self.X_p[t-1].reshape((xdim, xdim))).reshape((xdim*xdim, 1))