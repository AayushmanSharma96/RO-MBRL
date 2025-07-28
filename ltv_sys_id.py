'''
copyright @ Aayushman Sharma - aayushmansharma@tamu.edu

Date: July 27, 2025
'''
#!/usr/bin/env python

import numpy as np
import scipy.linalg.blas as blas
#from PFM import *
from material_params import *
#from mujoco_py import load_model_from_path, MjSim, MjViewer
import math
import PFM
import burgers as simulator
import PFM_CH
import sys
from testnlse import NLSE
import cmath


class ltv_sys_id_class(object):

	def __init__(self, model_xml_string,  state_size, action_size, n_substeps=1, n_samples=500):

		self.n_x = state_size
		self.n_u = action_size
		self.n_a = horizon#3

		# Standard deviation of the perturbation 
		self.sigma = 1e-3#7
		self.n_samples = n_samples

		# self.sim = NLSE()

		#self.sim = ACsolver()#MjSim(load_model_from_path(model_xml_string), nsubsteps=n_substeps)
		


	def sys_id(self, x_t, u_t, central_diff, activate_second_order=0, V_x_=None):

		'''
			system identification for a given nominal state and control
			returns - a numpy array with F_x and F_u horizantally stacked
		'''
		################## defining local functions & variables for faster access ################

		simulate = self.simulate
		n_x = self.n_x
		n_u = self.n_u

		##########################################################################################
		
		XU = np.random.normal(0.0, self.sigma, (self.n_samples, n_x + n_u))
		X_ = np.copy(XU[:, :n_x])
		U_ = np.copy(XU[:, n_x:])

		Cov_inv = (XU.T) @ XU
		V_x_F_XU_XU = None

		
		if central_diff:
			
			F_X_f = simulate((x_t.T) + X_, (u_t.T) + U_)
			F_X_b = simulate((x_t.T) - X_, (u_t.T) - U_)
			Y = 0.5*(F_X_f - F_X_b)
			
		else:

			Y = (simulate((x_t.T) + X_, (u_t.T) + U_) - simulate((x_t.T), (u_t.T)))

			
		
		F_XU = np.linalg.solve(Cov_inv, (XU.T @ Y)).T
		


		# If the second order terms in dynamics are activated as in the original DDP (not by default)
		if activate_second_order:

			assert (central_diff == activate_second_order)
			assert V_x_ is not None

			
			Z = (F_X_f + F_X_b - 2 * simulate((x_t.T), (u_t.T))).T

			D_XU = self.khatri_rao(XU.T, XU.T)
			
			triu_indices = np.triu_indices((n_x + n_u))
			linear_triu_indices = (n_x+n_u)*triu_indices[0] + triu_indices[1]
			
			D_XU_lin = np.copy(D_XU[linear_triu_indices,:])
			V_x_F_XU_XU_ = np.linalg.solve(10**12 * D_XU_lin @ D_XU_lin.T, 10**(12)*D_XU_lin @ (V_x_.T @ Z).T)
			D = np.zeros((n_x+n_u, n_x+n_u))
			# for ind, v in zip(list(np.array(triu_indices).T), V_x_F_XU_XU_):
			# 	V_x_F_XU_XU[ind] = v
			j=0
			for ind in np.array(triu_indices).T:
				D[ind[0]][ind[1]] = V_x_F_XU_XU_[j]
				j += 1
			
			V_x_F_XU_XU = (D + D.T)/2
			


		return F_XU, V_x_F_XU_XU	#(n_samples*self.sigma**2)


	def sys_id_ro(self, x_t, u_t, modes, central_diff, activate_second_order=0, V_a_=None, p_mask= None, m_mask= None, n_a=None):

		'''
			system identification for a given nominal state and control
			returns - a numpy array with F_x and F_u horizantally stacked
		'''
		################## defining local functions & variables for faster access ################

		simulate = self.simulate
		n_x = self.n_x
		n_u = self.n_u
		self.n_a = n_a

		if p_mask is not None:
			self.p_mask = p_mask
			self.m_mask = m_mask

		##########################################################################################
	
		# XU = np.random.normal(0.0, self.sigma, (self.n_samples, n_x + n_u))
		# X_ = np.copy(XU[:, :n_x])
		# U_ = np.copy(XU[:, n_x:])
		# alpha_ = X_ @ modes
		alphaU_ = np.random.normal(0.0, self.sigma, (self.n_samples, self.n_a + n_u))
		alpha_ = np.copy(alphaU_[:, :self.n_a])
		U_ = np.copy(alphaU_[:, self.n_a:])
		X_ = alpha_ @ modes.T 
		# alpha_ = X_ @ modes
		# print('X = ', X_)
		# print('alpha = ', alpha_)
		# alphaU_=np.hstack((alpha_.reshape((self.n_samples,self.n_a)), U_))

		Cov_inv = (alphaU_.T) @ alphaU_
		V_a_F_XU_XU = None

		
		if central_diff:
			
			F_X_f = simulate((x_t.T) + X_, (u_t.T) + U_)
			F_X_b = simulate((x_t.T) - X_, (u_t.T) - U_)
			# print(np.shape(modes))
			# print(np.shape(F_X_f))
			Y = 0.5*(F_X_f - F_X_b) @ modes
			
		else:

			Y =  (simulate((x_t.T) + X_, (u_t.T) + U_)@ modes - simulate((x_t.T), (u_t.T)))@ modes

		# print(np.shape(F_X_f))
		# print(np.shape(modes))
		# sys.exit()	
		# print('cov_inv = ', Cov_inv)
		# print('second = ',(alphaU_.T @ Y))
		F_XU = np.linalg.solve(Cov_inv, (alphaU_.T @ Y)).T
		


		# If the second order terms in dynamics are activated as in the original DDP (not by default)
		if activate_second_order:

			assert (central_diff == activate_second_order)
			assert V_x_ is not None

			
			Z = (F_X_f + F_X_b - 2 * simulate((x_t.T), (u_t.T))).T

			D_XU = self.khatri_rao(XU.T, XU.T)
			
			triu_indices = np.triu_indices((n_x + n_u))
			linear_triu_indices = (n_x+n_u)*triu_indices[0] + triu_indices[1]
			
			D_XU_lin = np.copy(D_XU[linear_triu_indices,:])
			V_x_F_XU_XU_ = np.linalg.solve(10**12 * D_XU_lin @ D_XU_lin.T, 10**(12)*D_XU_lin @ (V_x_.T @ Z).T)
			D = np.zeros((n_x+n_u, n_x+n_u))
			# for ind, v in zip(list(np.array(triu_indices).T), V_x_F_XU_XU_):
			# 	V_x_F_XU_XU[ind] = v
			j=0
			for ind in np.array(triu_indices).T:
				D[ind[0]][ind[1]] = V_x_F_XU_XU_[j]
				j += 1
			
			V_x_F_XU_XU = (D + D.T)/2
			


		return F_XU, V_a_F_XU_XU	#(n_samples*self.sigma**2)


	
	def sys_id_ro_arma(self, x_0, u_nom, modes, central_diff, activate_second_order=0, V_a_=None, p_mask= None, m_mask= None, n_a=None):

		'''
			system identification for a given nominal state and control
			returns - a numpy array with F_x and F_u horizontally stacked
		'''
		################## defining local functions & variables for faster access ################
		simulate = self.forward_simulate_arma
		n_x = self.n_x
		n_u = self.n_u


		if p_mask is not None:
			self.p_mask = p_mask
			self.m_mask = m_mask
		
		##########################################################################################
		# print('shape of U_nom = ', np.shape(u_nom))
		A_aug = np.zeros((horizon, n_a, n_a))
		B_aug = np.zeros((horizon, n_a, n_u))
		x_norm = np.zeros((n_x, horizon+1))
		alpha_norm = np.zeros((n_a, horizon+1))
		x_norm[:,0] = x_0.flatten()
		

		# Generating nominal traj
		for t in range(horizon):
			x_norm[:,t+1]=simulate(None, x_norm[:,t], u_nom[t]).flatten()
		
		alpha_norm = modes.T @ x_norm


		#Random Rollouts in control

		u_max = np.max(abs(u_nom))
		U_ = 0.2*u_max*np.random.normal(0, self.sigma, (self.n_samples, n_u*(horizon+1)))
		print(U_)

		delta_x = np.zeros((self.n_samples, n_x*(horizon+1)))
		delta_a = np.zeros((self.n_samples, n_a*(horizon+1)))

		X = np.zeros((n_x, horizon+1))
		alpha = np.zeros((n_a, horizon+1))
		
		ctrl = np.zeros((n_u, 1))

		# Generating delta_z for all rollouts
		for j in range(self.n_samples):
			X[:,0] = x_0.flatten()
			alpha[:,0]=modes.T @ X[:,0]
			
			for i in range(horizon):
				ctrl[:] = u_nom[i] + U_ [j, n_u*(horizon-i):n_u*(horizon-i+1)].reshape(np.shape(u_nom[i]))
				X[:, i+1] = simulate(None, X[:,i], ctrl).flatten()
				alpha[:,i+1]=modes.T @ X[:,i+1]

				delta_x[j,n_x*(horizon-i-1):n_x*(horizon-i)] = X[:,i+1] - x_norm[:,i+1]
				delta_a[j,n_a*(horizon-i-1):n_a*(horizon-i)] = alpha[:,i+1] - alpha_norm[:,i+1]
				print(delta_a[j,:])
				print(u_nom[i])
				print(U_ [j, n_u*(horizon-i):n_u*(horizon-i+1)].reshape(np.shape(u_nom[i])))
				sys.exit()


		fitcoef=np.zeros((n_x,n_x+n_u,horizon)); # M1 * fitcoef = delta_z
		fitcoef_alpha=np.zeros((n_a,n_a+n_u,horizon)); # M1 * fitcoef = delta_z

		
		for i in range(horizon):
			
			# M1 = np.hstack([delta_x[:, n_x*(horizon-i):n_x*(horizon-i+1)], U_[:,n_u*(horizon-i):n_u*(horizon-i+1)]])
			# delta = delta_x[:, n_x*(horizon-i-1):n_x*(horizon-i)]
			# mat, res, rank, S = np.linalg.lstsq(M1, delta, rcond=None)
			
			# fitcoef[:, :, i] =  mat.T
			# A_aug[i,:,:]= fitcoef[:,:n_x,i]
			# B_aug[i,:,:]= fitcoef[:,n_x:n_x+n_u,i]

			M1 = np.hstack([delta_a[:, n_a*(horizon-i):n_a*(horizon-i+1)], U_[:,n_u*(horizon-i):n_u*(horizon-i+1)]])
			delta = delta_a[:, n_a*(horizon-i-1):n_a*(horizon-i)]

			mat, res, rank, S = np.linalg.lstsq(M1, delta, rcond=None)
			
			fitcoef_alpha[:, :, i] =  mat.T
			A_aug[i,:,:]= fitcoef_alpha[:,:n_a,i]
			B_aug[i,:,:]= fitcoef_alpha[:,n_a:n_a+n_u,i]
			

		V_a_F_XU_XU = None


		# return F_ZU, V_x_F_XU_XU
		return A_aug, B_aug, V_a_F_XU_XU#, X


	def sys_id_FD(self, x_t, u_t, central_diff, activate_second_order=0, V_x_=None):

		'''
			system identification by a forward finite-difference for a given nominal state and control
			returns - a numpy array with F_x and F_u horizantally stacked
		'''
		################## defining local functions & variables for faster access ################

		simulate = self.simulate
		n_x = self.n_x
		n_u = self.n_u
		##########################################################################################
		
		XU = np.random.normal(0.0, self.sigma, (n_x, n_x + n_u))
		X_ = np.copy(XU[:, :n_x])
		U_ = np.copy(XU[:, n_x:])

		F_XU = np.zeros((n_x, n_x + n_u))
		V_x_F_XU_XU = None

		x_t_next = simulate(x_t.T, u_t.T)

		if central_diff:
			
			for i in range(0, n_x):
				for j in range(0, n_x):

					delta = np.zeros((1, n_x))
					delta[:, j] = XU[i, j]
					
					F_XU[i, j] = (simulate(x_t.T + delta, u_t.T)[:, i] - x_t_next[:, i])/XU[i, j]

			for i in range(0, n_x):
				for j in range(0, n_u):

					delta = np.zeros((1, n_u))
					delta[:, j] = XU[i, n_x + j]
					F_XU[i, n_x + j] = (simulate(x_t.T , u_t.T + delta)[:, i] - x_t_next[:, i])/XU[i, n_x + j]
					
		else:

			Y = (simulate((x_t.T) + X_, (u_t.T) + U_) - simulate((x_t.T), (u_t.T)))

			
	

		return F_XU, V_x_F_XU_XU	





	def simulate(self, X, U):
		
		'''
		Function to simulate a batch of inputs given a batch of control inputs and current states
		X - vector of states vertically stacked
		U - vector of controls vertically stacked
		'''
		################## defining local functions & variables for faster access ################

		#sim = self.sim
		forward_simulate = self.forward_simulate
		state_output = self.state_output

		##########################################################################################
		X_next = []

		# Augmenting X by adding a zero column corresponding to time
		X = np.hstack((np.zeros((X.shape[0], 1)), X))

		for i in range(X.shape[0]):

			X_next.append(state_output(forward_simulate(None, X[i] , U[i])))
			
		return np.asarray(X_next)[:,:,0]
	



	def vec2symm(self, ):
		pass




	def forward_simulate(self, sim, x, u):

		'''
			Function to simulate a single input and a single current state
			Note : The initial time is set to be zero. So, this can only be used for independent simulations
			x - append time (which is zero here due to above assumption) before state
		'''
		
		ctrl = np.zeros(self.n_u)
		ctrl[:] = u.flatten()
		xdim = int(math.sqrt(state_dimension))
		N = xdim


		# ctrl_nlse = np.zeros((int(self.n_x/2), 1))
		# N = self.n_x
		# dx = (2)/self.n_x
		# dt = 1e-4
		# nu = 0.1#0.01/np.pi
		# tperiod = 1000
		# return simulator.simulation(tperiod, nu, dt, ctrl[:int(len(ctrl)/2)], ctrl[int(len(ctrl)/2):], x[1:])
		
		##GSO ONLY##############################################
		ctrl_T = self.p_mask*ctrl[0]+self.m_mask*ctrl[1]
		ctrl_h = self.p_mask*ctrl[2]+self.m_mask*ctrl[3]
		# ctrl_nlse[0]=ctrl[0]
		# ctrl_nlse[-1] = ctrl[1]
		##GSO ONLY##############################################
		
		#self.sim.step(x[1:].reshape((sdim, sdim)), np.transpose(ctrl[0:int(len(ctrl)/2)]), np.transpose(ctrl[int(len(ctrl)/2):]), 25)
		#sim.render(mode='window')

		# return PFM.simulation(n_ac, 0.001, 0.001, ctrl_T, ctrl_h, x[1:].reshape((xdim, xdim))).reshape((xdim*xdim, 1))
		# return PFM.simulation(n_ac, 0.001, 0.001, ctrl_T, ctrl_h, x[1:].reshape((xdim, xdim))).reshape((xdim*xdim, 1))
		# return PFM.simulation(n_ac, 0.001, 0.001, ctrl[:int(len(ctrl)/2)].reshape((xdim, xdim)), ctrl[int(len(ctrl)/2):].reshape((xdim, xdim)), x[1:].reshape((xdim, xdim))).reshape((xdim*xdim, 1))
		return PFM_CH.simulation(n_ch, 0.001, dt_ch, ctrl_T, ctrl_h, x[1:].reshape((xdim, xdim))).reshape((xdim*xdim, 1))
		# return simulator.simulation(tperiod, nu, dt, ctrl[:int(len(ctrl)/2)], ctrl[int(len(ctrl)/2):], x[1:])
		# return PFM.simulation(n_ac, 0.001, 0.001, np.transpose(ctrl[0:int(len(ctrl)/2)]).reshape((sdim, sdim)), np.transpose(ctrl[int(len(ctrl)/2):]).reshape((sdim, sdim)), x[1:].reshape((sdim, sdim))).reshape((sdim*sdim, 1))#sim.get_state()
		# return PFM_CH.simulation(n_ch, 0.001, dt_ch, np.transpose(ctrl[0:int(len(ctrl)/2)]).reshape((sdim, sdim)), np.transpose(ctrl[int(len(ctrl)/2):]).reshape((sdim, sdim)), x[1:].reshape((sdim, sdim))).reshape((sdim*sdim, 1))#sim.get_state()
		#return PFM.simulation(25, 0.001, 0.001, np.zeros((sdim, sdim)), np.transpose(ctrl[:]).reshape((sdim, sdim)), x[1:].reshape((sdim, sdim))).reshape((sdim*sdim, 1))
		
		# x_cp = x[1:int(self.n_x/2)+1]+1j*x[1+int(self.n_x/2):]
		
		# return self.sim.simulate(0.05, x_cp.flatten(), ctrl_nlse).reshape((self.n_x,1))
		

	def traj_sys_id(self, x_nominal, u_nominal):	
		
		'''
			System identification for a nominal trajectory mentioned as a set of states
		'''
		
		Traj_jac = []
		
		for i in range(u_nominal.shape[0]):
			
			Traj_jac.append(self.sys_id(x_nominal[i], u_nominal[i]))

		return np.asarray(Traj_jac)
		

	
	def state_output(state):

		pass


	def khatri_rao(self, B, C):
	    """
	    Calculate the Khatri-Rao product of 2D matrices. Assumes blocks to
	    be the columns of both matrices.
	 
	    See
	    http://gmao.gsfc.nasa.gov/events/adjoint_workshop-8/present/Friday/Dance.pdf
	    for more details.
	 
	    Parameters
	    ----------
	    B : ndarray, shape = [n, p]
	    C : ndarray, shape = [m, p]
	 
	 
	    Returns
	    -------
	    A : ndarray, shape = [m * n, p]
	 
	    """
	    if B.ndim != 2 or C.ndim != 2:
	        raise ValueError("B and C must have 2 dimensions")
	 
	    n, p = B.shape
	    m, pC = C.shape
	 
	    if p != pC:
	        raise ValueError("B and C must have the same number of columns")
	 
	    return np.einsum('ij, kj -> ikj', B, C).reshape(m * n, p)

	def forward_simulate_arma(self, sim, x, u):

		'''
			Function to simulate a single input and a single current state
			Note : The initial time is set to be zero. So, this can only be used for independent simulations
			x - append time (which is zero here due to above assumption) before state
		'''
		
		ctrl = np.zeros(self.n_u)
		ctrl[:] = u.flatten()
		xdim = int(math.sqrt(state_dimension))
		N = xdim

		# N = self.n_x
		# # dx = (2*np.pi)/self.n_x
		# dt = 1e-4
		# nu = 0.1#0.01/np.pi
		# tperiod = 1000
		
		##GSO ONLY##############################################
		ctrl_T = self.p_mask*ctrl[0]+self.m_mask*ctrl[1]
		ctrl_h = self.p_mask*ctrl[2]+self.m_mask*ctrl[3]
		##GSO ONLY##############################################
		
		#self.sim.step(x[1:].reshape((sdim, sdim)), np.transpose(ctrl[0:int(len(ctrl)/2)]), np.transpose(ctrl[int(len(ctrl)/2):]), 25)
		#sim.render(mode='window')

		return PFM.simulation(n_ac, 0.001, 0.001, ctrl_T, ctrl_h, x.reshape((xdim, xdim))).reshape((xdim*xdim, 1))