'''
copyright @ Aayushman Sharma - aayushmansharma@tamu.edu

Date: July 27, 2025
'''
#!/usr/bin/env python

import numpy as np
from model_free_DDP import DDP
import time
from ltv_sys_id import ltv_sys_id_class
from material_params import *
import os
import math
import json
import numpy as np
import matplotlib.pyplot as plt
import math
import sys


class model_free_material_DDP(DDP, ltv_sys_id_class):
	
	def __init__(self, initial_state, final_state, MODEL_XML, alpha, horizon, state_dimension, control_dimension, Q, Q_final, R):
		
		'''
			Declare the matrices associated with the cost function
		'''
		self.Q = Q
		self.Q_final = Q_final
		self.R = R


		DDP.__init__(self, None, state_dimension, control_dimension, alpha, horizon, initial_state, final_state)
		ltv_sys_id_class.__init__(self, None, state_dimension, control_dimension, n_substeps, n_samples=feedback_n_samples)

	def state_output(self, state):
		'''
			Given a state in terms of Mujoco's MjSimState format, extract the state as a numpy array, after some preprocessing
		'''
		
		return state


	def cost(self, x, u):
		'''
			Incremental cost in terms of state and controls
		'''
		return (((x - self.X_g).T @ self.Q) @ (x - self.X_g)) + (((u.T) @ self.R) @ u)
	
	
	def cost_final(self, x):
		'''
			Cost in terms of state at the terminal time-step
		'''
		return (((x - self.X_g).T @ self.Q_final) @ (x - self.X_g))

	def initialize_traj(self, path=None, init=None):#Add check for replan, and add an input for U_0 as initial guess
		'''
		Initial guess for the nominal trajectory by default is produced by zero controls
		'''

		

		
		if path is None:
			
			for t in range(0, self.N):
			
				if init is None:
					self.U_p[t, :] = np.random.normal(0, nominal_init_stddev, (self.n_u, 1))	#np.random.normal(0, 0.01, self.n_u).reshape(self.n_u,1)#DM(array[t, 4:6])
				else:
					self.U_p[t, :] = init[t,:]

			# np.save('U_INIT_TEST.npy', self.U_p)
			self.U_p = np.load('U_INIT_TEST.npy')	
			np.copyto(self.U_p_temp, self.U_p)
			
			self.forward_pass_sim()
			self.pod(self.X_p_0, self.X_p, it=-1) 
			
			np.copyto(self.X_p_temp, self.X_p)

		else:

			array = np.loadtxt('../rrt_path.txt')
			### INCOMPLETE

	def replan(self, init_state, t1, horizon, ver, cost):

		path_to_model_free_DDP = "/home/karthikeya/D2C-2.0"
		MODEL_XML = path_to_model_free_DDP + "/models/fish_old.xml"
		path_to_exp = path_to_model_free_DDP + "/experiments/Burgers/exp_1"

		path_to_file = path_to_exp + "/material_policy_"+str(ver-1)+".txt"
		path_to_file_re = path_to_exp + "/material_policy_"+str(ver)+".txt"
		training_cost_data_file = path_to_exp + "/training_cost_data.txt"
		path_to_data = path_to_exp + "/material_D2C_DDP_data.txt"

		print(path_to_file)
		
		n_iterations=5
		alpha = .7
		U_p_rep = np.zeros((horizon, self.n_u, 1))
		xdim = int(math.sqrt(self.n_x))
		check_flag=True
		if horizon==2:
			check_flag=False

		with open(path_to_file) as f:

				Pi = json.load(f)

		for i in range(0, horizon):
			U_p_rep[i, :] = (np.array(Pi['U'][str(i)])).flatten().reshape(np.shape(U_p_rep[i, :]))


		
		
		updated_model = model_free_material_DDP(init_state, self.X_g, MODEL_XML, alpha, horizon-1, self.n_x, self.n_u, Q, Q_final, R)
		updated_model.iterate_ddp(n_iterations, finite_difference_gradients_flag=False, u_init=None)#U_p_rep[t1:, :])
		updated_model.save_policy(path_to_file_re)

		testCL = updated_model.test_episode(1, path=path_to_file_re, noise_stddev=noise_std_test, check_plan = check_flag, init=init_state, version=ver, cst=cost)
		if check_flag==False:
			print("Ver + "+str(ver)+" Replanned Goal diff = "+str(np.linalg.norm(testCL-final_state)))
			print("Final state = ", testCL.reshape((xdim,xdim)))
			


if __name__=="__main__":

	# Path of the model file
	path_to_model_free_DDP = "/home/karthikeya/D2C-2.0"
	MODEL_XML = path_to_model_free_DDP + "/models/fish_old.xml"
	path_to_exp = path_to_model_free_DDP + "/experiments/Burgers/exp_1"

	path_to_file = path_to_exp + "/material_policy_0.txt"
	training_cost_data_file = path_to_exp + "/training_cost_data.txt"
	path_to_data = path_to_exp + "/material_D2C_DDP_data.txt"


	# Declare other parameters associated with the problem statement
	
	# alpha is the line search parameter during forward pass
	alpha = .3

	# Declare the initial state and the final state in the problem

	sdim = int(math.sqrt(state_dimension))
	cdim = int(math.sqrt(control_dimension/2))
	# initial_state = np.zeros((sdim, sdim))
	# initial_state = initial_state.reshape((state_dimension, 1))
	initial_state = np.round(np.sin(np.linspace(-np.pi, np.pi, state_dimension)),2).reshape(state_dimension, 1)#np.ones((state_dimension, 1))#np.ones((state_dimension, 1))
	
	final_state = -0.5*np.ones((state_dimension, 1))#np.load('relaxed_out_ch.npy')#np.ones((sdim,sdim))#np.loadtxt('Final.txt')#0.9999*np.ones([sdim, sdim])#.reshape((state_dimension*state_dimension, ))
	# final_state[int(state_dimension/2):]*=-1

	final_state = final_state.reshape((state_dimension, 1))

	# print('Initial phase : \n', initial_state)
	# print('Goal phase : \n', final_state)
	# No. of ILQR iterations to run
	n_iterations =50#20#5#40

	# Initiate the above class that contains objects specific to this problem
	model = model_free_material_DDP(initial_state, final_state, MODEL_XML, alpha, horizon, state_dimension, control_dimension, Q, Q_final, R)


	# ---------------------------------------------------------------------------------------------------------
	# -----------------------------------------------Training---------------------------------------------------
	# ---------------------------------------------------------------------------------------------------------
	# Train the policy

	training_flag_on = True

	if training_flag_on:

		with open(path_to_data, 'w') as f:

			f.write("D2C training performed for a micro-structure control task:\n\n")

			f.write("System details : {}\n".format(os.uname().sysname + "--" + os.uname().nodename + "--" + os.uname().release + "--" + os.uname().version + "--" + os.uname().machine))
			f.write("-------------------------------------------------------------\n")

		time_1 = time.time()

		# Run the DDP algorithm
		# To run using our LLS-CD jacobian estimation (faster), make 'finite_difference_gradients_flag = False'
		# To run using forward difference for jacobian estimation (slower), make 'finite_difference_gradients_flag = True'

		model.iterate_ddp(n_iterations, finite_difference_gradients_flag=False)
		
		time_2 = time.time()

		D2C_algorithm_run_time = time_2 - time_1

		print("D2C-2 algorithm run time taken: ", D2C_algorithm_run_time)

		# Save the history of episodic costs 
		with open(training_cost_data_file, 'w') as f:
			for cost in model.episodic_cost_history:
				f.write("%s\n" % cost)

		# Test the obtained policy
		model.save_policy(path_to_file)

		with open(path_to_data, 'a') as f:

				f.write("\nTotal time taken: {}\n".format(D2C_algorithm_run_time))
				f.write("------------------------------------------------------------------------------------------------------------------------------------\n")

		# Display the final state in the deterministic policy on a noiseless system
		# print(model.X_p[-1])
		print('l2 = ', np.linalg.norm(final_state-model.X_p[-1],2))
		print('t = ', model.N)
		print('Relative MSE =',100*np.linalg.norm(final_state-model.X_p[-1],2)/np.linalg.norm(final_state,2))
		print('Final cost = ', model.episodic_cost_history[-1])
		# np.save('roilqr_burgers.npy',model.X_p)
		np.save('burgers_fo_2.npy', model.episodic_cost_history)
		sys.exit()

		# Plot the episodic cost during the training
		model.plot_episodic_cost_history(save_to_path=path_to_exp+"/episodic_cost_training.png")
		plt.figure()
		plt.plot(np.linspace(-np.pi, np.pi, state_dimension), initial_state, 'b')
		# plt.plot(np.linspace(-1, 1, state_dimension), model.X_p[5,:], '--r')
		plt.plot(np.linspace(-np.pi, np.pi, state_dimension), model.X_p[-1], 'g')
		plt.legend(['t_0', 't_f'])
		plt.xlabel('X')
		plt.ylabel('Velocity')
		np.save('Vel_profile.npy', model.X_p)
		# plt.show()

		plt.figure()
		plt.plot(np.arange(horizon), model.U_p[:,0], 'b')
		plt.plot(np.arange(horizon), model.U_p[:,1], 'r')
		# plt.plot(np.linspace(-1, 1, 10), model.X_p[-1], 'g')
		plt.legend(['U_1', 'U_2'])
		plt.xlabel('t')
		plt.ylabel('Control Inputs')
		plt.show()


		state_history_nominal=model.X_p[-1]
		
		# episodic_cost_data = np.zeros((np.shape(model.episodic_cost_history)[0], 20))
		# l2_data = np.zeros((20,1))
		# for i in range(20):
		# 	model = model_free_material_DDP(initial_state, final_state, MODEL_XML, alpha, horizon, state_dimension, control_dimension, Q, Q_final, R)
		# 	print('Iteration number  = ', i+1)
		# 	model.iterate_ddp(n_iterations, finite_difference_gradients_flag=False)
		# 	episodic_cost_data[:,i] = np.asarray(model.episodic_cost_history)
		# 	l2_data[i] = np.linalg.norm(final_state-model.X_p[-1],2)
		# np.save('burgers_episodic_cost.npy', episodic_cost_data)
		# np.save('burgers_l2.npy', l2_data)
	


	


	# ---------------------------------------------------------------------------------------------------------
	# -----------------------------------------------Testing---------------------------------------------------
	# ---------------------------------------------------------------------------------------------------------
	# Test the obtained policy

	test_flag_on = False
	#np.random.seed(1)

	if test_flag_on:

		f = open(path_to_exp + "/material_testing_data.txt", "a")

		def frange(start, stop, step):
			i = start
			while i < stop:
				yield i
				i += step
		
		u_max = 3.5

		try:

			for i in frange(0.0, 1.02, 0.02):

				print(i)
				print("\n")
				terminal_mse = 0
				Var_terminal_mse = 0
				n_samples = 100

				for j in range(n_samples):	

					terminal_state = model.test_episode(render=0, path=path_to_file, noise_stddev=i*u_max)
					terminal_mse += np.linalg.norm(terminal_state[0:3] - final_state[0:3], axis=0)
					Var_terminal_mse += (np.linalg.norm(terminal_state[0:3] - final_state[0:3], axis=0))**2

				terminal_mse_avg = terminal_mse/n_samples
				Var_terminal_mse_avg = (1/(n_samples*(n_samples-1)))*(n_samples*Var_terminal_mse - terminal_mse**2)

				std_dev_mse = np.sqrt(Var_terminal_mse_avg)

				f.write(str(i)+",\t"+str(terminal_mse_avg[0])+",\t"+str(std_dev_mse[0])+"\n")
		except:

			print("Testing failed!")
			f.close()

		f.close()

	xdim=int(math.sqrt(state_dimension))

	

	# end_disc = np.array([])
	# np.save('end_disc.npy',end_disc)
	# for i in range(1):
	# 	testCL = model.test_episode(1, path=path_to_file, noise_stddev=noise_std_test, check_plan=False, init=initial_state)
	# 	print("Goal diff = "+str(np.linalg.norm(testCL-final_state)))
	# AA=np.load('end_disc.npy')
	# np.save('end_disc.npy',np.append(AA, np.linalg.norm(testCL-final_state)))
	# # plt.matshow(testCL.reshape((sdim, sdim)))
	#plt.show()
	# for i in range(1):
	# 	testCL2 = model.test_episode(1, path=path_to_file, noise_stddev=noise_std_test, check_plan=True, init=initial_state)
	
	