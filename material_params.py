'''
copyright @ Aayushman Sharma - aayushmansharma@tamu.edu

Date: July 27, 2025
'''
import numpy as np


state_dimension = 20*20#50*50#50*50#50*50#20*20#6*6
control_dimension = 2*2#state_dimension#2#2*20*20#2*50*50#200

# Cost parameters for nominal design
Q = 9*np.eye(state_dimension)#*state_dimension)
Q_final =9000*np.eye(state_dimension)#*state_dimension)
R = .05*np.eye(control_dimension)#2*control_dimension*control_dimension)
## Mujoco simulation parameters
# Number of substeps in simulation

horizon = 10#10#20#50#800
nominal_init_stddev = 0.1
n_substeps = 5

# Cost parameters for feedback design

W_x_LQR = 10*np.eye(state_dimension)#*state_dimension)
W_u_LQR = 2*np.eye(control_dimension)#*control_dimension)
W_x_LQR_f = 100*np.eye(state_dimension)#*state_dimension)


# D2C parameters
feedback_n_samples = 500#30#600#2800#3000#50#200#200#100#control_dimension+horizon+10#50#100#100#8000#2000


# Cahn-Hiliard params
dt_ch=0.5*1e-5#1e-5
n_ch=int(0.025/dt_ch)

n_ac=30

noise_std_test = 0.0

save_buffer_traj = np.zeros((horizon, state_dimension, 21))
save_buffer_control = np.zeros((horizon, control_dimension, 20))