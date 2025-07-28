import numpy as np
import PFM
import PFM_CH
import math
import burgers as sim
from progressbar import *
import modred as mr
from material_params import dt_ch, n_ch, n_ac

x_nom = np.load('X_nominal.npy')
x = np.zeros((50*50, 11))
# x[:,0] = x_init.flatten()
x[:,1:] = x_nom.T

# # Compute POD
num_modes = 10
POD_res = mr.compute_POD_arrays_snaps_method(
    x, list(mr.range(num_modes)))

modes = POD_res.modes
proj = POD_res.proj_coeffs[:num_modes,:]
eigvals = POD_res.eigvals
# sum_eigvals=np.sum(eigvals)
# rel_energy=0

x1 = x+0.01*np.random.normal(np.zeros(np.shape(x)))
print(np.shape(x1))
POD_res2 = mr.compute_POD_arrays_snaps_method(
    x1, list(mr.range(num_modes)))

modes2 = POD_res2.modes
proj2 = POD_res2.proj_coeffs[:num_modes,:]
eigvals2 = POD_res2.eigvals

del_x = x1-x
del_alpha = modes.T @ del_x
alpha_nom = modes.T @ x
del_phi = modes2-modes

f1 = modes @ del_alpha
f2 = del_phi @ alpha_nom
f3 = del_phi @ del_alpha

print('del_x = ',del_x)
print('from approx = ',modes2 @ proj2-modes@ proj)
# print('f1 = ', f1)
# print('f2 = ', f2)
# print('f3 = ', f3)
# print('fnet = ', f1+f2+f3)