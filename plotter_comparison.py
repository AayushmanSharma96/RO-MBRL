import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import (mark_inset, InsetPosition)

x_fo = np.load('AC_fo_50_2.npy')
x_ro = np.load('AC_ro_50_2.npy')
# x_fo = np.load('burgers_fo_2.npy')
# x_ro = np.load('burgers_ro_2.npy')
# x_fo_1 = np.load('CH_fo_20_2_fin.npy')
# x_ro_1 = np.load('CH_ro_20_2.npy')
eps = np.arange(x_fo.shape[0])

# x_fo_int = x_fo_1[-1]*np.ones((51,1))
# x_ro_int = x_ro_1[-1]*np.ones((51,1))

# x_ro_int[:31] = x_ro_1.reshape(np.shape(x_ro_int[:31]))
# x_fo_int[1:32] = x_fo_1.reshape(np.shape(x_ro_int[:31]))
# x_fo_int[0] = x_ro_1[0]-72000

# x_fo = x_fo_int
# x_ro = x_ro_int

fig, ax1 = plt.subplots()
ax1.plot(eps, x_fo, Linewidth=3, label='Standard ILQR')
ax1.plot(eps, x_ro, '--', Linewidth=3, label='Reduced Order ILQR')
ax1.set_ylabel('Total Cost', fontsize=14)
ax1.set_xlabel('Iteration', fontsize=14)
# ax1.set_yticks(np.arange(0,5*1e6, 1e6))
ax1.tick_params(axis='x', labelsize=14)
ax1.tick_params(axis='y', labelsize=14)
# ax1.set_title('1D Burgers Equation')
ax1.set_title('Allen-Cahn Equation')
# ax1.set_title('Cahn-Hilliard Equation', fontsize=14)
ax1.legend(loc=0, fontsize=14)

ax2 = plt.axes([0,0,1,1])
ip = InsetPosition(ax1, [0.4,0.2,0.5,0.5])
ax2.set_axes_locator(ip)
mark_inset(ax1, ax2, loc1=2, loc2=4, fc="none", ec='0.5')

ax2.plot(eps[49:], x_fo[49:], Linewidth=3, label='Standard ILQR')
ax2.plot(eps[49:], x_ro[49:], '--', Linewidth=3, label='Reduced Order ILQR')
# ax2.legend(loc=0)

# Some ad hoc tweaks.
# ax1.set_ylim(0,26)
ax2.yaxis.set_major_formatter(FormatStrFormatter('%g'))
ax2.set_yticks(np.arange(42000,52000,2000))
# ax2.set_yticks([1297.8,1297.82,1297.84])#np.arange(1297.8,1297.9,0.01))
# ax2.set_yticks(np.arange(9840,9880,10))
ax2.set_xticks([49,50])#np.arange(49,50,1))
# ax2.set_xticklabels(ax2.get_xticks(), backgroundcolor='w')
ax2.tick_params(axis='x', which='major', pad=8, labelsize=14)
ax2.tick_params(axis='y', labelsize=14)
plt.tight_layout()

plt.savefig("roilqr_AC_comp.png")
plt.savefig("roilqr_AC_comp.pdf")
plt.savefig("roilqr_AC_comp.tif")
plt.savefig("roilqr_AC_comp.eps")

# plt.savefig("roilqr_CH_comp.png")
# plt.savefig("roilqr_CH_comp.pdf")
# plt.savefig("roilqr_CH_comp.tif")
# plt.savefig("roilqr_CH_comp.eps")

# plt.savefig("roilqr_burgers_comp.png")
# plt.savefig("roilqr_burgers_comp.pdf")
# plt.savefig("roilqr_burgers_comp.tif")
# plt.savefig("roilqr_burgers_comp.eps")
plt.show()