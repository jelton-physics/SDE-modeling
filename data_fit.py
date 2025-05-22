import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta


path = "C:\\Users\\jelto\\OneDrive\\Documents\\Research_Projects\\stock_suite\\"

xls = pd.ExcelFile(path + "Stock_Tracker.xlsx")
df = pd.read_excel(xls, 'Peaks')



# c_1: climb time for peak 1

# r_1: relaxation time for peak 1

# w_1: peak width 1

# p_1_perc_rise: percent rise from initial bottom to top for peak 1

# p_1_perc_fall: percent fall from top to final bottom for peak 1

# t_p: time between peak maxes

# del_t: time from final bottom of peak 1 to initial bottom of peak 2

# p_12_perc_fall: percent fall from max of peak 1 to max of peak 2 (could be a rise)




#### params for Green's function damped forced oscilltor and linear trend

# Natural oscillation period w_0 can be read off from chart. gam is the decay rate of forcing function spike, beta is the damping rate for oscillations
# F_0 controls how strong of an impulse

w_0 = 1.0
beta = 0.3*w_0
gam = 0.1*w_0
m = 1.0
F_0 = 1.0
b = beta*2*m
w_1 = np.sqrt(w_0*w_0 - beta*beta)
t_0 = 0.0

t_f = 50.0
t_step_max = 0.01
use_equal_times = False
tfirst = None # first time for showing solution curve on plot
t_eval = np.arange(t_0, t_f, t_step_max)  # If using equally spaced predetermined times

x_single = (b/w_1)*np.exp(-beta*(t_eval - t_0))*np.sin(w_1*(t_eval - t_0))

coeff = (F_0/m)/((gam - beta)*(gam - beta) + w_1*w_1)

x_decay = coeff*(np.exp(-gam*t_eval) - np.exp(-beta*t_eval)*(np.cos(w_1*t_eval) - ((gam-beta)/w_1)*np.sin(w_1*t_eval)))


# Time series plot
plt.plot(t_eval, x_decay, linestyle = 'solid')
plt.xlim((0,50))
plt.ylim((-0.2,1.5))
plt.title(f'Time Series Plot')
plt.xlabel('time')
plt.ylabel('x')
# plt.xticks(xticks) 
# plt.xticks(rotation='vertical')
# plt.grid()
# plt.tight_layout()
# cursor(hover=True)
plt.show()



zzz = 1