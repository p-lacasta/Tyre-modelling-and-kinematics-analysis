#---------------------------------------------------------
# OXFORD BROOKES UNIVERSITY - Motorsport Engineering MSc
# Advanced Vehicle dynamics
#
# Coursework 1 - "Tyre modelling and kinematic analysis"
#--------------------------------------------------------- 

#---------------------------------------------------------
# CORNERING STIFFNESS CURVE FITTING
#---------------------------------------------------------

import matplotlib.pyplot as plt
import math
import os
from scipy.optimize import curve_fit
import numpy as np
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import plot_style as ps

ps.apply_style()

#---------------------------------------------------------
# Input data
#---------------------------------------------------------

Fz = np.array([  # [N]
    0, 
    1000, 
    2000, 
    3000, 
    4000, 
    5000, 
    6000, 
    7000, 
    8000
]) 

CFa_deg = np.array([  # [N/deg]
    0.000, 
    487.352, 
    1122.378, 
    1461.080, 
    1864.081, 
    2062.300, 
    2176.319,
    2293.333,
    2258.657,
])

CFa_rad = CFa_deg * (180 / np.pi) # Convert from deg to rad

#---------------------------------------------------------
# Curve fitting
#---------------------------------------------------------

def MF_simplified (Fz, Pky1, Pky2):
    Fz0 = 5000 
    Kyalpha = Fz0*Pky1*np.sin(2*np.arctan(Fz/(Pky2*Fz0)))
    return Kyalpha

popt, pcov = curve_fit(MF_simplified, Fz, CFa_rad, p0=[10.00,1.00])


Fz_fit = np.linspace(0, 8000, 100)
CFa_fit_rad = MF_simplified(Fz_fit, *popt)
CFa_fit_deg = CFa_fit_rad * (np.pi / 180)  # Convert from rad to deg


#---------------------------------------------------------
# Evaluation parameters
#---------------------------------------------------------

# Prediction at measured points
CFa_pred_deg = MF_simplified(Fz, *popt) * (np.pi / 180)


# Residuals
residuals = CFa_deg - CFa_pred_deg

# RMSE
rmse = np.sqrt(np.mean(residuals**2))

# R-Squared
leftover_spread = np.sum(residuals**2)
total_spread = np.sum((CFa_deg - np.mean(CFa_deg))**2)
r_squared = 1-(leftover_spread / total_spread)


#---------------------------------------------------------
# Data plotting
#---------------------------------------------------------

fig, ax = ps.plot_curve_fit(
    Fz, CFa_deg, Fz_fit, CFa_fit_deg, rmse, r_squared,
    title="Cornering stiffness vs vertical load",
    xlabel="Vertical load, $F_z$ [N]",
    ylabel=r"Cornering stiffness, $C_{F\alpha}$ [N/deg]",
    rmse_unit="N/deg",
)
ps.save_figure(fig, __file__, "CorneringStiffness.png")

print("Fitted parameters:", popt)