#---------------------------------------------------------
# OXFORD BROOKES UNIVERSITY - Motorsport Engineering MSc
# Advanced Vehicle dynamics
#
# Coursework 1 - "Tyre modelling and kinematic analysis"
#--------------------------------------------------------- 

#---------------------------------------------------------
# LATERAL FORCE CURVE FITTING
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

alpha_deg = np.array([  # [deg]
    0.0,
    1.5,
    3.0,
    4.5,
    6.0,
    7.5,
    9.0,
    10.5,
    12.0,
] )

Fy = np.array([ # [N]
    0.000,
    2562.915,
    3952.965,
    4116.338,
    4116.000,
    4064.562,
    3802.260,
    3877.155,
    3533.645,
] )

#---------------------------------------------------------
# Curve fitting
#---------------------------------------------------------

alpha_rad = np.deg2rad(alpha_deg) # Convert from deg to rad

def MF (alpha_rad, Pcy1, Pdy1, Pey1):
    Fz = 5000
    kya = 117683.4892 

    ay = np.tan(alpha_rad)
    Cy = Pcy1
    Dy = Fz * Pdy1
    By = kya / (Cy*Dy)
    Ey = Pey1

    Fy = Dy * np.sin(Cy*np.arctan(By*ay-np.arctan(Ey*(By*ay-np.arctan(By*ay)))))
    
    return Fy

popt, pcov = curve_fit(MF, alpha_rad, Fy, p0=[0.50,0.50,0.50])



alpha_fit_deg = np.linspace(0, 12, 100)
alpha_fit_rad = np.deg2rad(alpha_fit_deg)
Fy_fit = MF(alpha_fit_rad, *popt)


#---------------------------------------------------------
# Evaluation parameters
#---------------------------------------------------------

# Prediction at measured points
Fy_pred = MF(alpha_rad, *popt)


# Residuals
residuals = Fy - Fy_pred

# RMSE
rmse = np.sqrt(np.mean(residuals**2))

# R-Squared
leftover_spread = np.sum(residuals**2)
total_spread = np.sum((Fy - np.mean(Fy))**2)
r_squared = 1-(leftover_spread / total_spread)

#---------------------------------------------------------
# Data plotting
#---------------------------------------------------------

fig, ax = ps.plot_curve_fit(
    alpha_deg, Fy, alpha_fit_deg, Fy_fit, rmse, r_squared,
    title="Lateral force vs slip angle",
    xlabel=r"Slip angle, $\alpha$ [deg]",
    ylabel="Lateral force, $F_y$ [N]",
    rmse_unit="N",
)
ps.save_figure(fig, __file__, "LateralForce.png")

print("Fitted parameters:", popt)