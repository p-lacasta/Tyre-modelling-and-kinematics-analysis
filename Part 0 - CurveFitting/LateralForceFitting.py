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
from scipy.optimize import curve_fit
import numpy as np

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

fig, ax = plt.subplots()

ax.scatter(
    alpha_deg,
    Fy,
    s=50, 
    facecolor="white", 
    edgecolors="blue", 
    alpha=0.7, 
    label="Dataset points",
    zorder=3,
)

ax.plot(
    alpha_fit_deg, 
    Fy_fit, 
    color="red", 
    label="Fitted curve",
    zorder=2,
)

ax.set_title("Lateral Force Curve Fitting")
ax.set_xlabel("Slip Angle - α [deg]")
ax.set_ylabel("Lateral Force - Fy [N]")
ax.text(0.05, 0.95, f"RMSE: {rmse:.2f}\nR²: {r_squared:.4f}", transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

ax.legend()

plt.show()

print("Fitted parameters:", popt)