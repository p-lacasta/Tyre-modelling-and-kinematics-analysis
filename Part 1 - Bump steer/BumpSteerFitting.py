#---------------------------------------------------------
# OXFORD BROOKES UNIVERSITY - Motorsport Engineering MSc
# Advanced Vehicle dynamics
#
# Coursework 1 - "Tyre modelling and kinematic analysis"
#--------------------------------------------------------- 

#---------------------------------------------------------
# BUMP STEER CURVE FITTING
#---------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

from CorneringStiffnessFitting import Fz_fit

z = np.array([
    -40,
    -30,
    -20,
    -10,
    0,
    10,
    20,
    30,
    40,
])

Toe = np.array([
    -0.939,
    -0.597,
    -0.318,
    -0.103,
    0.050,
    0.140,
    0.166,
    0.130,
    0.031
])

def bump_steer(z, a0, a1, a2):
    ToeAngle = a0 + a1*z + a2*z**2
    return ToeAngle

popt, pcov = opt.curve_fit(bump_steer, z, Toe)

z_fit = np.linspace(-40, 40, 100)
Toe_fit = bump_steer(z_fit, *popt)



Toe_pred = bump_steer(z, *popt)

residuals = Toe - Toe_pred

rmse = np.sqrt(np.mean(residuals**2))

leftover_spread = np.sum(residuals**2)
total_spread = np.sum((Toe - np.mean(Toe))**2)
r_squared = 1 - (leftover_spread / total_spread)

#---------------------------------------------------------
# Data plotting
#---------------------------------------------------------

fig, ax = plt.subplots()

ax.scatter(
    z,
    Toe,
    s=50, 
    facecolor="white", 
    edgecolors="blue", 
    alpha=0.7, 
    label="Dataset points",
    zorder=3,
)

ax.plot(
    z_fit, 
    Toe_fit, 
    color="red", 
    label="Fitted curve",
    zorder=2,
)

ax.set_title("Bump Steer Curve Fitting")
ax.set_xlabel("Vertical Displacement - z [mm]")
ax.set_ylabel("Toe Angle - Toe [deg]")
ax.legend()
ax.text(0.05, 0.95, f"RMSE: {rmse:.2f}\nR²: {r_squared:.4f}", transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

plt.show()

print("Fitted parameters:", popt)
