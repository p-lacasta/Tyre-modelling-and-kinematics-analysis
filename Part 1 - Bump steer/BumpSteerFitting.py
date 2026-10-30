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
import os
import matplotlib.pyplot as plt
import scipy.optimize as opt
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import plot_style as ps

ps.apply_style()


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

fig, ax = ps.plot_curve_fit(
    z, Toe, z_fit, Toe_fit, rmse, r_squared,
    title="Bump steer: toe angle vs wheel travel",
    xlabel="Vertical wheel displacement, $z$ [mm]",
    ylabel="Toe angle [deg]",
    rmse_unit="deg",
    legend_loc="lower center",
)
ps.save_figure(fig, __file__, "BumpSteerFitting.png")

print("Fitted parameters:", popt)
