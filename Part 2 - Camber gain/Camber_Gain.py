import matplotlib.pyplot as plt
import numpy as np

## --- INPUT DATA ---

IB_LW = np.array([-350,180])
OB_LW = np.array([-730, 160])
IB_UP = np.array([-380, 428])
OB_UP = np.array([-714, 440])

WHEEL_CENTRE = np.array([-775, 300])
CONTACT_PATCH_CENTRE = np.array([-775, 0])
WHEEL_TOP = np.array([-775, 600])

WHEEL_VERTICAL_DISPLACEMENT = 15 #mm
BODY_ROLL_ANGLE = 1.8 #deg

VEHICLE_TRACK = abs(WHEEL_CENTRE[0]*2)

# --- WHEEL GEOMETRY PLOT ---

def plot_wheel_geometry(ax1):
    ax1.plot(*IB_LW, 'ko')
    ax1.plot(*OB_LW, 'ko')
    ax1.plot(*IB_UP, 'ko')
    ax1.plot(*OB_UP, 'ko')

    ax1.plot(*WHEEL_CENTRE, 'bo')
    ax1.plot(*CONTACT_PATCH_CENTRE, 'bo')

    ax1.plot([IB_LW[0], OB_LW[0]], [IB_LW[1], OB_LW[1]], 'k-', label="Control arms")
    ax1.plot([IB_UP[0], OB_UP[0]], [IB_UP[1], OB_UP[1]], 'k-')
    ax1.plot([CONTACT_PATCH_CENTRE[0], WHEEL_TOP[0]], [CONTACT_PATCH_CENTRE[1], WHEEL_TOP[1]], 'b--', label='Wheel Centreline')
    ax1.set_xlabel("Y [mm]")
    ax1.set_ylabel("Z [mm]")
    ax1.legend()


fig1, ax1 = plt.subplots()
plot_wheel_geometry(ax1)
ax1.set_title("Wheel Geometry")

# --- INSTANT CENTRE PLOT ---

d_low = OB_LW - IB_LW
d_up = OB_UP - IB_UP

A = np.column_stack((d_low, -d_up))
b = OB_UP - IB_LW
t, s = np.linalg.solve(A, b)
IC = IB_LW + t*d_low

virtual_swing_arm_length = np.linalg.norm(IC - WHEEL_CENTRE)

fig2, ax2 = plt.subplots()
plot_wheel_geometry(ax2)
ax2.plot(*IC, 'ro', label="Instant Centre")
ax2.plot([IB_LW[0], IC[0]], [IB_LW[1], IC[1]], 'r--')
ax2.plot([IB_UP[0], IC[0]], [IB_UP[1], IC[1]], 'r--')
ax2.plot([WHEEL_CENTRE[0], IC[0]], [WHEEL_CENTRE[1], IC[1]], 'g--', label="Virtual Swing Arm")
ax2.set_title("Instant Centre Location")
ax2.legend()

print("Instant Centre Location: ", IC)
print("Virtual Swing Arm Length: ", virtual_swing_arm_length)

# --- ROLL CENTRE PLOT ---
d_roll = IC - CONTACT_PATCH_CENTRE
d_centre = np.array([0, 1])  # vehicle centreline direction (Y = 0)

A = np.column_stack((d_roll, -d_centre))
b = np.array([0, 0]) - CONTACT_PATCH_CENTRE
t, s = np.linalg.solve(A, b)
RC = CONTACT_PATCH_CENTRE + t*d_roll

fig3, ax3 = plt.subplots()
plot_wheel_geometry(ax3)
ax3.plot(*IC, 'ro', label="Instant Centre")
ax3.plot([CONTACT_PATCH_CENTRE[0], IC[0]], [CONTACT_PATCH_CENTRE[1], IC[1]], 'y--')
ax3.axvline(0, color="gray", linestyle="--", label="Vehicle Centreline")
ax3.plot(*RC, 'o', color="yellow", label="Roll Centre")
ax3.set_title("Roll Centre Location")
ax3.legend()

print("Roll Centre Location: ", RC)

# --- HEAVE CAMBER GAIN ---

heave_camber_gain = -(1/virtual_swing_arm_length)*WHEEL_VERTICAL_DISPLACEMENT
heave_camber_gain_DegPer10mm = np.degrees(heave_camber_gain)/WHEEL_VERTICAL_DISPLACEMENT*10   #camber gain in degrees per 10mm of wheel travel
print("Heave Camber Gain (deg/10mm): ", heave_camber_gain_DegPer10mm)

# --- ROLL CAMBER GAIN ---

roll_camber_gain = np.deg2rad(BODY_ROLL_ANGLE) - VEHICLE_TRACK/(2*virtual_swing_arm_length)*(np.deg2rad(BODY_ROLL_ANGLE))
roll_camber_gain_DegPerDeg = np.degrees(roll_camber_gain)/BODY_ROLL_ANGLE    #camber gain in degrees per degree of body roll
print("Roll Camber Change (deg): ", np.degrees(roll_camber_gain))
print("Roll Camber Gain (deg/deg roll): ", roll_camber_gain_DegPerDeg)

# --- TOTAL CAMBER GAIN --

total_camber_outside = np.degrees(roll_camber_gain) + np.degrees(heave_camber_gain)
total_camber_inside = np.degrees(roll_camber_gain) - np.degrees(heave_camber_gain)

print("Total Camber outside: ", total_camber_outside)
print("Total Camber inside: ", total_camber_inside)

# --- CAMBER RECOVERY RATIO ---

cr = VEHICLE_TRACK/(2*virtual_swing_arm_length)*100
print("Camber Recovery (%): ", cr)

# --- ROLL CENTRE MIGRATION ---

body_roll_rad = np.deg2rad(BODY_ROLL_ANGLE)

# Inboard points rotation
r = np.array([[np.cos(body_roll_rad), -np.sin(body_roll_rad)],
              [np.sin(body_roll_rad), np.cos(body_roll_rad)]])

IB_LW_R = r @ IB_LW + np.array([0, WHEEL_VERTICAL_DISPLACEMENT])
IB_UP_R = r @ IB_UP + np.array([0, WHEEL_VERTICAL_DISPLACEMENT])

# Outboard points rotation
r = np.array([[np.cos(roll_camber_gain), -np.sin(roll_camber_gain)],
              [np.sin(roll_camber_gain), np.cos(roll_camber_gain)]])

OB_LW_R = np.array([OB_LW[0], OB_LW[1]+WHEEL_VERTICAL_DISPLACEMENT+(VEHICLE_TRACK/2)*np.sin(body_roll_rad)])
OB_UP_R = OB_LW_R + r @ (OB_UP - OB_LW)

print("New Inboard Lower Point:", IB_LW_R)
print("New Inboard Upper Point:", IB_UP_R)
print("New Outboard Lower Point:", OB_LW_R)
print("New Outboard Upper Point:", OB_UP_R)

# --- NEW GEOEMTRY PLOT ---

def plot_new_wheel_geometry(ax1):
    ax1.plot(*IB_LW_R, 'ko')
    ax1.plot(*OB_LW_R, 'ko')
    ax1.plot(*IB_UP_R, 'ko')
    ax1.plot(*OB_UP_R, 'ko')

    ax1.plot(*CONTACT_PATCH_CENTRE, 'bo')

    ax1.plot([IB_LW_R[0], OB_LW_R[0]], [IB_LW_R[1], OB_LW_R[1]], 'k-', label="Control arms")
    ax1.plot([IB_UP_R[0], OB_UP_R[0]], [IB_UP_R[1], OB_UP_R[1]], 'k-')
    ax1.set_xlabel("Y [mm]")
    ax1.set_ylabel("Z [mm]")
    ax1.legend()

fig4, ax4 = plt.subplots()
plot_new_wheel_geometry(ax4)

# --- NEW INSTANT CENTRE PLOT ---

d_low = OB_LW_R - IB_LW_R
d_up = OB_UP_R - IB_UP_R

A = np.column_stack((d_low, -d_up))
b = OB_UP_R - IB_LW_R
t, s = np.linalg.solve(A, b)
IC_R = IB_LW_R + t*d_low

fig5, ax5 = plt.subplots()
plot_new_wheel_geometry(ax5)
ax5.plot(*IC_R, 'ro', label="Instant Centre")
ax5.plot([IB_LW_R[0], IC_R[0]], [IB_LW_R[1], IC_R[1]], 'r--')
ax5.plot([IB_UP_R[0], IC_R[0]], [IB_UP_R[1], IC_R[1]], 'r--')
ax5.set_title("Instant Centre Location")
ax5.legend()

print("New Instant Centre Location: ", IC_R)

# --- NEW ROLL CENTRE PLOT ---
d_roll = IC_R - CONTACT_PATCH_CENTRE
d_centre = np.array([0, 1])  # vehicle centreline direction (Y = 0)

A = np.column_stack((d_roll, -d_centre))
b = np.array([0, 0]) - CONTACT_PATCH_CENTRE
t, s = np.linalg.solve(A, b)
RC_R = CONTACT_PATCH_CENTRE + t*d_roll

fig6, ax6 = plt.subplots()
plot_new_wheel_geometry(ax6)
ax6.plot(*IC_R, 'ro', label="Instant Centre")
ax6.plot([CONTACT_PATCH_CENTRE[0], IC_R[0]], [CONTACT_PATCH_CENTRE[1], IC_R[1]], 'y--')
ax6.axvline(0, color="gray", linestyle="--", label="Vehicle Centreline")
ax6.plot(*RC_R, 'o', color="yellow", label="Roll Centre")
ax6.set_title("Roll Centre Location")
ax6.legend()

print("New Roll Centre Location: ", RC_R)
print("Roll Centre Migration: ", RC[1] - RC_R[1])

# --- ORIGINAL vs NEW GEOMETRY / ROLL CENTRE MIGRATION PLOT ---

def plot_rc_migration(ax7):
    # Original geometry
    ax7.plot([IB_LW[0], OB_LW[0]], [IB_LW[1], OB_LW[1]], 'k-', label="Original control arms")
    ax7.plot([IB_UP[0], OB_UP[0]], [IB_UP[1], OB_UP[1]], 'k-')
    ax7.plot(*np.array([IB_LW, OB_LW, IB_UP, OB_UP]).T, 'ko')
    ax7.plot([IB_LW[0], IC[0]], [IB_LW[1], IC[1]], 'k:', linewidth=0.8)
    ax7.plot([IB_UP[0], IC[0]], [IB_UP[1], IC[1]], 'k:', linewidth=0.8)
    ax7.plot([CONTACT_PATCH_CENTRE[0], RC[0]], [CONTACT_PATCH_CENTRE[1], RC[1]], 'k--', linewidth=0.8)
    ax7.plot(*IC, 'ks', label="Original instant centre")
    ax7.plot(*RC, 'k^', markersize=9, label="Original roll centre")

    # New (rolled + heaved) geometry
    ax7.plot([IB_LW_R[0], OB_LW_R[0]], [IB_LW_R[1], OB_LW_R[1]], 'r-', label="New control arms")
    ax7.plot([IB_UP_R[0], OB_UP_R[0]], [IB_UP_R[1], OB_UP_R[1]], 'r-')
    ax7.plot(*np.array([IB_LW_R, OB_LW_R, IB_UP_R, OB_UP_R]).T, 'ro')
    ax7.plot([IB_LW_R[0], IC_R[0]], [IB_LW_R[1], IC_R[1]], 'r:', linewidth=0.8)
    ax7.plot([IB_UP_R[0], IC_R[0]], [IB_UP_R[1], IC_R[1]], 'r:', linewidth=0.8)
    ax7.plot([CONTACT_PATCH_CENTRE[0], RC_R[0]], [CONTACT_PATCH_CENTRE[1], RC_R[1]], 'r--', linewidth=0.8)
    ax7.plot(*IC_R, 'rs', label="New instant centre")
    ax7.plot(*RC_R, 'r^', markersize=9, label="New roll centre")

    # Wheel and centreline
    ax7.plot(*CONTACT_PATCH_CENTRE, 'bo')
    ax7.plot([CONTACT_PATCH_CENTRE[0], WHEEL_TOP[0]], [CONTACT_PATCH_CENTRE[1], WHEEL_TOP[1]], 'b--', label='Wheel Centreline')
    ax7.axvline(0, color="gray", linestyle="--", label="Vehicle Centreline")

    # Roll centre migration arrow
    ax7.annotate("", xy=RC_R, xytext=RC, arrowprops=dict(arrowstyle="->", color="green", linewidth=2))
    ax7.annotate(f"RC migration: {RC_R[1] - RC[1]:+.1f} mm", xy=RC_R, xytext=(15, 0),
                 textcoords="offset points", color="green", va="center")

    ax7.set_xlabel("Y [mm]")
    ax7.set_ylabel("Z [mm]")
    ax7.grid(True, alpha=0.3)

fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(14, 6))
plot_rc_migration(ax7a)
ax7a.set_title("Full view (incl. instant centres)")
fig7.legend(*ax7a.get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=9)

plot_rc_migration(ax7b)
ax7b.set_xlim(-850, 450)
ax7b.set_ylim(-50, 650)
ax7b.set_aspect('equal', adjustable='box')
ax7b.set_title("Zoom: suspension and roll centre")

fig7.suptitle(f"Original vs New Geometry - Roll Centre Migration ({BODY_ROLL_ANGLE}° roll, {WHEEL_VERTICAL_DISPLACEMENT} mm heave)")
fig7.tight_layout(rect=(0, 0.1, 1, 1))

plt.show()