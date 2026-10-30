"""
ENGR7015 CW1 Part 2 - Camber gain, front SLA suspension (2-D front view, y-z plane)

Conventions
-----------
* ISO 8855, body-fixed frame: +x forward, +y driver's left, +z up. Viewed from the
  front looking rearward, so in the plots +y points to the right and the front-right
  (FR) corner sits at negative y.
* Camber (ISO sign) = positive rotation about +x for BOTH wheels, i.e. counter-clockwise
  in the y-z plot. RH wheel tilting outwards -> +; LH wheel tilting outwards -> -.
  "Physical" camber (outward lean +) is also printed: physical = ISO for RH, -ISO for LH.
* Static camber = 0.
* Track data dZc: suspension compression (bump) positive = wheel moves UP relative to
  the body. The brief's body heave z_is (body up +) is therefore z_is = -dZc.
  Eq. (3) of the brief, d_gamma = -z/c1, only gives the physically correct sign
  (bump -> negative camber on the RH wheel) when z is upward WHEEL travel, so it is
  applied here with z = dZc.
* Corner assumed LEFT-hand: the RH (analysed FR) wheel is the outside wheel, the body
  rolls towards -y, which is a positive rotation phi about +x.
* Vehicle assumed laterally symmetric: LH corner = mirror (y -> -y) of the RH data,
  t/2 = |y| of the contact patch.
* Exact solution: body rolls by phi about a longitudinal axis through the STATIC roll
  centre and heaves down by dZc; each corner's linkage is then solved (rigid arms,
  rigid upright) so that its contact patch stays on the ground plane z = 0.
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import plot_style as ps

ps.apply_style()

# Plot colours by role (shared palette, see plot_style.py)
C_STATIC = ps.SLATE        # static geometry
C_DISP = ps.PRIMARY           # displaced geometry / exact solution
C_ACCENT = ps.SECONDARY       # roll-centre migration / first-order estimate
C_VSAL = ps.TERTIARY           # virtual swing arm

# =============================================================================
# INPUT DATA  (individualised datasheet, scenario A)
# =============================================================================

IB_LW = np.array([-350.0, 180.0])     # chassis lower inboard
OB_LW = np.array([-730.0, 160.0])     # outboard lower ball joint
IB_UP = np.array([-380.0, 428.0])     # chassis upper inboard
OB_UP = np.array([-714.0, 440.0])     # outboard upper ball joint
WHEEL_CENTRE = np.array([-775.0, 300.0])
CONTACT_PATCH_CENTRE = np.array([-775.0, 0.0])

WHEEL_JOUNCE = 15.0       # dZc [mm], compression positive (both wheels, heave)
BODY_ROLL_ANGLE = 1.8     # phi [deg], positive about +x (rolls towards the RH wheel)

HALF_TRACK = abs(CONTACT_PATCH_CENTRE[0])   # t/2 from contact patch (symmetric vehicle)
VEHICLE_TRACK = 2 * HALF_TRACK

PARALLEL_TOL = 1e-9       # |det| below this -> arms treated as parallel (IC at infinity)


# =============================================================================
# GEOMETRY HELPERS
# =============================================================================

def rot(angle_rad):
    """2-D rotation matrix, positive = counter-clockwise in the y-z plot = +x rotation."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s], [s, c]])


def line_intersection(p1, d1, p2, d2):
    """Intersection of lines p1 + t*d1 and p2 + s*d2. Returns None if parallel."""
    A = np.column_stack((d1, -d2))
    if abs(np.linalg.det(A)) < PARALLEL_TOL * np.linalg.norm(d1) * np.linalg.norm(d2):
        return None
    t, _ = np.linalg.solve(A, p2 - p1)
    return p1 + t * d1


def circle_circle(c0, r0, c1, r1, near):
    """Intersection of two circles; returns the root closest to `near`."""
    d = np.linalg.norm(c1 - c0)
    if d > r0 + r1 or d < abs(r0 - r1):
        raise ValueError("Circles do not intersect - travel outside linkage range")
    a = (r0**2 - r1**2 + d**2) / (2 * d)
    h = np.sqrt(max(r0**2 - a**2, 0.0))
    u = (c1 - c0) / d
    p = c0 + a * u
    n = np.array([-u[1], u[0]])
    roots = (p + h * n, p - h * n)
    return min(roots, key=lambda q: np.linalg.norm(q - near))


def instant_centre(ib_lw, ob_lw, ib_up, ob_up):
    """IC = intersection of the lower and upper arm lines (inboard -> outboard)."""
    return line_intersection(ib_lw, ob_lw - ib_lw, ib_up, ob_up - ib_up)


def swing_arm_length(ic, wheel_centreline_y):
    """VSAL: HORIZONTAL distance from IC to the wheel's vertical centreline (lecture
    definition). Infinite if the arms are parallel."""
    return np.inf if ic is None else abs(ic[0] - wheel_centreline_y)


def mirror(p):
    """Mirror a point about the vehicle centreline (RH -> LH corner)."""
    return np.array([-p[0], p[1]])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# =============================================================================
# 1) STATIC GEOMETRY: IC, VSAL, RC
# =============================================================================

IC = instant_centre(IB_LW, OB_LW, IB_UP, OB_UP)
c1 = swing_arm_length(IC, WHEEL_CENTRE[0])

if IC is None:
    RC = np.array([0.0, 0.0])      # parallel arms: force line parallel to arms; RC at ground for horizontal arms
    print("Arms parallel: IC at infinity, VSAL infinite")
else:
    RC = line_intersection(CONTACT_PATCH_CENTRE, IC - CONTACT_PATCH_CENTRE,
                           np.array([0.0, 0.0]), np.array([0.0, 1.0]))

print("=== STATIC GEOMETRY ===")
def arm_inclination(ib, ob):
    """Arm angle to horizontal [deg], positive = arm rises from outboard to inboard."""
    return np.degrees(np.arctan2(ib[1] - ob[1], abs(ib[0] - ob[0])))


print(f"Lower arm inclination      : {arm_inclination(IB_LW, OB_LW):+.2f} deg (+ = rises inboard)")
print(f"Upper arm inclination      : {arm_inclination(IB_UP, OB_UP):+.2f} deg")
print(f"Arm convergence angle      : {arm_inclination(IB_LW, OB_LW) - arm_inclination(IB_UP, OB_UP):.2f} deg")
print(f"Instant centre (y, z)      : ({IC[0]:.1f}, {IC[1]:.1f}) mm")
print(f"Virtual swing arm c1       : {c1:.1f} mm (horizontal, IC to wheel centreline)")
print(f"c1 / track                 : {c1 / VEHICLE_TRACK:.2f}")
print(f"Roll centre (y, z)         : ({RC[0]:.1f}, {RC[1]:.1f}) mm")

# =============================================================================
# 2) FIRST-ORDER CAMBER (brief eqs 3-7), ISO sign
# =============================================================================

phi = np.deg2rad(BODY_ROLL_ANGLE)

dgamma_dz = -np.arctan(1 / c1)                      # rad/mm of wheel jounce (eq. 2)
camber_rate_deg_per_10mm = np.degrees(dgamma_dz) * 10

heave_term = dgamma_dz * WHEEL_JOUNCE               # rad, RH wheel (eq. 3 with z = dZc)
roll_recovery = HALF_TRACK / c1 * phi               # rad, (t/2c1)*phi
roll_term = phi - roll_recovery                     # rad, eq. 5 (same ISO sign both wheels)

gamma_out_fo = np.degrees(roll_term + heave_term)   # RH, outside   (eq. 6, i = 1)
gamma_in_fo = np.degrees(roll_term - heave_term)    # LH, inside    (eq. 6, i = 2)
CR = HALF_TRACK / c1 * 100                          # eq. 7

print("\n=== FIRST-ORDER CAMBER (ISO sign) ===")
print(f"Camber change rate         : {camber_rate_deg_per_10mm:.4f} deg/10 mm jounce")
print(f"Heave term (RH)            : {np.degrees(heave_term):+.3f} deg  (dZc = {WHEEL_JOUNCE} mm)")
print(f"Chassis tilt term          : {np.degrees(phi):+.3f} deg")
print(f"Roll recovery term         : {-np.degrees(roll_recovery):+.3f} deg")
print(f"Outside (RH) camber        : {gamma_out_fo:+.3f} deg ISO  -> {gamma_out_fo:+.3f} deg physical")
print(f"Inside  (LH) camber        : {gamma_in_fo:+.3f} deg ISO  -> {-gamma_in_fo:+.3f} deg physical")
print(f"Camber recovery ratio CR   : {CR:.1f} %")

# =============================================================================
# 3) EXACT KINEMATIC SOLUTION
# =============================================================================

L_LOWER = np.linalg.norm(OB_LW - IB_LW)
L_UPPER = np.linalg.norm(OB_UP - IB_UP)
L_UPRIGHT = np.linalg.norm(OB_UP - OB_LW)
UPRIGHT_ANGLE_0 = np.arctan2(*(OB_UP - OB_LW)[::-1])


def make_corner(side):
    """Hardpoints in the body frame for 'RH' (data) or 'LH' (mirrored)."""
    pts = dict(IB_LW=IB_LW, OB_LW=OB_LW, IB_UP=IB_UP, OB_UP=OB_UP,
               WC=WHEEL_CENTRE, CP=CONTACT_PATCH_CENTRE)
    return pts if side == "RH" else {k: mirror(v) for k, v in pts.items()}


def solve_linkage(corner, beta):
    """Rotate the lower arm by beta about its inboard pivot, then close the loop.
    Returns the displaced body-frame points and the upright rotation (= body-frame camber)."""
    ob_lw = corner["IB_LW"] + rot(beta) @ (corner["OB_LW"] - corner["IB_LW"])
    ob_up = circle_circle(ob_lw, L_UPRIGHT, corner["IB_UP"], L_UPPER, near=corner["OB_UP"])
    a0 = np.arctan2(*(corner["OB_UP"] - corner["OB_LW"])[::-1])
    a1 = np.arctan2(*(ob_up - ob_lw)[::-1])
    d_gamma = wrap(a1 - a0)                                   # rigid upright rotation
    R = rot(d_gamma)
    wc = ob_lw + R @ (corner["WC"] - corner["OB_LW"])
    cp = ob_lw + R @ (corner["CP"] - corner["OB_LW"])
    return dict(IB_LW=corner["IB_LW"], OB_LW=ob_lw, IB_UP=corner["IB_UP"], OB_UP=ob_up,
                WC=wc, CP=cp), d_gamma


def body_to_ground(p, phi_rad, heave_down, roll_axis):
    """Body pose: roll by phi about roll_axis, then translate down by heave."""
    return roll_axis + rot(phi_rad) @ (p - roll_axis) - np.array([0.0, heave_down])


def ground_to_body(p, phi_rad, heave_down, roll_axis):
    return roll_axis + rot(-phi_rad) @ (p + np.array([0.0, heave_down]) - roll_axis)


def bisect(f, a, b, tol=1e-10, it=200):
    fa = f(a)
    for _ in range(it):
        m = 0.5 * (a + b)
        fm = f(m)
        if abs(fm) < tol or (b - a) < tol:
            return m
        if np.sign(fm) == np.sign(fa):
            a, fa = m, fm
        else:
            b = m
    return 0.5 * (a + b)


def solve_corner(side, phi_rad, heave_down, roll_axis):
    """Find the lower-arm rotation that puts this corner's contact patch on z = 0."""
    corner = make_corner(side)

    def cp_height(beta):
        pts, _ = solve_linkage(corner, beta)
        return body_to_ground(pts["CP"], phi_rad, heave_down, roll_axis)[1]

    beta = bisect(cp_height, -0.3, 0.3)
    pts_body, d_gamma_body = solve_linkage(corner, beta)

    # rigid-link check
    assert np.isclose(np.linalg.norm(pts_body["OB_LW"] - pts_body["IB_LW"]), L_LOWER)
    assert np.isclose(np.linalg.norm(pts_body["OB_UP"] - pts_body["IB_UP"]), L_UPPER)
    assert np.isclose(np.linalg.norm(pts_body["OB_UP"] - pts_body["OB_LW"]), L_UPRIGHT)

    ic_body = instant_centre(pts_body["IB_LW"], pts_body["OB_LW"], pts_body["IB_UP"], pts_body["OB_UP"])
    pts_ground = {k: body_to_ground(v, phi_rad, heave_down, roll_axis) for k, v in pts_body.items()}
    ic_ground = None if ic_body is None else body_to_ground(ic_body, phi_rad, heave_down, roll_axis)

    return dict(
        body=pts_body, ground=pts_ground, IC_body=ic_body, IC_ground=ic_ground,
        jounce=pts_body["CP"][1] - corner["CP"][1],                # wheel travel rel. body (+ bump)
        camber_iso=np.degrees(d_gamma_body + phi_rad),             # camber relative to the road
        vsal=swing_arm_length(ic_body, pts_body["WC"][0]),
    )


def roll_centre_two_sides(rh, lh):
    """Dynamic RC = intersection of the RH and LH force lines (CP -> IC), ground frame."""
    if rh["IC_ground"] is None or lh["IC_ground"] is None:
        return None
    return line_intersection(rh["ground"]["CP"], rh["IC_ground"] - rh["ground"]["CP"],
                             lh["ground"]["CP"], lh["IC_ground"] - lh["ground"]["CP"])


roll_axis = RC.copy()               # assumption: body rolls about the static roll centre
rh = solve_corner("RH", phi, WHEEL_JOUNCE, roll_axis)
lh = solve_corner("LH", phi, WHEEL_JOUNCE, roll_axis)
RC_dyn = roll_centre_two_sides(rh, lh)
RC_dyn_body = ground_to_body(RC_dyn, phi, WHEEL_JOUNCE, roll_axis)

# body-fixed point that was the static RC, in the ground frame (for reference)
RC_static_ground = body_to_ground(RC, phi, WHEEL_JOUNCE, roll_axis)

print("\n=== EXACT SOLUTION (heave + roll about static RC) ===")
for name, side, c, fo in (("Outside (RH)", "RH", rh, gamma_out_fo), ("Inside  (LH)", "LH", lh, gamma_in_fo)):
    phys = c["camber_iso"] if side == "RH" else -c["camber_iso"]
    err = c["camber_iso"] - fo
    print(f"{name}: jounce {c['jounce']:+.2f} mm | camber {c['camber_iso']:+.3f} deg ISO "
          f"({phys:+.3f} deg physical) | first-order {fo:+.3f} | error {err:+.3f} deg "
          f"({100 * err / c['camber_iso']:+.1f} %)")
    print(f"              IC body ({c['IC_body'][0]:.1f}, {c['IC_body'][1]:.1f}) mm | "
          f"VSAL {c['vsal']:.1f} mm | CP lateral shift (ground) {c['ground']['CP'][0] - make_corner(side)['CP'][0]:+.2f} mm")
print(f"First-order roll jounce (t/2)sin(phi): {HALF_TRACK * np.sin(phi):.2f} mm")
print(f"Dynamic RC, ground frame (y, z): ({RC_dyn[0]:+.1f}, {RC_dyn[1]:.1f}) mm")
print(f"Dynamic RC, body frame   (y, z): ({RC_dyn_body[0]:+.1f}, {RC_dyn_body[1]:.1f}) mm")
print(f"RC migration (dynamic - static, ground frame): "
      f"dy = {RC_dyn[0] - RC[0]:+.1f} mm, dz = {RC_dyn[1] - RC[1]:+.1f} mm")

# =============================================================================
# 4) CAMBER vs JOUNCE SWEEP (pure heave, RH wheel): exact vs first-order
# =============================================================================

sweep = np.linspace(-40, 40, 81)
cam_exact, ic_z_sweep, rc_z_sweep = [], [], []
for dz in sweep:
    r = solve_corner("RH", 0.0, dz, roll_axis)
    cam_exact.append(r["camber_iso"])
    ic_z_sweep.append(r["IC_body"][1])
    # pure heave stays symmetric -> RC on the centreline
    rc = line_intersection(r["body"]["CP"], r["IC_body"] - r["body"]["CP"],
                           np.array([0.0, 0.0]), np.array([0.0, 1.0]))
    rc_z_sweep.append(body_to_ground(rc, 0.0, dz, roll_axis)[1])   # height above ground
cam_exact = np.array(cam_exact)
cam_fo = np.degrees(dgamma_dz) * sweep

i10 = np.argmin(abs(sweep - 10))
print(f"\nExact camber at +10 mm pure jounce: {cam_exact[i10]:+.4f} deg "
      f"(first-order {camber_rate_deg_per_10mm:+.4f})")

# =============================================================================
# 5) PLOTS
# =============================================================================

def draw_corner(ax, p, ic, colour, label, ls="-"):
    ax.plot([p["IB_LW"][0], p["OB_LW"][0]], [p["IB_LW"][1], p["OB_LW"][1]], color=colour, ls=ls, lw=2, label=label)
    ax.plot([p["IB_UP"][0], p["OB_UP"][0]], [p["IB_UP"][1], p["OB_UP"][1]], color=colour, ls=ls, lw=2)
    ax.plot([p["OB_LW"][0], p["OB_UP"][0]], [p["OB_LW"][1], p["OB_UP"][1]], color=colour, ls=ls, lw=3)  # upright
    ax.plot(*np.array([p["IB_LW"], p["OB_LW"], p["IB_UP"], p["OB_UP"]]).T, "o", color=colour,
            mfc=ps.SURFACE, mew=1.4, ms=5, ls="none")
    # wheel plane through CP and WC
    w = p["WC"] - p["CP"]
    ax.plot([p["CP"][0], p["CP"][0] + 2 * w[0]], [p["CP"][1], p["CP"][1] + 2 * w[1]], color=colour, ls="-.", lw=1.2)
    ax.plot(*p["WC"], "x", color=colour, ms=7, mew=1.6)
    if ic is not None:
        ax.plot([p["IB_LW"][0], ic[0]], [p["IB_LW"][1], ic[1]], color=colour, ls=":", lw=0.8)
        ax.plot([p["IB_UP"][0], ic[0]], [p["IB_UP"][1], ic[1]], color=colour, ls=":", lw=0.8)
        ax.plot([p["CP"][0], ic[0]], [p["CP"][1], ic[1]], color=colour, ls="--", lw=0.9)
        ax.plot(*ic, "s", color=colour, ms=7, mec=ps.SURFACE, mew=1.2)


def finish_geometry_axes(ax):
    ax.set_xlabel("Lateral position, $y$ [mm]  (+ = driver's left)")
    ax.set_ylabel("Vertical position, $z$ [mm]")


from matplotlib.patches import Polygon

TYRE_WIDTH = 250.0          # mm, ILLUSTRATIVE ONLY - not part of the datasheet
TYRE_DIAMETER = 2 * WHEEL_CENTRE[1]   # loaded radius = wheel-centre height
CORNER_RADIUS = 45.0        # mm, sidewall rounding for the drawing


def tyre_outline(cp, gamma_deg, n=12):
    """Front-view tyre outline (rounded rectangle) standing on `cp`, rotated by
    gamma (ISO sign, CCW in the y-z plot) about the contact patch."""
    hw, d, r = TYRE_WIDTH / 2, TYRE_DIAMETER, CORNER_RADIUS
    pts = []
    for (cy, cz, a0) in ((hw - r, r, -90), (hw - r, d - r, 0), (-hw + r, d - r, 90), (-hw + r, r, 180)):
        for a in np.linspace(np.radians(a0), np.radians(a0 + 90), n):
            pts.append([cy + r * np.cos(a), cz + r * np.sin(a)])
    pts = np.array(pts) @ rot(np.radians(gamma_deg)).T
    return pts + cp


def draw_tyre(ax, cp, gamma_deg, colour, alpha=0.18, label=None, centreline=True):
    outline = tyre_outline(cp, gamma_deg)
    ax.add_patch(Polygon(outline, closed=True, facecolor=colour, edgecolor="none",
                         alpha=alpha, label=label))
    ax.add_patch(Polygon(outline, closed=True, fill=False, edgecolor=colour, lw=1.2))
    if centreline:
        top = cp + rot(np.radians(gamma_deg)) @ np.array([0.0, TYRE_DIAMETER])
        ax.plot([cp[0], top[0]], [cp[1], top[1]], color=colour, ls="-.", lw=1.5)
        wc = cp + rot(np.radians(gamma_deg)) @ np.array([0.0, TYRE_DIAMETER / 2])
        ax.plot(*wc, "x", color=colour, ms=8)


static_RH = make_corner("RH")
static_LH = make_corner("LH")
IC_LH = mirror(IC)

# --- Figure 1: static geometry -------------------------------------------------
fig1, a1 = plt.subplots(figsize=(12, 6))
draw_corner(a1, static_RH, IC, C_STATIC, "Static arms / upright")
a1.plot([IC[0], WHEEL_CENTRE[0]], [IC[1], IC[1]], color=C_VSAL, lw=2, label=f"VSAL $c_1$ = {c1:.0f} mm")
a1.plot(*RC, "^", color=C_ACCENT, ms=10, mec=ps.SURFACE, mew=1.2,
        label=f"RC ({RC[0]:.0f}, {RC[1]:.1f}) mm")
a1.axvline(0, color=ps.MUTED, ls="--", lw=0.9, label="Vehicle centreline")
a1.axhline(0, color=ps.GROUND, lw=1.5, label="Ground")
finish_geometry_axes(a1)

# on-plot labels: IC, virtual swing arm length, roll centre
a1.annotate(f"IC ({IC[0]:.0f}, {IC[1]:.0f}) mm", IC, xytext=(0, 14), textcoords="offset points",
            ha="right", fontsize=10, color=ps.INK_2)
a1.annotate(f"VSAL $c_1$ = {c1:.0f} mm\n(horizontal, IC to wheel centreline)",
            (400.0, IC[1]), xytext=(0, -8), textcoords="offset points",
            ha="center", va="top", fontsize=10, color=C_VSAL)
a1.annotate(f"RC ({RC[0]:.0f}, {RC[1]:.1f}) mm\nheight {RC[1]:.1f} mm above ground",
            RC, xytext=(40, -18), textcoords="offset points", ha="left", va="center",
            fontsize=10, color=C_ACCENT,
            arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=1.2))
a1.set_ylim(-40, 640)
a1.set_title("Full view incl. instant centre (z scale exaggerated)")
a1.legend(loc="upper center", ncol=3)
fig1.suptitle("Static FR corner geometry (front view, ISO 8855 body frame)")
fig1.tight_layout()
ps.save_figure(fig1, __file__, "CamberGain_static.png")

# --- Figure 2a: static vs displaced, full view incl. instant centres ----------
def draw_both_corners(ax):
    draw_corner(ax, static_RH, IC, C_STATIC, "Static", ls="-")
    draw_corner(ax, static_LH, IC_LH, C_STATIC, None, ls="-")
    draw_corner(ax, rh["ground"], rh["IC_ground"], C_DISP, "Displaced (ground frame)")
    draw_corner(ax, lh["ground"], lh["IC_ground"], C_DISP, None)
    ax.plot(*RC, "^", color=C_STATIC, ms=10, mec=ps.SURFACE, mew=1.2,
            label=f"Static RC ({RC[0]:.0f}, {RC[1]:.1f}) mm")
    ax.plot(*RC_dyn, "^", color=C_DISP, ms=10, mec=ps.SURFACE, mew=1.2,
            label=f"Dynamic RC ({RC_dyn[0]:+.1f}, {RC_dyn[1]:.1f}) mm")
    ax.annotate("", xy=RC_dyn, xytext=RC, arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=2))
    ax.axhline(0, color=ps.GROUND, lw=1.5)
    finish_geometry_axes(ax)


scenario_txt = (rf"$\Delta Z_c$ = {WHEEL_JOUNCE:.0f} mm heave, $\phi$ = {BODY_ROLL_ANGLE}° roll "
                f"(roll axis through static RC)")

fig2a, a2 = plt.subplots(figsize=(12, 6.5))
draw_both_corners(a2)
a2.axvline(0, color=ps.MUTED, ls="--", lw=0.9, label="Vehicle centreline (static)")
for ic, static, side in ((IC, True, "RH"), (IC_LH, True, "LH"),
                         (rh["IC_ground"], False, "RH"), (lh["IC_ground"], False, "LH")):
    a2.annotate(f"IC {side}\n({ic[0]:.0f}, {ic[1]:.0f})", ic, xytext=(0, 12 if static else -14),
                textcoords="offset points", ha="center", va="bottom" if static else "top",
                fontsize=8, color=ps.INK_2)
a2.set_title("Instant centres and force lines, static vs displaced (z scale exaggerated)")
a2.legend(loc="upper center", ncol=3)
fig2a.suptitle(f"Static vs displaced geometry: {scenario_txt}")
fig2a.tight_layout()
ps.save_figure(fig2a, __file__, "CamberGain_displaced_full.png")

# --- Figure 2b: detailed view, true scale, with shaded chassis and tyres --------
# Chassis drawn as the body-fixed polygon through the four inboard pivots
# (illustrative outline only), moved with the same heave + roll as the body.
chassis_body = np.array([IB_LW, IB_UP, mirror(IB_UP), mirror(IB_LW)])
chassis_ground = np.array([body_to_ground(p, phi, WHEEL_JOUNCE, roll_axis) for p in chassis_body])

fig2b, b2 = plt.subplots(figsize=(13, 6.5))
b2.add_patch(Polygon(chassis_body, closed=True, facecolor=ps.MUTED, edgecolor=C_STATIC, alpha=0.18,
                     lw=1.2, zorder=0, label="Chassis, static"))
b2.add_patch(Polygon(chassis_ground, closed=True, facecolor=C_DISP, edgecolor=C_DISP, alpha=0.12,
                     lw=1.2, zorder=0, label="Chassis, displaced"))
for s_pts, c in ((static_RH, rh), (static_LH, lh)):
    draw_tyre(b2, s_pts["CP"], 0.0, ps.MUTED, alpha=0.20, centreline=False)
    draw_tyre(b2, c["ground"]["CP"], c["camber_iso"], C_DISP, alpha=0.12, centreline=False)
draw_both_corners(b2)

# body centreline: static (vertical) and displaced (rolled with the body)
b2.plot([0, 0], [-60, 660], color=ps.MUTED, ls="--", lw=0.9, label="Body centreline, static")
cl = np.array([body_to_ground(np.array([0.0, z]), phi, WHEEL_JOUNCE, roll_axis) for z in (-60.0, 660.0)])
b2.plot(cl[:, 0], cl[:, 1], color=C_DISP, ls="--", lw=0.9, label="Body centreline, displaced")

for c in (rh, lh):
    top = c["ground"]["CP"] + 2 * (c["ground"]["WC"] - c["ground"]["CP"])
    b2.annotate(f"γ = {c['camber_iso']:+.2f}°", top, xytext=(0, 8), textcoords="offset points",
                ha="center", color=ps.INK, fontsize=10)
b2.set_xlim(-950, 950)
b2.set_ylim(-60, 680)
b2.set_aspect("equal", adjustable="box")
b2.set_title(f"Detailed view (true scale): outside (RH) camber {rh['camber_iso']:+.2f}°, "
             f"inside (LH) camber {lh['camber_iso']:+.2f}° (ISO sign)\n"
             f"Chassis outline and {TYRE_WIDTH:.0f} mm tyre width are illustrative",
             fontweight="normal", fontsize=10)
fig2b.legend(*b2.get_legend_handles_labels(), loc="lower center", ncol=4)
fig2b.suptitle(f"Static vs displaced geometry: {scenario_txt}")
fig2b.tight_layout(rect=(0, 0.08, 1, 1))
ps.save_figure(fig2b, __file__, "CamberGain_displaced_detail.png")

# --- Figure 3: camber and RC height vs jounce ---------------------------------
fig3, (a3, b3) = plt.subplots(1, 2, figsize=(12, 4.6))
a3.plot(sweep, cam_exact, color=C_DISP, lw=2, label="Exact (linkage solve)")
a3.plot(sweep, cam_fo, color=C_ACCENT, ls="--", lw=2,
        label=f"First-order, {camber_rate_deg_per_10mm:.3f} deg/10 mm")
a3.axhline(0, color=ps.MUTED, lw=0.8)
a3.axvline(0, color=ps.MUTED, lw=0.8)
a3.set_xlabel(r"Wheel jounce, $\Delta Z_c$ [mm]  (bump +)")
a3.set_ylabel(r"RH camber, $\gamma$ [deg]  (ISO)")
a3.set_title("Camber vs wheel jounce (pure heave)")
a3.legend(loc="upper right")
b3.plot(sweep, rc_z_sweep, color=C_DISP, lw=2)
b3.axvline(0, color=ps.MUTED, lw=0.8)
b3.set_xlabel(r"Wheel jounce, $\Delta Z_c$ [mm]  (bump +)")
b3.set_ylabel("RC height above ground [mm]")
b3.set_title("Roll centre height vs jounce (pure heave)")
fig3.tight_layout()
ps.save_figure(fig3, __file__, "CamberGain_sweep.png")

plt.close("all")
