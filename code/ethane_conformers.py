"""
ethane_conformers.py
====================
Ethane's internal rotation as a descent in symmetry:

    eclipsed        D3h    order 12     E, 2C3, 3C2, sigma_h, 2S3, 3sigma_v
    anything else   D3     order  6     E, 2C3, 3C2
    staggered       D3d    order 12     E, 2C3, 3C2, i, 2S6, 3sigma_d

The point worth making is that the order does NOT simply fall and rise again by
accident. C3 and the three C2 axes survive the whole way --- that D3 core is
what every conformer has in common. What comes and goes are the extras: the
horizontal mirror at the eclipsed end, and the inversion centre at the
staggered end. Only at the two extremes does anything additional hold.

Needs symmetry_operation_gifs.py alongside it, for the drawing style.

Run:  python ethane_conformers.py [outdir]

Produces
    ethane_conformers.gif          the rotation, eclipsed -> staggered
    ethane_conformer_eclipsed.png  D3h
    ethane_conformer_skew.png      D3, a general conformer
    ethane_conformer_staggered.png D3d
plus a _plain variant of each still, with the labelling removed.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
import symmetry_operation_gifs as S

# =====================================================================
#  SETTINGS
# =====================================================================
D_CC = 1.54          # C-C bond length
R_CH = 1.09          # C-H bond length
HCC = 111.0          # H-C-C angle in degrees
VIEW = (12, -68)     # 3D camera: (elevation, azimuth)
LIM = 1.62           # half-width of the 3D box
TRACK = 2            # index of the hydrogen drawn crimson (2 = first top H)

AXIS = S.AXIS_C           # C3 and C2 axes, blue
PLANE = S.PLANE_C         # mirror plane fill, gold
INV = S.INV_C             # inversion centre, green
NAVY, RUST, GREY = "#1D3B5A", "#B5341F", "#6A6A6A"
# =====================================================================

d = np.deg2rad
TOP_AZ = np.array([90.0, 210.0, 330.0])       # the three top hydrogens


def ethane(tau):
    """Ethane with the BOTTOM methyl rotated by tau degrees.
    tau = 0 is eclipsed, tau = 60 is staggered."""
    zc = D_CC / 2
    phi = d(180 - HCC)                        # C-H tilt away from the C-C axis
    topC, botC = np.array([0, 0, zc]), np.array([0, 0, -zc])

    def H(C, az_deg, up):
        az = d(az_deg)
        return C + R_CH * np.c_[np.sin(phi) * np.cos(az),
                                np.sin(phi) * np.sin(az),
                                up * np.cos(phi) * np.ones_like(az)]

    coords = np.vstack([topC, botC, H(topC, TOP_AZ, 1), H(botC, TOP_AZ + tau, -1)])
    bonds = [(0, 1)] + [(0, i) for i in (2, 3, 4)] + [(1, i) for i in (5, 6, 7)]
    return coords, ["C", "C"] + ["H"] * 6, bonds


def point_group(tau):
    """Which group, and which elements, at this dihedral offset."""
    t = tau % 120.0
    if np.isclose(t, 0.0, atol=0.4) or np.isclose(t, 120.0, atol=0.4):
        return ("$D_{3h}$", 12, "eclipsed",
                "E, 2C$_3$, 3C$_2$, $\\sigma_h$, 2S$_3$, 3$\\sigma_v$", True, False)
    if np.isclose(t, 60.0, atol=0.4):
        return ("$D_{3d}$", 12, "staggered",
                "E, 2C$_3$, 3C$_2$, $i$, 2S$_6$, 3$\\sigma_d$", False, True)
    return ("$D_3$", 6, "skew", "E, 2C$_3$, 3C$_2$", False, False)


# ----------------------------------------------------------- drawing
def _axes3d(ax, tau, a_sh, a_i):
    """C3 up the C-C bond, the three C2 perpendicular to it, and whichever of
    sigma_h / i currently applies."""
    v = np.array([0, 0, LIM * 0.98])
    ax.plot(*zip(-v, v), color=AXIS, lw=2.6, alpha=0.95, zorder=6)

    # the C2 axes bisect a top H and the bottom H nearest it, so they turn
    # through half the dihedral angle as the methyl rotates
    for a in TOP_AZ + tau / 2.0:
        u = np.array([np.cos(d(a)), np.sin(d(a)), 0.0]) * LIM * 0.92
        ax.plot(*zip(-u, u), color=AXIS, lw=1.5, alpha=0.60, zorder=5)

    if a_i > 0.02:                             # inversion centre, staggered only
        ax.scatter(0, 0, 0, s=520, facecolors="none", edgecolors=INV,
                   linewidths=2.4, alpha=a_i, depthshade=False, zorder=8)

    if a_sh > 0.02:                            # horizontal mirror, eclipsed only
        g = LIM * 0.86
        q = np.array([[g, g, 0], [g, -g, 0], [-g, -g, 0], [-g, g, 0]], float)
        ax.add_collection3d(Poly3DCollection([q], facecolor=PLANE,
                                             alpha=0.16 * a_sh, edgecolor="none",
                                             zorder=0))
        loop = np.vstack([q, q[0]])
        ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], color=PLANE, lw=1.7,
                alpha=0.9 * a_sh, zorder=1)


def _draw3d(ax, tau, a_sh, a_i):
    P, elems, bonds = ethane(tau)
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_zlim(-LIM, LIM)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
    ax.view_init(*VIEW)
    _axes3d(ax, tau, a_sh, a_i)
    for i, j in bonds:
        ax.plot(*zip(P[i], P[j]), color=S.BOND_C, lw=3.0, alpha=0.95, zorder=2)
    for k, e in enumerate(elems):
        st = S.TRACK if k == TRACK else S.STYLE[e]
        ax.scatter(*P[k], color=st["color"], s=S.STYLE[e]["s"],
                   edgecolor=st["edge"], linewidth=0.8, depthshade=True, zorder=4)


def _draw_newman(ax, tau, a_i, cr=0.34, r=1.05):
    """Looking straight down the C-C bond: the clearest view of the dihedral."""
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
    ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(plt.Circle((0, 0), cr, fill=False, ec="#555", lw=1.7, zorder=2))

    for a in TOP_AZ + tau / 2.0:               # C2 axes, seen end-on
        u = np.array([np.cos(d(a)), np.sin(d(a))]) * 1.42
        ax.plot(*zip(-u, u), color=AXIS, lw=1.3, alpha=0.55, zorder=1)

    # drawing-only nudge so the back atoms are not perfectly hidden at the
    # eclipsed end; it fades out once the real dihedral separates them
    nudge = 5.0 * float(np.clip(1.0 - min(tau, 120 - tau) / 12.0, 0, 1))
    for front, azs in ((0, TOP_AZ + tau - nudge), (1, TOP_AZ + nudge)):
        for n, a in enumerate(azs):
            th = d(a); dirn = np.array([np.cos(th), np.sin(th)])
            pos = r * dirn
            inner = (1 - front) * cr * dirn
            ax.plot([inner[0], pos[0]], [inner[1], pos[1]], color="#6f6f6f",
                    lw=2.4 + 0.9 * front, alpha=0.55 + 0.45 * front,
                    zorder=3 + front, solid_capstyle="round")
            trk = (front == 1 and n == 0)
            ax.scatter(*pos, s=330 + 140 * front,
                       color=S.TRACK["color"] if trk else S.STYLE["H"]["color"],
                       edgecolor=S.TRACK["edge"] if trk else S.STYLE["H"]["edge"],
                       lw=1.0, zorder=5 + front)
    ax.scatter(0, 0, s=95, color=S.STYLE["C"]["color"], zorder=7)
    if a_i > 0.02:                             # inversion centre, staggered only
        ax.scatter(0, 0, s=620, facecolors="none", edgecolors=INV,
                   linewidths=2.4, alpha=a_i, zorder=8)


def _figure(dpi):
    fig = plt.figure(figsize=(8.6, 4.9), dpi=dpi)
    ax3 = fig.add_axes([0.02, 0.17, 0.44, 0.74], projection="3d")
    ax2 = fig.add_axes([0.53, 0.17, 0.44, 0.74])
    return fig, ax3, ax2


def _label(fig, tau, pg, order, name, elems):
    fig.text(0.5, 0.955, f"Ethane: internal rotation as a descent in symmetry",
             ha="center", fontsize=13, color=NAVY, weight="bold")
    fig.text(0.24, 0.085, "3D view", ha="center", fontsize=9.5, color=GREY,
             style="italic")
    fig.text(0.75, 0.095, "Newman projection, down the C\u2013C bond",
             ha="center", fontsize=9.5, color=GREY, style="italic")
    if min(tau, 120 - tau) < 6:        # only when the atoms would coincide
        fig.text(0.75, 0.062, "back atoms nudged apart, to be visible",
                 ha="center", fontsize=8, color=GREY, style="italic")
    fig.text(0.5, 0.035,
             f"dihedral {tau:.0f}\u00b0   \u2014   {name}   \u2014   {pg}, "
             f"order {order}   \u2014   {elems}",
             ha="center", fontsize=11.5, color=RUST, weight="bold")


# ------------------------------------------------------------- outputs
def still(tau, name, outdir=".", labelled=True, dpi=200):
    pg, order, conf, elems, sh, inv = point_group(tau)
    fig, ax3, ax2 = _figure(dpi)
    _draw3d(ax3, tau, 1.0 if sh else 0.0, 1.0 if inv else 0.0)
    _draw_newman(ax2, tau, 1.0 if inv else 0.0)
    if labelled:
        _label(fig, tau, pg, order, conf, elems)
    base = os.path.join(outdir,
                        f"ethane_conformer_{name}" + ("" if labelled else "_plain"))
    fig.savefig(base + ".png", transparent=True, facecolor="none")
    plt.close(fig)
    im = Image.open(base + ".png").convert("RGBA")
    bb = im.split()[3].getbbox(); pad = 12
    im.crop((max(0, bb[0] - pad), max(0, bb[1] - pad),
             min(im.width, bb[2] + pad), min(im.height, bb[3] + pad))
            ).save(base + "_tight.png")
    print("wrote", base + ".png")


def animation(outdir=".", dpi=110):
    # hold at each end so the two high-symmetry conformers can be read
    segs = [("hold0", 16), ("turn", 34), ("hold60", 22)]
    plan = [(nm, k / max(1, n - 1)) for nm, n in segs for k in range(n)]

    fig, ax3, ax2 = _figure(dpi)
    head = fig.text(0.5, 0.955, "", ha="center", fontsize=13, color=NAVY,
                    weight="bold")
    foot = fig.text(0.5, 0.022, "", ha="center", fontsize=11.5, color=RUST,
                    weight="bold")
    fig.text(0.24, 0.095, "3D view", ha="center", fontsize=9.5, color=GREY,
             style="italic")
    fig.text(0.75, 0.095, "Newman projection, down the C\u2013C bond",
             ha="center", fontsize=9.5, color=GREY, style="italic")

    def update(k):
        nm, t = plan[k]
        tau = 0.0 if nm == "hold0" else (60.0 if nm == "hold60"
                                         else 60.0 * S.ease(np.array([t]))[0])
        pg, order, conf, elems, sh, inv = point_group(tau)
        # fade the extras in and out over the first and last few degrees
        a_sh = float(np.clip(1.0 - tau / 8.0, 0, 1))
        a_i = float(np.clip((tau - 52.0) / 8.0, 0, 1))
        ax3.clear(); ax2.clear()
        _draw3d(ax3, tau, a_sh, a_i)
        _draw_newman(ax2, tau, a_i)
        head.set_text("Ethane: internal rotation as a descent in symmetry")
        foot.set_text(f"dihedral {tau:.0f}\u00b0   \u2014   {conf}   \u2014   "
                      f"{pg}, order {order}   \u2014   {elems}")
        return []

    out = os.path.join(outdir, "ethane_conformers.gif")
    FuncAnimation(fig, update, frames=len(plan), interval=1000 / S.FPS,
                  blit=False).save(out, writer=S.PlayOnceWriter(fps=S.FPS))
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out, exist_ok=True)
    for tau, name in [(0.0, "eclipsed"), (30.0, "skew"), (60.0, "staggered")]:
        still(tau, name, out, labelled=True)
        still(tau, name, out, labelled=False)
    animation(out)
