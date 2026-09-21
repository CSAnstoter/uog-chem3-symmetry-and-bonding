"""
ml6_gifs.py
===========
Octahedral (Oh) additions to the symmetry-operation GIF engine.

This file does not stand alone --- it imports symmetry_operation_gifs.py and
adds an octahedral ML6 complex to it, so keep the two side by side.

WHAT IT MAKES
-------------
  * one GIF per Oh operation class, with the unmoved ligands ringed in green
    so the contribution to the character can be counted off the figure
  * a C3-axis reveal, showing one axis then the other three
  * a C4-axis reveal, and the same for the two C2 classes, sigma_d and S4
  * a "C3 through two opposite faces" figure, drawn as a Newman projection
    alongside the 3D view --- the octahedron seen down a C3 axis is exactly
    the staggered-ethane picture
  * an assemble-and-reveal animation of the octahedron and the complex inside it
  * the a1g / eg / t1u M-L stretching modes

RUNNING IT
----------
    python ml6_gifs.py              # everything, into the current folder
    python ml6_gifs.py my_output    # everything, into my_output/

Or import it and call one renderer at a time:

    import ml6_gifs as M
    import symmetry_operation_gifs as S
    S.render("ml6", "sigma_d", outdir=".", mark_unmoved=True,
             unmoved_pool=range(1, 7), track_with_unmoved=True)
    M.render_ml6_c3_axes(".")

MAKING CHANGES
--------------
Colours, atom sizes, animation speed and resolution all live in the SETTINGS
block of symmetry_operation_gifs.py and apply here too --- this file has no
separate copy of them. What IS local is in the SETTINGS block below: the M-L
distance, the camera angle, and which ligand is followed in crimson.
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import symmetry_operation_gifs as S

d = np.deg2rad                # shorthand: d(120) is 120 degrees in radians

# =====================================================================
#  SETTINGS  --  local to this file; the rest are in the engine
# =====================================================================
R = 2.0          # metal-ligand distance. Everything else scales with it.
VIEW = (20, -55) # default camera: (elevation, azimuth) in degrees.
                 # elevation is how far above the equator you are looking from;
                 # azimuth is how far round. Change these if an atom is hidden.
TRACK = 1        # which ligand is drawn crimson and followed. 1 is the +x
                 # ligand. Ligands are numbered 1..6 as +x -x +y -y +z -z,
                 # matching phi_1..phi_6 in the lecture notes. 0 is the metal.
TRACK_C4 = 1     # the ligand followed in the C4 reveal. It must be one that
                 # actually moves under the first C4 axis, or nothing is visible.
# =====================================================================


# =====================================================================
#  GEOMETRY
# =====================================================================
def ml6():
    """M at the origin; six equivalent L on the Cartesian axes.
    Index order matches the phi_1..phi_6 of the projection table:
    1:+x  2:-x  3:+y  4:-y  5:+z  6:-z"""
    L = np.array([[R, 0, 0], [-R, 0, 0], [0, R, 0],
                  [0, -R, 0], [0, 0, R], [0, 0, -R]], float)
    return (np.vstack([[0, 0, 0], L]), ["A"] + ["L"] * 6,
            [(0, i) for i in range(1, 7)], 1)


S.MASS.update({"A": 60.0, "L": 20.0})
S.DISPLAY["ml6"] = "ML$_6$"

_C3 = [np.array(v, float) for v in ([1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1])]
_FACES = {            # each C3 axis pierces these two opposite triangular faces
    0: ([1, 3, 5], [2, 4, 6]),
    1: ([1, 4, 6], [2, 3, 5]),
    2: ([2, 3, 6], [1, 4, 5]),
    3: ([2, 4, 5], [1, 3, 6]),
}



def _un(n_lig, extra=""):
    return (f"green ring = unmoved by this operation \u2192 counts toward \u03c7  "
            f"({n_lig} of the 6 ligands{extra})")


S.OPS["ml6"] = {
 "E":       dict(mode="rot", axis=[0, 0, 1], angle=0, view=VIEW, elem=("none", None),
                 label="$E$ \u2014 identity",
                 unmoved_note=_un(6) + " \u2014 contribution 6"),
 "C3":      dict(mode="rot", axis=[1, 1, 1], angle=120, view=(24, -50), elem=("axis", [1, 1, 1]),
                 label="$8C_3$ \u2014 120\u00b0 about a body diagonal (through two opposite faces)",
                 unmoved_note=_un(0) + " \u2014 no ligand lies on a $C_3$ axis, so the contribution is 0"),
 "C2p":     dict(mode="rot", axis=[1, 1, 0], angle=180, view=(22, -52), elem=("axis", [1, 1, 0]),
                 label="$6C_2'$ \u2014 180\u00b0 about an axis through two opposite edge midpoints",
                 unmoved_note=_un(0) + " \u2014 the axis misses every ligand, so the contribution is 0"),
 "C4":      dict(mode="rot", axis=[0, 0, 1], angle=90, view=VIEW, elem=("axis", [0, 0, 1]),
                 label="$6C_4$ \u2014 90\u00b0 about an axis through two trans ligands",
                 unmoved_note=_un(2) + " \u2014 the two on the axis; contribution 2"),
 "C2":      dict(mode="rot", axis=[0, 0, 1], angle=180, view=VIEW, elem=("axis", [0, 0, 1]),
                 label="$3C_2$ ($=C_4^2$) \u2014 180\u00b0 about the same axis",
                 unmoved_note=_un(2) + " \u2014 the two on the axis; contribution 2"),
 "i":       dict(mode="inv", view=(22, -52), elem=("point", None),
                 label="$i$ \u2014 inversion through the metal",
                 unmoved_note=_un(0, ", but the metal is on the centre and never moves")
                              + " \u2014 every ligand swaps with its trans partner; contribution 0"),
 "S4":      dict(mode="improper", axis=[0, 0, 1], angle=90, view=VIEW, elem=("axis", [0, 0, 1]),
                 label="$6S_4$ \u2014 rotate 90\u00b0, then reflect $\\perp$ to that axis",
                 unmoved_note=_un(0) + " \u2014 the reflection swaps the two axial ligands; contribution 0"),
 "S6":      dict(mode="improper", axis=[1, 1, 1], angle=60, view=(24, -50), elem=("axis", [1, 1, 1]),
                 label="$8S_6$ \u2014 rotate 60\u00b0 about a body diagonal, then reflect $\\perp$",
                 unmoved_note=_un(0) + " \u2014 contribution 0"),
 "sigma_h": dict(mode="refl", normal=[0, 0, 1], view=(16, -55), elem=("plane", [0, 0, 1]),
                 label="$3\\sigma_h$ \u2014 a plane containing four ligands",
                 unmoved_note=_un(4) + " \u2014 the four in the plane; contribution 4"),
 "sigma_d": dict(mode="refl", normal=[1, -1, 0], view=(20, -52), elem=("plane", [1, -1, 0]),
                 label="$6\\sigma_d$ \u2014 a diagonal plane containing two trans ligands",
                 unmoved_note=_un(2) + " \u2014 the two in the plane; contribution 2"),
}

S.MOLECULES["ml6"] = dict(geom=ml6, lim=2.55,
                          numbers=[1, 2, 3, 4, 5, 6],
                          curated=list(S.OPS["ml6"].keys()))


# ---------------------------------------------------------- normal modes
def _radial():
    P, _, _, _ = ml6()
    return np.array([S._u(P[i]) for i in range(1, 7)])


def ml6_modes():
    rad = _radial()
    z = np.zeros(3)

    def build(coeffs):
        return np.vstack([z] + [c * rad[i] for i, c in enumerate(coeffs)])

    # coefficient order follows the geometry: +x -x +y -y +z -z
    return {
      "a1g": dict(disp=build([1, 1, 1, 1, 1, 1]), amp=0.30, view=VIEW,
                  label="$a_{1g}$ \u2014 symmetric stretch ($\\nu_1$)",
                  note="all six M\u2013L bonds stretch together \u2014 the breathing mode"),
      "eg_z2": dict(disp=build([-1, -1, -1, -1, 2, 2]), amp=0.26, view=VIEW,
                  label="$e_g$ \u2014 stretch, $z^2$ partner ($\\nu_2$)",
                  note="the axial pair lengthens as the equatorial four contract"),
      "eg_x2y2": dict(disp=build([1, 1, -1, -1, 0, 0]), amp=0.30, view=(26, -55),
                  label="$e_g$ \u2014 stretch, $x^2-y^2$ partner ($\\nu_2$)",
                  note="the second, orthogonal partner of the same $e_g$ pair"),
      "t1u_z": dict(disp=np.vstack([[0, 0, -0.6667], z, z, z, z,
                                     [0, 0, 1.0], [0, 0, 1.0]]), amp=0.30,
                  view=(16, -55),
                  label="$t_{1u}$ \u2014 antisymmetric stretch, $z$ partner ($\\nu_3$)",
                  note="the metal recoils \u2014 the dipole changes, so $t_{1u}$ is IR-active"),
    }


S.MODES["ml6"] = ml6_modes()
S.MODE_VIEW["ml6"] = VIEW
S.MODE_LIM["ml6"] = 2.55


# --------------------------------------------------- custom: the 8C3 reveal
def _tri(ax, idx, P, colour, alpha, z=1):
    C = np.array([P[i] for i in idx])
    ax.add_collection3d(Poly3DCollection([C], facecolors=[to_rgba(colour, alpha)],
                                         edgecolors=[to_rgba(colour, min(1, alpha * 2.6))],
                                         linewidths=1.4, zorder=z))


def _axis_line(ax, v, lim, alpha, colour=None, lw=2.2):
    if alpha <= 0.02:
        return
    v = S._u(v) * lim * 1.18
    ax.plot(*zip(-v, v), color=colour or S.AXIS_C, lw=lw, alpha=alpha, zorder=6)


def render_ml6_c3_axes(outdir="."):
    P0, elems, bonds, _ = ml6()
    lim = 2.45
    segs = [("intro", 12), ("axis1", 14), ("spin", 22), ("hold1", 8),
            ("axis2", 11), ("axis3", 11), ("axis4", 11), ("hold_all", 22)]
    names = [nm for nm, _ in segs]
    i_spin = names.index("spin")
    plan = []
    for si, (name, n) in enumerate(segs):
        for k in range(n):
            plan.append((name, k / max(1, n - 1), si > i_spin))
    NF = len(plan)

    fig = plt.figure(figsize=(5, 5), dpi=110)
    ax = fig.add_subplot(111, projection="3d")
    title = "ML$_6$:  the four $C_3$ axes of $O_h$"

    def update(k):
        name, t, after_spin = plan[k]
        ax.clear()
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
        ax.view_init(elev=24, azim=-50)

        # how many axes are showing, and with what opacity
        alphas = [0.0] * 4
        # the turn is a symmetry operation, so once it has run we STAY in the new
        # arrangement: snapping back would undo the point and make the tracked
        # ligand jump
        spin = 1.0 if after_spin else 0.0
        if name == "intro":
            note = "An octahedron has eight triangular faces."
        elif name == "axis1":
            alphas[0] = S.ease(np.array([t]))[0]
            note = "One $C_3$ axis runs through the centres of two opposite faces."
        elif name == "spin":
            alphas[0] = 1.0
            spin = S.ease(np.array([t]))[0]
            note = "A 120\u00b0 turn about it maps the octahedron onto itself."
        elif name == "hold1":
            alphas[0] = 1.0
            note = "No ligand sits on the axis \u2014 all six move."
        elif name == "axis2":
            alphas[0] = 1.0; alphas[1] = S.ease(np.array([t]))[0]
            note = "\u2026and there are three more, one per pair of opposite faces."
        elif name == "axis3":
            alphas[:2] = [1.0, 1.0]; alphas[2] = S.ease(np.array([t]))[0]
            note = "\u2026and there are three more, one per pair of opposite faces."
        elif name == "axis4":
            alphas[:3] = [1.0] * 3; alphas[3] = S.ease(np.array([t]))[0]
            note = "\u2026and there are three more, one per pair of opposite faces."
        else:
            alphas = [1.0] * 4
            note = "4 axes \u00d7 2 rotations each ($C_3$ and $C_3^2$)  =  the 8$C_3$ class."

        P = (S.Rmat(_C3[0], d(120) * spin) @ P0.T).T if spin > 0 else P0

        # the two faces belonging to each revealed axis
        for a in range(4):
            if alphas[a] > 0.02:
                for face in _FACES[a]:
                    _tri(ax, face, P, S.AXIS_C if a == 0 else "#7a8ea8",
                         0.16 * alphas[a])
        for a in range(4):
            _axis_line(ax, _C3[a], lim, alphas[a],
                       colour=S.AXIS_C if a == 0 else "#5c7a9e",
                       lw=2.4 if a == 0 else 1.8)

        for i, j in bonds:
            ax.plot(*zip(P[i], P[j]), color=S.BOND_C, lw=3.0, alpha=0.95, zorder=2)
        for idx, e in enumerate(elems):
            st = S.TRACK if idx == TRACK else S.STYLE[e]
            ax.scatter(*P[idx], color=st["color"], s=S.STYLE[e]["s"],
                       edgecolor=st["edge"], linewidth=0.8, depthshade=True, zorder=4)

        ax.set_title(title, fontsize=11.5, pad=2)
        ax.text2D(0.5, 0.015, S._wrap(note, 60), transform=ax.transAxes, ha="center",
                  va="bottom", fontsize=8.5, style="italic", color="#666")
        return []

    out = f"{outdir}/ml6_C3_axes.gif"
    FuncAnimation(fig, update, frames=NF, interval=1000 / S.FPS, blit=False).save(
        out, writer=S.PlayOnceWriter(fps=S.FPS))
    plt.close(fig); print("wrote", out)


# ------------------------------------------- custom: assemble and reveal
_OCTA_FACES = [[1, 3, 5], [3, 2, 5], [2, 4, 5], [4, 1, 5],
               [1, 3, 6], [3, 2, 6], [2, 4, 6], [4, 1, 6]]


def render_ml6_net(outdir="."):
    P0, elems, bonds, _ = ml6()
    lim = 2.95
    segs = [("apart", 10), ("close", 24), ("solid", 10), ("reveal", 22), ("hold", 18)]
    plan = []
    for name, n in segs:
        for k in range(n):
            plan.append((name, k / max(1, n - 1)))
    NF = len(plan)
    normals = [S._u(np.mean([P0[i] for i in f], axis=0)) for f in _OCTA_FACES]

    fig = plt.figure(figsize=(5, 5), dpi=110)
    ax = fig.add_subplot(111, projection="3d")

    def update(k):
        name, t = plan[k]
        ax.clear()
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
        ax.view_init(elev=22, azim=-55 + 40 * (k / NF))

        if name == "apart":
            explode, face_a, mol_a = 1.0, 0.55, 0.0
            note = "Eight faces, one for every triangle of three mutually cis ligands."
        elif name == "close":
            e = S.ease(np.array([t]))[0]
            explode, face_a, mol_a = 1.0 - e, 0.55, 0.0
            note = "Close them up and the octahedron appears."
        elif name == "solid":
            explode, face_a, mol_a = 0.0, 0.55, 0.0
            note = "The bare polyhedron \u2014 six vertices, eight faces, twelve edges."
        elif name == "reveal":
            e = S.ease(np.array([t]))[0]
            explode, face_a, mol_a = 0.0, 0.55 - 0.40 * e, e
            note = "Every vertex is a ligand; the metal sits at the centre."
        else:
            explode, face_a, mol_a = 0.0, 0.08, 1.0
            note = "ML$_6$ \u2014 point group $O_h$, 48 operations in 10 classes."

        wire = mol_a > 0.35          # once the molecule shows, edges only: filled
                                     # faces are depth-sorted in front of the spheres
        for f, n_hat in zip(_OCTA_FACES, normals):
            C = np.array([P0[i] + explode * 1.25 * n_hat for i in f])
            if wire:
                loop = np.vstack([C, C[0]])
                ax.plot(loop[:, 0], loop[:, 1], loop[:, 2],
                        color="#3f556e", lw=1.1, alpha=0.45, zorder=1)
            else:
                ax.add_collection3d(Poly3DCollection(
                    [C], facecolors=[to_rgba("#7a8ea8", 0.30 * face_a / 0.55)],
                    edgecolors=[to_rgba("#3f556e", 0.95 * face_a / 0.55)],
                    linewidths=1.5, zorder=1))

        if mol_a > 0.02:
            for i, j in bonds:
                ax.plot(*zip(P0[i], P0[j]), color=S.BOND_C, lw=3.0, alpha=0.95 * mol_a, zorder=3)
            for idx, e in enumerate(elems):
                st = S.TRACK if idx == TRACK else S.STYLE[e]
                ax.scatter(*P0[idx], color=to_rgba(st["color"], mol_a),
                           s=S.STYLE[e]["s"],
                           edgecolor=to_rgba(st["edge"], mol_a), linewidth=0.8,
                           depthshade=True, zorder=5)

        ax.set_title("ML$_6$:  the octahedron and the complex inside it",
                     fontsize=11.5, pad=2)
        ax.text2D(0.5, 0.015, S._wrap(note, 60), transform=ax.transAxes, ha="center",
                  va="bottom", fontsize=8.5, style="italic", color="#666")
        return []

    out = f"{outdir}/ml6_octahedron_reveal.gif"
    FuncAnimation(fig, update, frames=NF, interval=1000 / S.FPS, blit=False).save(
        out, writer=S.PlayOnceWriter(fps=S.FPS))
    plt.close(fig); print("wrote", out)


# --------------------------------------------------------------------- main



# ------------------------------------------- custom: the 6C4 class reveal
_C4 = [np.array(v, float) for v in ([0, 0, 1], [1, 0, 0], [0, 1, 0])]
_ON_AXIS = {0: (5, 6), 1: (1, 2), 2: (3, 4)}      # the trans pair each axis passes through
_SQUARE = {0: (1, 3, 2, 4), 1: (3, 5, 4, 6), 2: (1, 5, 2, 6)}   # the four that cycle


def render_ml6_c4_axes(outdir="."):
    """One C4 axis, the 90-degree turn it generates, then the other two.

    NB the class is 6C4 but there are only THREE axes: each carries two distinct
    rotations, 90 and 270 degrees. The same three lines are also the 3C2 axes,
    since C4 squared is C2."""
    P0, elems, bonds, _ = ml6()
    lim = 2.45
    segs = [("intro", 12), ("axis1", 13), ("spin", 22), ("hold1", 10),
            ("axis2", 12), ("axis3", 12), ("hold_all", 24)]
    names = [nm for nm, _ in segs]
    i_spin = names.index("spin")
    plan = [(name, k / max(1, n - 1), si > i_spin)
            for si, (name, n) in enumerate(segs) for k in range(n)]
    NF = len(plan)

    fig = plt.figure(figsize=(5, 5), dpi=110)
    ax = fig.add_subplot(111, projection="3d")

    def update(k):
        name, t, after_spin = plan[k]
        ax.clear()
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
        ax.view_init(elev=20, azim=-55)

        alphas = [0.0, 0.0, 0.0]
        # hold the rotated arrangement after the turn, so the ligands are ringed
        # where the operation actually left them
        spin = 1.0 if after_spin else 0.0
        ring = False
        if name == "intro":
            note = "Each $C_4$ axis runs through a pair of trans ligands."
        elif name == "axis1":
            alphas[0] = S.ease(np.array([t]))[0]
            note = "One $C_4$ axis: the four ligands around it form a square."
        elif name == "spin":
            alphas[0] = 1.0; spin = S.ease(np.array([t]))[0]
            note = "A 90\u00b0 turn cycles those four and leaves the axial pair in place."
        elif name == "hold1":
            alphas[0] = 1.0; ring = True
            note = "Two of the six ligands unmoved \u2014 the contribution to \u03c7 is 2."
        elif name == "axis2":
            alphas[0] = 1.0; alphas[1] = S.ease(np.array([t]))[0]
            note = "\u2026and there are two more, one per pair of trans ligands."
        elif name == "axis3":
            alphas[:2] = [1.0, 1.0]; alphas[2] = S.ease(np.array([t]))[0]
            note = "\u2026and there are two more, one per pair of trans ligands."
        else:
            alphas = [1.0] * 3
            note = ("3 axes \u00d7 2 rotations each ($C_4$ and $C_4^3$) = the 6$C_4$ class. "
                    "The same three lines carry the 3$C_2$ class, since $C_4^2 = C_2$.")

        P = (S.Rmat(_C4[0], d(90) * spin) @ P0.T).T if spin > 0 else P0

        for a in range(3):
            if alphas[a] > 0.02:
                q = np.array([P[i] for i in _SQUARE[a]])
                ax.add_collection3d(Poly3DCollection(
                    [q], facecolors=[to_rgba(S.AXIS_C if a == 0 else "#5c7a9e",
                                             0.13 * alphas[a])],
                    edgecolors=[to_rgba(S.AXIS_C if a == 0 else "#5c7a9e",
                                        0.45 * alphas[a])],
                    linewidths=1.2, zorder=1))
        for a in range(3):
            _axis_line(ax, _C4[a], lim, alphas[a],
                       colour=S.AXIS_C if a == 0 else "#5c7a9e",
                       lw=2.4 if a == 0 else 1.8)

        for i, j in bonds:
            ax.plot(*zip(P[i], P[j]), color=S.BOND_C, lw=3.0, alpha=0.95, zorder=2)
        for idx, e in enumerate(elems):
            st = S.TRACK if idx == TRACK_C4 else S.STYLE[e]
            ax.scatter(*P[idx], color=st["color"], s=S.STYLE[e]["s"],
                       edgecolor=st["edge"], linewidth=0.8, depthshade=True, zorder=4)
        if ring:
            for idx in _ON_AXIS[0]:
                ax.scatter(*P[idx], s=S.STYLE["L"]["s"] * 2.7, facecolors="none",
                           edgecolors="#2E8B57", linewidths=2.6,
                           depthshade=False, zorder=6)

        ax.set_title("ML$_6$:  the three $C_4$ axes of $O_h$", fontsize=11.5, pad=2)
        ax.text2D(0.5, 0.015, S._wrap(note, 62), transform=ax.transAxes, ha="center",
                  va="bottom", fontsize=8.5, style="italic", color="#666")
        return []

    out = f"{outdir}/ml6_C4_axes.gif"
    FuncAnimation(fig, update, frames=NF, interval=1000 / S.FPS, blit=False).save(
        out, writer=S.PlayOnceWriter(fps=S.FPS))
    plt.close(fig); print("wrote", out)


# ------------------- custom: C3 acting through two opposing triangular faces
# Looking down a C3 axis, ML6 projects to exactly the staggered-ethane Newman
# picture: a near triangle of three ligands at 0/120/240 and a far triangle at
# 60/180/300. The 120-degree turn carries each triangle onto itself.
_NEAR = [(1, 0.0), (3, 120.0), (5, 240.0)]        # (atom index, Newman angle)
_FAR = [(6, 60.0), (2, 180.0), (4, 300.0)]
TRACK_C, TRACK_E = "#d11f4a", "#7a1020"   # crimson fill / darker edge


def _newman_ml6(ax, spin_deg, cr=0.36, r=1.05):
    """Flat Newman view down the C3 axis. Near-face bonds run to the centre, far-face
    bonds from the circle edge, as in the ethane S6 figure."""
    import matplotlib.pyplot as _plt
    ax.add_patch(_plt.Circle((0, 0), cr, fill=False, ec="#555", lw=1.7, zorder=2))
    for grp, front in ((_FAR, 0), (_NEAR, 1)):
        pts = []
        for idx, a0 in grp:
            th = np.radians(a0 + spin_deg)
            dirn = np.array([np.cos(th), np.sin(th)])
            pts.append(r * dirn)
        poly = np.array(pts)
        ax.add_patch(_plt.Polygon(poly, closed=True,
                                  facecolor=to_rgba(S.AXIS_C if front else "#5c7a9e",
                                                    0.16 if front else 0.09),
                                  edgecolor=to_rgba(S.AXIS_C if front else "#5c7a9e",
                                                    0.85 if front else 0.45),
                                  lw=1.8 if front else 1.4,
                                  ls="-" if front else (0, (4, 3)), zorder=1 + front))
        for (idx, a0), pos in zip(grp, pts):
            inner = (1 - front) * cr * pos / np.linalg.norm(pos)
            ax.plot([inner[0], pos[0]], [inner[1], pos[1]], color="#6f6f6f",
                    lw=2.4 + 0.9 * front, alpha=0.55 + 0.45 * front,
                    zorder=3 + front, solid_capstyle="round")
            trk = (idx == TRACK)
            ax.scatter(*pos, s=340 + 130 * front,
                       color=TRACK_C if trk else S.STYLE["L"]["color"],
                       edgecolor=TRACK_E if trk else S.STYLE["L"]["edge"],
                       lw=1.0, zorder=5 + front)
    ax.scatter(0, 0, s=150, color=S.STYLE["A"]["color"],
               edgecolor=S.STYLE["A"]["edge"], lw=1.0, zorder=7)


VIEW3 = (0, 10)     # well off the C3 axis, so the two faces read as faces;
                    # the right-hand panel is the view straight down it


def render_ml6_c3_faces(outdir="."):
    P0, elems, bonds, _ = ml6()
    axis = _C3[0]
    lim3, lim2 = 2.35, 1.5
    segs = [("start", 14), ("spin", 26), ("done", 22)]
    plan = [(name, k / max(1, n - 1)) for name, n in segs for k in range(n)]
    NF = len(plan)

    fig = plt.figure(figsize=(9.2, 4.9), dpi=110)
    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)
    fig.suptitle("ML$_6$:  $C_3$ through two opposite faces", fontsize=12.5, y=0.97)
    # created once: fig.text inside the update loop would stack up every frame
    note_txt = fig.text(0.5, 0.045, "", ha="center", va="bottom",
                        fontsize=9, style="italic", color="#666")

    def update(k):
        name, t = plan[k]
        spin = 0.0 if name == "start" else (S.ease(np.array([t]))[0] if name == "spin" else 1.0)
        P = (S.Rmat(axis, d(120) * spin) @ P0.T).T

        ax3.clear()
        ax3.set_xlim(-lim3, lim3); ax3.set_ylim(-lim3, lim3); ax3.set_zlim(-lim3, lim3)
        ax3.set_box_aspect((1, 1, 1)); ax3.set_axis_off()
        ax3.view_init(*VIEW3)
        _axis_line(ax3, axis, lim3, 1.0, colour=S.AXIS_C, lw=2.4)
        for face, col, al in ((_FACES[0][0], S.AXIS_C, 0.20),
                              (_FACES[0][1], "#5c7a9e", 0.13)):
            _tri(ax3, face, P, col, al)
        for i, j in bonds:
            ax3.plot(*zip(P[i], P[j]), color=S.BOND_C, lw=3.0, alpha=0.95, zorder=2)
        for idx, e in enumerate(elems):
            st = S.STYLE[e]
            trk = (idx == TRACK)
            ax3.scatter(*P[idx], color=TRACK_C if trk else st["color"], s=st["s"],
                        edgecolor=TRACK_E if trk else st["edge"], linewidth=0.8,
                        depthshade=True, zorder=4)
        ax3.set_title("the two opposite faces, in 3D", fontsize=10, pad=0)

        ax2.clear()
        ax2.set_xlim(-lim2, lim2); ax2.set_ylim(-lim2, lim2)
        ax2.set_aspect("equal"); ax2.axis("off")
        _newman_ml6(ax2, 120.0 * spin)
        ax2.set_title("the same thing, straight down the axis", fontsize=10, pad=0)

        note = ("Two opposing triangular faces, staggered by 60\u00b0 \u2014 exactly the "
                "staggered-ethane Newman picture."
                if name == "start" else
                ("Turning 120\u00b0: each triangle maps onto itself, and the two never mix."
                 if name == "spin" else
                 "Back to an indistinguishable arrangement. No ligand stayed put, so "
                 "$8C_3$ contributes 0 to \u03c7."))
        note_txt.set_text(S._wrap(note, 96))
        return []

    out = f"{outdir}/ml6_C3_faces.gif"
    FuncAnimation(fig, update, frames=NF, interval=1000 / S.FPS, blit=False).save(
        out, writer=S.PlayOnceWriter(fps=S.FPS))
    plt.close(fig); print("wrote", out)



# ---------- custom: run one operation, then reveal the rest of its class
# For the classes where the number of ELEMENTS differs from the number of
# operations, or where "and where are the others?" is not obvious by eye.
_CLASS = {
    "C2":  dict(elem="axis", run="Turning 180\u00b0 about it.",
                vecs=[[0, 0, 1], [1, 0, 0], [0, 1, 0]],
                tail="3 axes, one per pair of trans ligands \u2014 the same three lines "
                     "that carry the 6$C_4$ class, since $C_2 = C_4^2$."),
    "C2p": dict(elem="axis", run="Turning 180\u00b0 about it.",
                hold="No ligand is left in place \u2014 the axis passes between them, "
                     "not through them. Contribution to \u03c7 is 0.",
                vecs=[[1, 1, 0], [1, -1, 0], [1, 0, 1],
                      [1, 0, -1], [0, 1, 1], [0, 1, -1]],
                tail="6 axes, one per pair of opposite edge midpoints. None passes "
                     "through a ligand, which is why 6$C_2'$ contributes 0 to \u03c7."),
    "sigma_d": dict(elem="plane", run="Reflecting in it.", orbit=True,
                    vecs=[[1, -1, 0], [1, 1, 0], [1, 0, -1],
                          [1, 0, 1], [0, 1, -1], [0, 1, 1]],
                    groups=[((0, 1), "The first two both contain the $\\pm z$ ligands, "
                                     "crossing at right angles."),
                            ((2, 3), "Two more contain the $\\pm y$ ligands."),
                            ((4, 5), "And two contain the $\\pm x$ ligands.")],
                    group_colours=["#c79a3a", "#4f8a7b", "#8a6fa8"],
                    tail="Three pairs, one pair per C$_4$ axis: 6 planes in all. "
                         "Each holds a pair of trans ligands, bisects the other two "
                         "pairs, and is perpendicular to one of the 6$C_2\'$ axes."),
    "S4":  dict(elem="axis", perp_plane=True,
                hold="Nothing is left in place: the turn moves the four equatorial "
                     "ligands, and the reflection swaps the two axial ones. "
                     "Contribution to \u03c7 is 0.",
                run="Turning 90\u00b0, then reflecting in the plane perpendicular "
                    "to the axis \u2014 one operation, not two.",
                vecs=[[0, 0, 1], [1, 0, 0], [0, 1, 0]],
                tail="3 axes, each carrying two improper rotations ($S_4$ and "
                     "$S_4^3$). They are the very same three lines as the 6$C_4$ "
                     "and 3$C_2$ classes \u2014 one set of axes, three classes."),
}


def _plane_patch(ax, normal, lim, alpha, filled=True, colour=None):
    """One mirror plane, in the engine's plane colours, with a fadeable alpha."""
    if alpha <= 0.02:
        return
    n = np.asarray(normal, float); n = n / np.linalg.norm(n)
    a = np.array([1., 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1., 0])
    u = a - (a @ n) * n; u /= np.linalg.norm(u)
    v = np.cross(n, u); g = lim * 0.86
    sq = [u * g + v * g, u * g - v * g, -u * g - v * g, -u * g + v * g]
    face = colour or S.PLANE_C
    edge = colour or S.PLANE_EDGE
    if filled:
        ax.add_collection3d(Poly3DCollection(
            [np.array(sq)], facecolor=face, alpha=0.20 * alpha,
            edgecolor="none", zorder=0))
    loop = np.array(sq + [sq[0]])
    ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], color=edge,
            lw=1.6 if filled else 1.1, alpha=(0.85 if filled else 0.55) * alpha,
            zorder=1)


def render_ml6_class(kind, outdir="."):
    """The operation, a hold showing which ligands it left in place, then the
    remaining equivalent elements of the class fading in one at a time."""
    assert kind in _CLASS
    spec = _CLASS[kind]
    op = S.OPS["ml6"][kind]
    vecs = [np.array(v, float) for v in spec["vecs"]]
    P0, elems, bonds, _ = ml6()
    lim = 2.45
    # which ligands the operation leaves alone: read off, not hard-coded
    P1 = S.frame_coords(op, P0, 1.0)
    unmoved = [i for i in range(1, 7) if np.linalg.norm(P1[i] - P0[i]) < 1e-6]

    groups = spec.get("groups")
    n_extra = len(vecs) - 1
    n_rev = len(groups) * 14 if groups else 9 * n_extra
    segs = [("start", 12), ("run", 22), ("hold", 12),
            ("reveal", n_rev), ("hold_all", 30 if spec.get("orbit") else 22)]
    names = [nm for nm, _ in segs]
    i_run = names.index("run")
    plan = [(name, k / max(1, n - 1), si > i_run)
            for si, (name, n) in enumerate(segs) for k in range(n)]
    NF = len(plan)

    fig = plt.figure(figsize=(5, 5), dpi=110)
    ax = fig.add_subplot(111, projection="3d")
    note_txt = fig.text(0.5, 0.015, "", ha="center", va="bottom",
                        fontsize=8.5, style="italic", color="#666")
    head = op["label"].split("\u2014")[0].strip()
    fig.suptitle("ML$_6$:  " + head + "  and the rest of its class",
                 fontsize=11.5, y=0.955)

    def update(k):
        name, t, after = plan[k]
        ax.clear()
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
        ev, az = op["view"]
        if spec.get("orbit") and name == "hold_all":
            az = az + 70.0 * t          # slow orbit: six planes only read in motion
            ev = ev + 14.0 * np.sin(np.pi * t)
        ax.view_init(ev, az)

        frac = S.ease(np.array([t]))[0] if name == "run" else (1.0 if after else 0.0)
        alphas = [1.0] + [0.0] * n_extra
        ring = False
        if name == "start":
            note = "One element of the class, with the operation about to run."
        elif name == "run":
            note = spec["run"]
        elif name == "hold":
            ring = True
            note = spec.get("hold") or op.get("unmoved_note", "")
        elif name == "reveal":
            if groups:
                ng = len(groups)
                gi = min(int(t * ng), ng - 1)
                note = groups[gi][1]
                for g, (idxs, _) in enumerate(groups):
                    lo, hi = g / ng, (g + 1) / ng
                    a_g = float(np.clip((t - lo) / (hi - lo), 0, 1))
                    for i_v in idxs:
                        alphas[i_v] = max(alphas[i_v], a_g)
            else:
                for a in range(n_extra):
                    lo, hi = a / n_extra, (a + 1) / n_extra
                    alphas[a + 1] = float(np.clip((t - lo) / (hi - lo), 0, 1))
                note = "\u2026and these are the others, all equivalent to the first."
            ring = True
        else:
            alphas = [1.0] * len(vecs)
            ring = True
            note = spec["tail"]

        P = S.frame_coords(op, P0, frac)

        if spec.get("perp_plane") and name in ("run", "hold"):
            _plane_patch(ax, vecs[0], lim, 0.75, filled=True)
        gcol = {}
        if groups:
            for g, (idxs, _) in enumerate(groups):
                for i_v in idxs:
                    gcol[i_v] = spec["group_colours"][g % len(spec["group_colours"])]
        for a, v in enumerate(vecs):
            if spec["elem"] == "plane":
                _plane_patch(ax, v, lim, alphas[a],
                             filled=(a == 0 or (groups and 0.02 < alphas[a] < 0.999)),
                             colour=gcol.get(a))
            else:
                _axis_line(ax, v, lim, alphas[a],
                           colour=S.AXIS_C if a == 0 else "#5c7a9e",
                           lw=2.4 if a == 0 else 1.7)
        for i, j in bonds:
            ax.plot(*zip(P[i], P[j]), color=S.BOND_C, lw=3.0, alpha=0.95, zorder=2)
        for idx, e in enumerate(elems):
            st = S.TRACK if idx == TRACK else S.STYLE[e]
            ax.scatter(*P[idx], color=st["color"], s=S.STYLE[e]["s"],
                       edgecolor=st["edge"], linewidth=0.8, depthshade=True, zorder=4)
        if ring:
            for idx in unmoved:
                ax.scatter(*P[idx], s=S.STYLE["L"]["s"] * 2.7, facecolors="none",
                           edgecolors="#2E8B57", linewidths=2.6,
                           depthshade=False, zorder=6)

        note_txt.set_text(S._wrap(note, 62))
        return []

    out = f"{outdir}/ml6_{kind}_class.gif"
    FuncAnimation(fig, update, frames=NF, interval=1000 / S.FPS, blit=False).save(
        out, writer=S.PlayOnceWriter(fps=S.FPS))
    plt.close(fig); print("wrote", out)


# ------------------- custom: the 6 sigma_d planes, made legible
# Six translucent planes in one frame is an unreadable tangle. They come in three
# pairs, one pair per C4 axis, and a pair is obvious when you look straight down
# that axis: the two planes are then the diagonals of the square of equatorial
# ligands. So the sequence builds one pair, turns the camera down the axis to make
# the geometry plain, and only then adds the other two pairs.
SD_VECS = [[1, -1, 0], [1, 1, 0],          # contain the +-z ligands
           [1, 0, -1], [1, 0, 1],          # contain the +-y ligands
           [0, 1, -1], [0, 1, 1]]          # contain the +-x ligands
SD_COL = ["#c79a3a", "#4f8a7b", "#8a6fa8"]


def render_ml6_sigma_d_class(outdir="."):
    op = S.OPS["ml6"]["sigma_d"]
    vecs = [np.array(v, float) for v in SD_VECS]
    P0, elems, bonds, _ = ml6()
    lim = 2.45
    P1 = S.frame_coords(op, P0, 1.0)
    unmoved = [i for i in range(1, 7) if np.linalg.norm(P1[i] - P0[i]) < 1e-6]

    segs = [("start", 12), ("run", 20), ("hold", 12), ("pair", 12),
            ("turn", 18), ("down", 24), ("back", 16), ("others", 26),
            ("orbit", 30)]
    names = [nm for nm, _ in segs]
    i_run = names.index("run")
    plan = [(nm, k / max(1, n - 1), si > i_run)
            for si, (nm, n) in enumerate(segs) for k in range(n)]

    OBL = op["view"]              # the oblique view the other figures use
    DOWN = (86, -90)              # straight down the C4 (z) axis

    fig = plt.figure(figsize=(5, 5), dpi=110)
    ax = fig.add_subplot(111, projection="3d")
    note_txt = fig.text(0.5, 0.015, "", ha="center", va="bottom",
                        fontsize=8.5, style="italic", color="#666")
    fig.suptitle("ML$_6$:  6$\\sigma_d$  and the rest of its class",
                 fontsize=11.5, y=0.955)

    def update(k):
        nm, t, after = plan[k]
        ax.clear()
        ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()

        e = S.ease(np.array([t]))[0]
        frac = e if nm == "run" else (1.0 if after else 0.0)
        alphas = [0.0] * 6
        alphas[0] = 1.0
        view = OBL
        ring = False
        note = ""

        if nm == "start":
            note = "One of the six planes, with the reflection about to run."
        elif nm == "run":
            note = "Reflecting in it."
        elif nm == "hold":
            ring = True
            note = op.get("unmoved_note", "")
        elif nm == "pair":
            alphas[1] = e
            note = ("A second plane also holds the $\\pm z$ ligands, at right angles "
                    "to the first.")
        elif nm == "turn":
            alphas[1] = 1.0
            view = (OBL[0] + (DOWN[0] - OBL[0]) * e, OBL[1] + (DOWN[1] - OBL[1]) * e)
            note = "Turning to look straight down that $C_4$ axis\u2026"
        elif nm == "down":
            alphas[1] = 1.0
            view = DOWN
            note = ("\u2026and there they are: the two planes are the diagonals of the "
                    "square of equatorial ligands.")
        elif nm == "back":
            alphas[1] = 1.0
            view = (DOWN[0] + (OBL[0] - DOWN[0]) * e, DOWN[1] + (OBL[1] - DOWN[1]) * e)
            note = "Every C$_4$ axis has its own pair, exactly like this one."
        elif nm == "others":
            alphas[1] = 1.0
            alphas[2] = alphas[3] = float(np.clip(t / 0.5, 0, 1))
            alphas[4] = alphas[5] = float(np.clip((t - 0.5) / 0.5, 0, 1))
            note = ("Two more hold the $\\pm y$ ligands, and two hold the $\\pm x$."
                    if t > 0.5 else "Two more hold the $\\pm y$ ligands\u2026")
        else:
            alphas = [1.0] * 6
            view = (OBL[0] + 12.0 * np.sin(np.pi * t), OBL[1] + 80.0 * t)
            note = ("Three pairs, one per C$_4$ axis: 6 planes. Each holds a pair of "
                    "trans ligands and is perpendicular to one of the 6$C_2'$ axes.")

        ax.view_init(*view)
        P = S.frame_coords(op, P0, frac)

        for a, v in enumerate(vecs):
            if alphas[a] <= 0.02:
                continue
            col = SD_COL[a // 2]
            # fill only the pair being introduced; outlines keep six legible
            filled = (nm in ("start", "run", "hold", "pair", "turn", "down", "back")
                      or (nm == "others" and 0.02 < alphas[a] < 0.999))
            _plane_patch(ax, v, lim, alphas[a], filled=filled, colour=col)
        for i, j in bonds:
            ax.plot(*zip(P[i], P[j]), color=S.BOND_C, lw=3.0, alpha=0.95, zorder=2)
        for idx, el in enumerate(elems):
            st = S.TRACK if idx == TRACK else S.STYLE[el]
            ax.scatter(*P[idx], color=st["color"], s=S.STYLE[el]["s"],
                       edgecolor=st["edge"], linewidth=0.8, depthshade=True, zorder=4)
        if ring:
            for idx in unmoved:
                ax.scatter(*P[idx], s=S.STYLE["L"]["s"] * 2.7, facecolors="none",
                           edgecolors="#2E8B57", linewidths=2.6,
                           depthshade=False, zorder=6)
        note_txt.set_text(S._wrap(note, 62))
        return []

    out = f"{outdir}/ml6_sigma_d_class.gif"
    FuncAnimation(fig, update, frames=len(plan), interval=1000 / S.FPS,
                  blit=False).save(out, writer=S.PlayOnceWriter(fps=S.FPS))
    plt.close(fig); print("wrote", out)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    for op in S.OPS["ml6"]:
        S.render("ml6", op, outdir=out, mark_unmoved=True, unmoved_pool=range(1, 7),
                 track_with_unmoved=True)
    render_ml6_c3_axes(out)
    render_ml6_c4_axes(out)
    render_ml6_c3_faces(out)
    for kind in ("C2", "C2p", "S4"):
        render_ml6_class(kind, out)
    render_ml6_sigma_d_class(out)
    render_ml6_net(out)
    for m in S.MODES["ml6"]:
        S.render_mode("ml6", m, outdir=out)
