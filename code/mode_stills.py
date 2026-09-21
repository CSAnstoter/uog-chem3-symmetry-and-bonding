"""Unlabelled PNG stills of normal-mode displacements, for any molecule defined
in the GIF engine's MODES table.

Atoms sit at their equilibrium positions; red arrows give the displacement
direction and relative size, which is the conventional textbook figure and
avoids the ambiguity of a frozen mid-cycle frame. Displacements come from the
same MODES table the animations use, so the mode shapes are identical to the
GIFs, and the centre of mass is held fixed (hence the small recoil arrow on the
central atom).

Each mode produces two files:
    <mol>_<mode>.png         square, transparent, framed like the GIFs
    <mol>_<mode>_tight.png   transparent, cropped tight to the figure

Run:  python3 mode_stills.py [outdir] [molecule ...]
      python3 mode_stills.py .                 -> water, ammonia, co2
      python3 mode_stills.py . ammonia         -> just ammonia
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import symmetry_operation_gifs as S

VEC = "#b5341f"
FLOOR, SPAN = 0.22, 0.24      # shortest arrow, and the extra length of the longest
SIDE = 0.135                  # sideways shift, so an arrow never lies along a bond
STATIC = 0.05                 # below this fraction of the largest motion: no arrow

# how far from an atom's centre an arrow starts, so short arrows are not swallowed
EDGE = {"H": 0.155, "C": 0.205, "N": 0.210, "O": 0.200, "F": 0.170,
        "Cl": 0.185, "A": 0.225, "L": 0.180}

# a tighter frame than the animations use: a still needs no room for motion
LIM = {"water": 1.05, "ammonia": 1.22, "co2": 1.55, "ml6": 2.45}
# a still has no motion to help the eye, so ammonia gets a more elevated camera
# than the animations use, which reads the pyramid rather than a flat fan
VIEW = {"ammonia": (24, -30)}     # camera between two H, so all three stay visible

DEFAULT = ["water", "ammonia", "co2"]


def still(mol, key, outdir=".", dpi=220):
    P, elems, bonds, disp = S.mode_disp(mol, key)
    P = P - P.mean(axis=0)                                  # centre in frame
    centroid = np.zeros(3)
    lim = LIM.get(mol, S.MODE_LIM.get(mol, S.MOLECULES[mol]["lim"]))
    view = VIEW.get(mol, S.MODES[mol][key].get("view", S.MODE_VIEW.get(mol, (18, -66))))
    dmax = np.linalg.norm(disp, axis=1).max()

    fig = plt.figure(figsize=(5, 5), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
    ax.view_init(elev=view[0], azim=view[1])

    for i, j in bonds:
        ax.plot(*zip(P[i], P[j]), color=S.BOND_C, lw=3.6, alpha=0.95, zorder=2)
    for idx, e in enumerate(elems):
        st = S.STYLE[e]
        ax.scatter(*P[idx], color=st["color"], s=st["s"] * 1.9, edgecolor=st["edge"],
                   linewidth=1.0, depthshade=True, zorder=4)

    for i, e in enumerate(elems):
        n = np.linalg.norm(disp[i])
        if n / dmax < STATIC:
            continue
        u = disp[i] / n
        L = FLOOR + SPAN * (n / dmax)
        r = P[i] - centroid
        perp = r - u * float(np.dot(r, u))       # component of r normal to u
        pn = np.linalg.norm(perp)
        off = perp / pn * SIDE if pn > 0.25 else np.zeros(3)
        start = P[i] + u * EDGE.get(e, 0.18) + off
        ax.quiver(start[0], start[1], start[2], *(u * L),
                  color=VEC, lw=3.0, alpha=0.95, arrow_length_ratio=0.30, zorder=6)

    base = os.path.join(outdir, f"{mol}_{key}")
    fig.savefig(base + ".png", transparent=True, facecolor="none")
    plt.close(fig)

    im = Image.open(base + ".png").convert("RGBA")
    bb = im.split()[3].getbbox(); pad = 14
    im.crop((max(0, bb[0] - pad), max(0, bb[1] - pad),
             min(im.width, bb[2] + pad), min(im.height, bb[3] + pad))
            ).save(base + "_tight.png")
    print("wrote", base + ".png", "and", base + "_tight.png")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    mols = sys.argv[2:] or DEFAULT
    os.makedirs(out, exist_ok=True)
    for m in mols:
        if m == "ml6":
            import ml6_gifs            # registers the ML6 geometry and modes
        for k in S.MODES[m]:
            still(m, k, out)
