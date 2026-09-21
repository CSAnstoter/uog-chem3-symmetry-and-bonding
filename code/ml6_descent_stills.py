"""Descent in symmetry: ML6 and its substitution products.

    ML6            Oh    48 operations   C4, sigma_h and i all present
    ML5L'          C4v    8              C4 survives; sigma_h and i are lost
    trans-ML4L'2   D4h   16              C4, sigma_h and i all survive
    cis-ML4L'2     C2v    4              C4 is gone; only a C2 bisecting the
                                         two L' survives, and i with it

Each figure marks the highest-order rotation axis in orange, shades sigma_h where
it survives, and rings the metal where the molecule keeps its inversion centre.
That last mark is the one that matters for spectroscopy: i present means mutual
exclusion, i absent means IR and Raman bands can coincide.

Framed to match the other ML6 figures. Labelled and "_plain" variants.

Run:  python3 ml6_descent_stills.py [outdir]
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
import symmetry_operation_gifs as S

R = 1.42
LIM = 1.92
VIEW = (20, -52)
ORANGE = "#e8701a"
LP_C, LP_E = "#4f8a7b", "#2c5f4c"          # the substituted ligand L'
NAVY = "#1D3B5A"

LIG = np.array([[R, 0, 0], [-R, 0, 0], [0, R, 0],
                [0, -R, 0], [0, 0, R], [0, 0, -R]], float)

#            substituted ligands, principal axis, sigma_h?, i?, label, order
CASES = {
    "ML6":       dict(sub=[],     axis=[0, 0, 1], sh=True,  inv=True,
                      name="ML$_6$",            pg="$O_h$",    order=48),
    "ML5Lp":     dict(sub=[4],    axis=[0, 0, 1], sh=False, inv=False,
                      name="ML$_5$L$^{\\prime}$", pg="$C_{4v}$", order=8),
    "trans":     dict(sub=[4, 5], axis=[0, 0, 1], sh=True,  inv=True,
                      name="trans-ML$_4$L$^{\\prime}_2$", pg="$D_{4h}$", order=16),
    "cis":       dict(sub=[4, 0], axis=[1, 0, 1], sh=False, inv=False,
                      name="cis-ML$_4$L$^{\\prime}_2$", pg="$C_{2v}$", order=4),
}


def _frame(dpi, labelled=True):
    # the labelled variant reserves a band at the top for the two caption lines;
    # the plain one fills the canvas, so it can be placed small without waste
    fig = plt.figure(figsize=(5, 5.2) if labelled else (5, 5), dpi=dpi)
    ax = fig.add_axes([0.0, 0.0, 1.0, 0.88] if labelled else [0.0, 0.0, 1.0, 1.0],
                      projection="3d")
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_zlim(-LIM, LIM)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
    ax.view_init(*VIEW)
    return fig, ax


def still(key, outdir=".", labelled=True, dpi=220):
    c = CASES[key]
    fig, ax = _frame(dpi, labelled)

    if c["sh"]:                      # the plane holding the four equatorial L
        g = R * 1.30
        q = np.array([[g, g, 0], [g, -g, 0], [-g, -g, 0], [-g, g, 0]], float)
        ax.add_collection3d(Poly3DCollection([q], facecolor=ORANGE, alpha=0.13,
                                             edgecolor="none", zorder=0))
        loop = np.vstack([q, q[0]])
        ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], color=ORANGE, lw=1.6,
                alpha=0.85, zorder=1)

    u = np.array(c["axis"], float)
    u = u / np.linalg.norm(u) * LIM * 0.95
    ax.plot(*zip(-u, u), color=ORANGE, lw=2.8, alpha=0.95, zorder=5)

    for p in LIG:
        ax.plot(*zip([0, 0, 0], p), color=S.BOND_C, lw=2.8, alpha=0.9, zorder=2)
    sa = S.STYLE["A"]
    ax.scatter(0, 0, 0, color=sa["color"], s=sa["s"] * 1.15,
               edgecolor=sa["edge"], linewidth=0.8, depthshade=True, zorder=4)
    for i, p in enumerate(LIG):
        st = (dict(color=LP_C, edge=LP_E, s=S.STYLE["L"]["s"] * 1.30)
              if i in c["sub"] else
              dict(color=S.STYLE["L"]["color"], edge=S.STYLE["L"]["edge"],
                   s=S.STYLE["L"]["s"] * 1.15))
        ax.scatter(*p, color=st["color"], s=st["s"], edgecolor=st["edge"],
                   linewidth=0.9, depthshade=True, zorder=6)

    if c["inv"]:
        # A 3D marker at the origin is depth-sorted behind the metal and vanishes,
        # so the ring is a figure-level overlay. Both its centre and its radius are
        # computed rather than guessed: the centre from the actual projection of
        # (0,0,0), the radius from the metal marker's own area.
        from mpl_toolkits.mplot3d import proj3d
        fig.canvas.draw()
        xp, yp, _ = proj3d.proj_transform(0, 0, 0, ax.get_proj())
        px, py = ax.transData.transform((xp, yp))
        cx, cy = px / fig.bbox.width, py / fig.bbox.height
        r_pt = np.sqrt(sa["s"] * 1.15 / np.pi) * 1.30      # just clear of the sphere
        rad = r_pt / (fig.get_figwidth() * 72.0)
        fig.add_artist(plt.Circle((cx, cy), rad, transform=fig.transFigure,
                                  fill=False, ec=ORANGE, lw=2.4, zorder=20))

    if labelled:
        fig.text(0.5, 0.945, f"{c['name']}", ha="center", fontsize=15,
                 color=NAVY, weight="bold")
        fig.text(0.5, 0.893, f"{c['pg']}   \u2014   order {c['order']}",
                 ha="center", fontsize=13, color=ORANGE, weight="bold")

    base = os.path.join(outdir, f"ml6_descent_{key}" + ("" if labelled else "_plain"))
    fig.savefig(base + ".png", transparent=True, facecolor="none")
    plt.close(fig)
    im = Image.open(base + ".png").convert("RGBA")
    bb = im.split()[3].getbbox(); pad = 14
    im.crop((max(0, bb[0] - pad), max(0, bb[1] - pad),
             min(im.width, bb[2] + pad), min(im.height, bb[3] + pad))
            ).save(base + "_tight.png")
    print("wrote", base + ".png")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out, exist_ok=True)
    for k in CASES:
        still(k, out, labelled=True)
        still(k, out, labelled=False)
