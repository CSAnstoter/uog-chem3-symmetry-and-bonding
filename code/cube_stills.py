"""ML6 inscribed in a cube: one PNG per Oh operation class.

Each ligand sits at the centre of a cube face, which turns the symmetry elements
into easily-located features of the cube:

    6C4 / 3C2 / 6S4   axes through opposite FACE CENTRES  (3 axes)
    8C3 / 8S6         axes along the BODY DIAGONALS       (4 axes)
    6C2'              axes through opposite EDGE MIDPOINTS (6 axes)
    3sigma_h          planes parallel to the cube faces   (3 planes)
    6sigma_d          planes through opposite cube EDGES  (6 planes)
    i                 the cube centre

The exemplary element is drawn in orange; the others of the same class are faint
blue, so each figure shows both "here is one" and "here are all of them". For the improper rotations the
perpendicular mirror is added as a dashed orange section, since the axis alone
does not convey the second step. Note the difference: for S4 that plane IS one of
the 3 sigma_h, whereas Oh has no mirror perpendicular to a body diagonal, so the
plane shown for S6 is only the one the composite reflects in, not an element of
the group. Worth saying out loud.

Two variants of each:  cube_<class>.png  (labelled)  and  cube_<class>_plain.png

Run:  python3 cube_stills.py [outdir]
"""
import os
import sys
import itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import to_rgba
from PIL import Image
import symmetry_operation_gifs as S

R = 1.0                      # half-edge of the cube = M-L distance
LIM = 1.92
VIEW = (18, -52)
ORANGE, ORANGE_D = "#e8701a", "#a8480c"
FAINT = "#5c7a9e"
CUBE_C = "#b0b0b0"

FACE = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]                       # 3 face-centre axes
BODY = [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]      # 4 body diagonals
EDGE = [[1, 1, 0], [1, -1, 0], [1, 0, 1],
        [1, 0, -1], [0, 1, 1], [0, 1, -1]]                     # 6 edge-midpoint axes
# same six directions, reordered: as an AXIS you want the exemplary roughly across
# the view, but as a PLANE NORMAL you want it pointing at the camera so the plane
# faces us rather than appearing edge-on
EDGE_PLANES = [[1, -1, 0], [1, 1, 0], [1, 0, 1],
               [1, 0, -1], [0, 1, 1], [0, 1, -1]]

# per class: element kind, the full list of elements, and how far to draw an axis
CLASSES = {
    "E":       dict(kind="none",  vecs=[],    label="$E$"),
    "8C3":     dict(kind="axis",  vecs=BODY,  reach=1.82, label="8$C_3$"),
    "6C2p":    dict(kind="axis",  vecs=EDGE,  reach=1.62, label="6$C_2'$"),
    "6C4":     dict(kind="axis",  vecs=FACE,  reach=1.42, label="6$C_4$"),
    "3C2":     dict(kind="axis",  vecs=FACE,  reach=1.42, label="3$C_2$ ($=C_4^2$)"),
    "i":       dict(kind="point", vecs=[],    label="$i$"),
    "6S4":     dict(kind="axis",  vecs=FACE,  reach=1.42, perp=True, label="6$S_4$"),
    "8S6":     dict(kind="axis",  vecs=BODY,  reach=1.82, perp=True, label="8$S_6$"),
    "3sigma_h": dict(kind="plane", vecs=FACE, label="3$\\sigma_h$"),
    "6sigma_d": dict(kind="plane", vecs=EDGE_PLANES, label="6$\\sigma_d$"),
}


def _cube(ax):
    v = np.array(list(itertools.product([-R, R], repeat=3)), float)
    for a, b in itertools.combinations(range(8), 2):
        if np.isclose(np.linalg.norm(v[a] - v[b]), 2 * R):     # an edge, not a diagonal
            ax.plot(*zip(v[a], v[b]), color=CUBE_C, lw=1.3, alpha=0.75, zorder=0)


def _axis(ax, vec, reach, colour, lw, alpha):
    u = np.asarray(vec, float); u = u / np.linalg.norm(u) * reach
    ax.plot(*zip(-u, u), color=colour, lw=lw, alpha=alpha, zorder=5)


def _cube_section(normal):
    """The polygon where a plane through the cube centre cuts the cube."""
    n = np.asarray(normal, float); n = n / np.linalg.norm(n)
    verts = [np.array(v, float) for v in itertools.product([-R, R], repeat=3)]
    pts = []
    for a, b in itertools.combinations(range(8), 2):
        va, vb = verts[a], verts[b]
        if not np.isclose(np.linalg.norm(va - vb), 2 * R):
            continue                                   # cube edges only
        da, db = n @ va, n @ vb
        if np.isclose(da, 0):
            pts.append(va)
        if np.isclose(db, 0):
            pts.append(vb)
        if da * db < -1e-12:
            pts.append(va + da / (da - db) * (vb - va))
    uniq = []
    for p in pts:
        if not any(np.allclose(p, q, atol=1e-7) for q in uniq):
            uniq.append(p)
    a = np.array([1., 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1., 0])
    u = a - (a @ n) * n; u /= np.linalg.norm(u)
    w = np.cross(n, u)
    uniq.sort(key=lambda p: np.arctan2(p @ w, p @ u))
    return np.array(uniq)


def _plane(ax, normal, colour, fill, alpha, dashed=False):
    sq = list(_cube_section(normal))
    if fill:
        ax.add_collection3d(Poly3DCollection(
            [np.array(sq)], facecolor=colour, alpha=0.15 * alpha,
            edgecolor="none", zorder=1))
    loop = np.array(sq + [sq[0]])
    ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], color=colour, lw=1.7 if fill else 1.2,
            alpha=alpha, ls=(0, (5, 3)) if dashed else "-", zorder=2)


def still(key, outdir=".", labelled=True, dpi=220):
    spec = CLASSES[key]
    P = np.array([[0, 0, 0], [R, 0, 0], [-R, 0, 0], [0, R, 0],
                  [0, -R, 0], [0, 0, R], [0, 0, -R]], float)
    elems = ["A"] + ["L"] * 6

    fig = plt.figure(figsize=(5, 5), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_zlim(-LIM, LIM)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
    ax.view_init(*VIEW)

    _cube(ax)

    vecs = spec["vecs"]
    if spec["kind"] == "axis":
        a_min = 0.40 if len(vecs) <= 4 else 0.30     # 6 axes need a lighter touch
        for v in vecs[1:]:
            _axis(ax, v, spec["reach"], FAINT, 1.3, a_min)
        _axis(ax, vecs[0], spec["reach"], ORANGE, 2.8, 0.95)
        if spec.get("perp"):
            # dashed and unfilled: for S4 this plane is one of the 3 sigma_h, but
            # for S6 it is not an element of the group at all, only the plane the
            # composite operation reflects in
            _plane(ax, vecs[0], ORANGE_D, True, 0.60, dashed=True)
    elif spec["kind"] == "plane":
        a_min = 0.38 if len(vecs) <= 3 else 0.26     # 6 planes need a lighter touch
        for v in vecs[1:]:
            _plane(ax, v, FAINT, False, a_min)
        _plane(ax, vecs[0], ORANGE, True, 0.95)
    elif spec["kind"] == "point":
        for v in FACE:                    # every ligand maps through the centre
            u = np.asarray(v, float) * R
            ax.plot(*zip(-u, u), color=ORANGE_D, lw=1.5, alpha=0.5,
                    ls=(0, (4, 3)), zorder=5)

    for i in range(1, 7):
        ax.plot(*zip(P[0], P[i]), color=S.BOND_C, lw=2.6, alpha=0.9, zorder=3)
    for idx, e in enumerate(elems):
        st = S.STYLE[e]
        ax.scatter(*P[idx], color=st["color"], s=st["s"] * 1.15,
                   edgecolor=st["edge"], linewidth=0.8, depthshade=True, zorder=6)

    if spec["kind"] == "point":
        # a figure-level overlay: a 3D artist at the origin is depth-sorted behind
        # the metal atom and vanishes, and a 2D patch cannot live on a 3D axes
        fig.add_artist(plt.Circle((0.502, 0.498), 0.052, transform=fig.transFigure,
                                  fill=False, ec=ORANGE, lw=2.8, zorder=20))

    if labelled:
        fig.text(0.5, 0.925, spec["label"], ha="center", fontsize=19,
                 color="#1D3B5A", weight="bold")

    base = os.path.join(outdir, f"cube_{key}" + ("" if labelled else "_plain"))
    fig.savefig(base + ".png", transparent=True, facecolor="none")
    plt.close(fig)

    im = Image.open(base + ".png").convert("RGBA")
    bb = im.split()[3].getbbox(); pad = 14
    im.crop((max(0, bb[0] - pad), max(0, bb[1] - pad),
             min(im.width, bb[2] + pad), min(im.height, bb[3] + pad))
            ).save(base + "_tight.png")
    print("wrote", base + ".png")


def setup_still(outdir=".", face=True, dpi=220):
    """The construction on its own, for introducing the idea: ML6 inside a cube
    with each ligand at the centre of a face. With face=True one cube face is
    shaded and its ligand ringed, which is what makes the arrangement obvious."""
    P = np.array([[0, 0, 0], [R, 0, 0], [-R, 0, 0], [0, R, 0],
                  [0, -R, 0], [0, 0, R], [0, 0, -R]], float)
    elems = ["A"] + ["L"] * 6

    fig = plt.figure(figsize=(5, 5), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_zlim(-LIM, LIM)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
    ax.view_init(*VIEW)
    _cube(ax)

    if face:
        # the top face: visible at this elevation, and its ligand is not hidden
        # behind the molecule the way a near-side face would be
        q = np.array([[R, R, R], [R, -R, R], [-R, -R, R], [-R, R, R]], float)
        ax.add_collection3d(Poly3DCollection(
            [q], facecolor=ORANGE, alpha=0.13, edgecolor="none", zorder=1))
        loop = np.vstack([q, q[0]])
        ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], color=ORANGE, lw=1.9,
                alpha=0.9, zorder=2)

    for i in range(1, 7):
        ax.plot(*zip(P[0], P[i]), color=S.BOND_C, lw=2.6, alpha=0.9, zorder=3)
    for idx, e in enumerate(elems):
        st = S.STYLE[e]
        ax.scatter(*P[idx], color=st["color"], s=st["s"] * 1.15,
                   edgecolor=st["edge"], linewidth=0.8, depthshade=True, zorder=6)
    if face:
        ax.scatter(*P[5], s=S.STYLE["L"]["s"] * 3.0, facecolors="none",
                   edgecolors=ORANGE, linewidths=2.4, depthshade=False, zorder=7)

    base = os.path.join(outdir, "cube_setup" + ("" if face else "_bare"))
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
    setup_still(out, face=True)
    setup_still(out, face=False)
    for k in CLASSES:
        still(k, out, labelled=True)
        still(k, out, labelled=False)
