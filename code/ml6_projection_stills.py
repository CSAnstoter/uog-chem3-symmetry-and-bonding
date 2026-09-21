"""Where does the seed orbital go? One PNG per Oh class, panelled by element.

For the sigma SALC derivation: seed phi_1 on the +x ligand and show, for every
symmetry element of a class, where that orbital ends up. The seed is drawn
translucent in its original position and the destination(s) solid, with the
element itself in orange.

Panels are grouped by ELEMENT, not by destination, because that is what answers
"where does the 1 in the table come from?" - for 3C2 the axis along x leaves the
orbital alone (contributing the 1) while the axes along y and z both send it to
-x (contributing the 2). Likewise 6C4: the two rotations about x leave it, giving
the 2, and the four about y and z move it.

The 48 operations are generated and classified from scratch (signed permutation
matrices, split by determinant, trace and axis direction), so the counts are
derived rather than typed - they come out as 1, 8, 6, 6, 3, 1, 6, 8, 3, 6.

Run:  python3 ml6_projection_stills.py [outdir]
"""
import itertools
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import to_rgba
from PIL import Image
import symmetry_operation_gifs as S

BLUE = S.PHASE_P
ORANGE, ORANGE_D = "#e8701a", "#a8480c"
R_L = 1.42
LIM = 1.88
VIEW = (35, 25)
GHOST = 0.28                  # opacity of the seed in its original position

LIG = np.array([[R_L, 0, 0], [-R_L, 0, 0], [0, R_L, 0],
                [0, -R_L, 0], [0, 0, R_L], [0, 0, -R_L]], float)
NAMES = ["\u03c6\u2081", "\u03c6\u2082", "\u03c6\u2083",
         "\u03c6\u2084", "\u03c6\u2085", "\u03c6\u2086"]


# ----------------------------------------------------- the group, generated
def operations():
    """All 48 operations of Oh as matrices, each tagged with its class."""
    ops = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            M = np.zeros((3, 3))
            for i, (p, sg) in enumerate(zip(perm, signs)):
                M[i, p] = sg
            ops.append(M)
    out = []
    for M in ops:
        det = round(np.linalg.det(M))
        tr = round(np.trace(M))
        if det == 1:
            if tr == 3:
                cls, el = "E", None
            elif tr == 0:
                cls, el = "8C3", _axis(M, 1)
            elif tr == 1:
                cls, el = "6C4", _axis(M, 1)
            else:                                  # tr == -1, a C2
                ax = _axis(M, 1)
                cls = "3C2" if _is_face(ax) else "6C2p"
                el = ax
        else:
            if tr == -3:
                cls, el = "i", None
            elif tr == 1:                          # a reflection
                nrm = _axis(M, -1)
                cls = "3sigma_h" if _is_face(nrm) else "6sigma_d"
                el = nrm
            elif tr == -1:
                cls, el = "6S4", _axis(M, 1) if _axis(M, 1) is not None else _axis(-M, 1)
            else:                                  # tr == 0, an S6
                cls, el = "8S6", _axis(-M, 1)
        out.append((cls, M, el))
    return out


def _axis(M, eigval):
    w, v = np.linalg.eig(M)
    for i in range(3):
        if abs(w[i].real - eigval) < 1e-6 and abs(w[i].imag) < 1e-6:
            a = v[:, i].real
            a = a / np.linalg.norm(a)
            return _canon(a)
    return None


def _canon(a):
    """Fix the sign so +a and -a are the same element."""
    a = np.round(a / (np.max(np.abs(a))), 6)
    for c in a:
        if abs(c) > 1e-9:
            return a if c > 0 else -a
    return a


def _is_face(v):
    return sum(abs(c) > 1e-6 for c in v) == 1


def _key(v):
    return None if v is None else tuple(np.round(_canon(v), 4))


# -------------------------------------------------------------- lobe drawing
def _rot_to(a):
    a = np.asarray(a, float); a = a / np.linalg.norm(a); z = np.array([0, 0, 1.])
    if np.allclose(a, z):
        return np.eye(3)
    if np.allclose(a, -z):
        return np.diag([1., -1., -1.])
    v = np.cross(z, a); c = float(z @ a); s = np.linalg.norm(v)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)


def lobe(p, La=0.34, Lb=0.175, n=22):
    rhat = p / np.linalg.norm(p)
    u = np.linspace(0, 2 * np.pi, n); v = np.linspace(0, np.pi, n)
    U, V = np.meshgrid(u, v)
    pts = np.stack([Lb * np.cos(U) * np.sin(V),
                    Lb * np.sin(U) * np.sin(V),
                    La * np.cos(V)], -1)
    return pts @ _rot_to(-rhat).T + (p - rhat * (La + 0.10))


def _plane(ax, normal, alpha=0.95):
    n = np.asarray(normal, float); n = n / np.linalg.norm(n)
    a = np.array([1., 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1., 0])
    u = a - (a @ n) * n; u /= np.linalg.norm(u)
    w = np.cross(n, u); g = R_L * 1.12
    sq = [u * g + w * g, u * g - w * g, -u * g - w * g, -u * g + w * g]
    ax.add_collection3d(Poly3DCollection([np.array(sq)], facecolor=ORANGE,
                                         alpha=0.14 * alpha, edgecolor="none",
                                         zorder=1))
    loop = np.array(sq + [sq[0]])
    ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], color=ORANGE, lw=1.8,
            alpha=alpha, zorder=2)


def _panel(ax, cls, el, dests):
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_zlim(-LIM, LIM)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
    ax.view_init(*VIEW)

    # faint framework
    for p in LIG:
        ax.plot(*zip([0, 0, 0], p), color="#6f6f6f", lw=1.9, alpha=0.28, zorder=0)
    st = S.STYLE["L"]
    for p in LIG:
        ax.scatter(*p, color=st["color"], s=st["s"] * 1.4, edgecolor=st["edge"],
                   linewidth=0.7, alpha=0.34, depthshade=False, zorder=0)
    sa = S.STYLE["A"]
    ax.scatter(0, 0, 0, color=sa["color"], s=sa["s"] * 0.85, edgecolor=sa["edge"],
               linewidth=0.7, alpha=0.38, depthshade=False, zorder=0)

    # the element
    if cls in ("3sigma_h", "6sigma_d"):
        _plane(ax, el)
    elif el is not None:
        u = _canon(el); u = u / np.linalg.norm(u) * LIM * 0.96
        ax.plot(*zip(-u, u), color=ORANGE, lw=2.6, alpha=0.95, zorder=5)
        if cls in ("6S4", "8S6"):
            _plane(ax, el, alpha=0.45)
    if cls == "i":
        ax.scatter(0, 0, 0, s=560, facecolors="none", edgecolors=ORANGE,
                   linewidths=2.4, depthshade=False, zorder=7)

    # the seed, ghosted where it started
    ax.scatter(*LIG[0], s=S.STYLE["L"]["s"] * 3.1, facecolors="none",
               edgecolors=BLUE, linewidths=1.7, linestyle=(0, (4, 3)),
               depthshade=False, zorder=6)
    P = lobe(LIG[0])
    ax.plot_surface(P[..., 0], P[..., 1], P[..., 2], color=BLUE, alpha=GHOST,
                    rstride=1, cstride=1, linewidth=0, antialiased=True,
                    shade=False)
    # and solid wherever it lands
    for d in sorted(set(dests)):
        P = lobe(LIG[d])
        ax.plot_surface(P[..., 0], P[..., 1], P[..., 2], color=BLUE, alpha=0.95,
                        rstride=1, cstride=1, linewidth=0, antialiased=True,
                        shade=True)


LABEL = {"E": "$E$", "8C3": "$8C_3$", "6C2p": "$6C_2'$", "6C4": "$6C_4$",
         "3C2": "$3C_2$", "i": "$i$", "6S4": "$6S_4$", "8S6": "$8S_6$",
         "3sigma_h": "$3\\sigma_h$", "6sigma_d": "$6\\sigma_d$"}
EL_KIND = {"3sigma_h": "plane", "6sigma_d": "plane"}


def _el_name(cls, el):
    if el is None:
        return ""
    v = _canon(el)
    txt = "(" + ", ".join(("0" if abs(c) < 1e-9 else
                           ("1" if c > 0 else "\u22121")) for c in v) + ")"
    return ("plane \u22a5 " if EL_KIND.get(cls) == "plane" else "axis ") + txt


def render_class(cls, ops, outdir=".", dpi=200):
    groups = {}
    for c, M, el in ops:
        if c != cls:
            continue
        d = int(np.argmin([np.linalg.norm(M @ LIG[0] - q) for q in LIG]))
        groups.setdefault(_key(el), []).append(d)
    keys = sorted(groups, key=lambda k: (0 not in groups[k], k is not None, k))

    n = len(keys)
    cols = min(n, 4); rows = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(3.05 * cols, 3.30 * rows), dpi=dpi)
    total = 0
    for i, k in enumerate(keys):
        dests = groups[k]
        total += len(dests)
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        el = None if k is None else np.array(k, float)
        _panel(ax, cls, el, dests)
        head = _el_name(cls, el)
        tail = " + ".join(NAMES[d] for d in sorted(set(dests)))
        cnt = f"{len(dests)} operation" + ("s" if len(dests) > 1 else "")
        # captions go BELOW the molecule: above, they crowd the figure title
        ax.text2D(0.5, 0.015, f"{head}\n\u03c6\u2081 \u2192 {tail}   ({cnt})",
                  transform=ax.transAxes, ha="center", va="bottom",
                  fontsize=9.5, color="#333", linespacing=1.45)
    fig.suptitle(f"ML$_6$:  where {LABEL[cls]} sends the +x seed orbital "
                 f"\u2014 {total} operations in the class",
                 fontsize=12, y=0.995)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.0,
                        wspace=0.02, hspace=0.05)

    base = os.path.join(outdir, f"proj_{cls}")
    fig.savefig(base + ".png", transparent=True, facecolor="none")
    plt.close(fig)
    im = Image.open(base + ".png").convert("RGBA")
    bb = im.split()[3].getbbox(); pad = 12
    im.crop((max(0, bb[0] - pad), max(0, bb[1] - pad),
             min(im.width, bb[2] + pad), min(im.height, bb[3] + pad))
            ).save(base + "_tight.png")
    print(f"wrote {base}.png   ({n} panels, {total} operations)")


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out, exist_ok=True)
    ops = operations()
    counts = {}
    for c, _, _ in ops:
        counts[c] = counts.get(c, 0) + 1
    print("class sizes:", counts, "total", sum(counts.values()))
    for cls in LABEL:
        render_class(cls, ops, out)


# ------------- the operations that leave the seed alone: the stabiliser
# This is the figure that explains the "+x (self)" row of the projection table.
# The operations fixing one ligand form C4v, of order 8 = 48/6 - and the row is
# just those 8 sorted into their Oh classes. NB 6C4 and 3C2 share the SAME axis
# (the x axis) but are different classes, which is exactly where it is easy to
# put a count in the wrong column.
STAB = [
    ("E",         "$E$",              [],                        1,
     "no element \u2014 nothing moves"),
    ("6C4",       "$6C_4$",           [[1, 0, 0]],               2,
     "\u00b190\u00b0 about the x axis"),
    ("3C2",       "$3C_2$",           [[1, 0, 0]],               1,
     "180\u00b0 about the same x axis"),
    ("3sigma_h",  "$3\\sigma_h$",     [[0, 0, 1], [0, 1, 0]],    2,
     "the two coordinate planes holding x"),
    ("6sigma_d",  "$6\\sigma_d$",     [[0, 1, 1], [0, 1, -1]],   2,
     "the two diagonal planes holding x"),
]


def render_stabiliser(outdir=".", dpi=200):
    n = len(STAB)
    fig = plt.figure(figsize=(3.05 * n, 3.85), dpi=dpi)
    for i, (cls, lab, els, cnt, note) in enumerate(STAB):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")
        ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_zlim(-LIM, LIM)
        ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
        ax.view_init(*VIEW)

        for p in LIG:
            ax.plot(*zip([0, 0, 0], p), color="#6f6f6f", lw=1.9, alpha=0.28, zorder=0)
        st = S.STYLE["L"]
        for p in LIG:
            ax.scatter(*p, color=st["color"], s=st["s"] * 1.4, edgecolor=st["edge"],
                       linewidth=0.7, alpha=0.34, depthshade=False, zorder=0)
        sa = S.STYLE["A"]
        ax.scatter(0, 0, 0, color=sa["color"], s=sa["s"] * 0.85,
                   edgecolor=sa["edge"], linewidth=0.7, alpha=0.38,
                   depthshade=False, zorder=0)

        for el in els:
            if cls.startswith("3sigma") or cls.startswith("6sigma"):
                _plane(ax, el, alpha=0.9)
            else:
                u = np.array(el, float); u = u / np.linalg.norm(u) * LIM * 0.96
                ax.plot(*zip(-u, u), color=ORANGE, lw=2.6, alpha=0.95, zorder=5)

        ax.scatter(*LIG[0], s=S.STYLE["L"]["s"] * 3.1, facecolors="none",
                   edgecolors=BLUE, linewidths=1.7, linestyle=(0, (4, 3)),
                   depthshade=False, zorder=6)
        P = lobe(LIG[0])
        ax.plot_surface(P[..., 0], P[..., 1], P[..., 2], color=BLUE, alpha=0.95,
                        rstride=1, cstride=1, linewidth=0, antialiased=True,
                        shade=True)

        ax.text2D(0.5, 0.125, lab, transform=ax.transAxes, ha="center",
                  va="bottom", fontsize=15, color="#1D3B5A", weight="bold")
        ax.text2D(0.5, 0.005, f"{note}\n\u2192 contributes {cnt}",
                  transform=ax.transAxes, ha="center", va="bottom",
                  fontsize=9.5, color="#555", linespacing=1.45)

    fig.suptitle("ML$_6$:  the 8 operations that leave the +x ligand where it is "
                 "\u2014 sorted into Oh classes, these are the \u201c+x (self)\u201d row",
                 fontsize=12.5, y=0.985)
    fig.text(0.5, 0.012,
             "1 + 2 + 1 + 2 + 2 = 8 = 48/6.  Every other class contributes 0: no "
             "$C_3$, $C_2'$, $S_4$, $S_6$ axis or inversion centre leaves a ligand "
             "in place.",
             ha="center", fontsize=10, color="#666", style="italic")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.965, bottom=0.075,
                        wspace=0.02)

    base = os.path.join(outdir, "proj_selfrow")
    fig.savefig(base + ".png", transparent=True, facecolor="none")
    plt.close(fig)
    im = Image.open(base + ".png").convert("RGBA")
    bb = im.split()[3].getbbox(); pad = 12
    im.crop((max(0, bb[0] - pad), max(0, bb[1] - pad),
             min(im.width, bb[2] + pad), min(im.height, bb[3] + pad))
            ).save(base + "_tight.png")
    print("wrote", base + ".png")
