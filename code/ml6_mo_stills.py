"""The molecular orbitals of the sigma-only ML6 scheme, on the ML framework.

Fifteen orbitals: six metal AOs (4s, 4p, 3d) plus six sigma SALCs give six bonding
combinations, six antibonding, and the three t2g d orbitals which have no SALC
partner and so pass through unchanged.

    1a1g   2a1g*                    from 4s      + a1g SALC
    1t1u   2t1u*   (x, y, z)        from 4p      + t1u SALCs
    1eg    2eg*    (z2, x2-y2)      from 3d eg   + eg SALCs
    t2g            (xy, xz, yz)     3d t2g, non-bonding

Bonding combinations use the SALC exactly as the projection operator gives it, so
each ligand lobe meets the metal lobe pointing at it with the same phase.
Antibonding combinations are the same picture with every ligand coefficient
negated. The t2g trio carries no ligand lobes at all - there is no sigma SALC of
t2g symmetry to draw.

Framed to match the d_*_ml6 and SALC figures: same camera, same M-L distance.

Run:  python3 ml6_mo_stills.py [outdir]
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import symmetry_operation_gifs as S

BLUE, RED = S.PHASE_P, S.PHASE_N
R_L = 1.42
LIM = 1.78
VIEW = (35, 25)
AX = "#8a8a8a"
N = 24

LIG = np.array([[R_L, 0, 0], [-R_L, 0, 0], [0, R_L, 0],
                [0, -R_L, 0], [0, 0, R_L], [0, 0, -R_L]], float)
c45 = float(np.cos(np.deg2rad(45)))

# metal orbital, then the ligand coefficients in order +x -x +y -y +z -z
MOS = {
    "1a1g":        ("s",     [1, 1, 1, 1, 1, 1]),
    "2a1g_star":   ("s",     [-1, -1, -1, -1, -1, -1]),
    "1t1u_x":      ("px",    [1, -1, 0, 0, 0, 0]),
    "1t1u_y":      ("py",    [0, 0, 1, -1, 0, 0]),
    "1t1u_z":      ("pz",    [0, 0, 0, 0, 1, -1]),
    "2t1u_x_star": ("px",    [-1, 1, 0, 0, 0, 0]),
    "2t1u_y_star": ("py",    [0, 0, -1, 1, 0, 0]),
    "2t1u_z_star": ("pz",    [0, 0, 0, 0, -1, 1]),
    "1eg_z2":      ("dz2",   [-1, -1, -1, -1, 2, 2]),
    "1eg_x2y2":    ("dx2y2", [1, 1, -1, -1, 0, 0]),
    "2eg_z2_star": ("dz2",   [1, 1, 1, 1, -2, -2]),
    "2eg_x2y2_star": ("dx2y2", [-1, -1, 1, 1, 0, 0]),
    "t2g_xy":      ("dxy",   [0, 0, 0, 0, 0, 0]),
    "t2g_xz":      ("dxz",   [0, 0, 0, 0, 0, 0]),
    "t2g_yz":      ("dyz",   [0, 0, 0, 0, 0, 0]),
}


def _rot_to(a):
    a = np.asarray(a, float); a = a / np.linalg.norm(a); z = np.array([0, 0, 1.])
    if np.allclose(a, z):
        return np.eye(3)
    if np.allclose(a, -z):
        return np.diag([1., -1., -1.])
    v = np.cross(z, a); c = float(z @ a); s = np.linalg.norm(v)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)


def lobe(centre, direction, La, Lb):
    u = np.linspace(0, 2 * np.pi, N); v = np.linspace(0, np.pi, N)
    U, V = np.meshgrid(u, v)
    pts = np.stack([Lb * np.cos(U) * np.sin(V),
                    Lb * np.sin(U) * np.sin(V),
                    La * np.cos(V)], -1)
    return pts @ _rot_to(direction).T + np.asarray(centre, float)


def sphere(r):
    u = np.linspace(0, 2 * np.pi, N * 2); v = np.linspace(0, np.pi, N)
    U, V = np.meshgrid(u, v)
    return np.stack([r * np.cos(U) * np.sin(V), r * np.sin(U) * np.sin(V),
                     r * np.cos(V)], -1)


def torus(Rt=0.38, rt=0.135):
    u = np.linspace(0, 2 * np.pi, N * 2); v = np.linspace(0, 2 * np.pi, N)
    U, V = np.meshgrid(u, v)
    return np.stack([(Rt + rt * np.cos(V)) * np.cos(U),
                     (Rt + rt * np.cos(V)) * np.sin(U),
                     rt * np.sin(V)], -1)


def _metal(kind):
    """(surface, colour) pairs for the metal orbital, sized to meet but not
    swallow the ligand lobes."""
    P, Q = 0.44, 0.34                       # p-lobe offset and length
    D, E = 0.46, 0.32                       # d-lobe offset and length
    if kind == "s":
        return [(sphere(0.40), BLUE)]
    if kind in ("px", "py", "pz"):
        u = {"px": [1, 0, 0], "py": [0, 1, 0], "pz": [0, 0, 1]}[kind]
        u = np.array(u, float)
        return [(lobe(u * P, u, Q, 0.19), BLUE),
                (lobe(-u * P, -u, Q, 0.19), RED)]
    if kind == "dz2":
        u = np.array([0, 0, 1.])
        return [(lobe(u * 0.46, u, 0.34, 0.20), BLUE),
                (lobe(-u * 0.46, -u, 0.34, 0.20), BLUE),
                (torus(), RED)]
    if kind == "dx2y2":
        return [(lobe([D, 0, 0], [1, 0, 0], E, 0.185), BLUE),
                (lobe([-D, 0, 0], [-1, 0, 0], E, 0.185), BLUE),
                (lobe([0, D, 0], [0, 1, 0], E, 0.185), RED),
                (lobe([0, -D, 0], [0, -1, 0], E, 0.185), RED)]
    dirs = {"dxy": [(c45, c45, 0), (-c45, -c45, 0), (-c45, c45, 0), (c45, -c45, 0)],
            "dxz": [(c45, 0, c45), (-c45, 0, -c45), (-c45, 0, c45), (c45, 0, -c45)],
            "dyz": [(0, c45, c45), (0, -c45, -c45), (0, -c45, c45), (0, c45, -c45)]}[kind]
    out = []
    for i, d in enumerate(dirs):
        d = np.array(d, float)
        out.append((lobe(d * D, d, E, 0.185), BLUE if i < 2 else RED))
    return out


def _frame(dpi):
    fig = plt.figure(figsize=(5, 5), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_zlim(-LIM, LIM)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
    ax.view_init(*VIEW)
    return fig, ax


def _scaffold(ax, triad):
    for p in LIG:
        ax.plot(*zip([0, 0, 0], p), color="#6f6f6f", lw=2.0, alpha=0.30, zorder=0)
    st = S.STYLE["L"]
    for p in LIG:
        ax.scatter(*p, color=st["color"], s=st["s"] * 1.5, edgecolor=st["edge"],
                   linewidth=0.8, alpha=0.38, depthshade=False, zorder=0)
    sa = S.STYLE["A"]
    ax.scatter(0, 0, 0, color=sa["color"], s=sa["s"] * 0.9, edgecolor=sa["edge"],
               linewidth=0.8, alpha=0.40, depthshade=False, zorder=0)
    if triad:
        for vec, lab in (((1, 0, 0), "x"), ((0, 1, 0), "y"), ((0, 0, 1), "z")):
            ax.text(*(np.array(vec, float) * (R_L + 0.30)), lab, color=AX,
                    fontsize=11, ha="center", va="center", style="italic", zorder=0)


def _surf(ax, P, col):
    ax.plot_surface(P[..., 0], P[..., 1], P[..., 2], color=col, alpha=0.95,
                    rstride=1, cstride=1, linewidth=0, antialiased=True, shade=True)


def mo(key, outdir=".", triad=True, dpi=220):
    kind, coeffs = MOS[key]
    fig, ax = _frame(dpi)
    _scaffold(ax, triad)
    for surf, col in _metal(kind):
        _surf(ax, surf, col)
    for c, p in zip(coeffs, LIG):
        if c == 0:
            continue
        rhat = p / np.linalg.norm(p)
        s = abs(c) ** 0.5
        La, Lb = 0.28 * s, 0.150 * s
        _surf(ax, lobe(p - rhat * (La + 0.10), -rhat, La, Lb),
              BLUE if c > 0 else RED)

    base = os.path.join(outdir, f"ml6_mo_{key}" + ("" if triad else "_plain"))
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
    for k in MOS:
        mo(k, out, triad=True)
        mo(k, out, triad=False)
