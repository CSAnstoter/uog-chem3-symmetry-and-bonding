#!/usr/bin/env python3
"""
symmetry_operation_gifs.py
==========================
Animated ball-and-stick GIFs of molecular symmetry operations, written for the 2026/27
Chem-3 "Symmetry and Bonding" course at the University of Glasgow.

WHAT IT DOES
------------
For a chosen molecule and symmetry operation it renders an animated GIF showing
the operation being performed, with the symmetry element (axis, plane or
inversion centre) drawn in place.

  * PROPER rotations (C2, C3, C4 ...) are genuine rigid motions, so the molecule
    is rotated continuously.
  * REFLECTIONS, INVERSION and IMPROPER rotations cannot be carried out as a
    continuous rigid motion in three dimensions. Each atom is instead moved in a
    straight line onto its image and back, and the caption says "mapping" rather
    than pretending otherwise.
  * One atom is drawn in crimson so you can follow where it goes.
  * Where an operation genuinely leaves a planar molecule untouched (benzene's
    sigma_h, water's in-plane sigma_v) the figure says so rather than faking
    movement.

RUNNING IT
----------
    python symmetry_operation_gifs.py                    # the curated set
    python symmetry_operation_gifs.py water methane      # chosen molecules
    python symmetry_operation_gifs.py pf5:sigma_h        # one molecule:operation
    python symmetry_operation_gifs.py all                # everything

GIFs are written to the current folder unless you pass outdir= to render().

REQUIREMENTS
------------
    numpy, matplotlib, pillow          (see README for install instructions)

MAKING CHANGES
--------------
Everything you are likely to want to alter is in the SETTINGS section directly
below: colours, atom sizes, animation speed, output resolution. Further down,
three registries control the content --- search for the banner comments:

    GEOMETRIES   where each molecule's atoms are, and what they are bonded to
    OPS          the symmetry operations available for each molecule
    MOLECULES    which operations are rendered by default, and the frame size

Each has a worked comment showing how to add your own.
"""
import sys
import textwrap
import numpy as np
import matplotlib
matplotlib.use("Agg")                      # render to file, no window needed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d import proj3d
import matplotlib.patheffects as pe
from matplotlib.animation import FuncAnimation, PillowWriter


class PlayOnceWriter(PillowWriter):
    """A GIF writer that omits the loop flag, so the animation plays through
    once and rests on the final frame. Better for lecture delivery than an
    endlessly looping GIF. Swap PlayOnceWriter for PillowWriter in the render
    calls if you would rather they loop."""
    def finish(self):
        self._frames[0].save(
            self.outfile, save_all=True, append_images=self._frames[1:],
            duration=int(1000 / self.fps))          # loop omitted on purpose


# =====================================================================
#  SETTINGS  --  everything here is safe to change
# =====================================================================

# ---- animation -------------------------------------------------------
FRAMES = 60          # frames per animation. More = smoother but bigger files.
FPS    = 15          # frames per second. FRAMES/FPS = seconds, so 60/15 = 4 s.

# ---- output size -----------------------------------------------------
FIG_SIZE = (5, 5)    # figure size in INCHES (width, height)
DPI      = 110       # dots per inch. Pixel size = FIG_SIZE x DPI, so 550x550.
                     # Raise DPI to ~200 for print; file size grows quickly.

# ---- atom colours and sizes -----------------------------------------
# One entry per element. "color" is the fill, "edge" the outline, and "s" the
# marker area in points squared (bigger number = bigger circle, and it scales
# as the AREA, so doubling s makes the radius about 1.4x larger).
# Colours are hex codes: "#RRGGBB". Any colour picker will give you these.
#
# To recolour every oxygen, edit the "O" line. To add an element, copy any line
# and change the key, e.g.   "Na": dict(color="#b040d0", s=600, edge="#602070"),
STYLE = {
 "C": dict(color="#2b2b2b", s=460, edge="#000000"),
 "H": dict(color="#e9e9e9", s=250, edge="#9a9a9a"),
 "O": dict(color="#d33b2c", s=500, edge="#7a1f16"),
 "N": dict(color="#2f55c7", s=490, edge="#1a2f6e"),
 "P": dict(color="#e08a2b", s=580, edge="#8a5212"),
 "F": dict(color="#9ccf5a", s=300, edge="#5f8a2f"),
 "Si":dict(color="#c9a96a", s=560, edge="#8a7040"),
 "S": dict(color="#e8d44d", s=480, edge="#a89818"),
 "Ni":dict(color="#4a9e7f", s=560, edge="#2c5f4c"),
 "Br":dict(color="#b5432a", s=420, edge="#7a2a18"),
 "Fe":dict(color="#d97a2b", s=620, edge="#8a4a15"),
 "I": dict(color="#8b3fa8", s=560, edge="#5a2470"),
 "Cl":dict(color="#4caf50", s=360, edge="#2f6e33"),
 "B": dict(color="#f0a58c", s=420, edge="#a8543a"),
 # Generic placeholders for schematic molecules such as AB4 and ML6. "B" is
 # already taken by boron, so a generic ligand uses the key "L" and a generic
 # central atom "A". These keys only ever pick a colour and size, never a label.
 "A": dict(color="#5f7fa6", s=520, edge="#31506e"),
 "L": dict(color="#a9b4c2", s=330, edge="#6b7686"),
}

# How a molecule's name is printed in the figure title. Anything not listed here
# is just capitalised, so "water" becomes "Water". Use \u escapes for
# subscripts: \u2082 is a subscript 2, \u2083 a subscript 3, and so on.
DISPLAY = {"co2":"CO\u2082", "pf5":"PF\u2085", "nh3":"NH\u2083",
           "tcb":"1,3,5-C\u2086H\u2083Cl\u2083",
           "bcl3":"BCl\u2083", "ab4":"AB\u2084"}

# The single atom drawn in crimson and followed through the operation.
TRACK = dict(color="#d11f4a", s=320, edge="#7a0f2b")

# ---- colours of the symmetry elements themselves ---------------------
AXIS_C   = "#1f5fbf"   # rotation axes (blue)
PLANE_C  = "#c79a3a"   # mirror plane fill (gold)
PLANE_EDGE = "#a8801f" # mirror plane border
INV_C    = "#2e8b57"   # inversion centre marker (green)
BOND_C   = "#6f6f6f"   # bonds (grey)

# ---- orbital lobe phases --------------------------------------------
PHASE_P, PHASE_N = "#2f5fbf", "#c0392b"    # blue = positive, red = negative
# NOTE the negative phase is a brick red, deliberately NOT the crimson used for
# TRACK. In this engine crimson means "the atom being followed", so it must
# never appear on an orbital or the two meanings collide.

# =====================================================================
#  End of settings. Below here is machinery.
# =====================================================================

d = np.deg2rad          # shorthand: d(90) converts 90 degrees to radians

# =====================================================================
#  GEOMETRIES  --  where the atoms are
# =====================================================================
# Each function returns four things:
#
#   P      an (N x 3) array of atom positions, in arbitrary length units
#   elems  a list of N element keys, matching the keys of STYLE above
#   bonds  a list of (i, j) index pairs to draw a stick between
#   track  the index of the atom to follow in crimson (-1 for none)
#
# The first atom is conventionally the central one. Positions are chosen so the
# molecule sits sensibly about the origin, because every symmetry element passes
# through the origin.
#
# TO ADD YOUR OWN MOLECULE, copy a function and edit the numbers:
#
#     def hcl():
#         P = np.array([[0, 0, 0.6], [0, 0, -0.7]])   # H above, Cl below
#         return P, ["H", "Cl"], [(0, 1)], 0
#
# then register it in MOLECULES further down.
# ---------------------------------------------------------------------
def water():
    OH = 0.96; h = d(104.5/2)
    O = np.array([0, 0, 0.30])
    H1 = O + [ OH*np.sin(h), 0, -OH*np.cos(h)]
    H2 = O + [-OH*np.sin(h), 0, -OH*np.cos(h)]
    return np.vstack([O, H1, H2]), ["O","H","H"], [(0,1),(0,2)], 1

def _c3v_tilt(alpha_deg):
    """Bond angle from the C3 axis for a C3v AX3 with X-A-X = alpha.
    For three X at polar angle b from the axis, cos(alpha) = 1.5 cos^2(b) - 0.5,
    so b = arccos(sqrt((cos alpha + 0.5)/1.5)).  NB this is NOT 180 - alpha."""
    return np.arccos(np.sqrt((np.cos(d(alpha_deg)) + 0.5) / 1.5))

def ammonia():
    NH = 1.01; h = _c3v_tilt(106.7)           # H-N-H = 106.7 -> 67.9 deg from the C3 axis
    N = np.array([0, 0, 0.0])
    az = d(np.array([90, 210, 330]))
    H = N + np.c_[NH*np.sin(h)*np.cos(az), NH*np.sin(h)*np.sin(az), -NH*np.cos(h)*np.ones(3)]
    # lift so the N sits above the H3 base, roughly centred
    coords = np.vstack([N, H]); coords[:,2] += 0.30
    return coords, ["N","H","H","H"], [(0,1),(0,2),(0,3)], 1

def benzene():
    rC = 1.39; rH = rC + 1.09
    ang = d(np.arange(0, 360, 60))
    C = np.c_[rC*np.cos(ang), rC*np.sin(ang), np.zeros(6)]
    H = np.c_[rH*np.cos(ang), rH*np.sin(ang), np.zeros(6)]
    bonds = [(i,(i+1)%6) for i in range(6)] + [(i,6+i) for i in range(6)]
    return np.vstack([C,H]), ["C"]*6+["H"]*6, bonds, 6

def ethane():
    dCC=1.54; zc=dCC/2; rCH=1.09; phi=d(180-111.0)
    topC=np.array([0,0,zc]); botC=np.array([0,0,-zc])
    ta=d(np.array([90,210,330])); ba=d(np.array([30,150,270]))
    def H(C,az,up): return C+rCH*np.c_[np.sin(phi)*np.cos(az),np.sin(phi)*np.sin(az),up*np.cos(phi)*np.ones_like(az)]
    coords=np.vstack([topC,botC,H(topC,ta,1),H(botC,ba,-1)])
    bonds=[(0,1)]+[(0,2),(0,3),(0,4)]+[(1,5),(1,6),(1,7)]
    return coords, ["C","C"]+["H"]*6, bonds, 2

def methane():
    b=1.09/np.sqrt(3)
    H=np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]],float)*b
    return np.vstack([[0,0,0],H]), ["C"]+["H"]*4, [(0,i) for i in range(1,5)], 1

def pf5():
    req=1.53; rax=1.58
    eq=d(np.array([0,120,240]))
    Feq=np.c_[req*np.cos(eq),req*np.sin(eq),np.zeros(3)]
    Fax=np.array([[0,0,rax],[0,0,-rax]])
    coords=np.vstack([[0,0,0],Feq,Fax])
    return coords, ["P"]+["F"]*5, [(0,i) for i in range(1,6)], 4   # track axial top

def co2():
    r=1.16
    coords=np.vstack([[0,0,0.0],[0,0,r],[0,0,-r]])
    return coords, ["C","O","O"], [(0,1),(0,2)], 1   # linear, track one O

def bcl3():
    # D3h, planar. z = C3 = principal axis, perpendicular to the molecular plane;
    # x runs along a B-Cl bond (the in-plane axis through the most atoms).
    r=1.75
    ang=d(np.array([0,120,240]))
    Cl=np.c_[r*np.cos(ang), r*np.sin(ang), np.zeros(3)]
    coords=np.vstack([[0,0,0], Cl])
    return coords, ["B","Cl","Cl","Cl"], [(0,1),(0,2),(0,3)], 2   # track Cl_b

def ab4():
    # Square planar AB4 (D4h), schematic. z = C4 = principal axis, perpendicular
    # to the molecular plane; the four B sit on +/-x and +/-y.
    r=1.90
    Bs=np.array([[r,0,0],[0,r,0],[-r,0,0],[0,-r,0]],float)
    coords=np.vstack([[0,0,0], Bs])
    return coords, ["A","L","L","L","L"], [(0,i) for i in range(1,5)], 2

def tcb():
    # 1,3,5-trichlorobenzene (D3h, planar). Cl at the 1,3,5 carbons, H at 2,4,6.
    rC=1.39; rCl=rC+1.74; rH=rC+1.08
    ang=np.deg2rad(np.arange(0,360,60))           # 0,60,...,300
    C=np.c_[rC*np.cos(ang), rC*np.sin(ang), np.zeros(6)]
    cl=[0,2,4]; h=[1,3,5]
    Cl=np.c_[rCl*np.cos(ang[cl]), rCl*np.sin(ang[cl]), np.zeros(3)]
    H =np.c_[rH*np.cos(ang[h]),  rH*np.sin(ang[h]),  np.zeros(3)]
    coords=np.vstack([C,Cl,H]); elements=["C"]*6+["Cl"]*3+["H"]*3
    bonds=[(i,(i+1)%6) for i in range(6)]
    bonds+=[(cl[k],6+k) for k in range(3)] + [(h[k],9+k) for k in range(3)]
    return coords, elements, bonds, 6             # track one Cl

# ----------------------------- maths -----------------------------
def Rmat(axis, th):
    a=np.asarray(axis,float); a=a/np.linalg.norm(a); x,y,z=a; c,s=np.cos(th),np.sin(th); K=1-c
    return np.array([[c+x*x*K,x*y*K-z*s,x*z*K+y*s],[y*x*K+z*s,c+y*y*K,y*z*K-x*s],[z*x*K-y*s,z*y*K+x*s,c+z*z*K]])
def reflect(n):
    n=np.asarray(n,float); n=n/np.linalg.norm(n); return np.eye(3)-2*np.outer(n,n)
def ease(t): return 0.5-0.5*np.cos(np.pi*np.clip(t,0,1))

def _flat_plane(ax, corners, al, fill=PLANE_C, edge=PLANE_EDGE, a_fill=0.20, a_edge=0.85, ew=1.5, z=0):
    """Draw a flat quadrilateral mirror plane robustly across matplotlib versions:
    a filled Poly3DCollection with the alpha baked into the RGBA facecolor (the
    alpha= kwarg is handled inconsistently on some versions), plus a solid outline
    so the plane still reads even if a translucent fill is dropped. `corners` is 4
    points in loop order."""
    if al<=0.02: return
    C=np.array(corners,float)
    ax.add_collection3d(Poly3DCollection([C],facecolors=[to_rgba(fill,a_fill*al)],edgecolors="none",zorder=z))
    loop=np.vstack([C,C[0]]); ax.plot(loop[:,0],loop[:,1],loop[:,2],color=edge,lw=ew,alpha=a_edge*al,zorder=z)
def schedule(mode):
    if mode=="rot": return ease(np.linspace(0,1,FRAMES))
    if mode=="improper":
        n=max(14,FRAMES//4); h1=max(6,FRAMES//8); h2=max(10,FRAMES//5)
        rot=0.5*ease(np.linspace(0,1,n))          # step 1: rotate to full angle
        ref=0.5+0.5*ease(np.linspace(0,1,n))      # step 2: reflect perpendicular
        # forward only (Sn maps onto an equivalent frame, so the loop is clean):
        # rotate, pause, reflect, longer pause on the completed result
        return np.concatenate([rot, np.full(h1,0.5), ref, np.full(h2,1.0)])
    if mode in ("inv","refl"):
        # forward only (no boomerang back to E): brief pause, apply the operation,
        # then hold on the completed result. Play-once GIFs end on that result -
        # for a reflection that swaps atoms, the swapped arrangement stays on screen.
        pre=max(4,FRAMES//12); hold=max(10,FRAMES//4); n=FRAMES-pre-hold
        return np.concatenate([np.zeros(pre), ease(np.linspace(0,1,n)), np.full(hold,1.0)])
    half=FRAMES//2; fwd=ease(np.linspace(0,1,half)); return np.concatenate([fwd,fwd[::-1]])
def frame_coords(op,P0,s):
    m=op["mode"]
    if m=="rot": return (Rmat(op["axis"],d(op["angle"])*s)@P0.T).T
    if m=="refl": return (1-s)*P0+s*((reflect(op["normal"])@P0.T).T)
    if m=="inv": return (1-s)*P0+s*(-P0)
    if m=="improper":            # phase 1 (s<=.5): rotate;  phase 2 (s>.5): reflect
        n=op["axis"]
        if s<=0.5:
            return (Rmat(n,d(op["angle"])*(s/0.5))@P0.T).T
        v=(s-0.5)/0.5
        rot=(Rmat(n,d(op["angle"]))@P0.T).T
        tgt=(reflect(n)@rot.T).T
        return (1-v)*rot+v*tgt
    raise ValueError(m)

def choose_tracker(op, P0, default):
    """Highlight the atom that actually moves most under this operation.
    If NOTHING moves (a planar molecule reflected in its own plane), highlight
    nothing: a crimson atom means "this one moves", so marking a static atom
    tells the student the opposite of the truth."""
    P1 = frame_coords(op, P0, 1.0)
    disp = np.linalg.norm(P1 - P0, axis=1)
    return int(np.argmax(disp)) if disp.max() > 1e-6 else -1

# =====================================================================
#  OPS  --  the symmetry operations, per molecule
# =====================================================================
# OPS[molecule][name] is a dictionary describing one operation:
#
#   mode   "rot"      proper rotation           needs axis= and angle=
#          "refl"     mirror plane              needs normal=
#          "inv"      inversion through origin
#          "improper" rotate then reflect       needs axis= and angle=
#          "none"     identity, nothing moves
#   view   (elevation, azimuth) camera angles in degrees
#   elem   ("axis", vector) | ("plane", normal) | ("point", None) | ("none", None)
#          -- which symmetry element to draw, and where
#   label  the caption. LaTeX between $...$ renders as maths.
#
# TO ADD AN OPERATION, add an entry to the right molecule, for example:
#
#     "C2_new": dict(mode="rot", axis=[1, 1, 0], angle=180, view=(20, -60),
#                    elem=("axis", [1, 1, 0]),
#                    label="$C_2$ through two opposite edges"),
#
# then add its name to that molecule's "curated" list in MOLECULES.
# ---------------------------------------------------------------------
PLAN_B = "planar molecule: atoms in the $\\sigma_h$ plane are unmoved"
OPS = {
 "water": {
   "E":        dict(mode="rot", axis=[0,0,1], angle=0, view=(14,-78), elem=("none",None), label="$E$ — identity (everything stays in place)"),
   "C2":       dict(mode="rot", axis=[0,0,1], angle=180, view=(14,-78), elem=("axis",[0,0,1]), label="$C_2$ — 180° about the bisector axis"),
   "sigma_v":  dict(mode="refl", normal=[0,1,0], view=(16,-72), elem=("plane",[0,1,0]), label="$\\sigma_v(xz)$ — the molecular plane",
                    note="molecular plane: its atoms are unmoved by this reflection"),
   "sigma_vp": dict(mode="refl", normal=[1,0,0], view=(14,-78), elem=("plane",[1,0,0]), label="$\\sigma_v'(yz)$ — perpendicular plane (swaps the two H)"),
 },
 "ammonia": {
   "C3":       dict(mode="rot", axis=[0,0,1], angle=120, view=(22,-60), elem=("axis",[0,0,1]), label="$C_3$ — 120° about the principal axis"),
   "sigma_v":  dict(mode="refl", normal=[1,0,0], view=(20,-60), elem=("plane",[1,0,0]), label="$\\sigma_v$ — vertical plane through one N–H bond"),
 },
 "benzene": {
   "C6":   dict(mode="rot", axis=[0,0,1], angle=60,  view=(80,-60), elem=("axis",[0,0,1]),  label="$C_6$ — rotation by 60°"),
   "C3":   dict(mode="rot", axis=[0,0,1], angle=120, view=(80,-60), elem=("axis",[0,0,1]),  label="$C_3$ — rotation by 120°"),
   "C2":   dict(mode="rot", axis=[0,0,1], angle=180, view=(80,-60), elem=("axis",[0,0,1]),  label="$C_2$ — rotation by 180°"),
   "C2_principal": dict(mode="rot", axis=[0,0,1], angle=180, view=(15,-72), elem=("axis",[0,0,1]),
                       triad=True,
                       label="Principal axis, perpendicular to the plane of the molecule",
                       note="180° turn ($C_2$) about the axis that is also $C_6$ and $C_3$"),
   "C2p":  dict(mode="rot", axis=[1,0,0], angle=180, view=(22,-72), elem=("axis",[1,0,0]), triad=True, label="$C_2'$ — 180° through two atoms"),
   "C2pp": dict(mode="rot", axis=[np.cos(d(30)),np.sin(d(30)),0], angle=180, view=(22,-72), elem=("axis",[np.cos(d(30)),np.sin(d(30)),0]), triad=True, label="$C_2''$ — 180° through two bonds"),
   "sigma_h": dict(mode="refl", normal=[0,0,1], view=(16,-72), elem=("plane",[0,0,1]), label="$\\sigma_h$ — reflection in the molecular plane", note=PLAN_B),
   "sigma_v": dict(mode="refl", normal=[0,1,0], view=(38,-72), elem=("plane",[0,1,0]), label="$\\sigma_v$ — vertical plane (through atoms)"),
   "sigma_d": dict(mode="refl", normal=[np.cos(d(120)),np.sin(d(120)),0], view=(38,-72), elem=("plane",[np.cos(d(120)),np.sin(d(120)),0]), label="$\\sigma_d$ — vertical plane (between atoms)"),
   "i":    dict(mode="inv", view=(26,-58), elem=("point",None), label="$i$ — inversion through the centre"),
   "S6":   dict(mode="improper", axis=[0,0,1], angle=60,  view=(40,-60), elem=("axis",[0,0,1]), label="$S_6$ — rotate 60° then reflect in $\\sigma_h$", planar=True),
   "S3":   dict(mode="improper", axis=[0,0,1], angle=120, view=(40,-60), elem=("axis",[0,0,1]), label="$S_3$ — rotate 120° then reflect in $\\sigma_h$", planar=True),
 },
 "ethane": {
   "C3":   dict(mode="rot", axis=[0,0,1], angle=120, view=(20,-60), elem=("axis",[0,0,1]), label="$C_3$ — 120° about the C–C axis"),
   "C2":   dict(mode="rot", axis=[np.cos(d(60)),np.sin(d(60)),0], angle=180, view=(12,-60), elem=("axis",[np.cos(d(60)),np.sin(d(60)),0]), label="$C_2$ — 180° about an axis $\\perp$ to C–C"),
   "i":    dict(mode="inv", view=(20,-55), elem=("point",None), label="$i$ — inversion through the C–C midpoint"),
   "S6":   dict(mode="improper", axis=[0,0,1], angle=60, view=(22,-60), elem=("axis",[0,0,1]), label="$S_6$ — rotate 60° then reflect $\\perp$ to C–C"),
   "sigma_d": dict(mode="refl", normal=[1,0,0], view=(22,-60), elem=("plane",[1,0,0]), label="$\\sigma_d$ — plane containing the C–C axis"),
 },
 "methane": {
   "C3":   dict(mode="rot", axis=[1,1,1], angle=120, view=(20,-60), elem=("axis",[1,1,1]), label="$C_3$ — 120° about a C–H bond axis"),
   "C2":   dict(mode="rot", axis=[0,0,1], angle=180, view=(18,-55), elem=("axis",[0,0,1]), label="$C_2$ — 180° about an edge-midpoint axis"),
   "S4":   dict(mode="improper", axis=[0,0,1], angle=90, view=(20,-55), elem=("axis",[0,0,1]), label="$S_4$ — rotate 90° then reflect $\\perp$  (CH$_4$ has $S_4$ but no $C_4$)"),
   "sigma_d": dict(mode="refl", normal=[0,1,-1], view=(18,-55), elem=("plane",[0,1,-1]), label="$\\sigma_d$ — plane through two H, bisecting the other two"),
 },
 "pf5": {
   "C3":   dict(mode="rot", axis=[0,0,1], angle=120, view=(18,-60), elem=("axis",[0,0,1]), label="$C_3$ — 120° about the axial axis"),
   "C2":   dict(mode="rot", axis=[1,0,0], angle=180, view=(10,-60), elem=("axis",[1,0,0]), label="$C_2$ — 180° through an equatorial F"),
   "sigma_h": dict(mode="refl", normal=[0,0,1], view=(12,-60), elem=("plane",[0,0,1]), label="$\\sigma_h$ — equatorial plane (swaps the two axial F)"),
   "S3":   dict(mode="improper", axis=[0,0,1], angle=120, view=(18,-60), elem=("axis",[0,0,1]), label="$S_3$ — rotate 120° then reflect in $\\sigma_h$"),
   "sigma_v": dict(mode="refl", normal=[0,1,0], view=(16,-60), elem=("plane",[0,1,0]), label="$\\sigma_v$ — vertical plane through an equatorial F"),
 },
}
# =====================================================================
#  MOLECULES  --  the registry
# =====================================================================
#   geom     the geometry function from above
#   lim      half-width of the drawing box; enlarge if anything is clipped
#   curated  the operations rendered when you run the script with no arguments
#   numbers  (optional) atom indices to print a number beside
# ---------------------------------------------------------------------
MOLECULES = {
 "water":   dict(geom=water,   lim=1.6, curated=["C2","sigma_v","sigma_vp"], inv_test=True),
 "ammonia": dict(geom=ammonia, lim=1.7, curated=["C3","sigma_v"]),
 "benzene": dict(geom=benzene, lim=3.0, walks=[6], curated=["C6","C3","C2","C2_principal","C2p","C2pp","sigma_v","sigma_d","i","S6"]),
 "ethane":  dict(geom=ethane,  lim=2.3, curated=["C3","C2","i","S6","sigma_d"]),
 "methane": dict(geom=methane, lim=1.8, curated=["C3","C2","S4","sigma_d"], inv_test=True),
 "pf5":     dict(geom=pf5,     lim=2.3, curated=["C3","C2","sigma_h","S3","sigma_v"]),
}

OPS["co2"]={
 "i":  dict(mode="inv", view=(12,-70), elem=("point",None), label="$i$ \u2014 inversion through the centre"),
 "C2": dict(mode="rot", axis=[1,0,0], angle=180, view=(12,-70), elem=("axis",[1,0,0]), label="$C_2$ \u2014 180\u00b0 about a perpendicular axis"),
}
MOLECULES["co2"]=dict(geom=co2, lim=1.9, curated=["i","C2"])

OPS["bcl3"]={
 "sigma_h": dict(mode="refl", normal=[0,0,1], view=(20,-82), elem=("plane",[0,0,1]),
                 label="$\\sigma_h$ \u2014 the $xy$ (molecular) plane",
                 note=PLAN_B, triad=True, orbital=dict(atom=0, kind="pz"),
                 unmoved_note=("green ring = unmoved (4 of 4) \u2014 but $p_z$ changes sign, "
                               "so $\\chi(\\sigma_h)=-1$")),
 "sigma_v": dict(mode="refl", normal=[0,1,0], view=(20,-62), elem=("plane",[0,1,0]),
                 label="$\\sigma_v$ \u2014 an $xz$ plane (swaps two Cl)", triad=True,
                 unmoved_note=("green ring = unmoved (2 of 4) \u2014 the other two Cl "
                               "are exchanged")),
 "C3": dict(mode="rot", axis=[0,0,1], angle=120, view=(52,-62), elem=("axis",[0,0,1]),
            label="$C_3$ \u2014 120\u00b0 about the principal axis ($\\perp$ to the plane)"),
 "C2": dict(mode="rot", axis=[1,0,0], angle=180, view=(18,-66), elem=("axis",[1,0,0]),
            label="$C_2$ \u2014 180\u00b0 (in-plane, along a B\u2013Cl bond)"),
 "S3": dict(mode="improper", axis=[0,0,1], angle=120, view=(40,-62), elem=("axis",[0,0,1]),
            label="$S_3$ \u2014 rotate 120\u00b0 then reflect in $\\sigma_h$", planar=True),
}
MOLECULES["bcl3"]=dict(geom=bcl3, lim=2.3, curated=["sigma_h","sigma_v","C3","C2","S3"])

_AB4_VIEW=(42,-115)          # one shared camera so the three planes are comparable
OPS["ab4"]={
 # --- the three types of mirror plane (the teaching set) ---
 "sigma_h":  dict(plane_alpha=0.16, mode="refl", normal=[0,0,1], view=_AB4_VIEW, elem=("plane",[0,0,1]),
                  label="$\\sigma_h$ \u2014 horizontal: $\\perp$ to $C_4$",
                  triad=True,
                  unmoved_note=("horizontal = $\\perp$ to the principal axis. It is the "
                                "molecular plane, so all 5 atoms are unmoved")),
 "sigma_v":  dict(plane_alpha=0.16, mode="refl", normal=[0,1,0], view=_AB4_VIEW, elem=("plane",[0,1,0]),
                  label="$\\sigma_v$ \u2014 vertical: through two B",
                  triad=True,
                  unmoved_note=("vertical = contains the principal axis. This one runs "
                                "THROUGH two B, so 3 of 5 atoms are unmoved")),
 "sigma_d":  dict(plane_alpha=0.16, mode="refl", normal=[-1,1,0], view=_AB4_VIEW,
                  elem=("plane",[-1,1,0]),
                  label="$\\sigma_d$ \u2014 dihedral: between the B",
                  triad=True,
                  unmoved_note=("dihedral = vertical, but BISECTS the B\u2013A\u2013B angles, "
                                "so only A is unmoved (1 of 5)")),
 # --- the rest of D4h, for later use ---
 "C4":   dict(mode="rot", axis=[0,0,1], angle=90,  view=(58,-115), elem=("axis",[0,0,1]),
              label="$C_4$ \u2014 90\u00b0 about the principal axis"),
 "C2":   dict(mode="rot", axis=[0,0,1], angle=180, view=(58,-115), elem=("axis",[0,0,1]),
              label="$C_2$ \u2014 180\u00b0 about the principal axis ($=C_4^2$)"),
 "C2p":  dict(mode="rot", axis=[1,0,0], angle=180, view=(24,-115), elem=("axis",[1,0,0]),
              label="$C_2'$ \u2014 180\u00b0 in-plane, through two B"),
 "C2pp": dict(mode="rot", axis=[1,1,0], angle=180, view=(24,-115), elem=("axis",[1,1,0]),
              label="$C_2''$ \u2014 180\u00b0 in-plane, between the B"),
 "i":    dict(mode="inv", view=_AB4_VIEW, elem=("point",None),
              label="$i$ \u2014 inversion through the centre (A)",
              unmoved_note=("inversion sends every atom straight through the centre to "
                            "the opposite side: 1\u21943, 2\u21944; only A is unmoved (1 of 5)")),
 "S4":   dict(mode="improper", axis=[0,0,1], angle=90, view=(44,-115), elem=("axis",[0,0,1]),
              label="$S_4$ \u2014 rotate 90\u00b0 then reflect in $\\sigma_h$", planar=True),
}
MOLECULES["ab4"]=dict(geom=ab4, lim=2.85, numbers=[1,2,3,4],
                      curated=["sigma_h","sigma_v","sigma_d","i"])

OPS["tcb"]={
 "C3": dict(mode="rot", axis=[0,0,1], angle=120, view=(66,-60), elem=("axis",[0,0,1]), label="$C_3$ \u2014 120\u00b0 ($\\perp$ to the plane)"),
 "C2": dict(mode="rot", axis=[1,0,0], angle=180, view=(20,-66), elem=("axis",[1,0,0]), label="$C_2$ \u2014 180\u00b0 (in-plane, through a Cl)"),
 "sigma_h": dict(mode="refl", normal=[0,0,1], view=(18,-66), elem=("plane",[0,0,1]), label="$\\sigma_h$ \u2014 the molecular plane", note=PLAN_B),
}
MOLECULES["tcb"]=dict(geom=tcb, lim=3.7, curated=["C3","C2"])

# ----------------------------- rendering -----------------------------
def _wrap(s, width):
    """Wrap on spaces only; the $...$ spans used here contain no spaces, so
    inline maths is never split."""
    return textwrap.fill(s, width) if len(s) > width else s

def _fit_fontsize(fig, text, start, min_size=6.0, max_frac=0.95, **kw):
    """Return the largest fontsize <= start at which `text` fits inside the
    figure width. Measured once, then reused for every frame."""
    a = fig.text(0.5, 0.5, text, ha="center", fontsize=start, **kw)
    r = fig.canvas.get_renderer()
    fs = start
    while fs > min_size:
        if a.get_window_extent(renderer=r).width <= max_frac*fig.bbox.width:
            break
        fs -= 0.5
        a.set_fontsize(fs)
    a.remove()
    return fs

def draw_element(ax, kind, vec, lim, alpha=0.22):
    if kind=="axis":
        v=np.asarray(vec,float); v=v/np.linalg.norm(v)*lim*0.95
        ax.plot([-v[0],v[0]],[-v[1],v[1]],[-v[2],v[2]],color=AXIS_C,lw=2.2,alpha=0.9,zorder=0)
    elif kind=="plane":
        n=np.asarray(vec,float); n=n/np.linalg.norm(n)
        a=np.array([1,0,0]) if abs(n[0])<0.9 else np.array([0,1,0])
        u=a-(a@n)*n; u/=np.linalg.norm(u); v=np.cross(n,u); g=lim*0.9
        sq=[u*g+v*g, u*g-v*g, -u*g-v*g, -u*g+v*g]
        ax.add_collection3d(Poly3DCollection([np.array(sq)],facecolor=PLANE_C,alpha=alpha,edgecolor="none"))
        # Outline drawn separately: a translucent fill alone washes out to near-white
        # on bright displays, so the plane must not rely on the fill to be visible.
        loop=np.array(sq+[sq[0]])
        ax.plot(loop[:,0],loop[:,1],loop[:,2],color=PLANE_EDGE,lw=1.6,alpha=0.85,zorder=1)
    elif kind=="point":
        ax.scatter([0],[0],[0],color=INV_C,s=170,depthshade=False,zorder=10)

def _spheroid(c, rxy, rz, n=18):
    u=np.linspace(0,2*np.pi,n); v=np.linspace(0,np.pi,n)
    return (c[0]+rxy*np.outer(np.cos(u),np.sin(v)),
            c[1]+rxy*np.outer(np.sin(u),np.sin(v)),
            c[2]+rz *np.outer(np.ones_like(u),np.cos(v)))

def draw_orbital(ax, op, s, centre, spec):
    """Draw a p orbital on `centre` and carry it through the operation, so the
    phase change (or lack of it) is visible.  The lobes are points in space, so
    they follow exactly the same path as the atoms."""
    ax_i={"px":0,"py":1,"pz":2}[spec.get("kind","pz")]
    L=spec.get("length",0.52); rxy=spec.get("rxy",0.42)
    gap=spec.get("gap",0.30)          # clearance so the nucleus is not swallowed
    off=np.zeros(3); off[ax_i]=gap+L
    lobes=np.vstack([centre+off, centre-off])
    Lb=frame_coords(op, lobes, s)
    r=[rxy,rxy,rxy]; r[ax_i]=L
    for pt,col in ((Lb[0],PHASE_P), (Lb[1],PHASE_N)):
        ax.plot_surface(*_spheroid(pt,rxy,L), color=col, alpha=0.45,
                        shade=True, linewidth=0, zorder=1)
    # Signs drawn as a 2D overlay: 3D text is depth-sorted, so the far lobe
    # would otherwise hide its own sign behind the surface.
    for pt,col,sym in ((Lb[0],PHASE_P,"+"), (Lb[1],PHASE_N,"\u2212")):
        xp,yp,_=proj3d.proj_transform(pt[0],pt[1],pt[2],ax.get_proj())
        t=ax.text2D(xp,yp,sym,transform=ax.transData,color="white",fontsize=16,
                    weight="bold",ha="center",va="center",zorder=20)
        t.set_path_effects([pe.withStroke(linewidth=2.4, foreground=col)])

def draw_triad(ax, op, s, mirror=True):
    """Small x/y/z triad that undergoes the SAME operation as the molecule,
    so students can see e.g. z -> -z as it happens."""
    ax.clear(); ax.set_axis_off(); ax.set_facecolor("none")
    ax.set_xlim(-1,1); ax.set_ylim(-1,1); ax.set_zlim(-1,1)
    ax.set_box_aspect((1,1,1)); ax.view_init(elev=op["view"][0],azim=op["view"][1])
    if mirror and op["mode"]=="refl":
        n=np.asarray(op["normal"],float); n=n/np.linalg.norm(n)
        a=np.array([1,0,0]) if abs(n[0])<0.9 else np.array([0,1,0])
        u=a-(a@n)*n; u/=np.linalg.norm(u); v=np.cross(n,u); g=0.9
        sq=[u*g+v*g, u*g-v*g, -u*g-v*g, -u*g+v*g]
        ax.add_collection3d(Poly3DCollection([np.array(sq)],facecolor=PLANE_C,
                                             alpha=0.22,edgecolor="none"))
        loop=np.array(sq+[sq[0]])
        ax.plot(loop[:,0],loop[:,1],loop[:,2],color=PLANE_EDGE,lw=1.1,alpha=0.85)
    V0=np.eye(3); V=frame_coords(op, V0, s)
    Vf=frame_coords(op, V0, 1.0)                    # final images decide "cleanliness"
    for k,(lab,v0,v,vf) in enumerate(zip("xyz", V0, V, Vf)):
        # Show an axis ONLY if it maps exactly onto itself (+1) or its opposite
        # (-1). An axis that rotates to an intermediate direction (e.g. x, y under
        # C2'') gets no +/- label that tells the truth, so it is omitted here and
        # covered verbally instead. Decided from the FINAL image so a clean axis
        # never flickers off while passing through an intermediate mid-animation.
        if abs(float(vf@v0)) < 0.999:
            continue
        moved=np.linalg.norm(v-v0)>1e-6
        col=TRACK["color"] if moved else AXIS_C
        if moved:                                  # ghost of the starting direction
            ax.plot([0,v0[0]],[0,v0[1]],[0,v0[2]],color="#b9b9b9",lw=1.0,ls=(0,(2,3)))
        if np.linalg.norm(v)<0.06: continue
        ax.quiver(0,0,0,*v,color=col,lw=2.4 if moved else 1.8,arrow_length_ratio=0.22)
        txt=lab if v@v0>=0 else "$-$"+lab
        ax.text(v[0]*1.4,v[1]*1.4,v[2]*1.4,txt,color=col,fontsize=11,
                weight="bold",ha="center",va="center")

def render(mol, opname, outdir=".", mark_unmoved=False, suffix="",
           triad=None, orbital=None, numbers=None, unmoved_pool=None,
           track_with_unmoved=False):
    M=MOLECULES[mol]; op=OPS[mol][opname]; lim=M["lim"]
    P0,elems,bonds,deftrack=M["geom"]()
    track=op.get("track", choose_tracker(op, P0, deftrack))
    sched=schedule(op["mode"])
    static_note=op.get("note", "" if op["mode"]=="rot" else "atoms shown moving to their image positions")
    unmoved=[]
    if mark_unmoved:
        P1=frame_coords(op,P0,1.0)
        pool=range(len(elems)) if unmoved_pool is None else unmoved_pool
        unmoved=[i for i in pool if np.linalg.norm(P1[i]-P0[i])<1e-6]
        if not track_with_unmoved:
            track=-1                               # let the green ring carry the emphasis
        # with track_with_unmoved the two channels coexist: crimson FILL = "this atom
        # moves" (choose_tracker returns -1 if none does, so the meaning is safe), and
        # a green RING = "this atom is unmoved, so it counts toward chi"
        static_note=op.get("unmoved_note",
                    f"green ring = atom unmoved by the operation \u2192 it counts toward \u03c7"
                    f"   (here {len(unmoved)} of {len(elems)})")
    if triad   is None: triad   = op.get("triad", False)
    if orbital is None: orbital = op.get("orbital", None)
    if numbers is None: numbers = M.get("numbers", None)
    fig=plt.figure(figsize=FIG_SIZE, dpi=DPI); ax=fig.add_subplot(111,projection="3d")
    ax_t=fig.add_axes([0.745,0.225,0.230,0.230],projection="3d") if triad else None
    title_txt=_wrap(f"{DISPLAY.get(mol, mol.capitalize())}:  {op['label']}", 52)
    fig.canvas.draw()
    title_fs=_fit_fontsize(fig, title_txt, 11.0)
    note_fs ={}
    def update(k):
        ax.clear(); ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
        ax.set_box_aspect((1,1,1)); ax.set_axis_off()
        ax.view_init(elev=op["view"][0],azim=op["view"][1])
        ss=sched[k]
        P=frame_coords(op,P0,ss); draw_element(ax,op["elem"][0],op["elem"][1],lim,op.get("plane_alpha",0.22))
        if op["mode"]=="improper" and ss>=0.5-1e-9:        # reveal sigma_h once rotation is done
            draw_element(ax,"plane",op["axis"],lim,op.get("plane_alpha",0.22))
        if orbital is not None:                   # behind the atoms, so nuclei stay visible
            draw_orbital(ax, op, ss, P0[orbital.get("atom",0)], orbital)
        for i,j in bonds: ax.plot(*zip(P[i],P[j]),color=BOND_C,lw=3.0,alpha=0.95,zorder=2)
        for idx,e in enumerate(elems):
            st=TRACK if idx==track else STYLE[e]
            ax.scatter(*P[idx],color=st["color"],s=st["s"],edgecolor=st["edge"],linewidth=0.8,depthshade=True,zorder=4)
        for idx in unmoved:                       # ring the atoms left in place
            ax.scatter(*P[idx],s=STYLE[elems[idx]]["s"]*2.7,facecolors="none",
                       edgecolors="#2E8B57",linewidths=3.0,depthshade=False,zorder=6)
        if numbers:                               # index the equivalent atoms so swaps are followable
            # drawn as a 2D overlay: 3D text is depth-sorted and the rear atoms
            # would otherwise hide their own label behind the sphere.
            for n,idx in enumerate(numbers,1):
                xp,yp,_=proj3d.proj_transform(P[idx][0],P[idx][1],P[idx][2],ax.get_proj())
                ax.text2D(xp,yp,str(n),transform=ax.transData,color="#22313f",
                          fontsize=9.5,weight="bold",ha="center",va="center",zorder=20)
        if ax_t is not None:
            draw_triad(ax_t, op, ss)
            ax_t.text2D(0.5,-0.05,"axes follow the operation",transform=ax_t.transAxes,
                        ha="center",fontsize=7,style="italic",color="#999")
        ax.set_title(title_txt,fontsize=title_fs,pad=2)
        if op["mode"]=="improper":
            if ss<=0.5+1e-9:
                nt=f"Step 1 of 2:  rotate by {op['angle']:g}\u00b0  ($C_n$)"
            else:
                nt="Step 2 of 2:  reflect $\\perp$ to the axis  ($\\sigma_h$)"
                if "planar" in static_note: nt+="  — planar: atoms unmoved"
        else:
            nt=static_note
        if nt:
            if nt not in note_fs:
                note_fs[nt]=(_wrap(nt,64), _fit_fontsize(fig,_wrap(nt,64),8.5,style="italic"))
            w,fs=note_fs[nt]
            ax.text2D(0.5,0.015,w,transform=ax.transAxes,ha="center",va="bottom",
                      fontsize=fs,style="italic",color="#666")
        return []
    out=f"{outdir}/{mol}_{opname}{suffix}.gif"
    FuncAnimation(fig,update,frames=len(sched),interval=1000/FPS,blit=False).save(
        out, writer=PlayOnceWriter(fps=FPS))
    plt.close(fig); print("wrote", out)

# ----------------------------- normal-mode animations -----------------------------
MASS = {"H":1.0,"C":12.0,"N":14.0,"O":16.0,"F":19.0,"Cl":35.5,"P":31.0}
def _u(v):
    v=np.asarray(v,float); n=np.linalg.norm(v); return v/n if n>1e-12 else v
def _perp_in_plane(b):                 # in-plane perpendicular to bond b, pointing downward
    b=np.asarray(b,float)
    normal = np.array([1.0,0,0]) if abs(b[0])<abs(b[1]) else np.array([0,1.0,0])
    p=np.cross(normal,b)
    if p[2]>0: p=-p
    return _u(p)

# The normal-mode figures now use the SAME xz-plane water() as the symmetry-operation
# figures, so the whole course runs one convention: the molecular plane is sigma_v(xz)
# and the antisymmetric stretch (H motion along x) transforms as p_x = B_1, with its
# dipole derivative along x.
MODE_GEOM    = {}                                  # no geometry override; every figure uses water()
MODE_CHARGES = {"water": {"O":-2.0, "H":1.0}}      # schematic, net-neutral; sets the dipole direction

def water_modes():
    P,_,_,_=water(); b1=_u(P[1]-P[0]); b2=_u(P[2]-P[0])
    return {
      "sym_stretch": dict(disp=np.array([[0,0,0],b1,b2]), amp=0.30,
          label="$A_1$ \u2014 symmetric stretch ($\\nu_1$)", note="both O\u2013H stretch together, in phase"),
      "bend":        dict(disp=np.array([[0,0,0],_perp_in_plane(b1),_perp_in_plane(b2)]), amp=0.45,
          label="$A_1$ \u2014 bend ($\\nu_2$)", note="the H\u2013O\u2013H angle opens and closes"),
      "asym_stretch":dict(disp=np.array([[0,0,0],b1,-b2]), amp=0.30,
          label="$B_1$ \u2014 antisymmetric stretch ($\\nu_3$)", note="one O\u2013H stretches as the other compresses"),
    }
def ammonia_modes():
    P,_,_,_=ammonia()
    b=[_u(P[i]-P[0]) for i in (1,2,3)]
    rad=[_u([P[i,0],P[i,1],0.0]) for i in (1,2,3)]
    umb=np.array([[0,0,0]]+[0.5*rad[i]+np.array([0,0,-1.0]) for i in range(3)])
    return {
      "sym_stretch":dict(disp=np.array([[0,0,0],b[0],b[1],b[2]]), amp=0.30,
          label="$A_1$ \u2014 symmetric stretch ($\\nu_1$)", note="all three N\u2013H stretch together"),
      "umbrella":   dict(disp=umb, amp=0.42,
          label="$A_1$ \u2014 symmetric bend / umbrella ($\\nu_2$)", note="the H$_3$ umbrella flaps; N moves the opposite way"),
      "e_stretch":  dict(disp=np.array([[0,0,0],2*b[0],-b[1],-b[2]]), amp=0.26,
          label="$E$ \u2014 asymmetric stretch ($\\nu_3$)", note="one of a degenerate pair"),
      "e_bend":     dict(disp=np.array([[0,0,0],[1.0,0,0],[1.0,0,0],[1.0,0,0]]), amp=0.42,
          label="$E$ \u2014 asymmetric bend ($\\nu_4$)", note="one of a degenerate pair"),
    }
def co2_modes():
    # C at origin (index 0); O1 at +z (1); O2 at -z (2).  Ratios keep the COM fixed.
    return {
      "sym_stretch": dict(disp=np.array([[0,0,0.0],[0,0,1.0],[0,0,-1.0]]), amp=0.32,
          label="$\\Sigma_g^+$ \u2014 symmetric stretch", note="Raman active; IR inactive (no dipole change)"),
      "asym_stretch":dict(disp=np.array([[0,0,-2.6667],[0,0,1.0],[0,0,1.0]]), amp=0.34,
          label="$\\Sigma_u^+$ \u2014 antisymmetric stretch", note="IR active (the dipole changes); Raman inactive"),
      "bend":        dict(disp=np.array([[-2.6667,0,0.0],[1.0,0,0],[1.0,0,0]]), amp=0.34,
          label="$\\Pi_u$ \u2014 bend (doubly degenerate)", note="IR active; Raman inactive \u2014 one of a degenerate pair"),
    }
MODES = {"water": water_modes(), "ammonia": ammonia_modes(), "co2": co2_modes()}
# water now sits in the xz-plane, so the camera moves from the +x axis (azim 0, which
# showed the old yz geometry face-on) round to the -y axis, which shows xz face-on.
MODE_VIEW = {"water":(6,-90), "ammonia":(14,-62), "co2":(8,-86)}
MODE_LIM  = {"ammonia":1.35}

def mode_disp(mol,key):
    coords,elems,bonds,_=MODE_GEOM.get(mol, MOLECULES[mol]["geom"])()
    raw=np.array(MODES[mol][key]["disp"],float).copy()
    m=np.array([MASS[e] for e in elems])
    raw=raw-(m[:,None]*raw).sum(0)/m.sum()        # remove net translation (COM fixed)
    mx=np.linalg.norm(raw,axis=1).max()
    if mx>1e-9: raw=raw/mx                         # scale so the largest displacement = 1
    return coords,elems,bonds,raw

def render_mode(mol,key,outdir=".",show_dipole=False):
    M=MOLECULES[mol]; mode=MODES[mol][key]; lim=MODE_LIM.get(mol, M["lim"])
    P0,elems,bonds,disp=mode_disp(mol,key)
    amp=mode.get("amp",0.34); view=mode.get("view",MODE_VIEW.get(mol,(18,-66)))
    FIX=0.12                                           # below this fraction of the largest motion, an atom counts as stationary
    moving=np.array([np.linalg.norm(disp[i])>=FIX for i in range(len(elems))])
    move_disp=disp*moving[:,None]                      # stationary atoms are held EXACTLY put
    ph=np.linspace(0,1,FRAMES,endpoint=False)
    # NB: green is reserved throughout the course for "unmoved by a symmetry operation
    # -> counts toward chi". A vibrational figure is saying something different (this
    # atom's displacement is too small to draw), so it gets its own neutral marker.
    PINNED="#6b7a8a"; VEC="#b5341f"; DIP="#6a3d9a"
    charges=MODE_CHARGES.get(mol) if show_dipole else None
    if charges is not None:
        q=np.array([charges[e] for e in elems]); mvec=np.array([MASS[e] for e in elems])
        base=(mvec[:,None]*P0).sum(0)/mvec.sum()       # anchor the dipole at the (fixed) centre of mass
        mu_eq=-(q[:,None]*P0).sum(0); dscale=0.60      # sign chosen so mu points toward the delta- O
        dipnote={"bend":"$\\mu$ changes along $z$  ($A_1$)  \u2014 IR active",
                 "sym_stretch":"$\\mu$ changes along $z$  ($A_1$)  \u2014 IR active",
                 "asym_stretch":"$\\mu$ swings along $x$  ($B_1$, like $p_x$)  \u2014 IR active"}.get(key,"")
    fig=plt.figure(figsize=FIG_SIZE, dpi=DPI); ax=fig.add_subplot(111,projection="3d")
    def update(k):
        ax.clear(); ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
        ax.set_box_aspect((1,1,1)); ax.set_axis_off(); ax.view_init(elev=view[0],azim=view[1])
        f=amp*np.sin(2*np.pi*ph[k]); P=P0+f*move_disp
        for i,j in bonds: ax.plot(*zip(P[i],P[j]),color=BOND_C,lw=3.0,alpha=0.95,zorder=2)
        for idx,e in enumerate(elems):
            st=STYLE[e]; ax.scatter(*P[idx],color=st["color"],s=st["s"],edgecolor=st["edge"],linewidth=0.8,depthshade=True,zorder=4)
        # dashed slate ring marks the atom(s) whose motion is too small to draw here
        for idx in range(len(elems)):
            if not moving[idx]:
                ax.scatter(*P0[idx],s=STYLE[elems[idx]]["s"]*2.7,facecolors="none",
                           edgecolors=PINNED,linewidths=1.8,linestyle=(0,(4,3)),
                           depthshade=False,zorder=6)
        # a displacement vector rides with each moving atom, pointing along the mode
        for i in range(len(elems)):
            if moving[i]:
                v=disp[i]*0.55
                ax.quiver(P[i,0],P[i,1],P[i,2],v[0],v[1],v[2],color=VEC,lw=2.4,alpha=0.9,
                          arrow_length_ratio=0.34,zorder=5)
        # dipole-moment vector (schematic), with a faint equilibrium reference
        if charges is not None:
            mu=-(q[:,None]*P).sum(0)
            ax.quiver(*base,*(dscale*mu_eq),color=DIP,lw=1.6,alpha=0.30,arrow_length_ratio=0.12,zorder=3)
            ax.quiver(*base,*(dscale*mu),color=DIP,lw=3.2,alpha=0.95,arrow_length_ratio=0.14,zorder=7)
            tip=base+dscale*mu; ax.text(tip[0],tip[1],tip[2]+0.10,"$\\mu$",color=DIP,fontsize=13,weight="bold",ha="center")
        ax.set_title(f"{DISPLAY.get(mol,mol.capitalize())}:  {mode['label']}",fontsize=11,pad=2)
        if charges is not None and dipnote:
            ax.text2D(0.5,0.05,dipnote,transform=ax.transAxes,ha="center",fontsize=9,color=DIP)
        elif mode.get("note"):
            ax.text2D(0.5,0.055,mode["note"],transform=ax.transAxes,ha="center",fontsize=8.5,style="italic",color="#666")
        if (~moving).any(): ax.text2D(0.5,0.008,"dashed ring = motion too small to draw (held fixed here)",
                                      transform=ax.transAxes,ha="center",fontsize=8,color=PINNED)
        return []
    suffix="_dipole" if show_dipole else ""
    FuncAnimation(fig,update,frames=FRAMES,interval=1000/FPS,blit=False).save(
        f"{outdir}/{mol}_mode_{key}{suffix}.gif",writer=PlayOnceWriter(fps=FPS))
    plt.close(fig); print("wrote",f"{outdir}/{mol}_mode_{key}{suffix}.gif")

def _cn_equiv(k, n):
    """Label for the k-th step of a stepped C_n walk: the C_n^k name plus the
    lowest-order operation it is equivalent to (C6^2 = C3, C6^3 = C2, ...)."""
    from math import gcd
    if k == 0:
        return "start", "$E$  (identity)"
    g = gcd(k, n); order = n//g; power = k//g
    primary = f"$C_{{{n}}}^{{{k}}}$"
    if order == 1:
        equiv = "$=\\,E$  (back to start)"
    elif power == 1:
        equiv = f"$=\\,C_{{{order}}}$"
    else:
        equiv = f"$=\\,C_{{{order}}}^{{{power}}}$"
    # C6^1 and C6^5 have no simpler equivalent (gcd = 1, order = n)
    return primary, ("" if order == n else equiv)

def render_cn_walk(mol, n=None, axis=(0,0,1), outdir=".", view=None,
                   hold_frames=11, turn_frames=16, suffix=""):
    """Step through a full C_n cycle 360/n degrees at a time, pausing at each
    stop to name the cumulative operation C_n^k and the simpler operation it
    equals. One tracked atom (crimson) lets students count the turns; a wedge
    on the axis shows how far round we are."""
    M = MOLECULES[mol]; P0, elems, bonds, _ = M["geom"]()
    lim = M["lim"]; ax_v = np.asarray(axis, float); ax_v /= np.linalg.norm(ax_v)
    if n is None: n = M["geom"]()[3]            # principal-axis order stored by geom
    if view is None: view = (58, -60)
    step = 360.0/n
    track = 6 if mol == "benzene" else next((i for i,e in enumerate(elems)
                                             if np.linalg.norm(P0[i])>1e-6), 0)

    # timeline: start hold, then each step = turn + a hold whose length depends on
    # how noteworthy the stop is (the C6^2=C3, C6^3=C2, C6^6=E stops linger longer).
    from math import gcd
    def hold_len(k):
        if k == 0: return hold_frames
        order = n//gcd(k, n)
        if order == 1:         return hold_frames + 12    # back to E
        if order != n:         return hold_frames + 8     # has a simpler equivalent
        return hold_frames                                # C6^1, C6^5
    segs = [("hold", 0, 0.0)]*hold_len(0)
    for k in range(1, n+1):
        for j in range(turn_frames):
            segs.append(("turn", k, (j+1)/turn_frames))
        segs += [("hold", k, 0.0)]*hold_len(k)

    fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)
    ax = fig.add_subplot(111, projection="3d")
    title_txt = f"{DISPLAY.get(mol, mol.capitalize())}:  $C_{{{n}}}$ principal axis \u2014 stepping $C_{{{n}}}^k$"
    fig.canvas.draw(); title_fs = _fit_fontsize(fig, title_txt, 11.0)

    def update(i):
        kind, k, frac = segs[i]
        ang = d(step*((k-1)+frac) if kind=="turn" else step*k)
        ax.clear(); ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
        ax.set_box_aspect((1,1,1)); ax.set_axis_off(); ax.view_init(elev=view[0],azim=view[1])
        R = Rmat(ax_v, ang); P = (R@P0.T).T

        # principal axis
        av = ax_v*lim*0.95
        ax.plot([-av[0],av[0]],[-av[1],av[1]],[-av[2],av[2]],color=AXIS_C,lw=2.0,alpha=0.9,zorder=0)

        # progress indicator: a bold radial pointer from the axis to the tracked
        # atom, plus faint spokes at every n-fold position so the stops are marked.
        prog = step*((k-1)+frac) if kind=="turn" else step*k
        rr = lim*0.62
        for m in range(n):                       # the n equivalent stop directions
            a0 = d(step*m); sx, sy = rr*np.cos(a0), rr*np.sin(a0)
            ax.plot([0,sx],[0,sy],[0,0], color="#c9c9c9", lw=1.0, alpha=0.7, zorder=1)
        pa = d(prog); px_, py_ = rr*np.cos(pa), rr*np.sin(pa)
        ax.plot([0,px_],[0,py_],[0,0], color=TRACK["color"], lw=3.0, alpha=0.95, zorder=3)
        ax.scatter([px_],[py_],[0], color=TRACK["color"], s=40, zorder=3, depthshade=False)

        for i2,j2 in bonds: ax.plot(*zip(P[i2],P[j2]),color=BOND_C,lw=3.0,alpha=0.95,zorder=2)
        for idx,e in enumerate(elems):
            st = TRACK if idx==track else STYLE[e]
            ax.scatter(*P[idx],color=st["color"],s=st["s"],edgecolor=st["edge"],
                       linewidth=0.8,depthshade=True,zorder=4)

        ax.set_title(title_txt, fontsize=title_fs, pad=2)

        # centred stop label
        primary, equiv = _cn_equiv(k if kind=="hold" else k, n)
        if kind == "turn":
            big = f"turning\u2026  {primary}"; sub = f"rotating {step:g}\u00b0"
        elif k == 0:
            big = "START"; sub = "before any rotation"
        else:
            big = f"{primary}   {equiv}".strip()
            sub = f"turn {k}/{n}   ({step*k:g}\u00b0 total)"
        ax.text2D(0.5, 0.11, big, transform=ax.transAxes, ha="center",
                  fontsize=15, weight="bold", color="#22313f")
        ax.text2D(0.5, 0.055, sub, transform=ax.transAxes, ha="center",
                  fontsize=10, style="italic", color="#666")
        return []

    out = f"{outdir}/{mol}_Cn_walk{suffix}.gif"
    FuncAnimation(fig, update, frames=len(segs), interval=1000/FPS, blit=False).save(
        out, writer=PlayOnceWriter(fps=FPS))
    plt.close(fig); print("wrote", out); return out

def render_staggered(kind, outdir="."):
    """'Two staggered planes' view of a complex ring, undergoing its improper
    rotation. kind='pentagons' (S10) or 'cot' (S4, cyclooctatetraene drawn as two
    perpendicular staggered rectangles). Shows the two rings, then applies Sn and
    lands the tracked vertex on the other ring."""
    cz=0.5; view=(60,-90)
    if kind=="pentagons":
        r=1.2
        front=np.array([[r*np.cos(d(a)),r*np.sin(d(a)), cz] for a in range(0,360,72)])
        back =np.array([[r*np.cos(d(a)),r*np.sin(d(a)),-cz] for a in range(36,360,72)])
        fbonds=[(i,(i+1)%5) for i in range(5)]; bbonds=fbonds
        sn=10; sn_ang=36; lim=1.7
        title="Two staggered pentagons: the $S_{10}$ axis"
        note="each vertex maps onto the other ring: rotate 36°, then reflect"
    else:  # cot
        W,Hh=1.35,0.75
        front=np.array([[W,Hh,cz],[-W,Hh,cz],[-W,-Hh,cz],[W,-Hh,cz]])
        back =np.array([[-y,x,-cz] for x,y,_ in front])
        fbonds=[(0,1),(1,2),(2,3),(3,0)]; bbonds=fbonds
        sn=4; sn_ang=90; lim=1.9
        title="Cyclooctatetraene: two staggered rectangles ($S_4$)"
        note="rotate 90°, then reflect — the two rectangles exchange"

    FCOL,BCOL="#1f7a4d","#7a4d1f"; TCOL="#d11f4a"
    seg =[("start",0,0)]*max(6,FRAMES//10)
    nr=max(14,FRAMES//3); seg+=[("rot",ease((j+1)/nr),0) for j in range(nr)]
    seg+=[("rotpause",1,0)]*max(4,FRAMES//12)
    nf=max(12,FRAMES//3); seg+=[("ref",1,ease((j+1)/nf)) for j in range(nf)]
    seg+=[("done",1,1)]*max(16,FRAMES//3)

    fig=plt.figure(figsize=FIG_SIZE, dpi=DPI); ax=fig.add_subplot(111,projection="3d")
    title=_wrap(title,46); fig.canvas.draw(); tfs=_fit_fontsize(fig,title,12.5)

    def update(k):
        kind2,fr,fref=seg[k]
        ax.clear(); ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
        ax.set_box_aspect((1,1,1)); ax.set_axis_off(); ax.view_init(*view)
        ax.plot([0,0],[0,0],[-lim*0.9,lim*0.9],color=AXIS_C,lw=1.6,alpha=0.6,zorder=0)
        if kind2 in ("ref","done"):     # mid-plane appears for the reflection
            R=lim*0.75; _flat_plane(ax,[[-R,-R,0],[R,-R,0],[R,R,0],[-R,R,0]],1.0,a_fill=0.16)

        allP=np.vstack([front,back])
        Q=(Rmat([0,0,1],d(sn_ang)*fr)@allP.T).T
        Q[:,2]=(1-fref)*Q[:,2]+fref*(-Q[:,2])
        nf_=len(front); F,B=Q[:nf_],Q[nf_:]
        for (i,j) in fbonds: ax.plot(*zip(F[i],F[j]),color=FCOL,lw=2.4,alpha=0.9,zorder=3)
        for (i,j) in bbonds: ax.plot(*zip(B[i],B[j]),color=BCOL,lw=2.4,alpha=0.7,zorder=2)
        for idx,p in enumerate(F):
            c=TCOL if idx==0 else FCOL
            ax.scatter(*p,s=240,color=c,edgecolor="#123",lw=0.6,zorder=5,depthshade=True)
        for p in B:
            ax.scatter(*p,s=210,color=BCOL,edgecolor="#321",lw=0.6,zorder=3,depthshade=True)

        ax.set_title(title,fontsize=tfs,pad=2)
        cap={"start":"two staggered rings, one vertex marked",
             "rot":f"Step 1: rotate {sn_ang}° about the axis",
             "rotpause":f"Step 1 done: rotated {sn_ang}°",
             "ref":"Step 2: reflect in the mid-plane",
             "done":note}[kind2]
        ax.text2D(0.5,0.04,cap,transform=ax.transAxes,ha="center",fontsize=10,style="italic",color="#555")
        return []

    out=f"{outdir}/staggered_{kind}.gif"
    FuncAnimation(fig,update,frames=len(seg),interval=1000/FPS,blit=False).save(
        out,writer=PlayOnceWriter(fps=FPS))
    plt.close(fig); print("wrote",out); return out

def _draw_newman2d(ax, atoms, r_H=1.0, cr=0.34):
    """Flat Newman projection looking straight down the C3/S6 axis. Each atom is
    (angle_deg, frontness, tracked): frontness 1 = front carbon's H (bond drawn to
    the CENTRE), 0 = back carbon's H (bond drawn from the CIRCLE edge). Reflection
    in the plane perpendicular to the axis swaps front and back, which is animated
    as frontness 1<->0 with the bond origin sliding between centre and circle."""
    import matplotlib.pyplot as _plt
    ax.add_patch(_plt.Circle((0,0), cr, fill=False, ec="#555", lw=1.7, zorder=2))  # back C
    ax.scatter(0,0,s=95,color="#2b2b2b",zorder=7)                                   # front C
    order=sorted(range(len(atoms)), key=lambda i: atoms[i][1])   # back first, front on top
    for i in order:
        ang,f,trk=atoms[i]
        th=np.radians(ang); dirn=np.array([np.cos(th),np.sin(th)])
        pos=r_H*dirn; inner=(1-f)*cr*dirn                    # centre if front, circle edge if back
        col="#d11f4a" if trk else "#dcdcdc"
        edge="#7a1020" if trk else "#8a8a8a"
        ax.plot([inner[0],pos[0]],[inner[1],pos[1]],color="#6f6f6f",
                lw=2.4+0.9*f,alpha=0.55+0.45*f,zorder=3+f,solid_capstyle="round")
        ax.scatter(*pos,s=360+150*f,color=col,edgecolor=edge,lw=1.0,zorder=5+f)

# initial Newman state: front H (f=1) at 90/210/330, staggered back H (f=0) at 150/270/30
_NEWMAN_INIT=[(90,1),(210,1),(330,1),(150,0),(270,0),(30,0)]

def _newman_state(k, frac_rot, frac_ref):
    """Atom list after (k-1) complete S6 operations plus a partial k-th one
    (rotate by frac_rot of 60 deg, then reflect by frac_ref)."""
    out=[]
    for idx,(a0,f_init) in enumerate(_NEWMAN_INIT):
        base=a0+60*(k-1)
        f0=f_init if (k-1)%2==0 else 1-f_init
        ang=base+60*frac_rot
        f=(1-frac_ref)*f0+frac_ref*(1-f0)
        out.append((ang,f,idx==0))
    return out

def render_ethane_s6(kind="single", outdir="."):
    lim=1.75
    def setup(ax):
        ax.clear(); ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim)
        ax.set_aspect("equal"); ax.axis("off")

    if kind=="single":
        seg =[("start",0,0)]*max(6,FRAMES//10)
        nr=max(16,FRAMES//3); seg+=[("rot",ease((j+1)/nr),0) for j in range(nr)]
        seg+=[("rotpause",1,0)]*max(4,FRAMES//12)
        nf=max(14,FRAMES//3); seg+=[("ref",1,ease((j+1)/nf)) for j in range(nf)]
        seg+=[("done",1,1)]*max(16,FRAMES//3)
        title="Ethane $S_6$ (Newman view): rotate 60°, then reflect"
        fig=plt.figure(figsize=FIG_SIZE, dpi=DPI); ax=fig.add_subplot(111)
        fig.canvas.draw(); tfs=_fit_fontsize(fig,_wrap(title,50),12.5)
        def update(k):
            kind2,fr,fref=seg[k]; setup(ax)
            atoms=_NEWMAN_INIT if kind2=="start" else _newman_state(1,fr,fref)
            if kind2=="start": atoms=[(a,f,i==0) for i,(a,f) in enumerate(_NEWMAN_INIT)]
            _draw_newman2d(ax,atoms)
            ax.set_title(title,fontsize=tfs,pad=2)
            cap={"start":"front H bond to the centre; back H bond from the circle",
                 "rot":"Step 1: rotate 60° about the axis",
                 "rotpause":"Step 1 done: rotated 60°",
                 "ref":"Step 2: reflect $\\perp$ axis  —  front $\\leftrightarrow$ back",
                 "done":"marked H is now a BACK atom: $S_6$ maps the molecule onto itself"}[kind2]
            ax.text(0.5,-0.02,cap,transform=ax.transAxes,ha="center",va="top",
                    fontsize=10,style="italic",color="#555")
            return []
        out=f"{outdir}/ethane_S6_newman.gif"
    else:  # walk through S6^1..S6^6
        from math import gcd
        def equiv(k):
            if k==6: return "$S_6^6 = E$","a full turn — six reflections cancel"
            if k%2==1:
                if k==3: return "$S_6^3 = i$","180° + reflect = inversion"
                return f"$S_6^{k}$","improper: rotation + one reflection"
            g=gcd(60*k,360); order=360//g; power=(60*k)//g
            eq=f"C_{order}" if power==1 else f"C_{{{order}}}^{power}"
            return f"$S_6^{k} = {eq}$","even power — reflections cancel, a pure rotation"
        seg=[("start",0,0.0,0.0)]*max(5,FRAMES//12)
        for kk in range(1,7):
            nr=7
            seg+=[("go",kk,ease((j+1)/nr),0.0) for j in range(nr)]              # rotate 60
            seg+=[("go",kk,1.0,ease((j+1)/nr)) for j in range(nr)]              # reflect
            seg+=[("hold",kk,1.0,1.0)]*(13 if kk in (2,3,4,6) else 6)
        title="Ethane: powers of $S_6$ (which collapse to $C_3$, $i$, $E$)"
        fig=plt.figure(figsize=FIG_SIZE, dpi=DPI); ax=fig.add_subplot(111)
        fig.canvas.draw(); tfs=_fit_fontsize(fig,_wrap(title,50),12.5)
        def update(k):
            tagk=seg[k]; kind2=tagk[0]; setup(ax)
            if kind2=="start":
                atoms=[(a,f,i==0) for i,(a,f) in enumerate(_NEWMAN_INIT)]
            else:
                _,kk,fr,fref=tagk; atoms=_newman_state(kk,fr,fref)
            _draw_newman2d(ax,atoms)
            ax.set_title(title,fontsize=tfs,pad=2)
            if kind2=="start":
                big="START"; sub="staggered ethane, one H marked"
            elif kind2=="go":
                _,kk,fr,fref=tagk; big=f"applying $S_6$  (turn {kk} of 6)"; sub="rotate 60°, then reflect"
            else:
                _,kk,fr,fref=tagk; big,sub=equiv(kk)
            ax.text(0.5,0.105,big,transform=ax.transAxes,ha="center",fontsize=15,weight="bold",color="#22313f")
            ax.text(0.5,0.048,sub,transform=ax.transAxes,ha="center",fontsize=9.5,style="italic",color="#666")
            return []
        out=f"{outdir}/ethane_S6_walk.gif"

    FuncAnimation(fig,update,frames=len(seg),interval=1000/FPS,blit=False).save(
        out,writer=PlayOnceWriter(fps=FPS))
    plt.close(fig); print("wrote",out); return out

def _group_molecules():
    """Small library for the point-group SETS. Each entry: (atoms, bonds) where
    atoms = list of (element, xyz). Geometry is schematic but the point group of
    each is correct."""
    dd=lambda deg: np.radians(deg)
    T=np.array([[1,1,1],[-1,-1,1],[1,-1,-1],[-1,1,-1]],float)/np.sqrt(3)  # tetrahedral dirs
    def tripodH(C,O,r=1.09):
        u=(O-C); u=u/np.linalg.norm(u)
        a=np.array([1,0,0.]) if abs(u[0])<0.9 else np.array([0,1,0.])
        p=a-(a@u)*u; p/=np.linalg.norm(p); q=np.cross(u,p)
        return [C+r*((-1/3)*u+np.sqrt(8)/3*(np.cos(dd(120*k+30))*p+np.sin(dd(120*k+30))*q)) for k in range(3)]
    M={}

    # ---- C2v ----
    M["water"]=([("O",[0,0,0.30]),("H",[0.76,0,-0.29]),("H",[-0.76,0,-0.29])],[(0,1),(0,2)])
    C1=np.array([-0.67,0,0]); C2=np.array([0.67,0,0])
    def sub(C,deg,r): th=dd(deg); return C+r*np.array([np.cos(th),np.sin(th),0])
    M["cis-dichloroethene"]=([("C",C1),("C",C2),
        ("Cl",sub(C1,120,1.72)),("H",sub(C1,240,1.08)),
        ("Cl",sub(C2,60,1.72)),("H",sub(C2,-60,1.08))],
        [(0,1),(0,2),(0,3),(1,4),(1,5)])
    M["dichloromethane"]=([("C",[0,0,0]),
        ("Cl",T[0]*1.77),("Cl",T[1]*1.77),("H",T[2]*1.09),("H",T[3]*1.09)],
        [(0,1),(0,2),(0,3),(0,4)])
    O=np.array([0,0,0.45]); ha=dd(55.5)
    Cme1=O+1.41*np.array([np.sin(ha),0,-np.cos(ha)]); Cme2=O+1.41*np.array([-np.sin(ha),0,-np.cos(ha)])
    atoms=[("O",O),("C",Cme1),("C",Cme2)]; bonds=[(0,1),(0,2)]
    for C in (Cme1,Cme2):
        i0=len(atoms)
        for H in tripodH(C,O): atoms.append(("H",H))
        ci=1 if C is Cme1 else 2
        bonds+=[(ci,i0),(ci,i0+1),(ci,i0+2)]
    M["dimethyl ether"]=(atoms,bonds)

    # ---- D3h ----
    def trig(center_el, out_el, r, extra_ax=None):
        atoms=[(center_el,[0,0,0])]; bonds=[]
        for k,t in enumerate((0,120,240)):
            atoms.append((out_el,[r*np.cos(dd(t)),r*np.sin(dd(t)),0])); bonds.append((0,k+1))
        if extra_ax:
            atoms.append((out_el,[0,0,extra_ax])); atoms.append((out_el,[0,0,-extra_ax]))
            bonds+=[(0,len(atoms)-2),(0,len(atoms)-1)]
        return atoms,bonds
    M["BF3"]=trig("B","F",1.31)
    M["BCl3"]=trig("B","Cl",1.75)
    M["SO3"]=trig("S","O",1.42)
    M["PF5"]=trig("P","F",1.53,extra_ax=1.58)
    M["PCl5"]=trig("P","Cl",2.02,extra_ax=2.14)
    # eclipsed ethane (D3h): two CH3 with H aligned (same azimuth top and bottom)
    dcc=0.77
    at=[("C",[0,0,dcc]),("C",[0,0,-dcc])]; bo=[(0,1)]
    for sign,ci in ((+1,0),(-1,1)):
        Cc=np.array([0,0,sign*dcc])
        for th in (0,120,240):
            h=np.array([np.sqrt(8)/3*np.cos(dd(th)),np.sqrt(8)/3*np.sin(dd(th)),sign*1/3])
            at.append(("H",Cc+1.09*h)); bo.append((ci,len(at)-1))
    M["eclipsed ethane"]=(at,bo)

    # ---- Td ----
    def tetra(center_el,out_el,r):
        atoms=[(center_el,[0,0,0])]; bonds=[]
        for k,t in enumerate(T): atoms.append((out_el,t*r)); bonds.append((0,k+1))
        return atoms,bonds
    M["methane"]=tetra("C","H",1.09)
    M["CCl4"]=tetra("C","Cl",1.77)
    M["SiF4"]=tetra("Si","F",1.56)
    from itertools import combinations
    Pv=T*1.28
    M["P4"]=([("P",v) for v in Pv],[(i,j) for i,j in combinations(range(4),2)])
    def _fillH(C,ndirs,r=1.09):
        ndirs=[n/np.linalg.norm(n) for n in ndirs]
        if len(ndirs)==3:
            h=-(ndirs[0]+ndirs[1]+ndirs[2]); h/=np.linalg.norm(h); return [C+r*h]
        d1,d2=ndirs; bis=-(d1+d2); bis/=np.linalg.norm(bis)
        n=np.cross(d1,d2); n/=np.linalg.norm(n); th=dd(54.75)
        return [C+r*(np.cos(th)*bis+np.sin(th)*n),C+r*(np.cos(th)*bis-np.sin(th)*n)]
    # Ni(CO)4: Ni centre, 4 linear Ni-C-O
    at=[("Ni",[0,0,0])]; bo=[]
    for t in T:
        c=t*1.82; o=t*(1.82+1.15); at+=[("C",c),("O",o)]; bo+=[(0,len(at)-2),(len(at)-2,len(at)-1)]
    M["Ni(CO)4"]=(at,bo)
    # adamantane C10H16: 4 bridgehead CH + 6 bridge CH2 (+H)
    sca=0.889
    BH=np.array([(1,1,1),(1,-1,-1),(-1,1,-1),(-1,-1,1)],float)*sca
    BR=np.array([(2,0,0),(-2,0,0),(0,2,0),(0,-2,0),(0,0,2),(0,0,-2)],float)*sca
    Cc=np.vstack([BH,BR]); at=[("C",p) for p in Cc]; bo=[]
    for i in range(10):
        for j in range(i+1,10):
            if np.linalg.norm(Cc[i]-Cc[j])<sca*2.0: bo.append((i,j))
    bset={(min(a,b),max(a,b)) for a,b in bo}
    for i in range(10):
        nd=[Cc[j]-Cc[i] for j in range(10) if (min(i,j),max(i,j)) in bset]
        for H in _fillH(Cc[i],nd): at.append(("H",H)); bo.append((i,len(at)-1))
    M["adamantane"]=(at,bo)
    # P4O10: P4 tetrahedron, 6 bridging O, 4 terminal O
    sP=0.924
    Pp=np.array([(1,1,1),(1,-1,-1),(-1,1,-1),(-1,-1,1)],float)*sP
    Obr=np.array([(2,0,0),(-2,0,0),(0,2,0),(0,-2,0),(0,0,2),(0,0,-2)],float)*sP
    at=[("P",p) for p in Pp]+[("O",p) for p in Obr]; bo=[]
    for i in range(4):
        for j in range(6):
            if np.linalg.norm(Pp[i]-Obr[j])<sP*2.0: bo.append((i,4+j))
        term=Pp[i]+1.45*(Pp[i]/np.linalg.norm(Pp[i])); at.append(("O",term)); bo.append((i,len(at)-1))
    M["P4O10"]=(at,bo)

    # ---- linear molecules (along x), for the infinity groups ----
    def lin(seq):   # seq = list of (element, x-position)
        atoms=[(el,[x,0,0]) for el,x in seq]
        bonds=[(i,i+1) for i in range(len(seq)-1)]
        return atoms,bonds
    # D_inf_h (centrosymmetric linear)
    M["H2"]=lin([("H",-0.37),("H",0.37)])
    M["N2"]=lin([("N",-0.55),("N",0.55)])
    M["CO2"]=lin([("O",-1.16),("C",0.0),("O",1.16)])
    M["C2H2"]=lin([("H",-1.66),("C",-0.60),("C",0.60),("H",1.66)])
    # C_inf_v (polar linear, no inversion centre)
    M["HBr"]=lin([("Br",-0.70),("H",0.71)])
    M["HCN"]=lin([("N",-1.15),("C",0.0),("H",1.06)])
    M["NCS"]=lin([("S",-1.61),("C",0.0),("N",1.17)])

    # ---- low-symmetry examples (no principal axis) ----
    Tn=np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]],float)/np.sqrt(3)
    # C1: CHFClBr (four different substituents -> no symmetry)
    M["CHFClBr"]=([("C",[0,0,0]),("H",Tn[0]*1.09),("F",Tn[1]*1.38),("Cl",Tn[2]*1.77),("Br",Tn[3]*1.94)],
                  [(0,1),(0,2),(0,3),(0,4)])
    # Cs: CHFCl2 (mirror plane through H,C,F, bisecting Cl-C-Cl)
    M["CHFCl2"]=([("C",[0,0,0]),("H",1.09*np.array([0,0,1.])),("F",1.38*np.array([0.943,0,-0.333])),
                  ("Cl",1.77*np.array([-0.471,0.816,-0.333])),("Cl",1.77*np.array([-0.471,-0.816,-0.333]))],
                 [(0,1),(0,2),(0,3),(0,4)])
    # Ci: anti CHFCl-CHFCl (inversion centre at the C-C midpoint)
    dcc=0.77; stt=np.sin(dd(70.5)); ctt=np.cos(dd(70.5)); C1=np.array([0,0,dcc])
    at=[("C",C1),("C",-C1)]; bo=[(0,1)]
    for el,az,r in (("H",0,1.09),("F",120,1.38),("Cl",240,1.77)):
        v=np.array([stt*np.cos(dd(az)),stt*np.sin(dd(az)),ctt]); p=C1+r*v
        at.append((el,p)); bo.append((0,len(at)-1)); at.append((el,-p)); bo.append((1,len(at)-1))
    M["CHFClanti"]=(at,bo)
    return M

def _cn_molecules():
    """Molecules for the C_n family, oriented with the principal axis along z.
    Returns key -> (atoms, bonds, sigma_v_angles). sigma_v_angles: list of azimuths
    for C_nv vertical planes; None for C_nh (uses sigma_h); [] for C_n (no planes)."""
    dd=lambda t: np.radians(t); M={}
    # C3h boric acid B(OH)3 (planar OH propeller)
    at=[("B",[0,0,0])]; bo=[]
    for a in (0,120,240):
        O=np.array([1.36*np.cos(dd(a)),1.36*np.sin(dd(a)),0.]); at.append(("O",O)); bo.append((0,len(at)-1))
        at.append(("H",O+0.96*np.array([np.cos(dd(a+130)),np.sin(dd(a+130)),0]))); bo.append((len(at)-2,len(at)-1))
    M["boric_acid"]=(at,bo,None)
    # C6h benzenehexol C6(OH)6 (planar OH propeller)
    at=[]; bo=[]; Ci=[]
    for k in range(6):
        a=60*k; at.append(("C",[1.4*np.cos(dd(a)),1.4*np.sin(dd(a)),0.])); Ci.append(len(at)-1)
    for k in range(6): bo.append((Ci[k],Ci[(k+1)%6]))
    for k in range(6):
        a=60*k; O=np.array([2.72*np.cos(dd(a)),2.72*np.sin(dd(a)),0.]); at.append(("O",O)); bo.append((Ci[k],len(at)-1))
        at.append(("H",O+0.96*np.array([np.cos(dd(a+130)),np.sin(dd(a+130)),0]))); bo.append((len(at)-2,len(at)-1))
    M["benzenehexol"]=(at,bo,None)
    # C2h trans-glyoxal
    at=[("C",[-0.75,0,0]),("C",[0.75,0,0]),("O",[-1.40,0.90,0]),("H",[-1.05,-1.02,0]),
        ("O",[1.40,-0.90,0]),("H",[1.05,1.02,0])]; bo=[(0,1),(0,2),(0,3),(1,4),(1,5)]
    M["glyoxal"]=(at,bo,None)
    # C2v water
    M["water"]=([("O",[0,0,0.3]),("H",[0.76,0,-0.29]),("H",[-0.76,0,-0.29])],[(0,1),(0,2)],[0,90])
    # C3v ammonia
    at=[("N",[0,0,0.38])]; bo=[]
    for a in (0,120,240): at.append(("H",[0.94*np.cos(dd(a)),0.94*np.sin(dd(a)),-0.32])); bo.append((0,len(at)-1))
    M["ammonia"]=(at,bo,[0,120,240])
    # C4v IF5 (square pyramidal)
    at=[("I",[0,0,0]),("F",[0,0,1.85])]; bo=[(0,1)]
    for a in (0,90,180,270): at.append(("F",[1.87*np.cos(dd(a)),1.87*np.sin(dd(a)),-0.30])); bo.append((0,len(at)-1))
    M["IF5"]=(at,bo,[0,45,90,135])
    # C2 hydrogen peroxide (skew)
    M["h2o2"]=([("O",[0,-0.74,0]),("O",[0,0.74,0]),("H",[0.88,-1.05,0.30]),("H",[-0.88,1.05,0.30])],
               [(0,1),(0,2),(1,3)],[])
    # C3 triphenylmethane (propeller of 3 phenyls about the C-H axis)
    at=[("C",[0,0,0]),("H",[0,0,1.09])]; bo=[(0,1)]; prop=dd(48)
    for a in (0,120,240):
        d=np.array([np.sin(dd(125))*np.cos(dd(a)),np.sin(dd(125))*np.sin(dd(a)),np.cos(dd(125))]); d/=np.linalg.norm(d)
        ipso=1.54*d
        n0=np.array([0,0,1.])-np.dot([0,0,1.],d)*d; n0/=np.linalg.norm(n0)
        # twist ring plane about d by prop
        n=np.cos(prop)*n0+np.sin(prop)*np.cross(d,n0)
        p=d; q=np.cross(n,d); q/=np.linalg.norm(q)
        centre=ipso+1.4*p; ring=[]; radial=[]
        for j,th in enumerate((180,240,300,0,60,120)):
            u=np.cos(dd(th))*p+np.sin(dd(th))*q          # outward, in the ring plane
            v=centre+1.4*u; at.append(("C",v)); ring.append(len(at)-1); radial.append(u)
        bo.append((0,ring[0]))               # central C to ipso
        for j in range(6): bo.append((ring[j],ring[(j+1)%6]))
        # ring hydrogens: one on each carbon except the ipso (ring[0]), which is
        # already bonded to the central C. Each C-H points radially outward from
        # the ring centre, in the plane of the ring.
        for j in range(1,6):
            at.append(("H", centre+(1.4+1.08)*radial[j])); bo.append((ring[j],len(at)-1))
    M["ph3ch"]=(at,bo,[])
    # S4 example: central atom + 4 arms generated by S4 (rotate 90 + z-flip). Pure S4.
    s4=lambda p: np.array([-p[1],p[0],-p[2]])
    A0=np.array([0.95,0.28,0.48]); B0=A0+np.array([0.60,0.50,0.55])
    at=[("C",[0,0,0])]; bo=[]; A=A0.copy(); B=B0.copy()
    for _ in range(4):
        at.append(("C",A.copy())); ai=len(at)-1; at.append(("Cl",B.copy())); bi=len(at)-1
        bo+=[(0,ai),(ai,bi)]; A=s4(A); B=s4(B)
    M["s4model"]=(at,bo,[])
    # D6h benzenehexol: radial OH (aligned outward, not a propeller)
    at=[]; bo=[]; Ci=[]
    for k in range(6):
        a=60*k; at.append(("C",[1.4*np.cos(dd(a)),1.4*np.sin(dd(a)),0.])); Ci.append(len(at)-1)
    for k in range(6): bo.append((Ci[k],Ci[(k+1)%6]))
    for k in range(6):
        a=60*k; O=np.array([2.72*np.cos(dd(a)),2.72*np.sin(dd(a)),0.]); at.append(("O",O)); bo.append((Ci[k],len(at)-1))
        at.append(("H",O+0.96*np.array([np.cos(dd(a)),np.sin(dd(a)),0]))); bo.append((len(at)-2,len(at)-1))
    M["benzenehexol_d6h"]=(at,bo,None)
    return M

def render_dxy_a2(outdir="."):
    """The a2 representation shown with a concrete orbital: d_xy on O. Four lobes in
    the xy-plane (+ blue, - red). Under C2(z) the pattern maps onto itself (+1); under
    either mirror plane the + and - lobes swap (-1). Row: 1, 1, -1, -1 = a2."""
    NAVY="#1D3B5A"; BLUE="#2f5fbf"; RED="#c0392b"; lim=1.45
    def rot_to(a):
        a=np.array(a,float); a=a/np.linalg.norm(a); z=np.array([0,0,1.])
        if np.allclose(a,z): return np.eye(3)
        if np.allclose(a,-z): return np.diag([1,-1,-1.])
        v=np.cross(z,a); c=float(z@a); s=np.linalg.norm(v)
        vx=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        return np.eye(3)+vx+vx@vx*((1-c)/s**2)
    def lobe(azdeg,dd_=0.60,La=0.50,Lb=0.27,N=20):
        u=np.linspace(0,2*np.pi,N); v=np.linspace(0,np.pi,N); U,V=np.meshgrid(u,v)
        x=Lb*np.cos(U)*np.sin(V); y=Lb*np.sin(U)*np.sin(V); z=La*np.cos(V)
        a=np.array([np.cos(np.radians(azdeg)),np.sin(np.radians(azdeg)),0.])
        return np.stack([x,y,z],-1)@rot_to(a).T + dd_*a
    lobes=[(lobe(45),BLUE),(lobe(135),RED),(lobe(225),BLUE),(lobe(315),RED)]
    ops=[("E",None,+1,"identity"),("C_2(z)","rot",+1,"pattern unchanged"),
         ("\\sigma_v(xz)","refly",-1,"+ and \u2212 lobes swap"),
         ("\\sigma_v'(yz)","reflx",-1,"+ and \u2212 lobes swap")]
    cols=["E","C_2","\\sigma_v(xz)","\\sigma_v'(yz)"]; chars=[1,1,-1,-1]
    show=6; nop=16; hold=14
    seg=[]
    for i in range(4): seg+=[("show",i,0.)]*show+[("op",i,(j+1)/nop) for j in range(nop)]+[("done",i,1.)]*hold
    fig=plt.figure(figsize=FIG_SIZE, dpi=DPI); ax=fig.add_subplot(111,projection="3d")
    def xf(P,opk,t):
        if opk is None: return P
        if opk=="rot": return P@Rmat([0,0,1],d(180)*t).T
        if opk=="refly": return P*np.array([1,1-2*t,1])
        return P*np.array([1-2*t,1,1])
    def draw(P,col,al): ax.plot_surface(P[...,0],P[...,1],P[...,2],color=col,alpha=al,rstride=1,cstride=1,linewidth=0,shade=True)
    def update(k):
        phase,i,t=seg[k]; name,opk,chi,why=ops[i]
        ax.clear(); ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
        ax.set_box_aspect((1,1,1)); ax.set_axis_off(); ax.view_init(20,-58)
        # operation element
        if opk=="rot":
            ax.plot([0,0],[0,0],[-1.25,1.25],color=AXIS_C,lw=2.0,alpha=0.9,zorder=1); ax.text(0,0,1.35,"$C_2$",color=AXIS_C,fontsize=11,weight="bold",ha="center")
        elif opk in ("refly","reflx"):
            R=1.2; g=np.linspace(-R,R,2); G1,G2=np.meshgrid(g,g); Z0=np.zeros_like(G1)
            crn=([[-R,0,-R],[R,0,-R],[R,0,R],[-R,0,R]] if opk=="refly" else
                 [[0,-R,-R],[0,R,-R],[0,R,R],[0,-R,R]])
            _flat_plane(ax,crn,1.0,a_fill=0.14)
        if phase!="show":
            for P,col in lobes: draw(P,col,0.09)
        for P,col in lobes: draw(xf(P,opk,t),col,0.9)
        ax.scatter(0,0,0,color="#222",s=45,zorder=8)
        fig.texts=[]
        fig.text(0.5,0.95,"$d_{xy}$ on O \u2014 the $a_2$ orbital",ha="center",fontsize=15,weight="bold",color=NAVY)
        fig.text(0.5,0.905,"symmetric under $C_2$, antisymmetric under both mirror planes",
                 ha="center",fontsize=9.5,color="#8a8f5c",style="italic")
        revealed=i+(1 if phase=="done" else 0); xs=[0.30,0.45,0.63,0.82]
        for c,(cx,cl,ch) in enumerate(zip(xs,cols,chars)):
            fig.text(cx,0.115,f"${cl}$",ha="center",fontsize=10,color="#555")
            if c<revealed:
                fig.text(cx,0.055,f"${'+1' if ch>0 else '-1'}$",ha="center",fontsize=15,weight="bold",color="#2e7d32" if ch>0 else "#c0392b")
            else: fig.text(cx,0.055,"\u00b7",ha="center",fontsize=15,color="#bbb")
        if phase=="op": fig.text(0.5,0.015,f"applying ${name}$\u2026",ha="center",fontsize=9.5,style="italic",color="#888")
        elif phase=="done": fig.text(0.5,0.015,f"${name}$: {why}",ha="center",fontsize=9.5,style="italic",color="#2e7d32" if chi>0 else "#c0392b")
        return []
    out=f"{outdir}/a2_dxy_c2v.gif"
    FuncAnimation(fig,update,frames=len(seg),interval=1000/FPS,blit=False).save(out,writer=PlayOnceWriter(fps=FPS))
    plt.close(fig); print("wrote",out); return out

def render_rotation_row(outdir="."):
    """The R_z (rotation about z) row of the C2v table, made explicit: R_z is a
    SPIN about the z-axis (curved arrow + a circulating dot showing its sense).
    Each operation either keeps the sense of spin (proper rotation, +1) or reverses
    it (reflection/improper, -1). Row: 1, 1, -1, -1 (the a2 representation)."""
    NAVY="#1D3B5A"; MAG="#b5179e"; lim=1.5
    th=np.linspace(np.radians(28),np.radians(332),64); rr=0.82
    arc0=np.stack([rr*np.cos(th),rr*np.sin(th),np.zeros_like(th)],1)
    ops=[("E",None,+1,"identity \u2014 nothing changes"),
         ("C_2(z)","rot",+1,"a proper rotation \u2014 spin sense kept"),
         ("\\sigma_v(xz)","refly",-1,"a reflection \u2014 spin sense reversed"),
         ("\\sigma_v'(yz)","reflx",-1,"a reflection \u2014 spin sense reversed")]
    cols=["E","C_2","\\sigma_v(xz)","\\sigma_v'(yz)"]; chars=[1,1,-1,-1]
    show=8; nop=16; hold=16
    seg=[]
    for i in range(4):
        seg+=[("show",i,0.0)]*show+[("op",i,(j+1)/nop) for j in range(nop)]+[("done",i,1.0)]*hold

    fig=plt.figure(figsize=FIG_SIZE, dpi=DPI); ax=fig.add_subplot(111,projection="3d")
    def xf(opk,t):
        if opk is None: return arc0.copy()
        if opk=="rot": return arc0@Rmat([0,0,1],d(180)*t).T
        if opk=="refly": return arc0*np.array([1,1-2*t,1])
        return arc0*np.array([1-2*t,1,1])
    def update(k):
        phase,i,t=seg[k]; name,opk,chi,why=ops[i]
        ax.clear(); ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
        ax.set_box_aspect((1,1,1)); ax.set_axis_off(); ax.view_init(38,-90)
        ax.plot([0,0],[0,0],[-1.25,1.25],color="#999",lw=1.3,alpha=0.7,zorder=0)
        ax.text(0,0,1.34,"$z$",color="#777",fontsize=10,ha="center")
        if opk=="rot":
            ax.plot([0,0],[0,0],[-1.25,1.25],color=AXIS_C,lw=2.2,alpha=0.9,zorder=1)
            ax.text(0,0,-1.4,"$C_2$",color=AXIS_C,fontsize=11,weight="bold",ha="center")
        elif opk in ("refly","reflx"):
            R=1.25; g=np.linspace(-R,R,2); G1,G2=np.meshgrid(g,g); Z0=np.zeros_like(G1)
            crn=([[-R,0,-R],[R,0,-R],[R,0,R],[-R,0,R]] if opk=="refly"
                 else [[0,-R,-R],[0,R,-R],[0,R,R],[0,-R,R]])
            _flat_plane(ax,crn,1.0,a_fill=0.15)
        if phase!="show":
            ax.plot(arc0[:,0],arc0[:,1],arc0[:,2],color=MAG,lw=2,alpha=0.14,zorder=1)
        arc=xf(opk,t)
        ax.plot(arc[:,0],arc[:,1],arc[:,2],color=MAG,lw=3.4,alpha=0.95,zorder=4)
        tip=arc[-1]; tan=arc[-1]-arc[-5]; tan=tan/np.linalg.norm(tan)*0.34
        ax.quiver(*(tip-tan),*tan,color=MAG,lw=3.4,arrow_length_ratio=0.7,zorder=5)
        ax.text(0,0,0.0,"$R_z$",color=MAG,fontsize=12,weight="bold",ha="center",zorder=6)
        # circulating dot shows the sense of spin (hidden while the operation morphs)
        if phase in ("show","done"):
            di=int((k*2.2)%(len(arc)-1)); ax.scatter(*arc[di],s=70,color=MAG,edgecolor="w",lw=0.8,zorder=7)
        fig.texts=[]
        fig.text(0.5,0.95,"$R_z$: a spin about the $z$ axis",ha="center",fontsize=15,weight="bold",color=NAVY)
        fig.text(0.5,0.905,"does the operation keep the sense of spin ($+1$) or reverse it ($-1$)?",
                 ha="center",fontsize=9.5,color="#8a8f5c",style="italic")
        revealed=i+(1 if phase=="done" else 0); xs=[0.30,0.45,0.63,0.82]
        for c,(cx,cl,ch) in enumerate(zip(xs,cols,chars)):
            fig.text(cx,0.115,f"${cl}$",ha="center",fontsize=10,color="#555")
            if c<revealed:
                fig.text(cx,0.055,f"${'+1' if ch>0 else '-1'}$",ha="center",fontsize=15,weight="bold",
                         color="#2e7d32" if ch>0 else "#c0392b")
            else: fig.text(cx,0.055,"\u00b7",ha="center",fontsize=15,color="#bbb")
        lab="applying" if phase=="op" else ("" if phase=="show" else "")
        if phase=="op":
            fig.text(0.5,0.015,f"applying ${name}$\u2026",ha="center",fontsize=9.5,style="italic",color="#888")
        elif phase=="done":
            fig.text(0.5,0.015,f"${name}$: {why}",ha="center",fontsize=9.5,style="italic",
                     color="#2e7d32" if chi>0 else "#c0392b")
        return []
    out=f"{outdir}/rotation_Rz_c2v.gif"
    FuncAnimation(fig,update,frames=len(seg),interval=1000/FPS,blit=False).save(out,writer=PlayOnceWriter(fps=FPS))
    plt.close(fig); print("wrote",out); return out

def render_nh3_sigma(k, tag, outdir="."):
    """Ammonia (C3v), one of its three sigma_v planes highlighted, with the
    reflection animated: the H in the plane (and N) stay put, the other two H swap.
    k in {0,1,2} selects which N-H the plane runs through."""
    dd=lambda t: np.radians(t); NAVY="#1D3B5A"; sc=1.7
    N=np.array([0,0,0.40])*sc; ang=[90,210,330]
    H=[np.array([0.94*np.cos(dd(a)),0.94*np.sin(dd(a)),-0.34])*sc for a in ang]
    ak=ang[k]; nrm=np.array([-np.sin(dd(ak)),np.cos(dd(ak)),0.]); Rref=np.eye(3)-2*np.outer(nrm,nrm)
    other=[j for j in range(3) if j!=k]
    lim=2.0; Zc=1.9; Rr=1.75
    frames=[]; azim=[-58.0]
    def push(nf,el0,el1,a0,a1,rf0,rf1,spin=1.0):
        for i in range(nf):
            t=i/max(1,nf-1); frames.append(dict(elev=el0+(el1-el0)*t,azim=azim[0],
                              ae=a0+(a1-a0)*t,rf=rf0+(rf1-rf0)*t)); azim[0]+=spin
    push(10,10,22,0,0,0,0); push(7,22,22,0,1,0,0); push(6,22,22,1,1,0,0)   # show plane
    push(16,22,22,1,1,0,1); push(16,22,22,1,1,1,1)                          # reflect + hold
    fig=plt.figure(figsize=FIG_SIZE, dpi=DPI); ax=fig.add_subplot(111,projection="3d")
    HCOL=["#d11f4a","#1f7a4d","#e8701a"]
    def update(kf):
        fr=frames[kf]; ae=fr["ae"]; rf=fr["rf"]
        ax.clear(); ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
        ax.set_box_aspect((1,1,1)); ax.set_axis_off(); ax.view_init(fr["elev"],fr["azim"])
        # C3 axis
        ax.plot([0,0],[0,0],[-Zc*0.5,Zc],color=AXIS_C,lw=1.6,alpha=0.5,zorder=0)
        # sigma_v plane through H_k (contains z and the H_k direction)
        if ae>0.02:
            u=np.array([np.cos(dd(ak)),np.sin(dd(ak)),0]); z=np.array([0,0,Zc*0.92])
            sq=np.array([u*Rr+z,u*Rr-z,-u*Rr-z,-u*Rr+z])
            ax.add_collection3d(Poly3DCollection([sq],facecolor=PLANE_C,alpha=0.18*ae,edgecolor="none"))
            loop=np.vstack([sq,sq[0]]); ax.plot(loop[:,0],loop[:,1],loop[:,2],color=PLANE_EDGE,lw=1.4,alpha=0.85*ae)
            ax.text(*(u*Rr*0.9+np.array([0,0,Zc*0.8])),"$\\sigma_v$",color=PLANE_EDGE,fontsize=12,weight="bold",alpha=ae)
        # atom positions with reflection interpolation on the two off-plane H
        pos=[N.copy()]+[H[j].copy() for j in range(3)]
        for j in other: pos[1+j]=(1-rf)*H[j]+rf*(Rref@H[j])
        # bonds
        for j in range(3): ax.plot(*zip(pos[0],pos[1+j]),color=BOND_C,lw=2.8,alpha=0.9,zorder=3)
        st=STYLE["N"]; ax.scatter(*pos[0],color=st["color"],s=st["s"]*0.8,edgecolor=st["edge"],lw=0.8,zorder=4)
        for j in range(3):
            inplane=(j==k)
            ax.scatter(*pos[1+j],color=HCOL[j],s=300 if inplane else 260,
                       edgecolor="#111" if inplane else "#333",lw=1.3 if inplane else 0.7,zorder=6)
            ax.text(*pos[1+j],str(j+1),color="white",fontsize=9,weight="bold",ha="center",va="center",zorder=8)
        fig.texts=[]
        fig.text(0.5,0.93,f"Ammonia $C_{{3v}}$: $\\sigma_v$ through N\u2013H$_{k+1}$",ha="center",fontsize=14,weight="bold",color=NAVY)
        if rf<0.5: cap="the plane contains the $C_3$ axis and one N\u2013H bond"
        else:      cap=f"reflection keeps H$_{k+1}$ (and N); swaps H$_{other[0]+1}$ and H$_{other[1]+1}$"
        fig.text(0.5,0.05,cap,ha="center",fontsize=10.5,style="italic",color="#555")
        return []
    out=f"{outdir}/ammonia_{tag}.gif"
    FuncAnimation(fig,update,frames=len(frames),interval=1000/FPS,blit=False).save(out,writer=PlayOnceWriter(fps=FPS))
    plt.close(fig); print("wrote",out); return out

def render_lowsym_single(molkey, symbol, kind, tag, outdir="."):
    """One low-symmetry point group per GIF, showing its element: kind 'cs' draws
    the single mirror plane (the xz-plane for CHFCl2), 'ci' marks the inversion
    centre, 'c1' shows a molecule with no symmetry at all."""
    atoms,bonds=_group_molecules()[molkey]
    pos=np.array([p for _,p in atoms],float); rmax=np.max(np.linalg.norm(pos,axis=1)) or 1.0
    pos=pos*(1.45/rmax); NAVY="#1D3B5A"; lim=1.9; Rr=1.55
    frames=[]; azim=[-56.0]
    def push(nf,el0,el1,a0,a1,spin=1.2):
        for i in range(nf):
            t=i/max(1,nf-1); frames.append(dict(elev=el0+(el1-el0)*t,azim=azim[0],ae=a0+(a1-a0)*t)); azim[0]+=spin
    push(10,8,24,0,0); push(7,24,24,0,1); push(18,24,24,1,1)
    fig=plt.figure(figsize=FIG_SIZE, dpi=DPI); ax=fig.add_subplot(111,projection="3d")
    nm={"CHFCl2":"CHFCl$_2$","CHFClanti":"CHFCl\u2013CHFCl (anti)","CHFClBr":"CHFClBr"}[molkey]
    note={"cs":"one mirror plane $\\sigma$","ci":"centre of inversion $i$","c1":"no symmetry \u2014 only $E$"}[kind]
    def update(k):
        fr=frames[k]; ae=fr["ae"]
        ax.clear(); ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
        ax.set_box_aspect((1,1,1)); ax.set_axis_off(); ax.view_init(fr["elev"],fr["azim"])
        if kind=="cs" and ae>0.02:                    # mirror plane = xz-plane (normal y)
            R=Rr; _flat_plane(ax,[[-R,0,-R],[R,0,-R],[R,0,R],[-R,0,R]],ae,a_fill=0.16)
            ax.text(R*0.8,0,R*0.8,"$\\sigma$",color=PLANE_EDGE,fontsize=12,weight="bold",alpha=ae)
        if kind=="ci" and ae>0.02:                    # inversion centre
            ax.scatter(0,0,0,s=90,color=INV_C,alpha=ae,zorder=6)
            ax.scatter(0,0,0,s=300,facecolors="none",edgecolors=INV_C,linewidths=1.8,alpha=ae,zorder=6)
            ax.text(0,0,0.28,"$i$",color=INV_C,fontsize=12,weight="bold",ha="center",alpha=ae)
        for i,j in bonds: ax.plot(*zip(pos[i],pos[j]),color=BOND_C,lw=2.6,alpha=0.9,zorder=3)
        for idx,(el,_) in enumerate(atoms):
            st=STYLE.get(el,STYLE["A"]); ax.scatter(*pos[idx],color=st["color"],s=st["s"]*0.75,
                edgecolor=st["edge"],lw=0.7,depthshade=True,zorder=4)
        fig.texts=[]
        fig.text(0.5,0.93,f"${symbol}$",ha="center",fontsize=28,weight="bold",color=NAVY)
        fig.text(0.5,0.065,nm,ha="center",fontsize=12.5,weight="bold",color="#333")
        fig.text(0.5,0.02,note,ha="center",fontsize=10,color="#666",style="italic")
        return []
    out=f"{outdir}/lowsym_{tag}.gif"
    FuncAnimation(fig,update,frames=len(frames),interval=1000/FPS,blit=False).save(out,writer=PlayOnceWriter(fps=FPS))
    plt.close(fig); print("wrote",out); return out

def render_mo_operation(orbital, op, tag, outdir="."):
    """A p orbital under a symmetry operation, showing the character +/-1. orbital
    in {px,py,pz}; op in {c2z (C2 about z), sxy (reflection in the xy-plane)}. Blue
    lobe = + phase, red lobe = - phase; if the moved orbital lands on itself chi=+1,
    if it lands on its negative chi=-1."""
    axis={"px":np.array([1.,0,0]),"py":np.array([0,1.,0]),"pz":np.array([0,0,1.])}[orbital]
    olab={"px":"p_x","py":"p_y","pz":"p_z"}[orbital]
    OPS_MO={"c2z":("C_2(z)","rot",np.diag([-1.,-1,1]),None),
            "sxy":("\\sigma(xy)","refl",np.diag([1.,1,-1]),2),
            "sxz":("\\sigma_v(xz)","refl",np.diag([1.,-1,1]),1),
            "syz":("\\sigma_v'(yz)","refl",np.diag([-1.,1,1]),0)}
    opname,kind,Rfull,mnorm=OPS_MO[op]
    chi = 1 if np.allclose(Rfull@axis,axis) else -1
    NAVY="#1D3B5A"; BLUE="#2f5fbf"; RED="#c0392b"; lim=1.55

    def rot_to(a):
        a=a/np.linalg.norm(a); z=np.array([0,0,1.])
        if np.allclose(a,z): return np.eye(3)
        if np.allclose(a,-z): return np.diag([1,-1,-1.])
        v=np.cross(z,a); c=float(z@a); s=np.linalg.norm(v)
        vx=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        return np.eye(3)+vx+vx@vx*((1-c)/s**2)
    Rax=rot_to(axis)
    def lobe(sign,d=0.52,La=0.60,Lb=0.33,N=22):
        u=np.linspace(0,2*np.pi,N); v=np.linspace(0,np.pi,N); U,V=np.meshgrid(u,v)
        x=Lb*np.cos(U)*np.sin(V); y=Lb*np.sin(U)*np.sin(V); z=La*np.cos(V)
        P=np.stack([x,y,z],-1)@Rax.T + sign*d*axis
        return P
    lp=lobe(+1); ln=lobe(-1)          # + lobe (blue), - lobe (red), standard position

    intro=8; nop=22; hold=18
    seg=[("intro",0.0)]*intro+[("op",(j+1)/nop) for j in range(nop)]+[("done",1.0)]*hold

    fig=plt.figure(figsize=FIG_SIZE, dpi=DPI); ax=fig.add_subplot(111,projection="3d")
    fig.text(0.5,0.93,f"${olab}$  under  ${opname}$",ha="center",fontsize=17,weight="bold",color=NAVY)

    def transform(P,t):
        if kind=="rot":
            R=Rmat([0,0,1],d(180)*t); return P@R.T
        else:                          # rigid translation of each lobe to its mirror image
            c=P[...,mnorm].mean(); shift=np.zeros(3); shift[mnorm]=-2*c*t; return P+shift
    def draw_lobe(P,col): ax.plot_surface(P[...,0],P[...,1],P[...,2],color=col,alpha=0.9,
                                          rstride=1,cstride=1,linewidth=0,antialiased=True,shade=True)

    def update(k):
        phase,t=seg[k]
        ax.clear(); ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
        ax.set_box_aspect((1,1,1)); ax.set_axis_off(); ax.view_init(32,-90)
        # operation element
        if op=="c2z":
            ax.plot([0,0],[0,0],[-1.4,1.4],color=AXIS_C,lw=2.0,alpha=0.9,zorder=0)
            ax.text(0,0,1.5,"$C_2$",color=AXIS_C,fontsize=11,weight="bold",ha="center")
        else:
            R=1.3; g=np.linspace(-R,R,2); G1,G2=np.meshgrid(g,g); Z0=np.zeros_like(G1)
            if mnorm==2:   X,Y,Zp=G1,G2,Z0; lpos=(1.15,1.15,0)                         # xy plane
            elif mnorm==1: X,Y,Zp=G1,Z0,G2; lpos=(1.15,0,1.15)                         # xz plane
            else:          X,Y,Zp=Z0,G1,G2; lpos=(0,1.15,1.15)                         # yz plane
            if mnorm==2:   crn=[[-R,-R,0],[R,-R,0],[R,R,0],[-R,R,0]]
            elif mnorm==1: crn=[[-R,0,-R],[R,0,-R],[R,0,R],[-R,0,R]]
            else:          crn=[[0,-R,-R],[0,R,-R],[0,R,R],[0,-R,R]]
            _flat_plane(ax,crn,1.0,a_fill=0.16,ew=1.6,a_edge=0.85)
            ax.text(*lpos,"$\\sigma$",color=PLANE_EDGE,fontsize=11,weight="bold")
        # faint ghost of the ORIGINAL orbital for comparison
        if phase!="intro":
            ax.plot_surface(lp[...,0],lp[...,1],lp[...,2],color=BLUE,alpha=0.10,shade=False,zorder=1)
            ax.plot_surface(ln[...,0],ln[...,1],ln[...,2],color=RED,alpha=0.10,shade=False,zorder=1)
        draw_lobe(transform(lp,t),BLUE); draw_lobe(transform(ln,t),RED)
        ax.scatter(0,0,0,color="#222",s=40,zorder=6)
        if phase=="done":
            if chi==1:
                msg="maps onto itself"; sub="$\\chi = +1$   (symmetric)"; col="#2e7d32"
            else:
                msg="maps onto its negative"; sub="$\\chi = -1$   (antisymmetric)"; col="#c0392b"
            ax.text2D(0.5,0.12,msg,transform=ax.transAxes,ha="center",fontsize=11,style="italic",color="#555")
            ax.text2D(0.5,0.05,sub,transform=ax.transAxes,ha="center",fontsize=15,weight="bold",color=col)
        else:
            cap={"intro":"blue = + lobe,  red = \u2212 lobe","op":f"applying ${opname}$\u2026"}[phase]
            ax.text2D(0.5,0.05,cap,transform=ax.transAxes,ha="center",fontsize=10.5,style="italic",color="#666")
        return []
    out=f"{outdir}/mo_{tag}.gif"
    FuncAnimation(fig,update,frames=len(seg),interval=1000/FPS,blit=False).save(out,writer=PlayOnceWriter(fps=FPS))
    plt.close(fig); print("wrote",out); return out

def render_c4_on_p(orbital, outdir="."):
    """C_4(z) acting on a single in-plane p orbital, to show that a rotation can
    turn one orbital into a DIFFERENT one (not merely +/- itself). This is exactly
    what forces p_x and p_y to travel together as a degenerate pair (the E irrep)
    in C_4v:
        C_4 on p_x  ->  p_y      (the + lobe swings from +x to +y; no sign change)
        C_4 on p_y  -> -p_x      (the + lobe swings from +y to -x; phase reversed)
    Blue lobe = + phase, red lobe = - phase. Same axis convention (view 20,-58) and
    lobe shape as the p-orbital MO figures. Companion to render_mo_operation, which
    handles the operations that DO send an orbital to +/- itself."""
    assert orbital in ("px","py")
    axis={"px":np.array([1.,0,0]),"py":np.array([0,1.,0])}[orbital]
    NAVY="#1D3B5A"; BLUE="#2f5fbf"; RED="#c0392b"; lim=1.55
    startlab={"px":"p_x","py":"p_y"}[orbital]
    reslab ={"px":"p_y","py":"-p_x"}[orbital]

    def rot_to(a):
        a=np.asarray(a,float); a=a/np.linalg.norm(a); z=np.array([0,0,1.])
        if np.allclose(a,z): return np.eye(3)
        if np.allclose(a,-z): return np.diag([1,-1,-1.])
        v=np.cross(z,a); c=float(z@a); s=np.linalg.norm(v)
        vx=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        return np.eye(3)+vx+vx@vx*((1-c)/s**2)
    Rax=rot_to(axis)
    def lobe(sign,dd=0.52,La=0.60,Lb=0.33,N=22):
        u=np.linspace(0,2*np.pi,N); v=np.linspace(0,np.pi,N); U,V=np.meshgrid(u,v)
        x=Lb*np.cos(U)*np.sin(V); y=Lb*np.sin(U)*np.sin(V); z=La*np.cos(V)
        return np.stack([x,y,z],-1)@Rax.T + sign*dd*axis
    lp=lobe(+1); ln=lobe(-1)                       # + lobe (blue), - lobe (red), start
    R90=Rmat([0,0,1],d(90))
    tp=lp@R90.T; tn=ln@R90.T                        # result = start turned +90 about z

    intro=10; nop=26; hold=20
    seg=[("intro",0.0)]*intro+[("op",(j+1)/nop) for j in range(nop)]+[("done",1.0)]*hold

    fig=plt.figure(figsize=FIG_SIZE, dpi=DPI); ax=fig.add_subplot(111,projection="3d")
    fig.text(0.5,0.93,f"$C_4(z)$  on  ${startlab}$",ha="center",fontsize=17,weight="bold",color=NAVY)

    def draw_lobe(P,col,al=0.9,shade=True):
        ax.plot_surface(P[...,0],P[...,1],P[...,2],color=col,alpha=al,rstride=1,cstride=1,
                        linewidth=0,antialiased=True,shade=shade)
    # static 90-degree curved arrow near the top of the z axis, giving the C4 sense
    tt=np.linspace(d(8),d(82),40); ra=0.95; zc=1.12
    arc=np.stack([ra*np.cos(tt),ra*np.sin(tt),np.full_like(tt,zc)],1)

    def update(k):
        phase,t=seg[k]; te=ease(t)
        ax.clear(); ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
        ax.set_box_aspect((1,1,1)); ax.set_axis_off(); ax.view_init(32,-90)
        # C4 axis (z) plus the curved sense arrow
        ax.plot([0,0],[0,0],[-1.35,1.35],color=AXIS_C,lw=2.0,alpha=0.9,zorder=0)
        ax.text(0,0,1.52,"$C_4$",color=AXIS_C,fontsize=12,weight="bold",ha="center")
        ax.plot(arc[:,0],arc[:,1],arc[:,2],color=AXIS_C,lw=2.6,alpha=0.9,zorder=2)
        tan=arc[-1]-arc[-4]; tan=tan/np.linalg.norm(tan)*0.30
        ax.quiver(*(arc[-1]-tan),*tan,color=AXIS_C,lw=2.6,arrow_length_ratio=0.8,zorder=3)
        # faint ghost of the RESULT in its final place, so the landing is anticipated
        draw_lobe(tp,BLUE,0.10,shade=False); draw_lobe(tn,RED,0.10,shade=False)
        # the orbital, rotating rigidly through 90 degrees
        Rt=Rmat([0,0,1],d(90)*te)
        draw_lobe(lp@Rt.T,BLUE); draw_lobe(ln@Rt.T,RED)
        ax.scatter(0,0,0,color="#222",s=40,zorder=6)
        if phase=="done":
            ax.text2D(0.5,0.12,f"the orbital has become ${reslab}$",transform=ax.transAxes,
                      ha="center",fontsize=11,style="italic",color="#555")
            ax.text2D(0.5,0.05,f"$C_4\\,{startlab} = {reslab}$",transform=ax.transAxes,
                      ha="center",fontsize=15,weight="bold",color=NAVY)
        else:
            cap={"intro":"blue = + lobe,  red = \u2212 lobe",
                 "op":"turning $90^\\circ$ about $z$\u2026"}[phase]
            ax.text2D(0.5,0.05,cap,transform=ax.transAxes,ha="center",fontsize=10.5,style="italic",color="#666")
        return []
    tag={"px":"c4_px","py":"c4_py"}[orbital]
    out=f"{outdir}/mo_{tag}.gif"
    FuncAnimation(fig,update,frames=len(seg),interval=1000/FPS,blit=False).save(out,writer=PlayOnceWriter(fps=FPS))
    plt.close(fig); print("wrote",out); return out

def render_c4_on_d(orbital, outdir="."):
    """C_4(z) on d_xz / d_yz - the SECOND E pair of C_4v, exactly parallel to the p
    pair. A rotation turns one member into the other (with a sign on the way back),
    so the two cannot be separated:
        C_4 on d_xz  ->  d_yz      (no sign change)
        C_4 on d_yz  -> -d_xz      (phase reversed)
    Four lobes; blue = + phase, red = - phase. Uses a symmetric view (26,-45) rather
    than the p-pair's (32,-90): at (32,-90) the yz-plane is nearly edge-on, which
    collapses d_yz, so the pair is shown with x and y both splayed and both members
    fully legible."""
    assert orbital in ("dxz","dyz")
    NAVY="#1D3B5A"; BLUE="#2f5fbf"; RED="#c0392b"; lim=1.55
    plane={"dxz":"xz","dyz":"yz"}[orbital]
    startlab={"dxz":"d_{xz}","dyz":"d_{yz}"}[orbital]
    reslab ={"dxz":"d_{yz}","dyz":"-d_{xz}"}[orbital]
    def rot_to(a):
        a=np.asarray(a,float); a=a/np.linalg.norm(a); z=np.array([0,0,1.])
        if np.allclose(a,z): return np.eye(3)
        if np.allclose(a,-z): return np.diag([1,-1,-1.])
        v=np.cross(z,a); c=float(z@a); s=np.linalg.norm(v)
        vx=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        return np.eye(3)+vx+vx@vx*((1-c)/s**2)
    def lobe(dirvec,dd=0.58,La=0.46,Lb=0.25,N=20):
        dirvec=np.asarray(dirvec,float)
        u=np.linspace(0,2*np.pi,N); v=np.linspace(0,np.pi,N); U,V=np.meshgrid(u,v)
        x=Lb*np.cos(U)*np.sin(V); y=Lb*np.sin(U)*np.sin(V); z=La*np.cos(V)
        return np.stack([x,y,z],-1)@rot_to(dirvec).T + dd*dirvec
    c=float(np.cos(d(45)))
    if plane=="xz": dirs=[(c,0,c),(-c,0,c),(-c,0,-c),(c,0,-c)]
    else:           dirs=[(0,c,c),(0,-c,c),(0,-c,-c),(0,c,-c)]
    cols=[BLUE,RED,BLUE,RED]
    start=[(lobe(dv),cl) for dv,cl in zip(dirs,cols)]
    R90=Rmat([0,0,1],d(90))
    target=[(P@R90.T,cl) for P,cl in start]        # result = the set turned +90 about z

    intro=10; nop=26; hold=20
    seg=[("intro",0.0)]*intro+[("op",(j+1)/nop) for j in range(nop)]+[("done",1.0)]*hold
    fig=plt.figure(figsize=FIG_SIZE, dpi=DPI); ax=fig.add_subplot(111,projection="3d")
    fig.text(0.5,0.93,f"$C_4(z)$  on  ${startlab}$",ha="center",fontsize=17,weight="bold",color=NAVY)
    def draw_lobe(P,col,al=0.9,shade=True):
        ax.plot_surface(P[...,0],P[...,1],P[...,2],color=col,alpha=al,rstride=1,cstride=1,
                        linewidth=0,antialiased=True,shade=shade)
    tt=np.linspace(d(8),d(82),40); ra=0.95; zc=1.12
    arc=np.stack([ra*np.cos(tt),ra*np.sin(tt),np.full_like(tt,zc)],1)
    def update(k):
        phase,t=seg[k]; te=ease(t)
        ax.clear(); ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
        ax.set_box_aspect((1,1,1)); ax.set_axis_off(); ax.view_init(26,-45)
        ax.plot([0,0],[0,0],[-1.35,1.35],color=AXIS_C,lw=2.0,alpha=0.9,zorder=0)
        ax.text(0,0,1.52,"$C_4$",color=AXIS_C,fontsize=12,weight="bold",ha="center")
        ax.plot(arc[:,0],arc[:,1],arc[:,2],color=AXIS_C,lw=2.6,alpha=0.9,zorder=2)
        tan=arc[-1]-arc[-4]; tan=tan/np.linalg.norm(tan)*0.30
        ax.quiver(*(arc[-1]-tan),*tan,color=AXIS_C,lw=2.6,arrow_length_ratio=0.8,zorder=3)
        for P,cl in target: draw_lobe(P,cl,0.10,shade=False)         # ghost of the result
        Rt=Rmat([0,0,1],d(90)*te)
        for P,cl in start: draw_lobe(P@Rt.T,cl)                       # the turning orbital
        ax.scatter(0,0,0,color="#222",s=40,zorder=6)
        if phase=="done":
            ax.text2D(0.5,0.12,f"the orbital has become ${reslab}$",transform=ax.transAxes,
                      ha="center",fontsize=11,style="italic",color="#555")
            ax.text2D(0.5,0.05,f"$C_4\\,{startlab} = {reslab}$",transform=ax.transAxes,
                      ha="center",fontsize=15,weight="bold",color=NAVY)
        else:
            cap={"intro":"blue = + lobe,  red = \u2212 lobe",
                 "op":"turning $90^\\circ$ about $z$\u2026"}[phase]
            ax.text2D(0.5,0.05,cap,transform=ax.transAxes,ha="center",fontsize=10.5,style="italic",color="#666")
        return []
    tag={"dxz":"c4_dxz","dyz":"c4_dyz"}[orbital]
    out=f"{outdir}/mo_{tag}.gif"
    FuncAnimation(fig,update,frames=len(seg),interval=1000/FPS,blit=False).save(out,writer=PlayOnceWriter(fps=FPS))
    plt.close(fig); print("wrote",out); return out

def render_sigmad_on_p(orbital, outdir="."):
    """A DIAGONAL mirror sigma_d (the plane through z and the x=y line) acting on an
    in-plane p orbital. Where sigma_v(xz) and sigma_v(yz) send p_x and p_y each to
    +/- itself, sigma_d SWAPS them:
        sigma_d p_x = p_y ,   sigma_d p_y = p_x
    So a reflection can mix the pair just as C_4 does - the other half of why
    {p_x,p_y} must travel together as E. Shown as a mapping (each lobe slides in a
    straight line onto its mirror image). View (32,-90), matching the p-orbital figures."""
    assert orbital in ("px","py")
    axis={"px":np.array([1.,0,0]),"py":np.array([0,1.,0])}[orbital]
    NAVY="#1D3B5A"; BLUE="#2f5fbf"; RED="#c0392b"; lim=1.55
    startlab={"px":"p_x","py":"p_y"}[orbital]; reslab={"px":"p_y","py":"p_x"}[orbital]
    n=np.array([1.,-1,0.])/np.sqrt(2); M=reflect(n)                   # sigma_d: swaps x<->y
    def rot_to(a):
        a=np.asarray(a,float); a=a/np.linalg.norm(a); z=np.array([0,0,1.])
        if np.allclose(a,z): return np.eye(3)
        if np.allclose(a,-z): return np.diag([1,-1,-1.])
        v=np.cross(z,a); c=float(z@a); s=np.linalg.norm(v)
        vx=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        return np.eye(3)+vx+vx@vx*((1-c)/s**2)
    Rax=rot_to(axis)
    def lobe(sign,dd=0.52,La=0.60,Lb=0.33,N=22):
        u=np.linspace(0,2*np.pi,N); v=np.linspace(0,np.pi,N); U,V=np.meshgrid(u,v)
        x=Lb*np.cos(U)*np.sin(V); y=Lb*np.sin(U)*np.sin(V); z=La*np.cos(V)
        return np.stack([x,y,z],-1)@Rax.T + sign*dd*axis
    lp=lobe(+1); ln=lobe(-1)
    tp=lp@M.T; tn=ln@M.T                                             # mirror images (result)
    intro=10; nop=26; hold=20
    seg=[("intro",0.0)]*intro+[("op",(j+1)/nop) for j in range(nop)]+[("done",1.0)]*hold
    fig=plt.figure(figsize=FIG_SIZE, dpi=DPI); ax=fig.add_subplot(111,projection="3d")
    fig.text(0.5,0.93,f"$\\sigma_d$  on  ${startlab}$",ha="center",fontsize=17,weight="bold",color=NAVY)
    def draw_lobe(P,col,al=0.9,shade=True):
        ax.plot_surface(P[...,0],P[...,1],P[...,2],color=col,alpha=al,rstride=1,cstride=1,
                        linewidth=0,antialiased=True,shade=shade)
    R=1.25; diag=np.array([1,1,0.])/np.sqrt(2); zc=np.array([0,0,1.])
    crn=[R*diag+R*zc, R*diag-R*zc, -R*diag-R*zc, -R*diag+R*zc]
    def update(k):
        phase,t=seg[k]; te=ease(t)
        ax.clear(); ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
        ax.set_box_aspect((1,1,1)); ax.set_axis_off(); ax.view_init(32,-90)
        _flat_plane(ax,crn,1.0,a_fill=0.15,ew=1.6,a_edge=0.85)
        ax.text(*(1.12*diag+1.28*zc),"$\\sigma_d$",color=PLANE_EDGE,fontsize=12,weight="bold",ha="center")
        draw_lobe(tp,BLUE,0.10,shade=False); draw_lobe(tn,RED,0.10,shade=False)   # ghost result
        draw_lobe((1-te)*lp+te*tp,BLUE); draw_lobe((1-te)*ln+te*tn,RED)           # sliding to image
        ax.scatter(0,0,0,color="#222",s=40,zorder=6)
        if phase=="done":
            ax.text2D(0.5,0.12,f"the mirror has swapped it to ${reslab}$",transform=ax.transAxes,
                      ha="center",fontsize=11,style="italic",color="#555")
            ax.text2D(0.5,0.05,f"$\\sigma_d\\,{startlab} = {reslab}$",transform=ax.transAxes,
                      ha="center",fontsize=15,weight="bold",color=NAVY)
        else:
            cap={"intro":"blue = + lobe,  red = \u2212 lobe",
                 "op":"reflecting in $\\sigma_d$ (the $x=y$ plane)\u2026"}[phase]
            ax.text2D(0.5,0.05,cap,transform=ax.transAxes,ha="center",fontsize=10.5,style="italic",color="#666")
        return []
    tag={"px":"sigmad_px","py":"sigmad_py"}[orbital]
    out=f"{outdir}/mo_{tag}.gif"
    FuncAnimation(fig,update,frames=len(seg),interval=1000/FPS,blit=False).save(out,writer=PlayOnceWriter(fps=FPS))
    plt.close(fig); print("wrote",out); return out

def render_c3_on_p(orbital, outdir="."):
    """C_3(z) acting on p_x / p_y - the E pair of C_3v. Unlike C_4, which swapped the
    pair, a 120-degree rotation sends each orbital to a genuine MIXTURE of both:
        C_3 p_x = -1/2 p_x + (sqrt3/2) p_y     (a p orbital pointing at 120 deg)
        C_3 p_y = -(sqrt3/2) p_x - 1/2 p_y     (a p orbital pointing at 210 deg)
    The 2x2 matrix [[-1/2,-sqrt3/2],[sqrt3/2,-1/2]] has trace 2*cos120 = -1 = chi(C_3)
    for E. Faint fixed ghosts mark the pure p_x and p_y directions, so the tilted
    result reads as a blend of the two. View (55,-90): p_x horizontal and p_y vertical
    as in the other p figures, but the xy-plane is opened up so the in-plane 120-degree
    sweep is visible (at (32,-90) the plane is edge-on and the mixing is hidden)."""
    assert orbital in ("px","py")
    axis={"px":np.array([1.,0,0]),"py":np.array([0,1.,0])}[orbital]
    NAVY="#1D3B5A"; BLUE="#2f5fbf"; RED="#c0392b"; lim=1.5
    startlab={"px":"p_x","py":"p_y"}[orbital]
    combo={"px":r"-\frac{1}{2}\,p_x + \frac{\sqrt{3}}{2}\,p_y",
           "py":r"-\frac{\sqrt{3}}{2}\,p_x - \frac{1}{2}\,p_y"}[orbital]
    def rot_to(a):
        a=np.asarray(a,float); a=a/np.linalg.norm(a); z=np.array([0,0,1.])
        if np.allclose(a,z): return np.eye(3)
        if np.allclose(a,-z): return np.diag([1,-1,-1.])
        v=np.cross(z,a); c=float(z@a); s=np.linalg.norm(v)
        vx=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        return np.eye(3)+vx+vx@vx*((1-c)/s**2)
    def plobe(dirv,sign,dd=0.52,La=0.60,Lb=0.33,N=22):
        dirv=np.asarray(dirv,float)
        u=np.linspace(0,2*np.pi,N); v=np.linspace(0,np.pi,N); U,V=np.meshgrid(u,v)
        x=Lb*np.cos(U)*np.sin(V); y=Lb*np.sin(U)*np.sin(V); z=La*np.cos(V)
        return np.stack([x,y,z],-1)@rot_to(dirv).T + sign*dd*dirv
    lp=plobe(axis,1); ln=plobe(axis,-1)
    xax=np.array([1.,0,0]); yax=np.array([0,1.,0])
    ghosts=[(plobe(xax,1),BLUE),(plobe(xax,-1),RED),(plobe(yax,1),BLUE),(plobe(yax,-1),RED)]
    intro=10; nop=30; hold=22
    seg=[("intro",0.0)]*intro+[("op",(j+1)/nop) for j in range(nop)]+[("done",1.0)]*hold
    fig=plt.figure(figsize=FIG_SIZE, dpi=DPI); ax=fig.add_subplot(111,projection="3d")
    fig.text(0.5,0.93,f"$C_3(z)$  on  ${startlab}$",ha="center",fontsize=17,weight="bold",color=NAVY)
    def draw_lobe(P,col,al=0.9,shade=True):
        ax.plot_surface(P[...,0],P[...,1],P[...,2],color=col,alpha=al,rstride=1,cstride=1,
                        linewidth=0,antialiased=True,shade=shade)
    tt=np.linspace(d(6),d(120),52); ra=0.62; zc=1.12
    arc=np.stack([ra*np.cos(tt),ra*np.sin(tt),np.full_like(tt,zc)],1)
    def update(k):
        phase,t=seg[k]; te=ease(t)
        ax.clear(); ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
        ax.set_box_aspect((1,1,1)); ax.set_axis_off(); ax.view_init(55,-90)
        ax.plot([0,0],[0,0],[-1.25,1.25],color=AXIS_C,lw=2.0,alpha=0.9,zorder=0)
        ax.text(0,0,1.42,"$C_3$",color=AXIS_C,fontsize=12,weight="bold",ha="center")
        ax.plot(arc[:,0],arc[:,1],arc[:,2],color=AXIS_C,lw=2.6,alpha=0.9,zorder=2)
        tan=arc[-1]-arc[-4]; tan=tan/np.linalg.norm(tan)*0.26
        ax.quiver(*(arc[-1]-tan),*tan,color=AXIS_C,lw=2.6,arrow_length_ratio=0.9,zorder=3)
        for P,c in ghosts: draw_lobe(P,c,0.10,shade=False)          # pure p_x / p_y basis
        ax.text(1.28,0,0,"$p_x$",color="#8a8a8a",fontsize=9,ha="center",zorder=4)
        ax.text(0,1.24,0,"$p_y$",color="#8a8a8a",fontsize=9,ha="center",zorder=4)
        Rt=Rmat([0,0,1],d(120)*te)
        draw_lobe(lp@Rt.T,BLUE); draw_lobe(ln@Rt.T,RED)             # the turning orbital
        ax.scatter(0,0,0,color="#222",s=40,zorder=6)
        if phase=="done":
            ax.text2D(0.5,0.135,"a mixture of $p_x$ and $p_y$",transform=ax.transAxes,
                      ha="center",fontsize=11,style="italic",color="#555")
            ax.text2D(0.5,0.055,f"$C_3\\,{startlab} = {combo}$",transform=ax.transAxes,
                      ha="center",fontsize=14,weight="bold",color=NAVY)
        else:
            cap={"intro":"blue = + lobe,  red = \u2212 lobe",
                 "op":"turning $120^\\circ$ about $z$\u2026"}[phase]
            ax.text2D(0.5,0.055,cap,transform=ax.transAxes,ha="center",fontsize=10.5,style="italic",color="#666")
        return []
    tag={"px":"c3_px","py":"c3_py"}[orbital]
    out=f"{outdir}/mo_{tag}.gif"
    FuncAnimation(fig,update,frames=len(seg),interval=1000/FPS,blit=False).save(out,writer=PlayOnceWriter(fps=FPS))
    plt.close(fig); print("wrote",out); return out

def render_c3_on_p(orbital, outdir="."):
    """C_3(z) acting on p_x / p_y, shown BIRDS-EYE (looking straight down z; x to the
    right, y up). A 120 degree rotation does not send p_x and p_y to +/- themselves,
    nor simply swap them - each becomes a genuine MIXTURE of the two:
        C_3 p_x = -1/2 p_x + (sqrt3/2) p_y
        C_3 p_y = -(sqrt3/2) p_x - 1/2 p_y
    The 2x2 block therefore has trace -1, i.e. chi(C_3) = -1 for the E irrep of C_3v.
    That inseparable mixing is exactly why {p_x,p_y} sit together as E. Blue = + lobe,
    red = - lobe."""
    assert orbital in ("px","py")
    axis={"px":np.array([1.,0,0]),"py":np.array([0,1.,0])}[orbital]
    NAVY="#1D3B5A"; BLUE="#2f5fbf"; RED="#c0392b"; AXG="#8a8a8a"; lim=1.5
    startlab={"px":"p_x","py":"p_y"}[orbital]
    eq={"px":r"$C_3\,p_x = -\tfrac{1}{2}\,p_x + \tfrac{\sqrt{3}}{2}\,p_y$",
        "py":r"$C_3\,p_y = -\tfrac{\sqrt{3}}{2}\,p_x - \tfrac{1}{2}\,p_y$"}[orbital]
    eq=eq.replace(r"\tfrac",r"\frac")
    def rot_to(a):
        a=np.asarray(a,float); a=a/np.linalg.norm(a); z=np.array([0,0,1.])
        if np.allclose(a,z): return np.eye(3)
        if np.allclose(a,-z): return np.diag([1,-1,-1.])
        v=np.cross(z,a); c=float(z@a); s=np.linalg.norm(v)
        vx=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        return np.eye(3)+vx+vx@vx*((1-c)/s**2)
    Rax=rot_to(axis)
    def lobe(sign,dd=0.52,La=0.60,Lb=0.33,N=22):
        u=np.linspace(0,2*np.pi,N); v=np.linspace(0,np.pi,N); U,V=np.meshgrid(u,v)
        x=Lb*np.cos(U)*np.sin(V); y=Lb*np.sin(U)*np.sin(V); z=La*np.cos(V)
        return np.stack([x,y,z],-1)@Rax.T + sign*dd*axis
    lp=lobe(+1); ln=lobe(-1)
    R120=Rmat([0,0,1],d(120))
    tp=lp@R120.T; tn=ln@R120.T                      # result = start turned +120 about z

    intro=10; nop=30; hold=22
    seg=[("intro",0.0)]*intro+[("op",(j+1)/nop) for j in range(nop)]+[("done",1.0)]*hold
    fig=plt.figure(figsize=FIG_SIZE, dpi=DPI); ax=fig.add_subplot(111,projection="3d")
    fig.text(0.5,0.93,f"$C_3(z)$  on  ${startlab}$",ha="center",fontsize=17,weight="bold",color=NAVY)
    def draw_lobe(P,col,al=0.9,shade=True):
        ax.plot_surface(P[...,0],P[...,1],P[...,2],color=col,alpha=al,rstride=1,cstride=1,
                        linewidth=0,antialiased=True,shade=shade)
    tt=np.linspace(d(18),d(102),44); ra=1.02
    arc=np.stack([ra*np.cos(tt),ra*np.sin(tt),np.zeros_like(tt)],1)
    def update(k):
        phase,t=seg[k]; te=ease(t)
        ax.clear(); ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
        ax.set_box_aspect((1,1,1)); ax.set_axis_off(); ax.view_init(90,-90)
        # in-plane reference axes:  x -> right,  y -> up
        ax.quiver(0,0,0,1.28,0,0,color=AXG,lw=1.7,arrow_length_ratio=0.05,zorder=1)
        ax.quiver(0,0,0,0,1.28,0,color=AXG,lw=1.7,arrow_length_ratio=0.05,zorder=1)
        ax.text(1.44,0,0,"x",color="#555",fontsize=13,weight="bold",ha="center",va="center")
        ax.text(0,1.44,0,"y",color="#555",fontsize=13,weight="bold",ha="center",va="center")
        # C_3 sense arrow (120 deg, anticlockwise from above)
        ax.plot(arc[:,0],arc[:,1],arc[:,2],color=AXIS_C,lw=2.5,alpha=0.9,zorder=2)
        tan=arc[-1]-arc[-4]; tan=tan/np.linalg.norm(tan)*0.28
        ax.quiver(*(arc[-1]-tan),*tan,color=AXIS_C,lw=2.5,arrow_length_ratio=0.9,zorder=3)
        ax.text2D(0.69,0.73,"$C_3$",transform=ax.transAxes,color=AXIS_C,fontsize=13,weight="bold")
        # faint ghost of the result, then the rotating orbital
        draw_lobe(tp,BLUE,0.10,shade=False); draw_lobe(tn,RED,0.10,shade=False)
        Rt=Rmat([0,0,1],d(120)*te)
        draw_lobe(lp@Rt.T,BLUE); draw_lobe(ln@Rt.T,RED)
        ax.scatter(0,0,0,color="#222",s=28,zorder=6)
        if phase=="done":
            ax.text2D(0.5,0.12,"now a mixture of $p_x$ and $p_y$",transform=ax.transAxes,
                      ha="center",fontsize=11,style="italic",color="#555")
            ax.text2D(0.5,0.045,eq,transform=ax.transAxes,ha="center",fontsize=13.5,weight="bold",color=NAVY)
        else:
            cap={"intro":"blue = + lobe,  red = \u2212 lobe",
                 "op":"rotating $120^\\circ$ about $z$\u2026"}[phase]
            ax.text2D(0.5,0.05,cap,transform=ax.transAxes,ha="center",fontsize=10.5,style="italic",color="#666")
        return []
    tag={"px":"c3_px","py":"c3_py"}[orbital]
    out=f"{outdir}/mo_{tag}.gif"
    FuncAnimation(fig,update,frames=len(seg),interval=1000/FPS,blit=False).save(out,writer=PlayOnceWriter(fps=FPS))
    plt.close(fig); print("wrote",out); return out

def render_sigmav_on_p(orbital, outdir="."):
    """A vertical mirror sigma_v (the plane containing x and z) acting on p_x / p_y,
    shown BIRDS-EYE (down z; x right, y up; the mirror is edge-on, a line along x).
    This particular sigma_v is DIAGONAL in the {p_x,p_y} basis:
        sigma_v p_x = +p_x   (in the plane        -> symmetric,     +1)
        sigma_v p_y = -p_y   (perpendicular to it -> antisymmetric, -1)
    so its character contribution is +1 + (-1) = 0 = chi(sigma_v) for E in C_3v. C_3 is
    NOT diagonal in the same basis, which is why {p_x,p_y} cannot be reduced. Shown as
    a mapping (each lobe slides onto its mirror image). Blue = + lobe, red = - lobe."""
    assert orbital in ("px","py")
    axis={"px":np.array([1.,0,0]),"py":np.array([0,1.,0])}[orbital]
    NAVY="#1D3B5A"; BLUE="#2f5fbf"; RED="#c0392b"; AXG="#8a8a8a"; lim=1.5
    POS="#2e7d32"; NEG="#c0392b"
    startlab={"px":"p_x","py":"p_y"}[orbital]
    reslab ={"px":"+p_x","py":"-p_y"}[orbital]
    verdict={"px":("maps onto itself","symmetric  $\\rightarrow +1$",POS),
             "py":("maps onto its negative","antisymmetric  $\\rightarrow -1$",NEG)}[orbital]
    n=np.array([0.,1.,0.]); M=reflect(n)             # sigma_v(xz): y -> -y
    def rot_to(a):
        a=np.asarray(a,float); a=a/np.linalg.norm(a); z=np.array([0,0,1.])
        if np.allclose(a,z): return np.eye(3)
        if np.allclose(a,-z): return np.diag([1,-1,-1.])
        v=np.cross(z,a); c=float(z@a); s=np.linalg.norm(v)
        vx=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        return np.eye(3)+vx+vx@vx*((1-c)/s**2)
    Rax=rot_to(axis)
    def lobe(sign,dd=0.52,La=0.60,Lb=0.33,N=22):
        u=np.linspace(0,2*np.pi,N); v=np.linspace(0,np.pi,N); U,V=np.meshgrid(u,v)
        x=Lb*np.cos(U)*np.sin(V); y=Lb*np.sin(U)*np.sin(V); z=La*np.cos(V)
        return np.stack([x,y,z],-1)@Rax.T + sign*dd*axis
    lp=lobe(+1); ln=lobe(-1)
    tp=lp@M.T; tn=ln@M.T                              # mirror images (result)

    intro=10; nop=26; hold=22
    seg=[("intro",0.0)]*intro+[("op",(j+1)/nop) for j in range(nop)]+[("done",1.0)]*hold
    fig=plt.figure(figsize=FIG_SIZE, dpi=DPI); ax=fig.add_subplot(111,projection="3d")
    fig.text(0.5,0.93,f"$\\sigma_v(xz)$  on  ${startlab}$",ha="center",fontsize=17,weight="bold",color=NAVY)
    def draw_lobe(P,col,al=0.9,shade=True):
        ax.plot_surface(P[...,0],P[...,1],P[...,2],color=col,alpha=al,rstride=1,cstride=1,
                        linewidth=0,antialiased=True,shade=shade)
    def update(k):
        phase,t=seg[k]; te=ease(t)
        ax.clear(); ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
        ax.set_box_aspect((1,1,1)); ax.set_axis_off(); ax.view_init(90,-90)
        # in-plane reference axes:  x -> right,  y -> up
        ax.quiver(0,0,0,1.28,0,0,color=AXG,lw=1.7,arrow_length_ratio=0.05,zorder=1)
        ax.quiver(0,0,0,0,1.28,0,color=AXG,lw=1.7,arrow_length_ratio=0.05,zorder=1)
        ax.text(1.44,0,0,"x",color="#555",fontsize=13,weight="bold",ha="center",va="center")
        ax.text(0,1.44,0,"y",color="#555",fontsize=13,weight="bold",ha="center",va="center")
        # sigma_v mirror, edge-on: a bold line along x
        ax.plot([-1.34,1.34],[0,0],[0,0],color=PLANE_EDGE,lw=3.0,alpha=0.95,zorder=2)
        ax.text2D(0.80,0.42,"$\\sigma_v$",transform=ax.transAxes,color=PLANE_EDGE,fontsize=14,weight="bold")
        draw_lobe(tp,BLUE,0.10,shade=False); draw_lobe(tn,RED,0.10,shade=False)   # ghost result
        draw_lobe((1-te)*lp+te*tp,BLUE); draw_lobe((1-te)*ln+te*tn,RED)           # sliding to image
        ax.scatter(0,0,0,color="#222",s=28,zorder=6)
        if phase=="done":
            ax.text2D(0.5,0.13,verdict[0],transform=ax.transAxes,ha="center",fontsize=10.5,
                      style="italic",color="#555")
            ax.text2D(0.5,0.055,f"$\\sigma_v\\,{startlab} = {reslab}$",transform=ax.transAxes,
                      ha="center",fontsize=14,weight="bold",color=NAVY)
            ax.text2D(0.5,-0.01,verdict[1],transform=ax.transAxes,ha="center",fontsize=11,
                      weight="bold",color=verdict[2])
        else:
            cap={"intro":"blue = + lobe,  red = \u2212 lobe",
                 "op":"reflecting in $\\sigma_v$ (the $xz$ plane)\u2026"}[phase]
            ax.text2D(0.5,0.05,cap,transform=ax.transAxes,ha="center",fontsize=10.5,style="italic",color="#666")
        return []
    tag={"px":"sigmav_px","py":"sigmav_py"}[orbital]
    out=f"{outdir}/mo_{tag}.gif"
    FuncAnimation(fig,update,frames=len(seg),interval=1000/FPS,blit=False).save(out,writer=PlayOnceWriter(fps=FPS))
    plt.close(fig); print("wrote",out); return out

def render_cn_single(molkey, n, kind, symbol, tag, outdir="."):
    """One C_n-family point group per GIF. kind: 'h' (C_nh: C_n + sigma_h),
    'v' (C_nv: C_n + n sigma_v), 'n' (C_n: axis only)."""
    dd=lambda t: np.radians(t); NAVY="#1D3B5A"
    atoms,bonds,sv=_cn_molecules()[molkey]
    pos=np.array([p for _,p in atoms],float)
    # centre on the centroid, not on whichever atom happens to sit at the origin,
    # or a lopsided molecule is drawn small and off to one side. The centroid of
    # a C_n molecule lies ON the axis, so this shift never moves the axis (nor
    # sigma_h, which by symmetry contains the centroid).
    pos=pos-pos.mean(axis=0)
    rmax=np.max(np.linalg.norm(pos,axis=1)) or 1.0
    pos=pos*(1.55/rmax)
    lim=1.95; Zc=1.8; Rr=1.7
    frames=[]; azim=[-58.0]
    def push(nf,el0,el1,ax0,ax1,sh0,sh1,pl0,pl1,spin=1.6):
        for i in range(nf):
            t=i/max(1,nf-1)
            frames.append(dict(elev=el0+(el1-el0)*t,azim=azim[0],aax=ax0+(ax1-ax0)*t,
                               ash=sh0+(sh1-sh0)*t,apl=pl0+(pl1-pl0)*t)); azim[0]+=spin
    push(10, 8,26, 0,0, 0,0, 0,0); push(7,26,26,0,1,0,0,0,0); push(6,26,26,1,1,0,0,0,0)
    if kind=="h":
        push(6,26,26,1,1,0,1,0,0); push(18,26,26,1,1,1,1,0,0)
    elif kind=="v":
        push(6,26,26,1,1,0,0,0,1); push(18,26,26,1,1,0,0,1,1)
    elif kind=="dh":
        push(6,26,26,1,1,0,1,0,0); push(6,26,26,1,1,1,1,0,1); push(16,26,26,1,1,1,1,1,1)
    elif kind=="s":
        push(20,26,26,1,1,0,0,0,0)
    else:
        push(20,26,26,1,1,0,0,0,0)
    fig=plt.figure(figsize=FIG_SIZE, dpi=DPI); ax=fig.add_subplot(111,projection="3d")
    def planepoly(corners,al,fill="#d9c48a",edge=PLANE_EDGE):
        # robust flat plane: Poly3DCollection with alpha baked into the colour,
        # plus a bold outline so the plane still reads even if a fill is dropped
        C=np.array(corners,float)
        pc=Poly3DCollection([C],facecolors=[to_rgba(fill,0.22*al)],edgecolors="none")
        ax.add_collection3d(pc)
        loop=np.vstack([C,C[0]]); ax.plot(loop[:,0],loop[:,1],loop[:,2],color=edge,lw=1.6,alpha=0.9*al)
    def vplane(a,al):
        u=np.array([np.cos(dd(a)),np.sin(dd(a)),0]); z=np.array([0,0,Zc*0.92])
        planepoly([u*Rr+z,u*Rr-z,-u*Rr-z,-u*Rr+z],al)
    def update(k):
        fr=frames[k]; aax,ash,apl=fr["aax"],fr["ash"],fr["apl"]
        ax.clear(); ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
        ax.set_box_aspect((1,1,1)); ax.set_axis_off(); ax.view_init(fr["elev"],fr["azim"])
        if ash>0.02:                                  # horizontal plane sigma_h
            R=Rr; planepoly([[-R,-R,0],[R,-R,0],[R,R,0],[-R,R,0]],ash,fill="#8bb8cf",edge="#5f92aa")
        if apl>0.02 and (sv or kind=="dh"):           # vertical planes sigma_v / sigma_d
            angs=[kk*180/n for kk in range(n)] if kind=="dh" else sv
            for a in angs: vplane(a,apl)
        if aax>0.02:
            if kind=="s":
                ax.plot([0,0],[0,0],[-Zc,Zc],color="#7d3c98",lw=2.2,ls=(0,(4,3)),alpha=0.9*aax,zorder=1)
                ax.text(0,0,Zc+0.12,f"$S_{n}$",color="#7d3c98",fontsize=11,weight="bold",ha="center",alpha=aax)
            else:
                ax.plot([0,0],[0,0],[-Zc,Zc],color=AXIS_C,lw=2.0,alpha=0.9*aax,zorder=1)
                ax.text(0,0,Zc+0.12,f"$C_{n}$",color=AXIS_C,fontsize=11,weight="bold",ha="center",alpha=aax)
                if kind=="dh":                       # in-plane C2 axes (radial)
                    for kk in range(n):
                        a=dd(kk*180/n); u=np.array([np.cos(a),np.sin(a),0])*Rr*0.98
                        ax.plot([-u[0],u[0]],[-u[1],u[1]],[0,0],color=AXIS_C,lw=1.5,alpha=0.8*aax,zorder=1)
        for i,j in bonds: ax.plot(*zip(pos[i],pos[j]),color=BOND_C,lw=2.4,alpha=0.9,zorder=3)
        for idx,(el,_) in enumerate(atoms):
            st=STYLE.get(el,STYLE["A"]); szf=0.30 if len(atoms)>=18 else 0.7
            ax.scatter(*pos[idx],color=st["color"],s=st["s"]*szf,
                edgecolor=st["edge"],lw=0.6,depthshade=True,zorder=4)
        fig.texts=[]
        fig.text(0.5,0.93,f"${symbol}$",ha="center",fontsize=28,weight="bold",color=NAVY)
        nm={"boric_acid":"boric acid B(OH)$_3$","benzenehexol":"benzenehexol C$_6$(OH)$_6$",
            "glyoxal":"trans-glyoxal","water":"water","ammonia":"ammonia NH$_3$","IF5":"IF$_5$",
            "h2o2":"hydrogen peroxide H$_2$O$_2$","ph3ch":"triphenylmethane",
            "s4model":"an $S_4$ molecule (4 arms related by $S_4$)","benzenehexol_d6h":"benzenehexol C$_6$(OH)$_6$"}[molkey]
        if kind=="h":
            cap=(f"$C_{n}$ axis and horizontal plane $\\sigma_h$" if ash>0.5 else
                 (f"$C_{n}$ principal axis" if aax>0.5 else nm))
        elif kind=="v":
            cap=(f"$C_{n}$ axis and {n} vertical planes $\\sigma_v$" if apl>0.5 else
                 (f"$C_{n}$ principal axis" if aax>0.5 else nm))
        elif kind=="dh":
            cap=(f"$C_{n}$, {n}$C_2$, $\\sigma_h$, {n}$\\sigma_v$" if apl>0.5 else
                 (f"$C_{n}$ + {n}$C_2$, horizontal plane $\\sigma_h$" if ash>0.5 else
                  (f"$C_{n}$ axis and {n} $C_2$ axes" if aax>0.5 else nm)))
        elif kind=="s":
            cap=(f"only $S_{n}$ (and $C_2=S_{n}^2$) \u2014 no mirror planes" if aax>0.5 else nm)
        else:
            cap=(f"only a $C_{n}$ axis \u2014 no mirror planes" if aax>0.5 else nm)
        fig.text(0.5,0.065,nm,ha="center",fontsize=12.5,weight="bold",color="#333")
        fig.text(0.5,0.02,cap,ha="center",fontsize=9.5,color="#666",style="italic")
        return []
    out=f"{outdir}/cn_{tag}.gif"
    FuncAnimation(fig,update,frames=len(frames),interval=1000/FPS,blit=False).save(out,writer=PlayOnceWriter(fps=FPS))
    plt.close(fig); print("wrote",out); return out

def render_dn_single(n, config, tag, outdir="."):
    """One point group per GIF from the stacked n-gon model. config: 'h' (eclipsed,
    D_nh), 'd' (staggered, D_nd), 'n' (unevenly staggered, D_n). Builds up the
    elements: C_n + nC2 always; sigma_h + n sigma_v for h; n sigma_d for d; none for n."""
    dd=lambda t: np.radians(t); NAVY="#1D3B5A"
    r=1.15; zt=0.72; lim=1.95; Zc=1.8; Rr=1.7
    off = 0.0 if config=="h" else (180/n if config=="d" else 0.55*180/n)
    symbol = f"D_{{{n}h}}" if config=="h" else (f"D_{{{n}d}}" if config=="d" else f"D_{{{n}}}")
    ptype  = "\\sigma_v" if config=="h" else ("\\sigma_d" if config=="d" else "")
    def poly(o,z): return np.array([[r*np.cos(dd(360*k/n+o)),r*np.sin(dd(360*k/n+o)),z] for k in range(n)])

    frames=[]; azim=[-58.0]
    def push(nf,el0,el1,ax0,ax1,sh0,sh1,pl0,pl1,spin=1.6):
        for i in range(nf):
            t=i/max(1,nf-1)
            frames.append(dict(elev=el0+(el1-el0)*t,azim=azim[0],aax=ax0+(ax1-ax0)*t,
                               ash=sh0+(sh1-sh0)*t,apl=pl0+(pl1-pl0)*t)); azim[0]+=spin
    push(10, 8,26, 0,0, 0,0, 0,0)          # intro: side -> 3/4
    push(7, 26,26, 0,1, 0,0, 0,0)          # Cn + nC2 in
    push(6, 26,26, 1,1, 0,0, 0,0)
    if config=="h":
        push(6, 26,26, 1,1, 0,1, 0,0)      # sigma_h in
        push(5, 26,26, 1,1, 1,1, 0,0)
        push(6, 26,26, 1,1, 1,1, 0,1)      # sigma_v in
        push(16,26,26, 1,1, 1,1, 1,1)
    elif config=="d":
        push(6, 26,26, 1,1, 0,0, 0,1)      # sigma_d in
        push(18,26,26, 1,1, 0,0, 1,1)
    else:
        push(20,26,26, 1,1, 0,0, 0,0)      # no planes, hold

    fig=plt.figure(figsize=FIG_SIZE, dpi=DPI); ax=fig.add_subplot(111,projection="3d")
    def vplane(a,al):
        u=np.array([np.cos(dd(a)),np.sin(dd(a)),0]); z=np.array([0,0,Zc*0.92])
        sq=np.array([u*Rr+z,u*Rr-z,-u*Rr-z,-u*Rr+z])
        ax.add_collection3d(Poly3DCollection([sq],facecolor=PLANE_C,alpha=0.12*al,edgecolor="none"))
        ax.plot([-u[0]*Rr,u[0]*Rr],[-u[1]*Rr,u[1]*Rr],[0,0],color=PLANE_EDGE,lw=1.2,alpha=0.7*al)

    def update(k):
        fr=frames[k]
        ax.clear(); ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
        ax.set_box_aspect((1,1,1)); ax.set_axis_off(); ax.view_init(fr["elev"],fr["azim"])
        aax,ash,apl=fr["aax"],fr["ash"],fr["apl"]
        if ash>0.02:
            R=Rr; _flat_plane(ax,[[-R,-R,0],[R,-R,0],[R,R,0],[-R,R,0]],ash,fill="#8bb8cf",edge="#5f92aa",a_fill=0.18)
        if apl>0.02:
            for kk in range(n): vplane(kk*180/n,apl)
        if aax>0.02:
            ax.plot([0,0],[0,0],[-Zc,Zc],color=AXIS_C,lw=2.0,alpha=0.9*aax,zorder=1)
            ax.text(0,0,Zc+0.12,f"$C_{n}$",color=AXIS_C,fontsize=11,weight="bold",ha="center",alpha=aax)
            for kk in range(n):
                a=dd(off/2+kk*180/n); u=np.array([np.cos(a),np.sin(a),0])*Rr*0.98
                ax.plot([-u[0],u[0]],[-u[1],u[1]],[0,0],color=AXIS_C,lw=1.6,alpha=0.85*aax,zorder=1)
        top=poly(off,zt); bot=poly(0,-zt)
        tt=np.vstack([top,top[0]]); ax.plot(tt[:,0],tt[:,1],tt[:,2],color="#c0392b",lw=2.6,alpha=0.95,zorder=5)
        bb=np.vstack([bot,bot[0]]); ax.plot(bb[:,0],bb[:,1],bb[:,2],color="#2f5fbf",lw=1.8,alpha=0.7,zorder=3)
        for p in top: ax.scatter(*p,color="#e9e9e9",s=170,edgecolor="#7a1f16",lw=1.2,zorder=6)
        for p in bot: ax.scatter(*p,color="#cfcfcf",s=140,edgecolor="#26468f",lw=1.0,alpha=0.85,zorder=4)
        fig.texts=[]
        fig.text(0.5,0.93,f"${symbol}$",ha="center",fontsize=28,weight="bold",color=NAVY)
        shp="squares" if n==4 else "pentagons"
        if config=="h":
            cap=(f"$C_{n}$, {n}$C_2$, $\\sigma_h$, {n}$\\sigma_v$" if apl>0.5 else
                 (f"$C_{n}$, {n}$C_2$, horizontal plane $\\sigma_h$" if ash>0.5 else
                  (f"$C_{n}$ axis and {n} $C_2$ axes" if aax>0.5 else f"two eclipsed {shp}")))
        elif config=="d":
            cap=(f"$C_{n}$, {n}$C_2$, {n}$\\sigma_d$  (no $\\sigma_h$)" if apl>0.5 else
                 (f"$C_{n}$ axis and {n} $C_2$ axes" if aax>0.5 else f"two staggered {shp}"))
        else:
            cap=(f"only $C_{n}$ + {n}$C_2$ \u2014 no mirror planes (rare)" if aax>0.5
                 else f"two unevenly staggered {shp}")
        fig.text(0.5,0.05,cap,ha="center",fontsize=10.5,color="#555",style="italic")
        return []

    out=f"{outdir}/dn_{tag}_{config}.gif"
    FuncAnimation(fig,update,frames=len(frames),interval=1000/FPS,blit=False).save(out,writer=PlayOnceWriter(fps=FPS))
    plt.close(fig); print("wrote",out); return out

def render_dn_family(n, tag, outdir="."):
    """Two stacked n-gons showing the D_nh -> D_nd -> D_n progression: perfectly
    eclipsed (C_n, nC2, sigma_h, n vertical planes = D_nh); perfectly staggered
    (sigma_h lost, the n vertical planes remain as sigma_d = D_nd); unevenly
    staggered (planes gone, only C_n + nC2 = D_n). n=4 squares, n=5 pentagons."""
    dd=lambda t: np.radians(t); NAVY="#1D3B5A"
    r=1.15; zt=0.72; lim=1.95; Zc=1.8; Rr=1.7
    off_stag=180/n; off_uneven=0.6*180/n
    def poly(off,z): return np.array([[r*np.cos(dd(360*k/n+off)),r*np.sin(dd(360*k/n+off)),z] for k in range(n)])

    # timeline: list of frame dicts. azim spins slowly throughout.
    frames=[]; azim=[-58.0]
    def push(nf,ci,off0,off1,el0,el1,ax0,ax1,sh0,sh1,pl0,pl1,spin=1.6):
        for i in range(nf):
            t=i/max(1,nf-1)
            frames.append(dict(ci=ci,off=off0+(off1-off0)*t,elev=el0+(el1-el0)*t,azim=azim[0],
                               aax=ax0+(ax1-ax0)*t,ash=sh0+(sh1-sh0)*t,apl=pl0+(pl1-pl0)*t))
            azim[0]+=spin
    push(10,0,0,0,      8,26, 0,0, 0,0, 0,0)          # intro: side -> 3/4
    push(7, 0,0,0,     26,26, 0,1, 0,0, 0,0)          # eclipsed: axes in
    push(5, 0,0,0,     26,26, 1,1, 0,0, 0,0)
    push(6, 0,0,0,     26,26, 1,1, 0,1, 0,0)          # sigma_h in
    push(5, 0,0,0,     26,26, 1,1, 1,1, 0,0)
    push(6, 0,0,0,     26,26, 1,1, 1,1, 0,1)          # vertical planes in
    push(9, 0,0,0,     26,26, 1,1, 1,1, 1,1)          # hold  D_nh
    push(13,1,0,off_stag, 26,26, 1,1, 1,0, 1,1)       # restack -> staggered, sigma_h fades
    push(10,1,off_stag,off_stag, 26,26, 1,1, 0,0, 1,1)# hold  D_nd
    push(13,2,off_stag,off_uneven, 26,26, 1,1, 0,0, 1,0)  # restack -> uneven, planes fade
    push(12,2,off_uneven,off_uneven, 26,26, 1,1, 0,0, 0,0)# hold  D_n

    def cfg(ci):
        if ci==0: return (f"D_{{{n}h}}","perfectly eclipsed","\\sigma_v",True)
        if ci==1: return (f"D_{{{n}d}}","perfectly staggered","\\sigma_d",False)
        return (f"D_{{{n}}}","unevenly staggered (rare)","",False)

    fig=plt.figure(figsize=FIG_SIZE, dpi=DPI); ax=fig.add_subplot(111,projection="3d")
    def vplane(a,al):
        u=np.array([np.cos(dd(a)),np.sin(dd(a)),0]); z=np.array([0,0,Zc*0.92])
        sq=np.array([u*Rr+z,u*Rr-z,-u*Rr-z,-u*Rr+z])
        ax.add_collection3d(Poly3DCollection([sq],facecolor=PLANE_C,alpha=0.12*al,edgecolor="none"))
        ax.plot([-u[0]*Rr,u[0]*Rr],[-u[1]*Rr,u[1]*Rr],[0,0],color=PLANE_EDGE,lw=1.2,alpha=0.7*al)

    def update(k):
        fr=frames[k]; ci=fr["ci"]; off=fr["off"]; symbol,name,ptype,has_sh=cfg(ci)
        ax.clear(); ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
        ax.set_box_aspect((1,1,1)); ax.set_axis_off(); ax.view_init(fr["elev"],fr["azim"])
        aax,ash,apl=fr["aax"],fr["ash"],fr["apl"]
        # sigma_h (horizontal)
        if has_sh and ash>0.02:
            R=Rr; _flat_plane(ax,[[-R,-R,0],[R,-R,0],[R,R,0],[-R,R,0]],ash,fill="#8bb8cf",edge="#5f92aa",a_fill=0.18)
        # vertical mirror planes at k*180/n
        if apl>0.02:
            for kk in range(n): vplane(kk*180/n,apl)
        # C_n axis (vertical) + n C2 axes (horizontal, at off/2 + k*180/n)
        if aax>0.02:
            ax.plot([0,0],[0,0],[-Zc,Zc],color=AXIS_C,lw=2.0,alpha=0.9*aax,zorder=1)
            ax.text(0,0,Zc+0.12,f"$C_{n}$",color=AXIS_C,fontsize=11,weight="bold",ha="center",alpha=aax)
            for kk in range(n):
                a=dd(off/2+kk*180/n); u=np.array([np.cos(a),np.sin(a),0])*Rr*0.98
                ax.plot([-u[0],u[0]],[-u[1],u[1]],[0,0],color=AXIS_C,lw=1.6,alpha=0.85*aax,zorder=1)
        # the two polygons: top bold, bottom fainter
        top=poly(off,zt); bot=poly(0,-zt)
        tt=np.vstack([top,top[0]]); ax.plot(tt[:,0],tt[:,1],tt[:,2],color="#c0392b",lw=2.6,alpha=0.95,zorder=5)
        bb=np.vstack([bot,bot[0]]); ax.plot(bb[:,0],bb[:,1],bb[:,2],color="#2f5fbf",lw=1.8,alpha=0.7,zorder=3)
        for p in top: ax.scatter(*p,color="#e9e9e9",s=170,edgecolor="#7a1f16",lw=1.2,zorder=6)
        for p in bot: ax.scatter(*p,color="#cfcfcf",s=140,edgecolor="#26468f",lw=1.0,alpha=0.85,zorder=4)

        fig.texts=[]
        fig.text(0.5,0.93,f"${symbol}$",ha="center",fontsize=27,weight="bold",color=NAVY)
        # caption reflects the current reveal stage
        if ci==0:
            if apl>0.5: cap=f"$C_{n}$, {n}$C_2$, $\\sigma_h$, {n}$\\sigma_v$"
            elif ash>0.5: cap=f"$C_{n}$, {n}$C_2$, and the horizontal plane $\\sigma_h$"
            elif aax>0.5: cap=f"$C_{n}$ axis and {n} $C_2$ axes"
            else: cap=f"two stacked {'squares' if n==4 else 'pentagons'} \u2014 eclipsed"
        elif ci==1:
            cap=f"staggered: $\\sigma_h$ lost, {n}$\\sigma_d$ remain"
        else:
            cap=f"unevenly staggered (rare): no mirror planes, only $C_{n}$ + {n}$C_2$"
        fig.text(0.5,0.05,cap,ha="center",fontsize=10.5,color="#555",style="italic")
        return []

    out=f"{outdir}/dn_family_{tag}.gif"
    FuncAnimation(fig,update,frames=len(frames),interval=1000/FPS,blit=False).save(out,writer=PlayOnceWriter(fps=FPS))
    plt.close(fig); print("wrote",out); return out

def render_dnh_elements(outdir="."):
    """Three D3h molecules in turn, each with the principal C3 axis (vertical) and
    the sigma_h plane (horizontal) marked: PF3Cl2 (trigonal bipyramidal), BF3
    (trigonal planar), eclipsed ethane. Orientation is common (C3 = z, sigma_h = xy)."""
    dd=lambda t: np.radians(t); lib=_group_molecules()
    def eqax(cen,eqel,axel,req,rax):
        at=[(cen,[0,0,0])]; bo=[]
        for a in (0,120,240): at.append((eqel,[req*np.cos(dd(a)),req*np.sin(dd(a)),0])); bo.append((0,len(at)-1))
        at+=[(axel,[0,0,rax]),(axel,[0,0,-rax])]; bo+=[(0,len(at)-2),(0,len(at)-1)]; return at,bo
    pf3cl2=eqax("P","F","Cl",1.53,1.58)
    bf3=([("B",[0,0,0])]+[("F",[1.31*np.cos(dd(a)),1.31*np.sin(dd(a)),0]) for a in (0,120,240)],
         [(0,1),(0,2),(0,3)])
    members=[("PF$_3$Cl$_2$",pf3cl2),("BF$_3$",bf3),("eclipsed ethane",lib["eclipsed ethane"])]
    NAVY="#1D3B5A"; lim=2.3; per=30; fin=4; fout=4; Zc=1.85; Rp=1.95
    seg=[]
    for mi in range(len(members)):
        for f in range(per):
            azim=-62+f*5
            al=(f+1)/fin if f<fin else ((per-f)/fout if f>=per-fout else 1.0)
            seg.append((mi,azim,min(1.0,al)))
    fig=plt.figure(figsize=FIG_SIZE, dpi=DPI); ax=fig.add_subplot(111,projection="3d")
    fig.text(0.5,0.93,"$D_{3h}$",ha="center",fontsize=30,weight="bold",color=NAVY)
    fig.text(0.5,0.875,"principal $C_3$ axis  ·  horizontal plane $\\sigma_h$",
             ha="center",fontsize=10,color="#8a8f5c",style="italic")
    def update(k):
        mi,azim,al=seg[k]; name,(atoms,bonds)=members[mi]
        pos=np.array([p for _,p in atoms],float); rmax=np.max(np.linalg.norm(pos,axis=1)) or 1.0
        pos=pos*(1.6/rmax)
        ax.clear(); ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
        ax.set_box_aspect((1,1,1)); ax.set_axis_off(); ax.view_init(16,azim)
        # sigma_h plane (horizontal)
        _flat_plane(ax,[[-Rp,-Rp,0],[Rp,-Rp,0],[Rp,Rp,0],[-Rp,Rp,0]],al,a_fill=0.16)
        # C3 axis (vertical)
        ax.plot([0,0],[0,0],[-Zc,Zc],color=AXIS_C,lw=2.0,alpha=0.9*al,zorder=1)
        ax.text(0,0,Zc+0.12,"$C_3$",color=AXIS_C,fontsize=11,weight="bold",ha="center",alpha=al)
        ax.text(Rp*0.82,Rp*0.82,0,"$\\sigma_h$",color=PLANE_EDGE,fontsize=11,weight="bold",alpha=al)
        for i,j in bonds: ax.plot(*zip(pos[i],pos[j]),color=BOND_C,lw=2.6,alpha=0.9*al,zorder=3)
        for idx,(el,_) in enumerate(atoms):
            st=STYLE.get(el,STYLE["A"]); ax.scatter(*pos[idx],color=st["color"],s=st["s"]*0.8,
                edgecolor=st["edge"],lw=0.7,alpha=al,depthshade=True,zorder=4)
        ax.text2D(0.5,0.05,name,transform=ax.transAxes,ha="center",fontsize=13,weight="bold",color="#333",alpha=al)
        return []
    out=f"{outdir}/dnh_elements.gif"
    FuncAnimation(fig,update,frames=len(seg),interval=1000/FPS,blit=False).save(out,writer=PlayOnceWriter(fps=FPS))
    plt.close(fig); print("wrote",out); return out

def render_dnd_ferrocene(outdir="."):
    """Staggered ferrocene (D5d): side-on sandwich with the S10/C5 axis, then
    rotate to look down the axis at two staggered cyclopentadienyl rings (top ring
    bold), then reveal the five sigma_d planes."""
    dd=lambda t: np.radians(t); NAVY="#1D3B5A"
    rC=1.15; zC=1.5
    topC=np.array([[rC*np.cos(dd(a)),rC*np.sin(dd(a)), zC] for a in range(0,360,72)])
    botC=np.array([[rC*np.cos(dd(a)),rC*np.sin(dd(a)),-zC] for a in range(36,360,72)])
    lim=2.0; Zc=2.15; sd_ang=[0,36,72,108,144]

    side=18; tilt=16; topA=12; planes=22
    seg=[]
    for f in range(side):  seg.append(("side",8,-64+f*3,0.0))
    for f in range(tilt):  seg.append(("tilt",8+(88-8)*(f+1)/tilt,-10,0.0))
    for f in range(topA):  seg.append(("top",88,-10,0.0))
    for f in range(planes):seg.append(("planes",88,-10,min(1.0,(f+1)/6)))

    fig=plt.figure(figsize=FIG_SIZE, dpi=DPI); ax=fig.add_subplot(111,projection="3d")
    fig.text(0.5,0.93,"$D_{5d}$",ha="center",fontsize=26,weight="bold",color=NAVY)

    def ring(ax,V,col,lw,al,sz,ec):
        loop=np.vstack([V,V[0]]); ax.plot(loop[:,0],loop[:,1],loop[:,2],color=col,lw=lw,alpha=al,zorder=5)
        for p in V: ax.scatter(*p,color="#2b2b2b",s=sz,edgecolor=ec,lw=1.0,alpha=al,zorder=6)

    def update(k):
        phase,elev,azim,pa=seg[k]
        ax.clear(); ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
        ax.set_box_aspect((1,1,1)); ax.set_axis_off(); ax.view_init(elev,azim)
        aax=0.9 if phase in ("side","tilt") else 0.4
        ax.plot([0,0],[0,0],[-Zc,Zc],color=AXIS_C,lw=2.0,alpha=aax,zorder=1)
        if phase in ("side","tilt"):
            ax.text(0,0,Zc+0.12,"$S_{10}/C_5$",color=AXIS_C,fontsize=10.5,weight="bold",ha="center")
        if pa>0:
            for a in sd_ang:
                u=np.array([np.cos(dd(a)),np.sin(dd(a)),0]); z=np.array([0,0,Zc*0.92])
                sq=np.array([u*1.75+z,u*1.75-z,-u*1.75-z,-u*1.75+z])
                ax.add_collection3d(Poly3DCollection([sq],facecolor="#c98a3a",alpha=0.11*pa,edgecolor="none"))
                ax.plot([-u[0]*1.75,u[0]*1.75],[-u[1]*1.75,u[1]*1.75],[0,0],color="#a8801f",lw=1.2,alpha=0.75*pa)
        # Fe and faint Fe-C lines
        for p in np.vstack([topC,botC]): ax.plot(*zip([0,0,0],p),color=BOND_C,lw=0.8,alpha=0.3,zorder=2)
        ax.scatter(0,0,0,color=STYLE["Fe"]["color"],s=STYLE["Fe"]["s"]*0.8,edgecolor=STYLE["Fe"]["edge"],lw=0.8,zorder=4)
        ring(ax,botC,"#2f5fbf",1.8,0.7,150,"#26468f")     # bottom ring fainter
        ring(ax,topC,"#c0392b",2.6,0.95,230,"#7a1f16")    # top ring bold
        cap={"side":"staggered ferrocene \u2014 the $S_{10}/C_5$ axis",
             "tilt":"rotating to look down the axis\u2026",
             "top":"two staggered pentagons (top ring bold)",
             "planes":"five $\\sigma_d$ planes"}[phase]
        ax.text2D(0.5,0.05,cap,transform=ax.transAxes,ha="center",fontsize=11,style="italic",color="#555")
        return []
    out=f"{outdir}/dnd_ferrocene.gif"
    FuncAnimation(fig,update,frames=len(seg),interval=1000/FPS,blit=False).save(out,writer=PlayOnceWriter(fps=FPS))
    plt.close(fig); print("wrote",out); return out

def render_dnd_ethane(outdir="."):
    """Staggered ethane (D3d): side-on with the S6/C3 axis, then rotate to a
    top-down view showing two staggered triangles (top set bold), then reveal the
    three sigma_d mirror planes."""
    dd=lambda t: np.radians(t); NAVY="#1D3B5A"; PURPLE="#7d3c98"
    dcc=0.77; st=np.sin(dd(70.5)); ct=np.cos(dd(70.5)); rH=1.09
    C1=np.array([0,0,dcc]); C2=np.array([0,0,-dcc])
    topH=np.array([C1+rH*np.array([st*np.cos(dd(a)),st*np.sin(dd(a)),ct]) for a in (0,120,240)])
    botH=np.array([C2+rH*np.array([st*np.cos(dd(a)),st*np.sin(dd(a)),-ct]) for a in (60,180,300)])
    lim=1.9; Zc=1.75
    sd_ang=[0,120,240]     # sigma_d planes through top H and the anti bottom H

    # timeline: side-on hold, tilt up, top-down hold, reveal sigma_d
    side=18; tilt=16; topA=12; planes=22
    seg=[]
    for f in range(side):  seg.append(("side",8,-64+f*3,0.0))
    for f in range(tilt):  seg.append(("tilt",8+(88-8)*(f+1)/tilt,-10,0.0))
    for f in range(topA):  seg.append(("top",88,-10,0.0))
    for f in range(planes):seg.append(("planes",88,-10,min(1.0,(f+1)/6)))

    fig=plt.figure(figsize=FIG_SIZE, dpi=DPI); ax=fig.add_subplot(111,projection="3d")
    fig.text(0.5,0.93,"$D_{3d}$",ha="center",fontsize=26,weight="bold",color=NAVY)

    def update(k):
        phase,elev,azim,pa=seg[k]
        ax.clear(); ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
        ax.set_box_aspect((1,1,1)); ax.set_axis_off(); ax.view_init(elev,azim)
        # S6/C3 axis (bold in side view, faint once top-down)
        aax=0.9 if phase in ("side","tilt") else 0.4
        ax.plot([0,0],[0,0],[-Zc,Zc],color=AXIS_C,lw=2.0,alpha=aax,zorder=1)
        if phase in ("side","tilt"):
            ax.text(0,0,Zc+0.12,"$S_6\\,/\\,C_3$",color=AXIS_C,fontsize=11,weight="bold",ha="center")
        # sigma_d planes, top-down reveal
        if pa>0:
            for a in sd_ang:
                u=np.array([np.cos(dd(a)),np.sin(dd(a)),0]); z=np.array([0,0,Zc*0.95])
                sq=np.array([u*1.7+z,u*1.7-z,-u*1.7-z,-u*1.7+z])
                ax.add_collection3d(Poly3DCollection([sq],facecolor="#c98a3a",alpha=0.14*pa,edgecolor="none"))
                ax.plot([-u[0]*1.7,u[0]*1.7],[-u[1]*1.7,u[1]*1.7],[0,0],color="#a8801f",lw=1.4,alpha=0.8*pa)
        # carbons + C-H bonds
        ax.plot(*zip(C1,C2),color=BOND_C,lw=3.0,zorder=2)
        for C in (C1,C2): ax.scatter(*C,color="#2b2b2b",s=230,edgecolor="#000",lw=0.7,zorder=4)
        for h in topH: ax.plot(*zip(C1,h),color=BOND_C,lw=2.2,alpha=0.9,zorder=3)
        for h in botH: ax.plot(*zip(C2,h),color=BOND_C,lw=2.2,alpha=0.6,zorder=2)
        # top triangle bold, bottom fainter (clearest top-down)
        tt=np.vstack([topH,topH[0]]); ax.plot(tt[:,0],tt[:,1],tt[:,2],color="#c0392b",lw=2.6,alpha=0.95,zorder=5)
        bb=np.vstack([botH,botH[0]]); ax.plot(bb[:,0],bb[:,1],bb[:,2],color="#2f5fbf",lw=1.8,alpha=0.7,zorder=3)
        for h in topH: ax.scatter(*h,color="#e9e9e9",s=230,edgecolor="#7a1f16",lw=1.2,zorder=6)
        for h in botH: ax.scatter(*h,color="#cfcfcf",s=180,edgecolor="#26468f",lw=1.0,alpha=0.85,zorder=4)

        cap={"side":"staggered ethane \u2014 the $S_6/C_3$ axis",
             "tilt":"rotating to look down the axis\u2026",
             "top":"two staggered triangles (top set bold)",
             "planes":"three $\\sigma_d$ planes (between the staggered H)"}[phase]
        ax.text2D(0.5,0.05,cap,transform=ax.transAxes,ha="center",fontsize=11,style="italic",color="#555")
        return []
    out=f"{outdir}/dnd_ethane.gif"
    FuncAnimation(fig,update,frames=len(seg),interval=1000/FPS,blit=False).save(out,writer=PlayOnceWriter(fps=FPS))
    plt.close(fig); print("wrote",out); return out

def render_lowsym(outdir="."):
    """The 'no principal axis' branch of the flowchart: one example each of Cs
    (a mirror plane), Ci (an inversion centre), and C1 (no symmetry). The point
    group symbol changes per molecule since these are three different groups."""
    lib=_group_molecules()
    members=[("CHFCl$_2$","CHFCl2","C_s","one mirror plane $\\sigma$",False),
             ("CHFCl\u2013CHFCl (anti)","CHFClanti","C_i","centre of inversion $i$",True),
             ("CHFClBr","CHFClBr","C_1","no symmetry (only $E$)",False)]
    NAVY="#1D3B5A"; lim=2.4; per=28; fin=4; fout=4
    seg=[]
    for mi in range(len(members)):
        for f in range(per):
            azim=-60+f*6
            al=(f+1)/fin if f<fin else ((per-f)/fout if f>=per-fout else 1.0)
            seg.append((mi,azim,min(1.0,al)))

    fig=plt.figure(figsize=FIG_SIZE, dpi=DPI); ax=fig.add_subplot(111,projection="3d")
    fig.text(0.5,0.875,"no principal axis \u2014 low-symmetry groups",ha="center",
             fontsize=10.5,color="#8a8f5c",style="italic")

    def update(k):
        mi,azim,al=seg[k]
        name,key,sym,note,mark=members[mi]; atoms,bonds=lib[key]
        pos=np.array([p for _,p in atoms],float); rmax=np.max(np.linalg.norm(pos,axis=1)) or 1.0
        pos=pos*(1.7/rmax)
        ax.clear(); ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
        ax.set_box_aspect((1,1,1)); ax.set_axis_off(); ax.view_init(18,azim)
        for i,j in bonds: ax.plot(*zip(pos[i],pos[j]),color=BOND_C,lw=2.6,alpha=0.9*al,zorder=2)
        for idx,(el,_) in enumerate(atoms):
            st=STYLE.get(el,STYLE["A"])
            ax.scatter(*pos[idx],color=st["color"],s=st["s"]*0.8,edgecolor=st["edge"],
                       lw=0.7,alpha=al,depthshade=True,zorder=4)
        if mark:                                     # inversion centre for the Ci example
            ax.scatter(0,0,0,s=70,facecolors="none",edgecolors=INV_C,linewidths=1.6,alpha=al,zorder=6)
        # per-molecule symbol (changes) + criterion note + name
        fig.texts=[fig.texts[0]]                     # keep only the standing subtitle
        fig.text(0.5,0.93,f"${sym}$",ha="center",fontsize=30,weight="bold",color=NAVY,alpha=al)
        ax.text2D(0.5,0.11,note,transform=ax.transAxes,ha="center",fontsize=10.5,
                  style="italic",color="#666",alpha=al)
        ax.text2D(0.5,0.05,name,transform=ax.transAxes,ha="center",fontsize=13,
                  weight="bold",color="#333",alpha=al)
        return []

    out=f"{outdir}/group_lowsym.gif"
    FuncAnimation(fig,update,frames=len(seg),interval=1000/FPS,blit=False).save(
        out,writer=PlayOnceWriter(fps=FPS))
    plt.close(fig); print("wrote",out); return out

def render_high_symmetry(kind, outdir="."):
    """Show a polyhedron, then reveal the molecule that shares its symmetry sitting
    inside it. kind='td' (tetrahedron -> methane), 'oh' (octahedron -> SF6),
    'ih' (icosahedron -> C60). The polyhedron dims to a faint cage as the molecule
    appears, so the shared point group is explicit."""
    from scipy.spatial import ConvexHull
    from itertools import combinations
    Ph=(1+5**0.5)/2
    def norm(V,R): V=np.array(V,float); return V*(R/np.max(np.linalg.norm(V,axis=1)))
    def medges(V):
        d=[(i,j,np.sum((V[i]-V[j])**2)) for i,j in combinations(range(len(V)),2)]
        m=min(x[2] for x in d); return [(i,j) for i,j,x in d if abs(x-m)<1e-3*m+1e-9]
    T=np.array([[1,1,1],[-1,-1,1],[1,-1,-1],[-1,1,-1]],float)

    if kind=="td":
        sym="T_d"; polyname="tetrahedron"; molname="methane  (CH$_4$)"; col="#2f5f8a"; dark="#1d3b5a"
        Vp=norm(T,1.2); Ep=[(i,j) for i,j in combinations(range(4),2)]
        matoms=[("C",[0,0,0])]+[("H",v) for v in Vp]; mbonds=[(0,i+1) for i in range(4)]
        molV=Vp
    elif kind=="oh":
        sym="O_h"; polyname="octahedron"; molname="SF$_6$"; col="#7d5ba6"; dark="#4f3a6b"
        Vp=norm([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]],1.2); Ep=medges(Vp)
        matoms=[("S",[0,0,0])]+[("F",v) for v in Vp]; mbonds=[(0,i+1) for i in range(6)]
        molV=Vp
    else:  # ih
        sym="I_h"; polyname="icosahedron"; molname="C$_{60}$ (buckminsterfullerene)"; col="#b23a5a"; dark="#7a2540"
        ic=[]
        for s1 in (1,-1):
            for s2 in (1,-1): ic+=[[0,s1,s2*Ph],[s1,s2*Ph,0],[s2*Ph,0,s1]]
        Vp=norm(ic,1.5); Ep=medges(Vp)
        base=[(0,1,3*Ph),(1,2+Ph,2*Ph),(2,1+2*Ph,Ph)]; vs=set()
        for tri in base:
            for pm in ((0,1,2),(1,2,0),(2,0,1)):
                p=[tri[pm[0]],tri[pm[1]],tri[pm[2]]]
                for a in ([1,-1] if abs(p[0])>1e-9 else [1]):
                    for b in ([1,-1] if abs(p[1])>1e-9 else [1]):
                        for c in ([1,-1] if abs(p[2])>1e-9 else [1]):
                            vs.add((round(a*p[0],5),round(b*p[1],5),round(c*p[2],5)))
        Vc=norm(np.array(sorted(vs)),1.2)
        matoms=[("C",v) for v in Vc]; mbonds=medges(Vc); molV=Vc

    lim=1.9; NAVY="#1D3B5A"
    intro=8; holdA=13; reveal=15; holdB=28
    seg=[]
    for f in range(intro):  seg.append(("intro",(f+1)/intro,0.0))
    for f in range(holdA):  seg.append(("polyhold",1.0,0.0))
    for f in range(reveal): seg.append(("reveal",1.0-0.6*(f+1)/reveal,(f+1)/reveal))
    for f in range(holdB):  seg.append(("molhold",0.4,1.0))

    fig=plt.figure(figsize=FIG_SIZE, dpi=DPI); ax=fig.add_subplot(111,projection="3d")
    fig.text(0.5,0.93,f"${sym}$",ha="center",fontsize=30,weight="bold",color=NAVY)

    def update(k):
        phase,pa,ma=seg[k]; azim=-60+k*3.2
        ax.clear(); ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
        ax.set_box_aspect((1,1,1)); ax.set_axis_off(); ax.view_init(18,azim)
        # polyhedron: translucent faces only while it is the focus
        if pa>0.02 and ma<0.5:
            try:
                hull=ConvexHull(Vp); faces=[Vp[s] for s in hull.simplices]
                ax.add_collection3d(Poly3DCollection(faces,facecolor=col,alpha=0.14*pa,edgecolor="none"))
            except Exception: pass
        for i,j in Ep: ax.plot(*zip(Vp[i],Vp[j]),color=dark,lw=1.6,alpha=0.9*pa,zorder=2)
        if ma<0.1:                                  # polyhedron vertices, before molecule appears
            ax.scatter(Vp[:,0],Vp[:,1],Vp[:,2],color=dark,s=36,alpha=pa,depthshade=True,zorder=3)
        # molecule inside
        if ma>0.02:
            pos=np.array([p for _,p in matoms],float)
            for i,j in mbonds: ax.plot(*zip(pos[i],pos[j]),color=("#444" if kind=="ih" else BOND_C),
                                       lw=2.0 if kind=="ih" else 2.6,alpha=0.9*ma,zorder=4)
            for idx,(el,_) in enumerate(matoms):
                st=STYLE.get(el,STYLE["A"]); sz=st["s"]*(0.24 if kind=="ih" else 0.8)
                ax.scatter(*pos[idx],color=st["color"],s=sz,edgecolor=st["edge"],lw=0.5,
                           alpha=ma,depthshade=True,zorder=5)
        # crossfade label: polyhedron name -> molecule name
        if pa>0 and ma<0.98:
            ax.text2D(0.5,0.055,polyname,transform=ax.transAxes,ha="center",fontsize=13,
                      weight="bold",color="#555",alpha=max(0,1-ma))
        if ma>0.02:
            ax.text2D(0.5,0.055,molname,transform=ax.transAxes,ha="center",fontsize=13,
                      weight="bold",color="#333",alpha=ma)
        return []

    out=f"{outdir}/high_{kind}.gif"
    FuncAnimation(fig,update,frames=len(seg),interval=1000/FPS,blit=False).save(
        out,writer=PlayOnceWriter(fps=FPS))
    plt.close(fig); print("wrote",out); return out

def render_group_set(symbol, members, tag, outdir=".", linear=False, subtitle=None, mark_centre=False):
    """Cycle through a SET of molecules that share one point group, holding the
    Schoenflies symbol fixed so the message is 'different molecules, same group'.
    members = list of (display_name, library_key).  linear=True lays the molecules
    along x and gently rocks the view (a line spun the usual way vanishes edge-on);
    mark_centre draws the inversion centre (for D_inf_h)."""
    lib=_group_molecules()
    NAVY="#1D3B5A"; lim=2.4; elev=15 if linear else 18; per=26; fin=4; fout=4
    seg=[]
    for mi in range(len(members)):
        for f in range(per):
            if linear: azim=-90+22*np.sin(2*np.pi*f/per)     # gentle rock, stays broadside
            else:      azim=-60+f*6                           # full turntable spin
            al=(f+1)/fin if f<fin else ((per-f)/fout if f>=per-fout else 1.0)
            seg.append((mi,azim,min(1.0,al)))

    fig=plt.figure(figsize=FIG_SIZE, dpi=DPI); ax=fig.add_subplot(111,projection="3d")
    fig.text(0.5,0.93,f"${symbol}$",ha="center",fontsize=30,weight="bold",color=NAVY)
    fig.text(0.5,0.875,subtitle or "different molecules, same point group",
             ha="center",fontsize=10.5,color="#8a8f5c",style="italic")

    def update(k):
        mi,azim,al=seg[k]
        name,key=members[mi]; atoms,bonds=lib[key]
        pos=np.array([p for _,p in atoms],float)
        rmax=np.max(np.linalg.norm(pos,axis=1)) or 1.0
        pos=pos*(1.7/rmax)
        ax.clear(); ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
        ax.set_box_aspect((1,1,1)); ax.set_axis_off(); ax.view_init(elev,azim)
        for i,j in bonds: ax.plot(*zip(pos[i],pos[j]),color=BOND_C,lw=2.6,alpha=0.9*al,zorder=2)
        for idx,(el,_) in enumerate(atoms):
            st=STYLE.get(el,STYLE["A"])
            ax.scatter(*pos[idx],color=st["color"],s=st["s"]*0.8,edgecolor=st["edge"],
                       lw=0.7,alpha=al,depthshade=True,zorder=4)
        if mark_centre:                                       # inversion centre (D_inf_h)
            ax.scatter(0,0,0,s=70,facecolors="none",edgecolors=INV_C,linewidths=1.6,alpha=al,zorder=6)
            ax.text2D(0.5,0.115,"$i$",transform=ax.transAxes,ha="center",fontsize=11,
                      color=INV_C,alpha=al*0.9)
        ax.text2D(0.5,0.055,name,transform=ax.transAxes,ha="center",fontsize=13,
                  weight="bold",color="#333",alpha=al)
        return []

    out=f"{outdir}/group_{tag}.gif"
    FuncAnimation(fig,update,frames=len(seg),interval=1000/FPS,blit=False).save(
        out,writer=PlayOnceWriter(fps=FPS))
    plt.close(fig); print("wrote",out); return out

def render_pointgroup_showcase(outdir="."):
    """Generic 'what is a point group' backdrop: a sequence of rotating polyhedra,
    each labelled with its Schoenflies symbol and order h. Not a specific molecule,
    the point is that different symmetric shapes belong to different point groups."""
    from scipy.spatial import ConvexHull
    from itertools import combinations
    P=(1+5**0.5)/2
    def norm(V): V=np.array(V,float); return V*(1.1/np.max(np.linalg.norm(V,axis=1)))
    def mindist_edges(V):
        d2=[(i,j,np.sum((V[i]-V[j])**2)) for i,j in combinations(range(len(V)),2)]
        m=min(x[2] for x in d2); return [(i,j) for i,j,x in d2 if abs(x-m)<1e-6]

    a,b=1.0,0.62
    pyr=norm([[a,b,-0.35],[-a,b,-0.35],[-a,-b,-0.35],[a,-b,-0.35],[0,0,0.95]])
    pyrE=[(0,1),(1,2),(2,3),(3,0),(0,4),(1,4),(2,4),(3,4)]
    eq=[[np.cos(np.radians(t)),np.sin(np.radians(t)),0] for t in (90,210,330)]
    tbp=norm(eq+[[0,0,1.25],[0,0,-1.25]])
    tbpE=[(0,1),(1,2),(2,0),(0,3),(1,3),(2,3),(0,4),(1,4),(2,4)]
    tet=norm([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])
    top=[[np.cos(np.radians(t)),np.sin(np.radians(t)),0.7] for t in range(0,360,60)]
    bot=[[np.cos(np.radians(t)),np.sin(np.radians(t)),-0.7] for t in range(0,360,60)]
    hexp=norm(top+bot)
    hexE=[(i,(i+1)%6) for i in range(6)]+[(6+i,6+(i+1)%6) for i in range(6)]+[(i,6+i) for i in range(6)]
    octa=norm([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
    ic=[]
    for s1 in (1,-1):
        for s2 in (1,-1): ic+=[[0,s1,s2*P],[s1,s2*P,0],[s2*P,0,s1]]
    ico=norm(ic)

    shapes=[
      ("rectangular pyramid","C_{2v}",4,pyr,pyrE),
      ("trigonal bipyramid","D_{3h}",12, tbp, tbpE),
      ("tetrahedron",   "T_d",   24, tet,  mindist_edges(tet)),
      ("hexagonal prism","D_{6h}",24, hexp, hexE),
      ("octahedron",    "O_h",   48, octa, mindist_edges(octa)),
      ("icosahedron",   "I_h",  120, ico,  mindist_edges(ico)),
    ]
    fills=["#1f7a8c","#b5643c","#2f5f8a","#7a8b3a","#7d5ba6","#b23a5a"]
    darks=["#12525e","#7a3f24","#1d3b5a","#55622a","#4f3a6b","#7a2540"]
    lim=1.45; elev=18; per=24; fin=4; fout=4
    seg=[]
    for si in range(len(shapes)):
        for f in range(per):
            azim=-60+f*7
            al=(f+1)/fin if f<fin else ((per-f)/fout if f>=per-fout else 1.0)
            seg.append((si,azim,min(1.0,al)))

    fig=plt.figure(figsize=FIG_SIZE, dpi=DPI); ax=fig.add_subplot(111,projection="3d")
    NAVY="#1D3B5A"
    fig.text(0.5,0.90,"every shape has its own point group",ha="center",
             fontsize=10.5,color="#8a8f5c",style="italic")

    def update(k):
        si,azim,al=seg[k]
        name,sym,order,V,E=shapes[si]
        ax.clear(); ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
        ax.set_box_aspect((1,1,1)); ax.set_axis_off(); ax.view_init(elev,azim)
        try:
            hull=ConvexHull(V); faces=[V[s] for s in hull.simplices]
            ax.add_collection3d(Poly3DCollection(faces,facecolor=fills[si],alpha=0.16*al,edgecolor="none"))
        except Exception:
            pass
        for i,j in E: ax.plot(*zip(V[i],V[j]),color=darks[si],lw=1.7,alpha=0.92*al,solid_capstyle="round")
        ax.scatter(V[:,0],V[:,1],V[:,2],color=darks[si],s=34,alpha=al,depthshade=True)
        ax.text2D(0.5,0.135,f"${sym}$",transform=ax.transAxes,ha="center",
                  fontsize=30,weight="bold",color=darks[si],alpha=al)
        ax.text2D(0.5,0.065,f"{name}  ·  order $h={order}$",transform=ax.transAxes,
                  ha="center",fontsize=11,color="#555",alpha=al)
        return []

    out=f"{outdir}/pointgroup_showcase.gif"
    FuncAnimation(fig,update,frames=len(seg),interval=1000/FPS,blit=False).save(
        out,writer=PlayOnceWriter(fps=FPS))
    plt.close(fig); print("wrote",out); return out

def render_water_tour(outdir="."):
    """Water (C2v) symmetry-element inventory: principal axis C2 (bisects H-O-H),
    the molecular plane sigma_v (xz, contains all atoms), and the perpendicular
    plane sigma_v' (yz, swaps the two H). Molecule fixed; each element drawn in.
    A fixed xyz triad distinguishes the two planes (molecular = xz)."""
    P0,elems,bonds,_=MOLECULES["water"]["geom"](); lim=1.5; view=(14,-64)
    NAVY="#1D3B5A"; Zh=1.25; R=1.15
    groups=[("c2","Principal axis  $C_2$  (bisects H\u2013O\u2013H)"),
            ("sigmav","$\\sigma_v$  \u2014  molecular plane (all 3 atoms)"),
            ("sigmavp","$\\sigma_v'$  \u2014  $\\perp$ plane (swaps the two H)"),
            ("summary","$C_{2v}$:  $E$,  $C_2$,  $\\sigma_v$,  $\\sigma_v'$")]
    header="Water  $C_{2v}$ \u2014 every symmetry element"

    def princ(ax,af):
        ax.plot([0,0],[0,0],[-Zh,Zh],color=AXIS_C,lw=2.2,alpha=af,zorder=5)
    def plane_xz(ax,af,alpha=0.22):   # molecular plane, y=0
        _flat_plane(ax,[[-R,0,-R],[R,0,-R],[R,0,R],[-R,0,R]],af,a_fill=alpha,ew=1.3,z=1)
    def plane_yz(ax,af,alpha=0.22):   # perpendicular plane, x=0
        _flat_plane(ax,[[0,-R,-R],[0,R,-R],[0,R,R],[0,-R,R]],af,fill="#c98a3a",edge="#a8801f",a_fill=alpha,ew=1.3,z=1)

    fade=5; hold=11; seg=[]
    for key,cap in groups:
        for j in range(fade): seg.append((key,cap,(j+1)/fade))
        for _ in range(hold): seg.append((key,cap,1.0))

    fig=plt.figure(figsize=FIG_SIZE, dpi=DPI); ax=fig.add_subplot(111,projection="3d")
    axt=fig.add_axes([0.79,0.07,0.19,0.19],projection="3d")     # fixed xyz triad
    axt.set_axis_off(); axt.set_xlim(-1,1); axt.set_ylim(-1,1); axt.set_zlim(-1,1)
    axt.set_box_aspect((1,1,1)); axt.view_init(*view)
    for v,lab,col in (([1,0,0],"x","#c0392b"),([0,1,0],"y","#2e8b57"),([0,0,1],"z","#1f5fbf")):
        v=np.array(v,float); axt.quiver(0,0,0,*v,color=col,lw=2.0,arrow_length_ratio=0.22)
        axt.text(*(v*1.45),lab,color=col,fontsize=10,weight="bold",ha="center",va="center")
    fig.text(0.885,0.045,"fixed axes",fontsize=7,color="#999",ha="center",style="italic")

    def draw_molecule():
        for i,j in bonds: ax.plot(*zip(P0[i],P0[j]),color=BOND_C,lw=3.4,alpha=0.9,zorder=3)
        for idx,eg in enumerate(elems):
            st=STYLE[eg]; ax.scatter(*P0[idx],color=st["color"],s=st["s"],
                                     edgecolor=st["edge"],lw=0.8,depthshade=True,zorder=4)

    def update(k):
        key,cap,af=seg[k]
        ax.clear(); ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
        ax.set_box_aspect((1,1,1)); ax.set_axis_off(); ax.view_init(*view)
        if key in ("sigmav","sigmavp","summary"): princ(ax,0.30)
        draw_molecule()
        if key=="c2":
            princ(ax,af); ax.text(0,0,Zh+0.12,"$C_2$",color=AXIS_C,fontsize=12,weight="bold",ha="center")
        elif key=="sigmav":
            plane_xz(ax,af)
        elif key=="sigmavp":
            plane_yz(ax,af)
        elif key=="summary":
            plane_xz(ax,af,alpha=0.12); plane_yz(ax,af,alpha=0.12); princ(ax,af)
        ax.set_title(header,fontsize=12.5,color=NAVY,pad=6,weight="bold")
        ax.text2D(0.5,0.04,cap,transform=ax.transAxes,ha="center",fontsize=10.5,color="#333",clip_on=False)
        ax.text2D(0.02,0.98,"THEORY",transform=ax.transAxes,ha="left",va="top",fontsize=7.5,color="#8a8f5c",weight="bold")
        return []

    out=f"{outdir}/water_tour.gif"
    FuncAnimation(fig,update,frames=len(seg),interval=1000/FPS,blit=False).save(
        out,writer=PlayOnceWriter(fps=FPS))
    plt.close(fig); print("wrote",out); return out

def render_pf5_tour(part, outdir="."):
    """Trigonal-bipyramidal PF5 (D3h) element inventory, in two parts:
      part='axes'   -> principal C3 (through axial F) + three C2 (through equatorial
                       F), with a fixed xyz triad at the side.
      part='planes' -> sigma_h (equatorial), three sigma_v (vertical), and S3.
    Static identification: the molecule is fixed and each element is drawn in."""
    P0,elems,bonds,_=pf5(); lim=2.5; view=(18,-60)
    NAVY="#1D3B5A"; PURPLE="#7d3c98"; Eang=[0,120,240]     # equatorial F / C2 / sigma_v dirs
    Rax=1.85; Zh=2.05

    def dirn(deg): return np.array([np.cos(d(deg)),np.sin(d(deg)),0.0])
    def inplane_axis(ax,deg,col,af):
        u=dirn(deg)*Rax; ax.plot([-u[0],u[0]],[-u[1],u[1]],[0,0],color=col,lw=3.0,alpha=1.0*af,zorder=7,solid_capstyle="round")
    def vplane(ax,deg,col,af):
        u=dirn(deg); z=np.array([0,0,Zh]); R=2.05
        sq=np.array([u*R+z,u*R-z,-u*R-z,-u*R+z])
        ax.add_collection3d(Poly3DCollection([sq],facecolor=col,alpha=0.16*af,edgecolor="none"))
        loop=np.vstack([sq,sq[0]]); ax.plot(loop[:,0],loop[:,1],loop[:,2],color=col,lw=1.3,alpha=0.8*af)
    def princ(ax,col,af,dashed=False):
        ax.plot([0,0],[0,0],[-Zh,Zh],color=col,lw=2.2,alpha=af,
                ls=(0,(4,3)) if dashed else "-",zorder=5)

    if part=="axes":
        groups=[("principal","Principal axis  $C_3$  (through axial F)"),
                ("c2","Three $C_2$ axes (equatorial)"),
                ("both","$C_3$ and 3$\\,C_2$: the rotation axes")]
        header="PF$_5$  $D_{3h}$ \u2014 rotation axes"
    else:
        groups=[("sigmah","Horizontal mirror plane $\\sigma_h$ (equatorial)"),
                ("sigmav","Three $\\sigma_v$ planes \u2014 vertical, through the axial F"),
                ("s3","Improper axis  $S_3$  (along the principal axis)"),
                ("summary","$D_{3h}$ \u2014 the planes and improper axis")]
        header="PF$_5$  $D_{3h}$ \u2014 planes and improper axis"

    fade=5; hold=11; seg=[]
    for key,cap in groups:
        for j in range(fade): seg.append((key,cap,(j+1)/fade))
        for _ in range(hold): seg.append((key,cap,1.0))

    fig=plt.figure(figsize=FIG_SIZE, dpi=DPI); ax=fig.add_subplot(111,projection="3d")

    if part=="axes":                              # fixed xyz triad at the side (drawn once)
        axt=fig.add_axes([0.79,0.07,0.19,0.19],projection="3d")
        axt.set_axis_off(); axt.set_xlim(-1,1); axt.set_ylim(-1,1); axt.set_zlim(-1,1)
        axt.set_box_aspect((1,1,1)); axt.view_init(*view)
        for v,lab,col in (([1,0,0],"x","#c0392b"),([0,1,0],"y","#2e8b57"),([0,0,1],"z","#1f5fbf")):
            v=np.array(v,float); axt.quiver(0,0,0,*v,color=col,lw=2.0,arrow_length_ratio=0.22)
            axt.text(*(v*1.45),lab,color=col,fontsize=10,weight="bold",ha="center",va="center")
        fig.text(0.885,0.045,"fixed axes",fontsize=7,color="#999",ha="center",style="italic")

    def draw_molecule():
        for i,j in bonds: ax.plot(*zip(P0[i],P0[j]),color=BOND_C,lw=2.8,alpha=0.9,zorder=2)
        for idx,eg in enumerate(elems):
            st=STYLE[eg]; ax.scatter(*P0[idx],color=st["color"],s=st["s"]*0.9,
                                     edgecolor=st["edge"],lw=0.7,depthshade=True,zorder=4)

    def update(k):
        key,cap,af=seg[k]
        ax.clear(); ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
        ax.set_box_aspect((1,1,1)); ax.set_axis_off(); ax.view_init(*view)
        draw_molecule()
        if key in ("sigmav","s3","summary"): princ(ax,AXIS_C,0.30)   # faint reference

        if key=="principal":
            princ(ax,AXIS_C,af); ax.text(0,0,Zh+0.15,"$C_3$",color=AXIS_C,fontsize=12,weight="bold",ha="center")
        elif key=="c2":
            for g in Eang: inplane_axis(ax,g,AXIS_C,af)
        elif key=="both":
            princ(ax,AXIS_C,af)
            for g in Eang: inplane_axis(ax,g,AXIS_C,af)
        elif key=="sigmah":
            R=2.05; _flat_plane(ax,[[-R,-R,0],[R,-R,0],[R,R,0],[-R,R,0]],af,a_fill=0.22,ew=1.3,z=1)
        elif key=="sigmav":
            for g in Eang: vplane(ax,g,PLANE_C,af)
        elif key=="s3":
            princ(ax,PURPLE,af,dashed=True); ax.text(0,0,Zh+0.15,"$S_3$",color=PURPLE,fontsize=12,weight="bold",ha="center")
        elif key=="summary":
            R=2.05; _flat_plane(ax,[[-R,-R,0],[R,-R,0],[R,R,0],[-R,R,0]],af,a_fill=0.10,a_edge=0.4,z=1)
            for gg in Eang: vplane(ax,gg,PLANE_C,0.5*af)
            princ(ax,PURPLE,af,dashed=True)

        ax.set_title(header,fontsize=12.5,color=NAVY,pad=6,weight="bold")
        ax.text2D(0.5,0.04,cap,transform=ax.transAxes,ha="center",fontsize=10.5,color="#333",clip_on=False)
        ax.text2D(0.02,0.98,"THEORY",transform=ax.transAxes,ha="left",va="top",fontsize=7.5,color="#8a8f5c",weight="bold")
        return []

    out=f"{outdir}/pf5_tour_{part}.gif"
    FuncAnimation(fig,update,frames=len(seg),interval=1000/FPS,blit=False).save(
        out,writer=PlayOnceWriter(fps=FPS))
    plt.close(fig); print("wrote",out); return out

def render_benzene_tour(outdir="."):
    """A guided inventory of every symmetry element of benzene (D6h), revealed one
    group at a time to accompany the 'finding every element' slide: principal axis,
    C2' and C2'' axes, centre of inversion, sigma_h, sigma_v, sigma_d, then the
    improper axes S6/S3, closing on the complete set. The molecule stays fixed;
    each element is drawn in and named. Static identification, not an operation."""
    P0,elems,bonds,_=benzene(); lim=3.15; view=(34,-58)
    NAVY="#1D3B5A"; PURPLE="#7d3c98"
    Cang=[0,60,120]; Dang=[30,90,150]         # C2'/sigma_v (atoms), C2''/sigma_d (bonds)
    Rax=2.95; Zh=1.65                          # in-plane axis half-length, plane half-height

    def dirn(deg): return np.array([np.cos(d(deg)),np.sin(d(deg)),0.0])
    def vplane(ax,deg,col,af):                 # vertical plane containing z and a direction
        u=dirn(deg); z=np.array([0,0,Zh]); R=2.75
        sq=np.array([u*R+z,u*R-z,-u*R-z,-u*R+z])
        ax.add_collection3d(Poly3DCollection([sq],facecolor=col,alpha=0.16*af,edgecolor="none"))
        loop=np.vstack([sq,sq[0]])
        ax.plot(loop[:,0],loop[:,1],loop[:,2],color=col,lw=1.3,alpha=0.8*af)
    def inplane_axis(ax,deg,col,af):
        u=dirn(deg)*Rax
        ax.plot([-u[0],u[0]],[-u[1],u[1]],[0,0],color=col,lw=2.4,alpha=0.95*af,zorder=6)

    groups=[
      ("principal","Principal axis:  $C_6$  (and collinear $C_3$, $C_2$)"),
      ("c2p",      "Three $C_2'$ axes \u2014 through opposite atoms"),
      ("c2pp",     "Three $C_2''$ axes \u2014 through opposite bonds"),
      ("i",        "Centre of inversion,  $i$"),
      ("sigmah",   "Horizontal mirror plane,  $\\sigma_h$"),
      ("sigmav",   "Three $\\sigma_v$ planes \u2014 vertical, through atoms"),
      ("sigmad",   "Three $\\sigma_d$ planes \u2014 vertical, through bonds"),
      ("improper", "Improper axes:  $S_6$  and  $S_3$  (along the principal axis)"),
      ("summary",  "$D_{6h}$ \u2014 the complete set of elements"),
    ]
    fade=5; hold=11
    seg=[]
    for key,cap in groups:
        for j in range(fade): seg.append((key,cap,(j+1)/fade))
        for _ in range(hold): seg.append((key,cap,1.0))

    fig=plt.figure(figsize=FIG_SIZE, dpi=DPI); ax=fig.add_subplot(111,projection="3d")
    header="Benzene $D_{6h}$ \u2014 every symmetry element"

    def draw_molecule():
        for i,j in bonds: ax.plot(*zip(P0[i],P0[j]),color=BOND_C,lw=2.6,alpha=0.9,zorder=2)
        for idx,eg in enumerate(elems):
            st=STYLE[eg]; ax.scatter(*P0[idx],color=st["color"],s=st["s"]*0.9,
                                     edgecolor=st["edge"],lw=0.7,depthshade=True,zorder=4)

    def princ_axis(afaint):
        ax.plot([0,0],[0,0],[-Zh*1.15,Zh*1.15],color=AXIS_C,lw=2.2,alpha=afaint,zorder=5)

    def update(k):
        key,cap,af=seg[k]
        ax.clear(); ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
        ax.set_box_aspect((1,1,0.9)); ax.set_axis_off(); ax.view_init(*view)
        draw_molecule()

        # principal axis is a reference for several later elements; keep it faint there
        if key in ("sigmav","sigmad","improper","summary"): princ_axis(0.35)

        if key=="principal":
            princ_axis(af); ax.text(0,0,Zh*1.15+0.2,"$C_6$",color=AXIS_C,fontsize=12,
                                    weight="bold",ha="center",zorder=8)
        elif key=="c2p":
            for g in Cang: inplane_axis(ax,g,AXIS_C,af)
        elif key=="c2pp":
            for g in Dang: inplane_axis(ax,g,AXIS_C,af)
        elif key=="i":
            ax.scatter(*np.zeros(3),s=120,color=INV_C,zorder=9,alpha=af)
            ax.scatter(*np.zeros(3),s=320,facecolors="none",edgecolors=INV_C,lw=1.6,alpha=af,zorder=9)
        elif key=="sigmah":
            R=2.75; _flat_plane(ax,[[-R,-R,0],[R,-R,0],[R,R,0],[-R,R,0]],af,a_fill=0.22,ew=1.3,z=1)
        elif key=="sigmav":
            for g in Cang: vplane(ax,g,PLANE_C,af)
        elif key=="sigmad":
            for g in Dang: vplane(ax,g,"#c98a3a",af)
        elif key=="improper":
            ax.plot([0,0],[0,0],[-Zh*1.15,Zh*1.15],color=PURPLE,lw=2.4,ls=(0,(4,3)),alpha=af,zorder=6)
            ax.text(0,0,Zh*1.15+0.2,"$S_6,\\,S_3$",color=PURPLE,fontsize=12,weight="bold",ha="center",zorder=8)
        elif key=="summary":
            for g in Cang: inplane_axis(ax,g,AXIS_C,0.5*af)
            for g in Dang: inplane_axis(ax,g,AXIS_C,0.5*af)
            R=2.75; _flat_plane(ax,[[-R,-R,0],[R,-R,0],[R,R,0],[-R,R,0]],af,a_fill=0.10,a_edge=0.4,z=1)
            ax.scatter(*np.zeros(3),s=90,color=INV_C,alpha=af,zorder=9)

        fig.text  # keep linter calm
        ax.set_title(header,fontsize=12.5,color=NAVY,pad=6,weight="bold")
        ax.text2D(0.5,0.045,cap,transform=ax.transAxes,ha="center",fontsize=11.5,
                  color="#333",zorder=10)
        # running list of what has been covered
        covered=[g[1] for g in groups[:[gg[0] for gg in groups].index(key)]]
        ax.text2D(0.02,0.98,"THEORY",transform=ax.transAxes,ha="left",va="top",
                  fontsize=7.5,color="#8a8f5c",weight="bold")
        return []

    out=f"{outdir}/benzene_tour.gif"
    FuncAnimation(fig,update,frames=len(seg),interval=1000/FPS,blit=False).save(
        out,writer=PlayOnceWriter(fps=FPS))
    plt.close(fig); print("wrote",out); return out

def render_methane_cube(op, outdir="."):
    """Methane's tetrahedron inscribed in a cube: the 4 H sit on alternating
    cube corners, and the C2 / S4 axis runs through the centres of two opposite
    cube faces (the z-axis here). The cube stays fixed as a reference frame while
    the H atoms move between its corners, so the permutation is easy to follow.
      C2: rotate 180 (top pair swaps, bottom pair swaps).
      S4: rotate 90, then reflect in the cube's horizontal mid-plane (a 4-cycle)."""
    P0,elems,bonds,_=MOLECULES["methane"]["geom"]()
    C=P0[0]; H=P0[1:].copy(); h=float(np.abs(H).max()); lim=h*2.15
    view=(20,-58)
    corners=np.array([[sx*h,sy*h,sz*h] for sx in (1,-1) for sy in (1,-1) for sz in (1,-1)])
    edges=[(i,j) for i in range(8) for j in range(i+1,8)
           if int(np.sum(np.abs(corners[i]-corners[j])>1e-9))==1]
    HCOL=["#d11f4a","#1f5fbf","#2e8b57","#e8701a"]     # per-H colours, for tracking only

    if op=="C2":
        seg=[("start",0.0)]*max(6,FRAMES//10)
        n=max(22,FRAMES//2)
        seg+=[("rot",ease((j+1)/n)) for j in range(n)]
        seg+=[("done",1.0)]*max(16,FRAMES//3)
        title="Methane $C_2$: 180° through opposite cube faces"
    else:  # S4
        seg=[("start",0.0)]*max(6,FRAMES//10)
        n1=max(16,FRAMES//3); n2=max(14,FRAMES//3)
        seg+=[("rot",ease((j+1)/n1)) for j in range(n1)]
        seg+=[("rotpause",1.0)]*max(4,FRAMES//12)
        seg+=[("ref",ease((j+1)/n2)) for j in range(n2)]
        seg+=[("done",1.0)]*max(16,FRAMES//3)
        title="Methane $S_4$: rotate 90°, then reflect (mid-plane)"

    def positions(kind,frac):
        if op=="C2":
            a = d(180)*frac if kind in ("rot",) else (d(180) if kind in ("done",) else 0.0)
            return (Rmat([0,0,1],a)@H.T).T
        # S4
        if kind in ("start",): return H.copy()
        if kind=="rot":       return (Rmat([0,0,1],d(90)*frac)@H.T).T
        Hrot=(Rmat([0,0,1],d(90))@H.T).T
        if kind=="rotpause":  return Hrot
        # ref / done: reflect z from Hrot
        Q=Hrot.copy(); Q[:,2]=(1-frac)*Hrot[:,2]+frac*(-Hrot[:,2]); return Q

    fig=plt.figure(figsize=FIG_SIZE, dpi=DPI); ax=fig.add_subplot(111,projection="3d")
    fig.canvas.draw(); tfs=_fit_fontsize(fig,_wrap(title,52),12.5)

    def update(k):
        kind,frac=seg[k]
        ax.clear(); ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
        ax.set_box_aspect((1,1,1)); ax.set_axis_off(); ax.view_init(*view)

        # cube (fixed reference frame)
        for i,j in edges:
            ax.plot(*zip(corners[i],corners[j]),color="#b9b9b9",lw=1.2,alpha=0.8,zorder=1)
        # the C2 / S4 axis through opposite face centres
        ax.plot([0,0],[0,0],[-lim*0.96,lim*0.96],color=AXIS_C,lw=1.8,alpha=0.9,zorder=0)
        ax.scatter([0,0],[0,0],[h,-h],s=26,color=AXIS_C,zorder=2)   # face centres
        # S4 mid-plane, shown during the reflection step
        if op=="S4" and kind in ("ref","done"):
            _flat_plane(ax,[[-h,-h,0],[h,-h,0],[h,h,0],[-h,h,0]],1.0,a_fill=0.20,ew=1.2)

        P=positions(kind,frac)
        # bonds C-H (follow the moving H)
        for q in P:
            ax.plot([C[0],q[0]],[C[1],q[1]],[C[2],q[2]],color=BOND_C,lw=2.6,alpha=0.9,zorder=2)
        # central carbon
        st=STYLE["C"]; ax.scatter(*C,color=st["color"],s=st["s"],edgecolor=st["edge"],lw=0.8,zorder=4)
        # hydrogens, coloured + numbered for tracking
        for idx,q in enumerate(P):
            ax.scatter(*q,color=HCOL[idx],s=300,edgecolor="#222",lw=0.6,depthshade=True,zorder=5)
            ax.text(q[0],q[1],q[2],str(idx+1),color="white",fontsize=9,weight="bold",
                    ha="center",va="center",zorder=8)

        ax.set_title(title,fontsize=tfs,pad=2)
        if op=="C2":
            cap={"start":"the 4 H sit on alternating corners of the cube",
                 "rot":"rotating 180° about the vertical face-to-face axis",
                 "done":"top pair 1,4 swap; bottom pair 2,3 swap"}[kind]
        else:
            cap={"start":"the 4 H sit on alternating corners of the cube",
                 "rot":"Step 1: rotate 90° about the face-to-face axis",
                 "rotpause":"Step 1 done: rotated 90°",
                 "ref":"Step 2: reflect in the horizontal mid-plane",
                 "done":"net effect is the 4-cycle 1$\\rightarrow$3$\\rightarrow$4$\\rightarrow$2$\\rightarrow$1"}[kind]
        ax.text2D(0.5,0.03,cap,transform=ax.transAxes,ha="center",fontsize=10,
                  style="italic",color="#555")
        return []

    out=f"{outdir}/methane_cube_{op}.gif"
    FuncAnimation(fig,update,frames=len(seg),interval=1000/FPS,blit=False).save(
        out,writer=PlayOnceWriter(fps=FPS))
    plt.close(fig); print("wrote",out); return out

def render_sn_collapse(n, outdir="."):
    """Illustrate why S1 = sigma and S2 = i. An improper rotation S_n is: rotate
    by 360/n about the axis, THEN reflect in the plane perpendicular to it. For
    n=1 the rotation is a full 360 turn (does nothing), so S1 collapses to the
    reflection. For n=2 rotate-180-then-reflect sends every point through the
    centre, i.e. inversion. A tracked point does the two steps and lands exactly
    where the simple operation (sigma or i) would put it."""
    lim=1.95; view=(16,-64)
    A=np.array([1.15,0.60,0.95])                      # tracked atom: off-axis, off-plane
    C=np.zeros(3)
    if n==1:
        B=A*np.array([1,1,-1]); ang=360
        eq="$S_1 = \\sigma$"
        why="a full 360° turn changes nothing, so $S_1$ is just the reflection"
        direct="reflection $\\sigma$: straight down through the plane"
    else:
        B=-A.copy(); ang=180
        eq="$S_2 = i$"
        why="rotate 180° then reflect = inversion through the centre"
        direct="inversion $i$: straight through the centre"
    A_col, B_col = "#2f5fbf", "#3f7d3f"

    # segments: start hold, rotate, pause, reflect, hold on result (play once)
    nr=max(18,FRAMES//2); p1=max(4,FRAMES//12); nf=max(12,FRAMES//3); hold=max(16,FRAMES//3)
    seg =[("start",0.0)]*max(6,FRAMES//10)
    seg+=[("rot", (j+1)/nr) for j in range(nr)]
    seg+=[("rotpause",1.0)]*p1
    seg+=[("ref", (j+1)/nf) for j in range(nf)]
    seg+=[("done",1.0)]*hold

    Arot=(Rmat([0,0,1],d(ang))@A)                     # position after the rotation step

    fig=plt.figure(figsize=FIG_SIZE, dpi=DPI); ax=fig.add_subplot(111,projection="3d")
    title=f"$S_{n}$: rotate 360°/{n}, then reflect $\\perp$ to the axis"
    fig.canvas.draw(); tfs=_fit_fontsize(fig,title,13.0)

    def cross(ax,q,col,r=0.14):
        ax.plot([q[0]-r,q[0]+r],[q[1]-r,q[1]+r],[q[2],q[2]],color=col,lw=2.2,zorder=9)
        ax.plot([q[0]-r,q[0]+r],[q[1]+r,q[1]-r],[q[2],q[2]],color=col,lw=2.2,zorder=9)

    def update(k):
        kind,frac=seg[k]
        ax.clear(); ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
        ax.set_box_aspect((1,1,1)); ax.set_axis_off(); ax.view_init(*view)

        # principal axis (vertical) and the perpendicular mirror plane (horizontal)
        ax.plot([0,0],[0,0],[-lim*0.95,lim*0.95],color=AXIS_C,lw=1.8,alpha=0.9,zorder=0)
        ax.text(0,0,lim*0.95,"axis",color=AXIS_C,fontsize=9,ha="center")
        plane_hot = kind in ("ref","done")
        _flat_plane(ax,[[-1.5,-1.5,0],[1.5,-1.5,0],[1.5,1.5,0],[-1.5,1.5,0]],1.0,
                    a_fill=0.22 if plane_hot else 0.10,a_edge=0.85 if plane_hot else 0.5,ew=1.3)
        # inversion centre dot for S2
        if n==2:
            ax.scatter(*C,s=70,facecolors="none",edgecolors=INV_C,linewidths=1.8,zorder=2)

        # reference: start atom A (blue) and target B (green, faint until reached)
        ax.scatter(*A,s=300,color=A_col,edgecolor="#1b3a70",linewidth=0.8,depthshade=True,zorder=4)
        ax.text(A[0],A[1],A[2]+0.28,"start",color=A_col,fontsize=9,weight="bold",ha="center",zorder=8)
        b_seen = kind=="done"
        ax.scatter(*B,s=300,color=B_col if b_seen else "#cfd8cf",
                   edgecolor="#2b5630" if b_seen else "#b8c0b8",linewidth=0.8,depthshade=True,zorder=4)

        # tracked point through the two steps
        if kind in ("start",):
            p=A
        elif kind=="rot":
            p=(Rmat([0,0,1],d(ang)*frac)@A)
        elif kind=="rotpause":
            p=Arot
        else:                                          # ref / done
            p=(1-frac)*Arot + frac*(Arot*np.array([1,1,-1]))
        # faint arc of the rotation, and the drop line for the reflection
        if kind in ("rot","rotpause","ref","done"):
            th=np.linspace(0,d(ang),40); rr=np.hypot(A[0],A[1])
            ax.plot(rr*np.cos(th+np.arctan2(A[1],A[0])),rr*np.sin(th+np.arctan2(A[1],A[0])),
                    np.full_like(th,A[2]),color=TRACK["color"],lw=1.0,ls=(0,(3,3)),alpha=0.5,zorder=1)
        if kind in ("ref","done"):
            ax.plot([Arot[0],Arot[0]],[Arot[1],Arot[1]],[Arot[2],-Arot[2]],
                    color=TRACK["color"],lw=1.0,ls=(0,(2,3)),alpha=0.6,zorder=1)
        ax.scatter(*p,s=300,facecolors="none",edgecolors=TRACK["color"],linewidths=2.6,
                   depthshade=False,zorder=6)

        if kind=="done":                               # the simple operation lands identically
            if n==1:
                ax.plot([A[0],A[0]],[A[1],A[1]],[A[2],B[2]],color=B_col,lw=1.6,ls=(0,(4,3)),zorder=3)
            else:
                ax.plot([A[0],B[0]],[A[1],B[1]],[A[2],B[2]],color=B_col,lw=1.6,ls=(0,(4,3)),zorder=3)

        ax.set_title(title,fontsize=tfs,pad=2)
        # phase caption
        cap={"start":"start: one point, off the axis and off the plane",
             "rot":f"Step 1: rotate {ang}° about the axis",
             "rotpause":f"Step 1 done: rotated {ang}°",
             "ref":"Step 2: reflect in the plane $\\perp$ to the axis",
             "done":why}[kind]
        ax.text2D(0.5,0.05,cap,transform=ax.transAxes,ha="center",fontsize=10,
                  style="italic",color="#555")
        if kind=="done":
            ax.text2D(0.5,0.125,eq,transform=ax.transAxes,ha="center",fontsize=16,
                      weight="bold",color=B_col)
            ax.text2D(0.5,0.008,direct,transform=ax.transAxes,ha="center",fontsize=8.5,
                      color=B_col,style="italic")
        return []

    out=f"{outdir}/S{n}_collapse.gif"
    FuncAnimation(fig,update,frames=len(seg),interval=1000/FPS,blit=False).save(
        out,writer=PlayOnceWriter(fps=FPS))
    plt.close(fig); print("wrote",out); return out

def render_inversion_test(mol, view=None, outdir=".", suffix=""):
    """Show WHY a molecule has no centre of inversion. Invert every atom through
    the candidate centre (the central atom). Solid atoms are the real molecule;
    a moving open ring is each atom's inverted image, travelling straight through
    the centre to the antipodal point. Where the image lands on a matching atom
    it turns green; where it lands on empty space it turns red - and a red landing
    means the molecule is NOT reproduced, so there is no centre of inversion."""
    M=MOLECULES[mol]; P0,elems,bonds,_=M["geom"]()
    lim=M["lim"]; c=P0[0].copy()                    # candidate centre = central atom
    if view is None: view={"methane":(20,-60),"water":(12,-72)}.get(mol,(22,-60))
    numbers=M.get("numbers")

    # classify each atom's inverted image: does an atom of the same element sit there?
    match=[]
    for i in range(len(P0)):
        q=2*c-P0[i]
        match.append(any(elems[k]==elems[i] and np.linalg.norm(P0[k]-q)<0.05
                         for k in range(len(P0))))

    pre=max(4,FRAMES//12); hold=max(14,FRAMES//3); n=FRAMES-pre-hold
    sched=np.concatenate([np.zeros(pre), ease(np.linspace(0,1,n)), np.full(hold,1.0)])

    fig=plt.figure(figsize=FIG_SIZE, dpi=DPI); ax=fig.add_subplot(111,projection="3d")
    title_txt=f"{DISPLAY.get(mol, mol.capitalize())}:  is there a centre of inversion?"
    fig.canvas.draw(); title_fs=_fit_fontsize(fig, title_txt, 12.5)

    def update(fr):
        s=sched[fr]
        ax.clear(); ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
        ax.set_box_aspect((1,1,1)); ax.set_axis_off(); ax.view_init(elev=view[0],azim=view[1])

        # candidate centre marker
        ax.scatter(*c, s=90, facecolors="none", edgecolors=INV_C, linewidths=1.6, zorder=2)

        # real molecule (reference), slightly muted so the moving images stand out
        for i,j in bonds:
            ax.plot(*zip(P0[i],P0[j]), color=BOND_C, lw=3.0, alpha=0.5, zorder=2)
        for idx,e in enumerate(elems):
            st=STYLE[e]
            ax.scatter(*P0[idx], color=st["color"], s=st["s"], edgecolor=st["edge"],
                       linewidth=0.8, alpha=0.55, depthshade=True, zorder=3)

        # inverted images: open rings travelling through the centre to 2c - p
        for idx in range(len(P0)):
            if idx==0: continue                     # central atom maps to itself
            p=(1-s)*P0[idx]+s*(2*c-P0[idx])
            ax.plot([P0[idx][0],2*c[0]-P0[idx][0]],[P0[idx][1],2*c[1]-P0[idx][1]],
                    [P0[idx][2],2*c[2]-P0[idx][2]], color="#b9b9b9", lw=1.0,
                    ls=(0,(2,3)), alpha=0.55, zorder=1)
            col=INV_C if match[idx] else TRACK["color"]
            ax.scatter(*p, s=STYLE[elems[idx]]["s"]*1.05, facecolors="none",
                       edgecolors=col, linewidths=2.6, depthshade=False, zorder=6)
            if numbers:
                ax.text(p[0],p[1],p[2],str(numbers[idx-1] if idx-1<len(numbers) else idx),
                        color=col, fontsize=8.5, weight="bold", ha="center", va="center", zorder=7)
            # at the end, mark a mismatched landing with a red cross = "no atom here"
            if s>0.98 and not match[idx]:
                q=2*c-P0[idx]; r=0.16
                ax.plot([q[0]-r,q[0]+r],[q[1]-r,q[1]+r],[q[2],q[2]],color=TRACK["color"],lw=2.4,zorder=8)
                ax.plot([q[0]-r,q[0]+r],[q[1]+r,q[1]-r],[q[2],q[2]],color=TRACK["color"],lw=2.4,zorder=8)

        ax.set_title(title_txt, fontsize=title_fs, pad=2)
        # verdict, once the images have landed (drawn on the axes so it clears cleanly)
        if s>0.55:
            outer=elems[1] if len(set(elems[1:]))==1 else "outer"
            ax.text2D(0.5,0.085,
                      f"the {outer} atoms land on empty space:  $i$ is absent",
                      transform=ax.transAxes,ha="center",fontsize=11.5,weight="bold",
                      color=TRACK["color"])
        ax.text2D(0.5,0.02,"open ring = each atom inverted through the centre",
                  transform=ax.transAxes,ha="center",fontsize=8.5,style="italic",color="#666")
        return []

    out=f"{outdir}/{mol}_inversion_test{suffix}.gif"
    FuncAnimation(fig, update, frames=len(sched), interval=1000/FPS, blit=False).save(
        out, writer=PlayOnceWriter(fps=FPS))
    plt.close(fig); print("wrote", out); return out

def render_water_salc(kind, outdir="."):
    """H(1s) symmetry-adapted linear combinations (SALCs) of water, carried through
    the four C2v operations so the character row - and hence the irrep - is read off
    directly.  kind in {"a1","b1"}:
      a1 = sigma_s : the two H(1s) in phase      (+,+)  ->  chi = ( 1,  1,  1,  1)
      b1 = sigma_a : the two H(1s) out of phase  (+,-)  ->  chi = ( 1, -1,  1, -1)
    Axis convention matches the rest of this module: the molecule lies in the xz
    plane, C2 is z, sigma_v = xz (the molecular plane), sigma_v' = yz (swaps the two
    H).  With this convention the antisymmetric SALC is b1; it would be b2 if the
    molecule were placed in the yz plane instead (a common source of confusion)."""
    P0, elems, bonds, _ = MOLECULES["water"]["geom"]()
    lim = 1.6; view = (14, -64); NAVY = "#1D3B5A"
    Hidx  = [1, 2]                                   # H1 at +x, H2 at -x
    phase = {"a1": [+1, +1], "b1": [+1, -1]}[kind]   # phase carried on H1, H2
    slab, mull = {"a1": ("\\sigma_s", "a_1"), "b1": ("\\sigma_a", "b_1")}[kind]
    combo = {"a1": "the two H(1s) in phase  $(+,+)$",
             "b1": "the two H(1s) out of phase  $(+,-)$"}[kind]
    transf = {"a1": "totally symmetric, transforms as $z$",
              "b1": "transforms as $x$  (node in the $yz$ plane)"}[kind]
    rsph = 0.34                                      # schematic s-orbital radius

    # the four C2v operations, in character-table order, reusing the module op dicts
    ops = [("E",        OPS["water"]["E"],        "$E$"),
           ("C2",       OPS["water"]["C2"],       "$C_2$"),
           ("sigma_v",  OPS["water"]["sigma_v"],  "$\\sigma_v(xz)$"),
           ("sigma_vp", OPS["water"]["sigma_vp"], "$\\sigma_v'(yz)$")]

    def char_of(opdict):
        """+1 if the operation returns the SALC to itself, -1 to its negative."""
        C0 = P0[Hidx]; C1 = frame_coords(opdict, C0, 1.0)
        newphase = [0, 0]
        for a in range(2):                          # sphere a lands on nearest site b
            b = int(np.argmin(np.linalg.norm(C0 - C1[a], axis=1)))
            newphase[b] = phase[a]                  # the phase travels with the sphere
        if newphase == phase:                 return +1
        if newphase == [-p for p in phase]:   return -1
        return 0
    chars = [char_of(od) for _, od, _ in ops]

    # frame schedule: intro, then per-op (apply ramp + hold on result), then summary
    intro, ramp, hold, summ = 14, 16, 20, 30
    seg = [("intro", None, 0.0)] * intro
    for oi, _ in enumerate(ops):
        for j in range(ramp): seg.append(("op",   oi, ease((j + 1) / ramp)))
        for _   in range(hold): seg.append(("hold", oi, 1.0))
    seg += [("summary", None, 1.0)] * summ

    fig = plt.figure(figsize=FIG_SIZE, dpi=DPI); ax = fig.add_subplot(111, projection="3d")
    fig.text(0.5, 0.945, f"Water H(1s) SALC:   ${slab}$   (${mull}$)",
             ha="center", fontsize=15.5, weight="bold", color=NAVY)

    def draw_salc(C):
        cols = [PHASE_P if p > 0 else PHASE_N for p in phase]
        syms = ["+" if p > 0 else "\u2212" for p in phase]
        for c, col in zip(C, cols):                 # spheres drawn behind the nuclei
            ax.plot_surface(*_spheroid(c, rsph, rsph), color=col, alpha=0.55,
                            shade=True, linewidth=0, zorder=1)
        for c, sgn, col in zip(C, syms, cols):      # phase sign as a 2D overlay
            xp, yp, _ = proj3d.proj_transform(c[0], c[1], c[2], ax.get_proj())
            t = ax.text2D(xp, yp, sgn, transform=ax.transData, color="white",
                          fontsize=15, weight="bold", ha="center", va="center", zorder=25)
            t.set_path_effects([pe.withStroke(linewidth=2.4, foreground=col)])

    def char_row(ax, revealed):
        xs = [0.315, 0.445, 0.575, 0.705]
        for c, (_, _, lab) in zip(xs, ops):
            ax.text2D(c, 0.155, lab, transform=ax.transAxes, ha="center", va="center",
                      fontsize=10.5, weight="bold", color="#22313f")
        for n, (c, chi) in enumerate(zip(xs, chars)):
            if n < revealed:
                col = "#2e7d32" if chi > 0 else "#c0392b"
                ax.text2D(c, 0.100, f"${chi:+d}$", transform=ax.transAxes, ha="center",
                          va="center", fontsize=13, weight="bold", color=col)
            else:
                ax.text2D(c, 0.100, "\u00b7", transform=ax.transAxes, ha="center",
                          va="center", fontsize=13, color="#bbb")

    def update(k):
        tag, oi, ss = seg[k]
        ax.clear(); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(*view)
        opdict = None if oi is None else ops[oi][1]
        if tag in ("op", "hold") and opdict is not None:
            draw_element(ax, opdict["elem"][0], opdict["elem"][1], lim, 0.20)
        P = frame_coords(opdict, P0, ss) if opdict is not None else P0
        draw_salc(P[Hidx])                          # spheres ride on the H nuclei
        for i, j in bonds:
            ax.plot(*zip(P[i], P[j]), color=BOND_C, lw=3.0, alpha=0.95, zorder=2)
        for idx, e in enumerate(elems):
            st = STYLE[e]
            ax.scatter(*P[idx], color=st["color"], s=st["s"], edgecolor=st["edge"],
                       linewidth=0.8, depthshade=True, zorder=4)
        revealed = {"intro": 0, "op": (oi or 0), "hold": (oi or 0) + 1,
                    "summary": 4}[tag]
        char_row(ax, revealed)
        # caption
        if tag == "intro":
            cap = combo + "\nblue $=+$ phase,  red $=-$ phase"
            ax.text2D(0.5, 0.015, cap, transform=ax.transAxes, ha="center", va="bottom",
                      fontsize=9.5, style="italic", color="#555")
        elif tag == "op":
            ax.text2D(0.5, 0.02, f"applying {ops[oi][2]}\u2026", transform=ax.transAxes,
                      ha="center", va="bottom", fontsize=10.5, style="italic", color="#666")
        elif tag == "hold":
            chi = chars[oi]
            msg = "maps onto itself" if chi > 0 else "maps onto its negative"
            if ops[oi][0] == "sigma_v": msg = "molecular plane: the H are unmoved"
            col = "#2e7d32" if chi > 0 else "#c0392b"
            ax.text2D(0.5, 0.02, f"$\\chi = {chi:+d}$   \u2014  {msg}", transform=ax.transAxes,
                      ha="center", va="bottom", fontsize=11.5, weight="bold", color=col)
        else:  # summary: completed row, irrep to its right, note cleanly beneath
            ax.text2D(0.855, 0.100, f"$\\Rightarrow\\,{mull}$", transform=ax.transAxes,
                      ha="center", va="center", fontsize=15, weight="bold", color=NAVY)
            ax.text2D(0.5, 0.035, transf, transform=ax.transAxes, ha="center", va="bottom",
                      fontsize=9.5, style="italic", color="#666")
        return []

    out = f"{outdir}/water_salc_{kind}.gif"
    FuncAnimation(fig, update, frames=len(seg), interval=1000 / FPS, blit=False).save(
        out, writer=PlayOnceWriter(fps=FPS))
    plt.close(fig); print("wrote", out); return out


def render_allyl_pi(outdir="."):
    """The three pi MOs of the allyl radical (C3H5), stacked by energy, carried
    through the four C2v operations so each MO's character row - and its irrep - is
    built up on screen.  The Huckel solutions are psi1 (1, sqrt2, 1)/2, bonding, 0
    nodes; psi2 (1, 0, -1)/sqrt2, non-bonding, node ON the central carbon; psi3
    (1, -sqrt2, 1)/2, antibonding, 2 nodes.  They reduce as Gamma(pi) = A2 + 2 B2,
    so psi1/psi3 are b2 and psi2 is a2, and they alternate symmetric / antisymmetric
    under sigma_v' (the mirror through the central C).

    Convention (stated for the course): the molecular plane is treated as sigma_v,
    the p orbitals are perpendicular to it, and the C2 axis bisects the C-C-C angle.
    With the molecular plane taken as sigma_v(xz) the pi set is A2 + 2 B2; taking it
    as sigma_v'(yz) instead relabels the b2 pair as b1 - the physics is identical."""
    from matplotlib.patches import FancyArrow, Rectangle
    dd = np.deg2rad
    CC = 1.42; half = dd(124 / 2); s, c = np.sin(half), np.cos(half)
    C2c = np.array([0, 0.62, 0.]); C1 = C2c + CC * np.array([s, -c, 0]); C3 = C2c + CC * np.array([-s, -c, 0])
    CARB = np.vstack([C1, C2c, C3])
    cmax = 0.7071
    # (key, label, energy, coeffs on (C1,C2,C3)) top -> bottom = high -> low energy
    MOS = [("\\psi_3", "\\alpha-\\sqrt{2}\\,\\beta", np.array([0.5, -0.7071, 0.5])),
           ("\\psi_2", "\\alpha",                    np.array([0.7071, 0., -0.7071])),
           ("\\psi_1", "\\alpha+\\sqrt{2}\\,\\beta", np.array([0.5, 0.7071, 0.5]))]
    IRREP = ["B_2", "A_2", "B_2"]
    # name, motion op-dict, element cue, lambda (p_z sign), carbon permutation
    OPS_A = [("E",           None,                                     None,      +1, [0, 1, 2]),
             ("C_2",         dict(mode="rot",  axis=[0, 1, 0], angle=180), "axis",    -1, [2, 1, 0]),
             ("\\sigma_v",   dict(mode="refl", normal=[0, 0, 1]),          "planexy", -1, [0, 1, 2]),
             ("\\sigma_v'",  dict(mode="refl", normal=[1, 0, 0]),          "planeyz", +1, [2, 1, 0])]
    VIEW = (24, -62); NAVY = "#1D3B5A"

    def chi_of(coef, lam, perm):
        return +1 if np.allclose(lam * coef[perm], coef, atol=1e-6) else -1

    def lobe_mesh(cen, rxy, L, n=16):
        u = np.linspace(0, 2 * np.pi, n); v = np.linspace(0, np.pi, n)
        x = cen[0] + rxy * np.outer(np.cos(u), np.sin(v))
        y = cen[1] + rxy * np.outer(np.sin(u), np.sin(v))
        z = cen[2] + L * np.outer(np.ones_like(u), np.cos(v))
        return x, y, z

    def lobes_of(coef):
        out = []
        for i, ci in enumerate(coef):
            if abs(ci) < 1e-6: continue
            sc = 0.8 + 0.2 * (abs(ci) / cmax); L = 0.5 * sc; rxy = 0.29 * sc; gap = 0.14
            tc = PHASE_P if ci > 0 else PHASE_N; bc = PHASE_N if ci > 0 else PHASE_P
            out.append((CARB[i] + [0, 0, gap + L], tc, L, rxy))
            out.append((CARB[i] + [0, 0, -(gap + L)], bc, L, rxy))
        return out

    def draw_panel(ax, coef, opdict, ss, elem):
        ax.clear(); ax.set_axis_off()
        ax.set_xlim(-1.7, 1.7); ax.set_ylim(-1.2, 1.2); ax.set_zlim(-1.05, 1.05)
        ax.set_box_aspect((3.4, 2.4, 2.1)); ax.view_init(*VIEW)
        if elem == "axis":
            ax.plot([0, 0], [-1.15, 1.15], [0, 0], color=AXIS_C, lw=1.8, alpha=0.85, zorder=0)
        elif elem == "planexy":
            cor = [[-1.55, -1.1, 0], [1.55, -1.1, 0], [1.55, 1.1, 0], [-1.55, 1.1, 0]]
            ax.add_collection3d(Poly3DCollection([np.array(cor)], facecolors=[to_rgba(PLANE_C, 0.16)], edgecolors="none", zorder=0))
        elif elem == "planeyz":
            cor = [[0, -1.1, -1.0], [0, 1.1, -1.0], [0, 1.1, 1.0], [0, -1.1, 1.0]]
            ax.add_collection3d(Poly3DCollection([np.array(cor)], facecolors=[to_rgba("#c98a3a", 0.16)], edgecolors="none", zorder=0))
            lp = np.array(cor + [cor[0]]); ax.plot(lp[:, 0], lp[:, 1], lp[:, 2], color=PLANE_EDGE, lw=1.0, alpha=0.7)
        P = frame_coords(opdict, CARB, ss) if opdict is not None else CARB
        for i, j in [(0, 1), (1, 2)]:
            ax.plot(*zip(P[i], P[j]), color=BOND_C, lw=3.0, alpha=0.9, zorder=2)
        for cen, col, L, rxy in lobes_of(coef):
            X, Y, Z = lobe_mesh(cen, rxy, L)
            if opdict is not None:                       # rigid-transform the whole lobe
                pts = np.stack([X, Y, Z], -1).reshape(-1, 3)
                pts = frame_coords(opdict, pts, ss).reshape(X.shape + (3,))
                X, Y, Z = pts[..., 0], pts[..., 1], pts[..., 2]
            ax.plot_surface(X, Y, Z, color=col, alpha=0.85, linewidth=0, shade=True, zorder=1)
        for i in range(3):
            ax.scatter(*P[i], color=STYLE["C"]["color"], s=250, edgecolor=STYLE["C"]["edge"],
                       linewidth=0.8, depthshade=True, zorder=5)

    # ---- figure + persistent axes ----
    fig = plt.figure(figsize=(7.0, 6.7), dpi=DPI)
    fig.text(0.5, 0.955, "Allyl radical: the three $\\pi$ MOs and their $C_{2v}$ symmetry",
             ha="center", fontsize=13.5, weight="bold", color=NAVY)
    fig.text(0.055, 0.90, "$E$", ha="center", fontsize=10, color="#888")           # energy arrow label
    fig.add_artist(FancyArrow(0.055, 0.20, 0, 0.66, transform=fig.transFigure,
                   width=0.001, head_width=0.012, head_length=0.02, color="#aaa", length_includes_head=True))
    panbot = [0.635, 0.395, 0.155]; panels = []
    for r in range(3):
        panels.append(fig.add_axes([0.055, panbot[r], 0.49, 0.225], projection="3d"))
        fig.text(0.095, panbot[r] + 0.145, f"${MOS[r][0]}$", ha="left", va="center", fontsize=12.5, weight="bold", color=NAVY)
        fig.text(0.095, panbot[r] + 0.100, f"${MOS[r][1]}$", ha="left", va="center", fontsize=8.5, color="#777")
    tb = fig.add_axes([0.575, 0.15, 0.405, 0.72]); tb.set_axis_off(); tb.set_xlim(0, 1); tb.set_ylim(0, 1)
    cap = fig.text(0.5, 0.055, "", ha="center", va="center", fontsize=10.5, style="italic", color="#555")

    cx = [0.15, 0.32, 0.50, 0.69, 0.90]; colhead = ["E", "C_2", "\\sigma_v", "\\sigma_v'", "\\Gamma"]
    ry = [0.833, 0.500, 0.167]                          # aligned to the three panels

    def draw_table(revealed, show_irrep, hi_col=None):
        tb.clear(); tb.set_axis_off(); tb.set_xlim(0, 1); tb.set_ylim(0, 1)
        tb.text(0.5, 0.965, "$C_{2v}$  character analysis", ha="center", fontsize=11, weight="bold", color="#22313f")
        for x, cl in zip(cx, colhead):
            tb.text(x, 0.905, f"${cl}$", ha="center", fontsize=10.5, weight="bold", color="#22313f")
        if hi_col is not None:                          # highlight the operation being applied
            tb.add_patch(Rectangle((cx[hi_col] - 0.075, 0.06), 0.15, 0.88,
                         facecolor="#eef3fb", edgecolor="none", zorder=0))
        for r, (plab, elab, coef) in enumerate(MOS):
            tb.text(0.0, ry[r], f"${plab}$", ha="left", va="center", fontsize=11, weight="bold", color=NAVY)
            for cidx in range(4):
                _, od, _, lam, perm = OPS_A[cidx]
                if cidx < revealed:
                    chi = chi_of(coef, lam, perm); col = "#2e7d32" if chi > 0 else "#c0392b"
                    tb.text(cx[cidx], ry[r], f"${chi:+d}$", ha="center", va="center", fontsize=11.5, weight="bold", color=col)
                else:
                    tb.text(cx[cidx], ry[r], "\u00b7", ha="center", va="center", fontsize=11, color="#ccc")
            tb.text(cx[4], ry[r], f"${IRREP[r]}$" if show_irrep else "\u00b7", ha="center", va="center",
                    fontsize=12, weight="bold", color=NAVY if show_irrep else "#ccc")

    # ---- schedule ----
    intro, appl, hold, relax, gap, outro = 18, 12, 16, 10, 4, 36
    seg = [("intro", None, 0.0)] * intro
    for oi in (1, 2, 3):
        for j in range(appl):  seg.append(("apply", oi, ease((j + 1) / appl)))
        for _ in range(hold):  seg.append(("hold",  oi, 1.0))
        for j in range(relax): seg.append(("relax", oi, ease(1 - (j + 1) / relax)))
        for _ in range(gap):   seg.append(("gap",   oi, 0.0))
    seg += [("outro", None, 0.0)] * outro

    def update(k):
        tag, oi, ss = seg[k]
        opdict = OPS_A[oi][1] if oi is not None else None
        elem   = OPS_A[oi][2] if oi is not None else None
        show_elem = elem if tag in ("apply", "hold", "relax") else None
        for r in range(3):
            draw_panel(panels[r], MOS[r][2], opdict if tag in ("apply", "hold", "relax") else None, ss, show_elem)
        revealed = {"intro": 1, "apply": (oi or 1), "hold": (oi or 1) + 1,
                    "relax": (oi or 1) + 1, "gap": (oi or 1) + 1, "outro": 4}[tag]
        show_irrep = (tag == "outro")
        hi = oi if tag in ("apply", "hold", "relax") else None
        draw_table(revealed, show_irrep, hi_col=hi)
        if tag == "intro":
            cap.set_text("Three $\\pi$ MOs from three C $2p$ orbitals  \u2014  blue $=+$, red $=-$;  node count 0, 1, 2 going up")
        elif tag in ("apply", "relax"):
            cap.set_text(f"applying ${OPS_A[oi][0]}$ to every MO\u2026")
        elif tag == "hold":
            cap.set_text(f"${OPS_A[oi][0]}$:  read the character of each MO  ($+1$ onto itself,  $-1$ onto its negative)")
        else:
            cap.set_text("$\\Gamma_\\pi = A_2 + 2\\,B_2$   \u2014   symmetric / antisymmetric alternate under $\\sigma_v'$  (node through the central C)")
        return []

    out = f"{outdir}/allyl_pi_mos.gif"
    FuncAnimation(fig, update, frames=len(seg), interval=1000 / FPS, blit=False).save(
        out, writer=PlayOnceWriter(fps=FPS))
    plt.close(fig); print("wrote", out); return out


def render_c3_on_d(orbital, outdir="."):
    """C_3(z) acting on d_xz / d_yz - the second E pair (parallel to {p_x,p_y}). A
    120-degree rotation sends each member to a genuine MIXTURE of both, never to
    +/- itself:
        C_3 d_xz = -1/2 d_xz + (sqrt3/2) d_yz
        C_3 d_yz = -(sqrt3/2) d_xz - 1/2 d_yz
    The 2x2 block [[-1/2,-sqrt3/2],[sqrt3/2,-1/2]] has trace 2*cos120 = -1 = chi(C_3)
    for E, so the two orbitals are inseparable. Faint fixed ghosts mark the pure d_xz
    and d_yz sets, so the turned orbital reads as a blend of the two. Four lobes;
    blue = + phase, red = - phase. View (26,-45) keeps both planes legible."""
    assert orbital in ("dxz", "dyz")
    NAVY = "#1D3B5A"; BLUE = "#2f5fbf"; RED = "#c0392b"; lim = 1.55
    startlab = {"dxz": "d_{xz}", "dyz": "d_{yz}"}[orbital]
    combo = {"dxz": r"-\frac{1}{2}\,d_{xz} + \frac{\sqrt{3}}{2}\,d_{yz}",
             "dyz": r"-\frac{\sqrt{3}}{2}\,d_{xz} - \frac{1}{2}\,d_{yz}"}[orbital]

    def rot_to(a):
        a = np.asarray(a, float); a = a / np.linalg.norm(a); z = np.array([0, 0, 1.])
        if np.allclose(a, z): return np.eye(3)
        if np.allclose(a, -z): return np.diag([1, -1, -1.])
        v = np.cross(z, a); c = float(z @ a); s = np.linalg.norm(v)
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)

    def lobe(dirvec, dd=0.58, La=0.50, Lb=0.22, N=20):
        dirvec = np.asarray(dirvec, float)
        u = np.linspace(0, 2 * np.pi, N); v = np.linspace(0, np.pi, N); U, V = np.meshgrid(u, v)
        x = Lb * np.cos(U) * np.sin(V); y = Lb * np.sin(U) * np.sin(V); z = La * np.cos(V)
        return np.stack([x, y, z], -1) @ rot_to(dirvec).T + dd * dirvec

    c = float(np.cos(d(45)))
    dirs_xz = [(c, 0, c), (-c, 0, c), (-c, 0, -c), (c, 0, -c)]
    dirs_yz = [(0, c, c), (0, -c, c), (0, -c, -c), (0, c, -c)]
    cols = [BLUE, RED, BLUE, RED]
    dirs = dirs_xz if orbital == "dxz" else dirs_yz
    start = [(lobe(dv), cl) for dv, cl in zip(dirs, cols)]
    # faint fixed ghosts of BOTH pure basis members (the reference frame of the pair)
    ghosts = [(lobe(dv), cl) for dv, cl in zip(dirs_xz, cols)] + \
             [(lobe(dv), cl) for dv, cl in zip(dirs_yz, cols)]

    intro = 10; nop = 30; hold = 22
    seg = [("intro", 0.0)] * intro + [("op", (j + 1) / nop) for j in range(nop)] + [("done", 1.0)] * hold
    fig = plt.figure(figsize=FIG_SIZE, dpi=DPI); ax = fig.add_subplot(111, projection="3d")
    fig.text(0.5, 0.93, f"$C_3(z)$  on  ${startlab}$", ha="center", fontsize=17, weight="bold", color=NAVY)

    def draw_lobe(P, col, al=0.9, shade=True):
        ax.plot_surface(P[..., 0], P[..., 1], P[..., 2], color=col, alpha=al, rstride=1, cstride=1,
                        linewidth=0, antialiased=True, shade=shade)

    tt = np.linspace(d(6), d(120), 52); ra = 0.92; zc = 1.14
    arc = np.stack([ra * np.cos(tt), ra * np.sin(tt), np.full_like(tt, zc)], 1)

    def update(k):
        phase, t = seg[k]; te = ease(t)
        ax.clear(); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(32, 58)
        ax.plot([0, 0], [0, 0], [-1.35, 1.35], color=AXIS_C, lw=2.0, alpha=0.9, zorder=0)
        ax.text(0, 0, 1.52, "$C_3$", color=AXIS_C, fontsize=12, weight="bold", ha="center")
        ax.plot(arc[:, 0], arc[:, 1], arc[:, 2], color=AXIS_C, lw=2.6, alpha=0.9, zorder=2)
        tan = arc[-1] - arc[-4]; tan = tan / np.linalg.norm(tan) * 0.28
        ax.quiver(*(arc[-1] - tan), *tan, color=AXIS_C, lw=2.6, arrow_length_ratio=0.85, zorder=3)
        for P, cl in ghosts: draw_lobe(P, cl, 0.09, shade=False)      # pure d_xz / d_yz basis
        Rt = Rmat([0, 0, 1], d(120) * te)
        for P, cl in start: draw_lobe(P @ Rt.T, cl)                   # the turning orbital
        ax.scatter(0, 0, 0, color="#222", s=40, zorder=6)
        if phase == "done":
            ax.text2D(0.5, 0.135, "a mixture of $d_{xz}$ and $d_{yz}$", transform=ax.transAxes,
                      ha="center", fontsize=11, style="italic", color="#555")
            ax.text2D(0.5, 0.05, f"$C_3\\,{startlab} = {combo}$", transform=ax.transAxes,
                      ha="center", fontsize=13.5, weight="bold", color=NAVY)
        else:
            cap = {"intro": "blue = + lobe,  red = \u2212 lobe",
                   "op": "turning $120^\\circ$ about $z$\u2026"}[phase]
            ax.text2D(0.5, 0.05, cap, transform=ax.transAxes, ha="center", fontsize=10.5, style="italic", color="#666")
        return []

    tag = {"dxz": "c3_dxz", "dyz": "c3_dyz"}[orbital]
    out = f"{outdir}/mo_{tag}.gif"
    FuncAnimation(fig, update, frames=len(seg), interval=1000 / FPS, blit=False).save(out, writer=PlayOnceWriter(fps=FPS))
    plt.close(fig); print("wrote", out); return out


def render_c3_ammonia_p(orbital, outdir="."):
    """C_3 acting on an N p_x / p_y orbital, now carried on the real NH3 framework, so
    the abstract E-pair mixing gains a chemical meaning. The clip builds up in stages:
    the bare atomic orbital in space, then the C3v framework fades in around it, then
    C_3 turns molecule and orbital together. As with the free-space figure a 120-degree
    rotation sends each orbital to a MIXTURE of both,
        C_3 p_x = -1/2 p_x + (sqrt3/2) p_y ,   C_3 p_y = -(sqrt3/2) p_x - 1/2 p_y ,
    (trace -1 = chi(C_3) for E). The chemical consequence: because (p_x,p_y) are E, they
    match the e SALCs of the three H(1s) orbitals and form the doubly-degenerate N-H
    bonding pair. Numbered H atoms make the 1->2->3 cycle followable."""
    assert orbital in ("px", "py")
    NAVY = "#1D3B5A"; lim = 1.7
    startlab = {"px": "p_x", "py": "p_y"}[orbital]
    combo = {"px": r"-\frac{1}{2}\,p_x + \frac{\sqrt{3}}{2}\,p_y",
             "py": r"-\frac{\sqrt{3}}{2}\,p_x - \frac{1}{2}\,p_y"}[orbital]
    P0, elems, bonds, _ = MOLECULES["ammonia"]["geom"]()
    N = P0[0]
    op = dict(mode="rot", axis=[0, 0, 1], angle=120)
    spec = dict(kind=orbital, length=0.60, rxy=0.34, gap=0.26)
    ai = {"px": 0, "py": 1}[orbital]

    def ghost_lobe(centre, axis_i, col):
        off = np.zeros(3); off[axis_i] = spec["gap"] + spec["length"]
        r = [spec["rxy"]] * 3; r[axis_i] = spec["length"]
        for c in (centre + off, centre - off):
            ax.plot_surface(*_spheroid(c, spec["rxy"] if axis_i != 2 else spec["rxy"],
                                       spec["length"] if axis_i == 2 else spec["rxy"]),
                            color=col, alpha=0.08, shade=False, linewidth=0, zorder=0)

    # need proper axis-aligned spheroid for ghosts; small helper
    def ghost(centre, axis_i, col):
        off = np.zeros(3); off[axis_i] = spec["gap"] + spec["length"]
        for c in (centre + off, centre - off):
            rx = spec["length"] if axis_i == 0 else spec["rxy"]
            ry = spec["length"] if axis_i == 1 else spec["rxy"]
            rz = spec["length"] if axis_i == 2 else spec["rxy"]
            u = np.linspace(0, 2 * np.pi, 16); v = np.linspace(0, np.pi, 16)
            X = c[0] + rx * np.outer(np.cos(u), np.sin(v))
            Y = c[1] + ry * np.outer(np.sin(u), np.sin(v))
            Z = c[2] + rz * np.outer(np.ones_like(u), np.cos(v))
            ax.plot_surface(X, Y, Z, color=col, alpha=0.08, shade=False, linewidth=0, zorder=0)

    flo, bui, set_, nop, hold = 12, 14, 10, 30, 26
    seg = [("float", 0.0, 0.0)] * flo
    seg += [("build", 0.0, ease((j + 1) / bui)) for j in range(bui)]
    seg += [("settle", 0.0, 1.0)] * set_
    seg += [("op", ease((j + 1) / nop), 1.0) for j in range(nop)]
    seg += [("done", 1.0, 1.0)] * hold

    fig = plt.figure(figsize=FIG_SIZE, dpi=DPI); ax = fig.add_subplot(111, projection="3d")
    fig.text(0.5, 0.93, f"$C_3$  on  N $ {startlab}$  in NH$_3$", ha="center", fontsize=16, weight="bold", color=NAVY)
    tt = np.linspace(d(6), d(120), 46); ra = 1.35; zc = N[2]
    arc = np.stack([ra * np.cos(tt), ra * np.sin(tt), np.full_like(tt, zc)], 1)

    def update(k):
        phase, ss, fw = seg[k]
        ax.clear(); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(90, -90)
        # top-down: the C3 axis is perpendicular to the page; show its sense as an arc
        if phase in ("settle", "op", "done"):
            ax.plot(arc[:, 0], arc[:, 1], arc[:, 2], color=AXIS_C, lw=2.4, alpha=0.9, zorder=2)
            tan = arc[-1] - arc[-4]; tan = tan / np.linalg.norm(tan) * 0.26
            ax.quiver(*(arc[-1] - tan), *tan, color=AXIS_C, lw=2.4, arrow_length_ratio=0.9, zorder=3)
            ax.text(1.55 * np.cos(d(60)), 1.55 * np.sin(d(60)), N[2], "$C_3$",
                    color=AXIS_C, fontsize=13, weight="bold", ha="center", va="center")
            # faint x / y reference axes so the turned orbital reads as a mix of the two
            for vec, lab in (((1.0, 0, 0), "x"), ((0, 1.0, 0), "y")):
                e = np.array(vec) * 1.25
                ax.plot([-e[0], e[0]], [-e[1], e[1]], [N[2], N[2]],
                        color="#b3b3b3", lw=1.1, ls=(0, (4, 3)), alpha=0.9, zorder=0)
                ax.text(e[0] * 1.14, e[1] * 1.14, N[2], f"${lab}$",
                        color="#8a8a8a", fontsize=11, ha="center", va="center", zorder=0)
        # framework (fades in)
        if fw > 0.01:
            P = frame_coords(op, P0, ss)
            for i, j in bonds:
                ax.plot(*zip(P[i], P[j]), color=BOND_C, lw=3.0, alpha=0.95 * fw, zorder=2)
            for idx, e in enumerate(elems):
                st = STYLE[e]
                ax.scatter(*P[idx], color=st["color"], s=st["s"], edgecolor=st["edge"],
                           linewidth=0.8, depthshade=True, alpha=min(1.0, fw), zorder=4)
            if fw > 0.5:
                for n, idx in enumerate([1, 2, 3], 1):
                    xp, yp, _ = proj3d.proj_transform(P[idx][0], P[idx][1], P[idx][2], ax.get_proj())
                    tnum = ax.text2D(xp, yp, str(n), transform=ax.transData, color="#22313f",
                              fontsize=9.5, weight="bold", ha="center", va="center", zorder=20)
                    tnum.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white")])
        # the p orbital on N (N is on the axis, so it stays put; lobes rotate with C_3)
        draw_orbital(ax, op, ss, N, spec)
        ax.scatter(*N, color="#222", s=18, zorder=6)
        # captions
        if phase == "float":
            cap = f"an N ${startlab}$ orbital  (blue $=+$, red $=-$)"
        elif phase == "build":
            cap = "place it on the nitrogen of NH$_3$\u2026"
        elif phase == "settle":
            cap = "NH$_3$ is $C_{3v}$: the $C_3$ axis runs through N"
        elif phase == "op":
            cap = "turning $120^\\circ$ about the $C_3$ axis\u2026"
        else:
            cap = None
        if cap is not None:
            ax.text2D(0.5, 0.045, cap, transform=ax.transAxes, ha="center", fontsize=10.5, style="italic", color="#666")
        else:
            ax.text2D(0.5, 0.150, f"$C_3\\,{startlab} = {combo}$", transform=ax.transAxes,
                      ha="center", fontsize=13, weight="bold", color=NAVY)
            ax.text2D(0.5, 0.020, "an $E$ pair: $p_x,p_y$ overlap the $e$ SALCs of the 3 H(1s)\n"
                                  "to give the degenerate N\u2013H bonding MOs",
                      transform=ax.transAxes, ha="center", fontsize=9, style="italic", color="#555")
        return []

    tag = {"px": "c3_nh3_px", "py": "c3_nh3_py"}[orbital]
    out = f"{outdir}/mo_{tag}.gif"
    FuncAnimation(fig, update, frames=len(seg), interval=1000 / FPS, blit=False).save(out, writer=PlayOnceWriter(fps=FPS))
    plt.close(fig); print("wrote", out); return out


def render_nh3_projection(key, outdir="."):
    """Projection-operator method for the three H(1s) SALCs of NH3 (C3v), viewed down
    the C3 axis. Label convention: s1 = bottom-right H, then clockwise s2 (bottom-left),
    s3 (top). Starting from one orbital, every group operation R is applied; the image
    R s_gen is weighted by the character chi(R) of the target irrep and deposited on the
    H it lands on. Summing the six terms gives the SALC:
        a1 :  s1 + s2 + s3                 (project s1, all chi = +1)
        e  :  2 s1 - s2 - s3               (project s1;  chi(E)=2, chi(C3)=-1, chi(sv)=0)
        e' :  2 s2 - s1 - s3               (project s2; the partner of the e pair)
    The three sigma_v terms carry chi = 0 for e, so they contribute nothing - the reason
    the e SALC comes only from E and the two rotations."""
    NAVY = "#1D3B5A"; BLUE = "#2f5fbf"; RED = "#c0392b"; GREY = "#c9ccd1"; lim = 1.7
    P0, elems, bonds, _ = MOLECULES["ammonia"]["geom"]()
    N = P0[0]
    idx = {"s1": 3, "s2": 2, "s3": 1}                 # label -> geom H index
    order = ["s1", "s2", "s3"]
    cfg = {"a1": dict(gen="s1", irr="a_1", chars=[1, 1, 1, 1, 1, 1]),
           "e1": dict(gen="s1", irr="e",   chars=[2, -1, -1, 0, 0, 0]),
           "e2": dict(gen="s2", irr="e",   chars=[2, -1, -1, 0, 0, 0])}[key]
    gen, chars, irr = cfg["gen"], cfg["chars"], cfg["irr"]
    targets = {"s1": ["s1", "s3", "s2", "s2", "s3", "s1"],
               "s2": ["s2", "s1", "s3", "s1", "s2", "s3"]}[gen]
    opnm = ["E", "C_3", "C_3^{2}", "\\sigma_a", "\\sigma_b", "\\sigma_c"]
    mir_az = {"\\sigma_a": 90, "\\sigma_b": 210, "\\sigma_c": 330}
    kind = {0: "id", 1: "rot", 2: "rot", 3: "mir", 4: "mir", 5: "mir"}
    rot_deg = {1: 120, 2: 240}
    # cumulative coefficients after each op
    cum = []; cur = {"s1": 0, "s2": 0, "s3": 0}
    for i in range(6):
        cur = dict(cur); cur[targets[i]] += chars[i]; cum.append(cur)
    final = cum[-1]
    Hpos = {lab: P0[idx[lab]] for lab in order}

    def ssphere(p, r, n=18):
        u = np.linspace(0, 2 * np.pi, n); v = np.linspace(0, np.pi, n)
        return (p[0] + r * np.outer(np.cos(u), np.sin(v)),
                p[1] + r * np.outer(np.sin(u), np.sin(v)),
                p[2] + r * np.outer(np.ones_like(u), np.cos(v)))

    def lab2d(p, s, dx, dy, color, fs, weight="bold", halo=None):
        xp, yp, _ = proj3d.proj_transform(p[0], p[1], p[2], ax.get_proj())
        t = ax.text2D(xp + dx, yp + dy, s, transform=ax.transData, color=color, fontsize=fs,
                      weight=weight, ha="center", va="center", zorder=25)
        if halo: t.set_path_effects([pe.withStroke(linewidth=2.4, foreground=halo)])

    flo, per, res = 12, 15, 40
    seg = [("intro", None)] * flo
    for i in range(6):
        seg += [("op", i)] * per
    seg += [("result", None)] * res

    fig = plt.figure(figsize=(5.2, 5.2), dpi=DPI); ax = fig.add_subplot(111, projection="3d")
    fig.text(0.5, 0.945, f"Projection operator  $\\rightarrow$  ${irr}$ SALC", ha="center",
             fontsize=15, weight="bold", color=NAVY)
    fig.text(0.5, 0.895, f"generating orbital: ${gen}$", ha="center", fontsize=10, color="#666")

    def coeffs_at(k):
        phase, i = seg[k]
        if phase == "intro": return {"s1": 0, "s2": 0, "s3": 0}, None
        if phase == "op": return cum[i], i
        return final, None

    def update(k):
        phase, i = seg[k]
        co, opi = coeffs_at(k)
        ax.clear(); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(90, -90)
        # framework
        for a, b in bonds:
            ax.plot(*zip(P0[a], P0[b]), color=BOND_C, lw=2.6, alpha=0.9, zorder=1)
        ax.scatter(*N, color=STYLE["N"]["color"], s=STYLE["N"]["s"], edgecolor=STYLE["N"]["edge"],
                   linewidth=0.8, depthshade=True, zorder=4)
        # operation geometry cue
        if phase == "op":
            if kind[opi] == "rot":
                tt = np.linspace(d(8), d(rot_deg[opi] - 8), 60); ra = 1.42
                arc = np.stack([ra * np.cos(tt), ra * np.sin(tt), np.full_like(tt, N[2])], 1)
                ax.plot(arc[:, 0], arc[:, 1], arc[:, 2], color=AXIS_C, lw=2.2, alpha=0.85, zorder=2)
                tan = arc[-1] - arc[-4]; tan = tan / np.linalg.norm(tan) * 0.24
                ax.quiver(*(arc[-1] - tan), *tan, color=AXIS_C, lw=2.2, arrow_length_ratio=0.9, zorder=3)
            elif kind[opi] == "mir":
                a = d(mir_az[opnm[opi]]); e = np.array([np.cos(a), np.sin(a), 0]) * 1.45
                ax.plot([-e[0], e[0]], [-e[1], e[1]], [N[2], N[2]], color="#c98a3a",
                        lw=1.6, ls=(0, (5, 3)), alpha=0.9, zorder=2)
        # s orbitals (spheres sized by |coeff|, coloured by sign)
        for lab in order:
            v = co[lab]; r = 0.11 + 0.17 * abs(v)
            col = GREY if v == 0 else (BLUE if v > 0 else RED)
            al = 0.35 if v == 0 else 0.9
            ax.plot_surface(*ssphere(Hpos[lab], r), color=col, alpha=al, linewidth=0, shade=True, zorder=5)
        # generating-orbital ring marker
        gp = Hpos[gen]
        ring = np.stack([gp[0] + 0.34 * np.cos(np.linspace(0, 2 * np.pi, 40)),
                         gp[1] + 0.34 * np.sin(np.linspace(0, 2 * np.pi, 40)),
                         np.full(40, gp[2])], 1)
        ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], color="#444", lw=1.0, ls=(0, (2, 2)), alpha=0.7, zorder=6)
        # labels + coefficient values
        for lab in order:
            p = Hpos[lab]; rad = np.linalg.norm(p[:2]); dxy = p[:2] / rad
            outer = np.array([dxy[0] * (rad + 0.62), dxy[1] * (rad + 0.62), p[2]])
            lab2d(outer, f"${lab}$", 0, 0, "#22313f", 11)
            v = co[lab]
            if not (phase == "intro"):
                txt = f"${v:+d}$" if v != 0 else "$0$"
                col = "#2e7d32" if v > 0 else ("#c0392b" if v < 0 else "#999")
                lab2d(p, txt, 0, 0, "white" if v != 0 else "#666", 12, halo=(col if v != 0 else None))
        # captions
        if phase == "intro":
            cap = f"start from ${gen}$;  apply every operation of $C_{{3v}}$ and sum $\\chi(R)\\,R\\,{gen}$"
        elif phase == "op":
            tgt = targets[opi]; ch = chars[opi]
            arrow = f"${opnm[opi]}\\,{gen} = {tgt}$"
            chtxt = f"$\\chi={ch:+d}$"
            note = "  (adds nothing)" if ch == 0 else ""
            cap = f"{arrow}     {chtxt}{note}"
        else:
            cap = None
        if cap is not None:
            ax.text2D(0.5, 0.03, cap, transform=ax.transAxes, ha="center", fontsize=10.5,
                      style="italic", color="#555")
        else:
            expr = {"a1": "s_1+s_2+s_3", "e1": "2s_1-s_2-s_3", "e2": "2s_2-s_1-s_3"}[key]
            norm = {"a1": "\\frac{1}{\\sqrt{3}}", "e1": "\\frac{1}{\\sqrt{6}}", "e2": "\\frac{1}{\\sqrt{6}}"}[key]
            ax.text2D(0.5, 0.075, f"${irr}$ SALC $= {norm}\\,({expr})$", transform=ax.transAxes,
                      ha="center", fontsize=13, weight="bold", color=NAVY)
            if key == "e2":
                ax.text2D(0.5, 0.02, "with $2s_1-s_2-s_3$, spans the degenerate $e$ pair",
                          transform=ax.transAxes, ha="center", fontsize=9, style="italic", color="#666")
        return []

    tag = {"a1": "nh3_proj_a1", "e1": "nh3_proj_e1", "e2": "nh3_proj_e2"}[key]
    out = f"{outdir}/{tag}.gif"
    FuncAnimation(fig, update, frames=len(seg), interval=1000 / FPS, blit=False).save(out, writer=PlayOnceWriter(fps=FPS))
    plt.close(fig); print("wrote", out); return out


def render_bf3_tour(outdir="."):
    """BF3 (D3h, trigonal planar) symmetry-element inventory in one clip. B at the
    centre, three F at 0/120/240 deg in the xy-plane; z is the principal axis. Each
    element is drawn onto the fixed molecule in turn: the principal C3 (perpendicular
    to the plane), the three C2 axes (in-plane, along the B-F bonds), sigma_h (the
    molecular plane, which contains all four atoms), the three sigma_v (vertical, each
    holding C3 and one B-F bond), and S3 (coincident with C3). A fixed xyz triad keeps
    the orientation clear. Element set: E, 2C3, 3C2, sigma_h, 2S3, 3sigma_v."""
    r = 1.31; Fang = [0, 120, 240]
    P0 = np.array([[0, 0, 0.]] + [[r * np.cos(d(a)), r * np.sin(d(a)), 0.] for a in Fang])
    elems = ["B", "F", "F", "F"]; bonds = [(0, 1), (0, 2), (0, 3)]
    lim = 1.9; view = (24, -58); NAVY = "#1D3B5A"; PURPLE = "#7d3c98"
    Zh = 1.6; Rax = 1.7; Rpl = 1.85

    def dirn(deg): return np.array([np.cos(d(deg)), np.sin(d(deg)), 0.0])
    def inplane_axis(ax, deg, col, af):
        u = dirn(deg) * Rax
        ax.plot([-u[0], u[0]], [-u[1], u[1]], [0, 0], color=col, lw=3.0, alpha=af,
                zorder=7, solid_capstyle="round")
    def vplane(ax, deg, col, af):
        u = dirn(deg); z = np.array([0, 0, Zh]); R = Rpl
        sq = np.array([u * R + z, u * R - z, -u * R - z, -u * R + z])
        ax.add_collection3d(Poly3DCollection([sq], facecolor=col, alpha=0.16 * af, edgecolor="none"))
        loop = np.vstack([sq, sq[0]]); ax.plot(loop[:, 0], loop[:, 1], loop[:, 2], color=col, lw=1.3, alpha=0.8 * af)
    def princ(ax, col, af, dashed=False):
        ax.plot([0, 0], [0, 0], [-Zh, Zh], color=col, lw=2.2, alpha=af,
                ls=(0, (4, 3)) if dashed else "-", zorder=5)

    groups = [("c3",     "Principal axis  $C_3$  ($\\perp$ molecular plane)"),
              ("c2",     "Three  $C_2$  axes  (through each F)"),
              ("sigmah", "$\\sigma_h$  \u2014  molecular plane (all 4 atoms)"),
              ("sigmav", "Three  $\\sigma_v$  planes  (through each B\u2013F)"),
              ("s3",     "$S_3$  \u2014  rotate 120\u00b0, then reflect in $\\sigma_h$"),
              ("summary","$D_{3h}$:   $E$,  $2C_3$,  $3C_2$,  $\\sigma_h$,  $2S_3$,  $3\\sigma_v$")]
    header = "BF$_3$   $D_{3h}$ \u2014 every symmetry element"

    fade = 5; hold = 12; seg = []
    for key, cap in groups:
        for j in range(fade): seg.append((key, cap, (j + 1) / fade))
        for _ in range(hold): seg.append((key, cap, 1.0))

    fig = plt.figure(figsize=FIG_SIZE, dpi=DPI); ax = fig.add_subplot(111, projection="3d")
    axt = fig.add_axes([0.79, 0.07, 0.19, 0.19], projection="3d")     # fixed xyz triad
    axt.set_axis_off(); axt.set_xlim(-1, 1); axt.set_ylim(-1, 1); axt.set_zlim(-1, 1)
    axt.set_box_aspect((1, 1, 1)); axt.view_init(*view)
    for v, lab, col in (([1, 0, 0], "x", "#c0392b"), ([0, 1, 0], "y", "#2e8b57"), ([0, 0, 1], "z", "#1f5fbf")):
        v = np.array(v, float); axt.quiver(0, 0, 0, *v, color=col, lw=2.0, arrow_length_ratio=0.22)
        axt.text(*(v * 1.45), lab, color=col, fontsize=10, weight="bold", ha="center", va="center")
    fig.text(0.885, 0.045, "fixed axes", fontsize=7, color="#999", ha="center", style="italic")

    def draw_molecule():
        for i, j in bonds: ax.plot(*zip(P0[i], P0[j]), color=BOND_C, lw=3.2, alpha=0.9, zorder=3)
        for idx, eg in enumerate(elems):
            st = STYLE[eg]; ax.scatter(*P0[idx], color=st["color"], s=st["s"],
                                       edgecolor=st["edge"], lw=0.8, depthshade=True, zorder=4)

    def update(k):
        key, cap, af = seg[k]
        ax.clear(); ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
        ax.set_box_aspect((1, 1, 1)); ax.set_axis_off(); ax.view_init(*view)
        if key in ("sigmav", "s3", "summary"): princ(ax, AXIS_C, 0.30)   # faint reference axis
        draw_molecule()
        if key == "c3":
            princ(ax, AXIS_C, af); ax.text(0, 0, Zh + 0.15, "$C_3$", color=AXIS_C, fontsize=12, weight="bold", ha="center")
        elif key == "c2":
            for g in Fang: inplane_axis(ax, g, AXIS_C, af)
            ax.text(Rax + 0.12, 0, 0.12, "$C_2$", color=AXIS_C, fontsize=11, weight="bold", ha="center")
        elif key == "sigmah":
            _flat_plane(ax, [[-Rpl, -Rpl, 0], [Rpl, -Rpl, 0], [Rpl, Rpl, 0], [-Rpl, Rpl, 0]], af, a_fill=0.22, ew=1.3, z=1)
        elif key == "sigmav":
            for g in Fang: vplane(ax, g, PLANE_C, af)
        elif key == "s3":
            princ(ax, PURPLE, af, dashed=True); ax.text(0, 0, Zh + 0.15, "$S_3$", color=PURPLE, fontsize=12, weight="bold", ha="center")
        elif key == "summary":
            _flat_plane(ax, [[-Rpl, -Rpl, 0], [Rpl, -Rpl, 0], [Rpl, Rpl, 0], [-Rpl, Rpl, 0]], af, a_fill=0.10, a_edge=0.4, z=1)
            for g in Fang: vplane(ax, g, PLANE_C, 0.5 * af); inplane_axis(ax, g, AXIS_C, 0.6 * af)
            princ(ax, PURPLE, af, dashed=True)
        ax.set_title(header, fontsize=12.5, color=NAVY, pad=6, weight="bold")
        ax.text2D(0.5, 0.04, cap, transform=ax.transAxes, ha="center", fontsize=9.5, color="#333", clip_on=False)
        ax.text2D(0.02, 0.98, "THEORY", transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color="#8a8f5c", weight="bold")
        return []

    out = f"{outdir}/bf3_tour.gif"
    FuncAnimation(fig, update, frames=len(seg), interval=1000 / FPS, blit=False).save(out, writer=PlayOnceWriter(fps=FPS))
    plt.close(fig); print("wrote", out); return out


def render_nh3_project_track(gen="s1", outdir="."):
    """Projection method as taught: label the three H(1s) orbitals s1,s2,s3, pick one
    (default s1), and TRACK where each C3v operation sends it, recording the images by
    class. Viewed down the C3 axis (same geometry/vantage as the other NH3 clips). The
    table built underneath reproduces the taught layout:
        C3v |  E  |  2C3   |   3sv
         s1 |  s1 | s2+s3  | s1+s2+s3
    A gold 'tracked' orbital starts on the chosen H and moves to its image under each
    operation (rotations follow the C3 arc; reflections cross their mirror line), then
    returns so the next operation again acts on the original orbital."""
    NAVY = "#1D3B5A"; BLUE = "#9db8e8"; GOLD = "#e0a52e"; BROWN = "#8a4b2a"; lim = 1.7
    P0, elems, bonds, _ = MOLECULES["ammonia"]["geom"]()
    N = P0[0]
    idx = {"s1": 3, "s2": 2, "s3": 1}; order = ["s1", "s2", "s3"]
    Hpos = {lab: P0[idx[lab]] for lab in order}
    azlab = {"s1": 330.0, "s2": 210.0, "s3": 90.0}
    genlab = gen
    gendisp = f"{gen[0]}_{gen[1]}"

    def lbl_of_az(a):
        a %= 360
        for k, v in azlab.items():
            if min(abs(a - v), 360 - abs(a - v)) < 1: return k
        return "?"

    # operations in taught order, grouped by class
    ops = [("E",          "E",        "id",  None),
           ("C_3",        "2C_3",     "rot", 120),
           ("C_3^{2}",    "2C_3",     "rot", 240),
           ("\\sigma_v",  "3\\sigma_v","mir", 330),   # through s1
           ("\\sigma_v",  "3\\sigma_v","mir", 210),   # through s2
           ("\\sigma_v",  "3\\sigma_v","mir", 90)]    # through s3
    g0 = azlab[gen]
    targets = []
    for nm, cls, kind, prm in ops:
        if kind == "id":  a = g0
        elif kind == "rot": a = g0 + prm
        else: a = 2 * prm - g0
        targets.append(lbl_of_az(a))
    # cumulative per-class cell contents after each op
    cells_after = []
    acc = {"E": [], "2C_3": [], "3\\sigma_v": []}
    for i, (nm, cls, kind, prm) in enumerate(ops):
        acc = {k: list(v) for k, v in acc.items()}
        acc[cls].append(targets[i])
        cells_after.append({k: sorted(set(v)) for k, v in acc.items()})

    def rotz(p, deg):
        a = d(deg); c, s = np.cos(a), np.sin(a)
        return np.array([c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]])
    def reflect(p, m_deg):
        a = d(2 * m_deg); c, s = np.cos(a), np.sin(a)
        return np.array([c * p[0] + s * p[1], s * p[0] - c * p[1], p[2]])
    def tracked_pos(i, t):
        nm, cls, kind, prm = ops[i]; p = Hpos[gen]
        if kind == "id":  return p
        if kind == "rot": return rotz(p, prm * t)
        return (1 - t) * p + t * reflect(p, prm)

    def ssphere(p, r, n=18):
        u = np.linspace(0, 2 * np.pi, n); v = np.linspace(0, np.pi, n)
        return (p[0] + r * np.outer(np.cos(u), np.sin(v)),
                p[1] + r * np.outer(np.sin(u), np.sin(v)),
                p[2] + r * np.outer(np.ones_like(u), np.cos(v)))

    # ---- schedule ----
    seg = [("intro", None, 0.0)] * 14
    for i, (nm, cls, kind, prm) in enumerate(ops):
        if kind == "id":
            seg += [("land", i, 1.0)] * 14
        else:
            seg += [("move", i, ease((j + 1) / 9)) for j in range(9)]
            seg += [("land", i, 1.0)] * 10
            seg += [("back", i, 1 - ease((j + 1) / 5)) for j in range(5)]
    seg += [("result", None, 0.0)] * 34

    fig = plt.figure(figsize=(5.4, 6.35), dpi=DPI)
    fig.text(0.5, 0.965, f"Projection method: track  ${gendisp}$  through $C_{{3v}}$",
             ha="center", fontsize=14.5, weight="bold", color=NAVY)
    axm = fig.add_axes([0.13, 0.40, 0.74, 0.52], projection="3d")
    axt = fig.add_axes([0.07, 0.055, 0.86, 0.30]); axt.set_axis_off()
    cx = [0.11, 0.34, 0.58, 0.84]; xb = [0.0, 0.22, 0.46, 0.70, 1.0]
    heads = ["C_{3v}", "E", "2C_3", "3\\sigma_v"]; clskey = [None, "E", "2C_3", "3\\sigma_v"]

    def draw_table(cells, active_col=None):
        axt.clear(); axt.set_axis_off(); axt.set_xlim(0, 1); axt.set_ylim(0, 1)
        axt.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="#b9b39a", lw=1.6))
        axt.plot([0, 1], [0.5, 0.5], color="#333", lw=1.4)
        for x in xb[1:-1]: axt.plot([x, x], [0, 1], color="#333", lw=1.2)
        if active_col is not None:
            axt.add_patch(plt.Rectangle((xb[active_col], 0), xb[active_col + 1] - xb[active_col], 1,
                          facecolor="#fbf3df", edgecolor="none", zorder=0))
        for c, h in zip(cx, heads):
            col = NAVY if h == "C_{3v}" else "#111"
            axt.text(c, 0.75, f"${h}$", ha="center", va="center", fontsize=14, weight="bold", color=col)
        axt.text(cx[0], 0.25, f"${gendisp}$", ha="center", va="center", fontsize=14, weight="bold", color="#5b6fd6")
        for j in (1, 2, 3):
            labs = cells[clskey[j]]
            txt = " + ".join(f"{l}" for l in labs)
            txt = txt.replace("s1", "s_1").replace("s2", "s_2").replace("s3", "s_3")
            if txt: axt.text(cx[j], 0.25, f"${txt}$", ha="center", va="center", fontsize=13.5, weight="bold", color=BROWN)

    def lab2d(p, s, color, fs, weight="bold", halo=None):
        xp, yp, _ = proj3d.proj_transform(p[0], p[1], p[2], axm.get_proj())
        t = axm.text2D(xp, yp, s, transform=axm.transData, color=color, fontsize=fs,
                       weight=weight, ha="center", va="center", zorder=25)
        if halo: t.set_path_effects([pe.withStroke(linewidth=2.4, foreground=halo)])

    def update(k):
        phase, i, t = seg[k]
        axm.clear(); axm.set_xlim(-lim, lim); axm.set_ylim(-lim, lim); axm.set_zlim(-lim, lim)
        axm.set_box_aspect((1, 1, 1)); axm.set_axis_off(); axm.view_init(90, -90)
        for a, b in bonds: axm.plot(*zip(P0[a], P0[b]), color=BOND_C, lw=2.6, alpha=0.9, zorder=1)
        axm.scatter(*N, color=STYLE["N"]["color"], s=STYLE["N"]["s"], edgecolor=STYLE["N"]["edge"],
                    linewidth=0.8, depthshade=True, zorder=4)
        # operation geometry cue
        if phase in ("move", "land") and i is not None:
            nm, cls, kind, prm = ops[i]
            if kind == "rot":
                tt = np.linspace(d(8), d(prm - 8), 60); ra = 1.44
                arc = np.stack([ra * np.cos(tt), ra * np.sin(tt), np.full_like(tt, N[2])], 1)
                axm.plot(arc[:, 0], arc[:, 1], arc[:, 2], color=AXIS_C, lw=2.0, alpha=0.85, zorder=2)
                v = arc[-1] - arc[-4]; v = v / np.linalg.norm(v) * 0.22
                axm.quiver(*(arc[-1] - v), *v, color=AXIS_C, lw=2.0, arrow_length_ratio=0.9, zorder=3)
            elif kind == "mir":
                a = d(prm); e = np.array([np.cos(a), np.sin(a), 0]) * 1.5
                axm.plot([-e[0], e[0]], [-e[1], e[1]], [N[2], N[2]], color="#c98a3a",
                         lw=1.6, ls=(0, (5, 3)), alpha=0.9, zorder=2)
        # reference orbitals + labels
        for lab in order:
            axm.plot_surface(*ssphere(Hpos[lab], 0.24), color=BLUE, alpha=0.55, linewidth=0, shade=True, zorder=5)
            p = Hpos[lab]; rad = np.linalg.norm(p[:2]); dxy = p[:2] / rad
            outer = np.array([dxy[0] * (rad + 0.5), dxy[1] * (rad + 0.5), p[2]])
            hot = (phase in ("land",) and i is not None and targets[i] == lab)
            lab2d(outer, f"${lab[0]}_{lab[1]}$", GOLD if hot else "#22313f", 12 if hot else 11)
        # tracked orbital
        tp = tracked_pos(i, t) if (phase in ("move", "land", "back") and i is not None) else Hpos[gen]
        axm.plot_surface(*ssphere(tp, 0.28), color=GOLD, alpha=0.95, linewidth=0, shade=True, zorder=8)
        # captions + table state
        if phase == "intro":
            cap = f"chosen orbital ${gendisp}$ (gold); apply each operation and note where it lands"
            cells = {"E": [], "2C_3": [], "3\\sigma_v": []}; acol = None
        elif phase == "result":
            cap = "images of $s_1$ by class \u2014 the row is the projection-method starting point"
            cells = cells_after[-1]; acol = None
        else:
            nm, cls, kind, prm = ops[i]
            cur = cells_after[i] if phase in ("land", "back") else (cells_after[i - 1] if i > 0 else {"E": [], "2C_3": [], "3\\sigma_v": []})
            cells = cur; acol = {"E": 1, "2C_3": 2, "3\\sigma_v": 3}[cls]
            opdisp = "$E$" if kind == "id" else (f"${nm}$")
            cap = f"{opdisp}:   ${gendisp} \\rightarrow {targets[i][0]}_{targets[i][1]}$"
        axm.text2D(0.5, -0.02, cap, transform=axm.transAxes, ha="center", fontsize=10, style="italic", color="#555")
        draw_table(cells, acol)
        return []

    out = f"{outdir}/nh3_proj_{gen}.gif"
    FuncAnimation(fig, update, frames=len(seg), interval=1000 / FPS, blit=False).save(out, writer=PlayOnceWriter(fps=FPS))
    plt.close(fig); print("wrote", out); return out


def main():
    _GROUP_SETS={
      "group_c2v":("C_{2v}",[("water","water"),("cis-dichloroethene","cis-dichloroethene"),("dichloromethane","dichloromethane"),("dimethyl ether","dimethyl ether")],"C2v",{}),
      "group_d3h":("D_{3h}",[("eclipsed ethane","eclipsed ethane"),("SO$_3$","SO3"),("PCl$_5$","PCl5")],"D3h",{}),
      "group_td":("T_d",[("CCl$_4$","CCl4"),("Ni(CO)$_4$","Ni(CO)4"),("adamantane","adamantane"),("P$_4$O$_{10}$","P4O10")],"Td",{}),
      "group_dinfh":("D_{\\infty h}",[("H$_2$","H2"),("CO$_2$","CO2"),("N$_2$","N2"),("C$_2$H$_2$","C2H2")],"Dinfh",{"linear":True,"subtitle":"linear \u00b7 centre of inversion (i)","mark_centre":True}),
      "group_cinfv":("C_{\\infty v}",[("HBr","HBr"),("HCN","HCN"),("NCS$^-$","NCS")],"Cinfv",{"linear":True,"subtitle":"linear \u00b7 no centre of inversion"}),
    }
    args=sys.argv[1:]; jobs=[]; walks=[]; inv_tests=[]; sn=[]; cubes=[]; tour=False; extras=[]
    if not args:
        for m in MOLECULES:
            jobs+=[(m,o) for o in MOLECULES[m]["curated"]]
            walks+=[(m,nn) for nn in MOLECULES[m].get("walks",[])]  # e.g. benzene C6 walk
            if MOLECULES[m].get("inv_test"): inv_tests.append(m)    # e.g. water, methane
        sn=[1,2]                                                    # S1=sigma, S2=i illustrations
        cubes=["C2","S4"]                                           # methane-in-cube illustrations
        tour=True                                                   # benzene element inventory
        extras=["ethane_s6","ethane_walk","stag_pentagons","stag_cot","pf5_axes","pf5_planes","water_tour","pointgroups","group_c2v","group_d3h","group_td","group_dinfh","group_cinfv","high_td","high_oh","high_ih","lowsym_cs","lowsym_ci","lowsym_c1","rot_rz","a2_dxy","nh3_sv1","nh3_sv2","nh3_sv3","dnh_elements","dnd_ethane","dnd_ferrocene","dn_sq_h","dn_sq_d","dn_sq_n","dn_pent_h","dn_pent_d","dn_pent_n","cn_boric_acid","cn_benzenehexol","cn_glyoxal","cn_water","cn_ammonia","cn_if5","cn_h2o2","cn_ph3ch","cn_s4","cn_bhx_d6h","mo_px_c2z","mo_py_c2z","mo_pz_c2z","mo_px_sxy","mo_py_sxy","mo_pz_sxy","mo_px_sxz","mo_py_sxz","mo_pz_sxz","mo_px_syz","mo_py_syz","mo_pz_syz","c4_px","c4_py","c4_dxz","c4_dyz","sigmad_px","sigmad_py","c3_px","c3_py","sigmav_px","sigmav_py","nm_water_sym_stretch","nm_water_bend","nm_water_asym_stretch","nmdip_water_bend","nmdip_water_asym_stretch","water_salc_a1","water_salc_b1","allyl_pi","c3_dxz","c3_dyz","c3_nh3_px","c3_nh3_py","nh3_proj_a1","nh3_proj_e1","nh3_proj_e2","bf3_tour","nh3_proj_s1"]
    elif args==["all"]:
        for m in MOLECULES:
            jobs+=[(m,o) for o in OPS[m]]
            walks+=[(m,nn) for nn in MOLECULES[m].get("walks",[])]
            if MOLECULES[m].get("inv_test"): inv_tests.append(m)
        sn=[1,2]; cubes=["C2","S4"]; tour=True
        extras=["ethane_s6","ethane_walk","stag_pentagons","stag_cot","pf5_axes","pf5_planes","water_tour","pointgroups","group_c2v","group_d3h","group_td","group_dinfh","group_cinfv","high_td","high_oh","high_ih","lowsym_cs","lowsym_ci","lowsym_c1","rot_rz","a2_dxy","nh3_sv1","nh3_sv2","nh3_sv3","dnh_elements","dnd_ethane","dnd_ferrocene","dn_sq_h","dn_sq_d","dn_sq_n","dn_pent_h","dn_pent_d","dn_pent_n","cn_boric_acid","cn_benzenehexol","cn_glyoxal","cn_water","cn_ammonia","cn_if5","cn_h2o2","cn_ph3ch","cn_s4","cn_bhx_d6h","mo_px_c2z","mo_py_c2z","mo_pz_c2z","mo_px_sxy","mo_py_sxy","mo_pz_sxy","mo_px_sxz","mo_py_sxz","mo_pz_sxz","mo_px_syz","mo_py_syz","mo_pz_syz","c4_px","c4_py","c4_dxz","c4_dyz","sigmad_px","sigmad_py","c3_px","c3_py","sigmav_px","sigmav_py","nm_water_sym_stretch","nm_water_bend","nm_water_asym_stretch","nmdip_water_bend","nmdip_water_asym_stretch"]
    else:
        for a in args:
            if a.startswith("walk:"):               # e.g. walk:benzene  or  walk:bcl3:3
                parts=a.split(":")
                m=parts[1]; nn=int(parts[2]) if len(parts)>2 else None
                if m in MOLECULES: walks.append((m,nn))
            elif a.startswith("inv_test:"):          # e.g. inv_test:methane
                m=a.split(":",1)[1]
                if m in MOLECULES: render_inversion_test(m)
            elif a.startswith("sn:"):                # e.g. sn:1  or  sn:2
                sn.append(int(a.split(":",1)[1]))
            elif a.startswith("cube:"):              # e.g. cube:C2  or  cube:S4
                cubes.append(a.split(":",1)[1])
            elif a=="tour":                          # benzene D6h element inventory
                tour=True
            elif a=="water_tour":                    # water C2v element inventory
                extras.append("water_tour")
            elif a in ("bf3_tour","bf3"):              # BF3 D3h element inventory
                extras.append("bf3_tour")
            elif a in ("pointgroups","point_groups"):# generic point-group showcase
                extras.append("pointgroups")
            elif a.startswith("group:"):             # group:c2v / group:d3h / group:td
                extras.append("group_"+a.split(":",1)[1].lower())
            elif a in ("ethane_s6","ethane_walk"):   # ethane Newman S6 / S6 powers walk
                extras.append(a)
            elif a.startswith("staggered:"):         # staggered:pentagons or staggered:cot
                extras.append("stag_"+a.split(":",1)[1])
            elif a.startswith("pf5_tour:"):           # pf5_tour:axes or pf5_tour:planes
                extras.append("pf5_"+a.split(":",1)[1])
            elif a.startswith("high:"):               # high:td / high:oh / high:ih
                extras.append("high_"+a.split(":",1)[1])
            elif a in ("lowsym","low_symmetry"):     # combined Cs/Ci/C1 cycle
                extras.append("lowsym")
            elif a in ("lowsym_cs","lowsym_ci","lowsym_c1"): extras.append(a)
            elif a in ("water_modes","nm_water"):                # all six water normal-mode figures at once
                extras+=["nm_water_sym_stretch","nm_water_bend","nm_water_asym_stretch",
                         "nmdip_water_sym_stretch","nmdip_water_bend","nmdip_water_asym_stretch"]
            elif a.startswith("nm_") or a.startswith("nmdip_"):  # e.g. nm_water_bend, nmdip_water_asym_stretch
                extras.append(a)
            elif a in ("rot_rz","rotation_rz"): extras.append("rot_rz")
            elif a in ("a2_dxy","a2","dxy"): extras.append("a2_dxy")
            elif a in ("c4_px","c4_py","c4_dxz","c4_dyz"): extras.append(a)
            elif a.startswith("c4:"):                # c4:px / c4:py / c4:dxz / c4:dyz
                extras.append("c4_"+a.split(":",1)[1])
            elif a in ("sigmad_px","sigmad_py"): extras.append(a)
            elif a.startswith("sigmad:"):            # sigmad:px / sigmad:py
                extras.append("sigmad_"+a.split(":",1)[1])
            elif a in ("c3_px","c3_py"): extras.append(a)
            elif a in ("c3_dxz","c3_dyz"): extras.append(a)
            elif a in ("c3_nh3_px","c3_nh3_py"): extras.append(a)
            elif a in ("nh3_proj_a1","nh3_proj_e1","nh3_proj_e2"): extras.append(a)
            elif a in ("nh3proj","projection","nh3_proj"): extras += ["nh3_proj_a1","nh3_proj_e1","nh3_proj_e2"]
            elif a.startswith("nh3proj:"): extras.append("nh3_proj_"+a.split(":",1)[1])
            elif a in ("nh3_proj_s1","nh3track","nh3_track"): extras.append("nh3_proj_s1")
            elif a.startswith("nh3track:"): extras.append("nh3_proj_"+a.split(":",1)[1])
            elif a in ("c3nh3","nh3_c3","c3_nh3"): extras += ["c3_nh3_px","c3_nh3_py"]
            elif a.startswith("c3nh3:"): extras.append("c3_nh3_"+a.split(":",1)[1])
            elif a in ("c3d","c3_d"): extras += ["c3_dxz","c3_dyz"]
            elif a.startswith("c3d:"): extras.append("c3_"+a.split(":",1)[1])
            elif a.startswith("c3:"):                # c3:px / c3:py
                extras.append("c3_"+a.split(":",1)[1])
            elif a in ("sigmav_px","sigmav_py"): extras.append(a)
            elif a.startswith("sigmav:"):            # sigmav:px / sigmav:py
                extras.append("sigmav_"+a.split(":",1)[1])
            elif a in ("nh3_sv1","nh3_sv2","nh3_sv3"): extras.append(a)
            elif a=="dnh_elements": extras.append("dnh_elements")
            elif a=="dnd_ethane": extras.append("dnd_ethane")
            elif a=="dnd_ferrocene": extras.append("dnd_ferrocene")
            elif a=="dn_square": extras.append("dn_square")
            elif a=="dn_pentagon": extras.append("dn_pentagon")
            elif a.startswith("dn:"):                # dn:square:h / dn:pentagon:d etc
                pp=a.split(":");
                if len(pp)==3: extras.append({"square":"dn_sq_","pentagon":"dn_pent_"}.get(pp[1],"dn_sq_")+pp[2])
            elif a.startswith("cn:"):                # cn:water / cn:ammonia / cn:if5 ...
                extras.append("cn_"+a.split(":",1)[1])
            elif a=="cn_s4": extras.append("cn_s4")
            elif a=="cn_bhx_d6h": extras.append("cn_bhx_d6h")
            elif a.startswith("mo:"):                # mo:px:c2z / mo:pz:sxy ...
                pp=a.split(":")
                if len(pp)==3: extras.append(f"mo_{pp[1]}_{pp[2]}")
            elif a.startswith("modes:"):             # modes:water  -> all normal modes of a molecule
                m=a.split(":",1)[1]
                if m in MODES: extras += [f"nm_{m}_{key}" for key in MODES[m]]
            elif a.startswith("mode:"):              # mode:water:bend  -> one normal mode
                pp=a.split(":")
                if len(pp)==3 and pp[1] in MODES and pp[2] in MODES[pp[1]]: extras.append(f"nm_{pp[1]}_{pp[2]}")
            elif a.startswith("dipole:"):            # dipole:water:asym_stretch -> normal mode with dipole vector
                pp=a.split(":")
                if len(pp)==3 and pp[1] in MODES and pp[2] in MODES[pp[1]]: extras.append(f"nmdip_{pp[1]}_{pp[2]}")
            elif a in ("water_salc_a1","water_salc_b1"): extras.append(a)
            elif a in ("allyl","allyl_pi","allyl_pi_mos"): extras.append("allyl_pi")
            elif a=="salc" or a=="water_salc":            # both water H(1s) SALCs
                extras += ["water_salc_a1","water_salc_b1"]
            elif a.startswith("salc:"):                    # salc:a1 / salc:b1
                extras.append("water_salc_"+a.split(":",1)[1])
            elif ":" in a:
                m,o=a.split(":",1)
                if m in OPS and o in OPS[m]: jobs.append((m,o))
            elif a in MOLECULES:
                jobs+=[(a,o) for o in MOLECULES[a]["curated"]]
    for m,o in jobs: render(m,o)
    for m,nn in walks: render_cn_walk(m, n=nn)
    for m in inv_tests: render_inversion_test(m)
    for k in sn: render_sn_collapse(k)
    for op in cubes: render_methane_cube(op)
    if tour: render_benzene_tour()
    for x in extras:
        if x=="ethane_s6": render_ethane_s6("single")
        elif x=="ethane_walk": render_ethane_s6("walk")
        elif x.startswith("nm_"): _m,_k=x[3:].split("_",1); render_mode(_m,_k)
        elif x.startswith("nmdip_"): _m,_k=x[6:].split("_",1); render_mode(_m,_k,show_dipole=True)
        elif x=="stag_pentagons": render_staggered("pentagons")
        elif x=="stag_cot": render_staggered("cot")
        elif x=="pf5_axes": render_pf5_tour("axes")
        elif x=="pf5_planes": render_pf5_tour("planes")
        elif x=="water_tour": render_water_tour()
        elif x=="bf3_tour": render_bf3_tour()
        elif x=="pointgroups": render_pointgroup_showcase()
        elif x in ("high_td","high_oh","high_ih"): render_high_symmetry(x.split("_",1)[1])
        elif x=="lowsym": render_lowsym()
        elif x=="lowsym_cs": render_lowsym_single("CHFCl2","C_s","cs","cs")
        elif x=="lowsym_ci": render_lowsym_single("CHFClanti","C_i","ci","ci")
        elif x=="lowsym_c1": render_lowsym_single("CHFClBr","C_1","c1","c1")
        elif x=="rot_rz": render_rotation_row()
        elif x=="a2_dxy": render_dxy_a2()
        elif x in ("c4_px","c4_py"): render_c4_on_p(x.split("_",1)[1])
        elif x in ("c4_dxz","c4_dyz"): render_c4_on_d(x.split("_",1)[1])
        elif x in ("sigmad_px","sigmad_py"): render_sigmad_on_p(x.split("_",1)[1])
        elif x in ("c3_px","c3_py"): render_c3_on_p(x.split("_",1)[1])
        elif x in ("c3_dxz","c3_dyz"): render_c3_on_d(x.split("_",1)[1])
        elif x in ("c3_nh3_px","c3_nh3_py"): render_c3_ammonia_p(x.split("_")[-1])
        elif x in ("nh3_proj_a1","nh3_proj_e1","nh3_proj_e2"): render_nh3_projection(x.split("_")[-1])
        elif x in ("nh3_proj_s1","nh3_proj_s2","nh3_proj_s3"): render_nh3_project_track(x.split("_")[-1])
        elif x in ("sigmav_px","sigmav_py"): render_sigmav_on_p(x.split("_",1)[1])
        elif x in ("nh3_sv1","nh3_sv2","nh3_sv3"): render_nh3_sigma(int(x[-1])-1, f"sv{x[-1]}")
        elif x=="dnh_elements": render_dnh_elements()
        elif x=="dnd_ethane": render_dnd_ethane()
        elif x=="dnd_ferrocene": render_dnd_ferrocene()
        elif x=="dn_square": render_dn_family(4,"square")
        elif x=="dn_pentagon": render_dn_family(5,"pentagon")
        elif x in ("dn_sq_h","dn_sq_d","dn_sq_n"): render_dn_single(4,x.split("_")[-1],"square")
        elif x in ("dn_pent_h","dn_pent_d","dn_pent_n"): render_dn_single(5,x.split("_")[-1],"pentagon")
        elif x=="cn_boric_acid": render_cn_single("boric_acid",3,"h","C_{3h}","boric_acid")
        elif x=="cn_benzenehexol": render_cn_single("benzenehexol",6,"h","C_{6h}","benzenehexol")
        elif x=="cn_glyoxal": render_cn_single("glyoxal",2,"h","C_{2h}","glyoxal")
        elif x=="cn_water": render_cn_single("water",2,"v","C_{2v}","water")
        elif x=="cn_ammonia": render_cn_single("ammonia",3,"v","C_{3v}","ammonia")
        elif x=="cn_if5": render_cn_single("IF5",4,"v","C_{4v}","if5")
        elif x=="cn_h2o2": render_cn_single("h2o2",2,"n","C_2","h2o2")
        elif x=="cn_ph3ch": render_cn_single("ph3ch",3,"n","C_3","ph3ch")
        elif x=="cn_s4": render_cn_single("s4model",4,"s","S_4","s4")
        elif x=="cn_bhx_d6h": render_cn_single("benzenehexol_d6h",6,"dh","D_{6h}","benzenehexol_d6h")
        elif x in ("water_salc_a1","water_salc_b1"): render_water_salc(x.split("_")[-1])
        elif x=="allyl_pi": render_allyl_pi()
        elif x.startswith("mo_"):
            _,orb,mop=x.split("_"); render_mo_operation(orb,mop,f"{orb}_{mop}")
        elif x in _GROUP_SETS:
            sym,mem,tg,kw=_GROUP_SETS[x]; render_group_set(sym,mem,tg,**kw)

if __name__=="__main__":
    main()
