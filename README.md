# Symmetry and Bonding — figures and code

Animated GIFs and still figures for **Chem-3 P2: Symmetry and Bonding**, together
with the Python that generates them.

Two ways to use this repository:

- **Just want the pictures?** Browse the `figures/` folders and click any file to
  view it, or use the *Download ZIP* button under the green **Code** button to get
  everything at once. Nothing needs installing.
- **Want to make your own?** Follow the setup below. The code is commented so you
  can change colours, atoms, molecules and symmetry operations without knowing
  much Python.

Everything here is teaching material for the course. You are welcome to use the
figures in your own notes and revision.

---

## What's here

| folder | contents |
|---|---|
| `code/` | the Python scripts |
| `figures/operations/` | symmetry operations for water, ammonia, methane, ethane, benzene, PF₅ and others |
| `figures/ml6/` | the octahedral complex: operations, SALCs, MOs, vibrations |
| `figures/orbitals/` | d orbitals, metal AOs, SALCs on the ML framework |
| `figures/conformers/` | ethane's eclipsed → staggered descent in symmetry |
| `docs/` | the full worked SALC derivation (LaTeX source and PDF) |

---

## Setting up

You need **Python** and three packages. About ten minutes, once.

### Do you already have Python?

Open a terminal — **Command Prompt** on Windows, **Terminal** on a Mac — and type:

```
python --version
```

on Windows, or

```
python3 --version
```

on a Mac. If you see `Python 3.9` or higher, skip to *Installing the packages*.
If you see an error, or a version starting with `2.`, install Python first.

---

### Windows

**1. Install Python.** Go to [python.org/downloads](https://www.python.org/downloads/)
and download the latest Windows installer.

> **Important:** on the first screen of the installer, tick the box marked
> **"Add python.exe to PATH"** before clicking Install. It is easy to miss and it
> is the single most common thing that goes wrong. If you have already installed
> without it, run the installer again and choose *Modify*.

**2. Open Command Prompt.** Press `Windows key`, type `cmd`, press Enter.

**3. Install the packages:**

```
py -m pip install numpy matplotlib pillow
```

**4. Check it worked:**

```
py -c "import numpy, matplotlib, PIL; print('all good')"
```

---

### macOS

macOS comes with an old Python that is not suitable, so install a current one.

**1. Install Python.** Go to [python.org/downloads](https://www.python.org/downloads/)
and download the macOS installer. Run it and accept the defaults.

> Apple Silicon (M1/M2/M3/M4) and Intel Macs both work. The installer picks the
> right version automatically.

**2. Open Terminal.** Press `Cmd + Space`, type `Terminal`, press Enter.

**3. Install the packages:**

```
python3 -m pip install numpy matplotlib pillow
```

If that reports a permissions error, add `--user`:

```
python3 -m pip install --user numpy matplotlib pillow
```

**4. Check it worked:**

```
python3 -c "import numpy, matplotlib, PIL; print('all good')"
```

---

### Versions

Developed against Python 3.12, numpy 2.4, matplotlib 3.10 and pillow 12, but
anything from **Python 3.9** and **matplotlib 3.5** onwards should work. The
oldest hard requirement is matplotlib 3.3, for 3D axis scaling.

---

## Running the code

Download the repository (green **Code** button → *Download ZIP*) and unzip it.
Then in your terminal, move into the `code` folder. The quickest way:

- **Windows:** open the `code` folder in File Explorer, click the address bar,
  type `cmd` and press Enter.
- **Mac:** type `cd ` in Terminal (with a space), then drag the `code` folder
  from Finder onto the Terminal window and press Enter.

Then run a script. On Windows use `py`, on a Mac use `python3`:

```
py symmetry_operation_gifs.py                 # Windows
python3 symmetry_operation_gifs.py            # Mac
```

That writes a set of GIFs into the current folder. Other things to try:

```
python3 symmetry_operation_gifs.py water              # just water
python3 symmetry_operation_gifs.py water methane      # two molecules
python3 symmetry_operation_gifs.py pf5:sigma_h        # one specific operation
python3 ml6_gifs.py my_output                         # octahedral set, into a folder
python3 ethane_conformers.py                          # the ethane conformers
```

Each script prints the name of every file as it writes it. A full run takes a
minute or two — the 3D rendering is the slow part.

---

## Making changes

This is the part worth trying even if you have never written Python.

Open `symmetry_operation_gifs.py` in any text editor (Notepad, TextEdit, VS Code)
and look at the block near the top marked **SETTINGS**. Everything there is safe
to change. For instance:

**Make the animations smaller and faster to test with:**
```python
FRAMES = 20          # was 60
DPI    = 70          # was 110
```

**Recolour an element** — oxygen in purple:
```python
"O": dict(color="#7b2fbf", s=500, edge="#3d1060"),
```
Colours are hex codes; any online colour picker will give you one. `s` is the
marker *area*, so doubling it makes the circle about 1.4× wider, not 2×.

**Change which atom is followed in crimson,** or the colour of the axes and
planes — both are in the same block.

Further down, three sections marked `GEOMETRIES`, `OPS` and `MOLECULES` control
the content, and each has a worked comment showing how to add your own molecule
or symmetry operation. Adding a diatomic takes about four lines.

If you break something, download a fresh copy — nothing is lost.

---

## If it goes wrong

**`'python' is not recognized`** (Windows) — Python is not on your PATH. Re-run
the installer, choose *Modify*, and tick *Add python.exe to PATH*. Or use `py`
instead of `python`, which usually works regardless.

**`No module named 'matplotlib'`** — the packages did not install, or installed
for a different Python. Run the install command again using exactly the same
command name (`py` or `python3`) you use to run the scripts.

**`No module named 'scipy'`** — you have an old copy of the code. Download the
current version; scipy is not needed. If you would rather not re-download,
`py -m pip install scipy` also fixes it.

**`No module named 'symmetry_operation_gifs'`** — you are running from the wrong
folder, or `ml6_gifs.py` has been separated from the main engine. They must sit in
the same folder.

**Nothing appears on screen** — that is expected. The scripts write image files
rather than opening a window. Look in the folder you ran them from.

**It is very slow** — reduce `FRAMES` and `DPI` in the SETTINGS block while you
experiment, and put them back when you want a final version.

**A figure looks clipped** — increase `lim` for that molecule in the `MOLECULES`
registry. It sets the half-width of the drawing box.

---

## A note on the code

The scripts are deliberately written to be read. The animations of reflections,
inversions and improper rotations are honest about something worth knowing:
those operations cannot be carried out as a continuous physical motion in three
dimensions. The figures move each atom in a straight line onto its image and say
"mapping" rather than pretending a rotation is happening. Proper rotations, which
*are* genuine rigid motions, are animated as such.

---

## Licence

Figures: Creative Commons Attribution 4.0 (CC BY 4.0) — use them anywhere with
credit.
Code: MIT Licence — use, modify and share freely.
