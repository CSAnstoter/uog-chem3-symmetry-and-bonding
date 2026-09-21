# Figure index, by lecture

Which figures go with which lecture. Many are used more than once — water appears
in Lectures 1, 4 and 5 — so each is listed under every lecture that uses it.

**File name conventions**, consistent across the whole set:

| suffix | meaning |
|---|---|
| *(none)* | square canvas, transparent background, with labels |
| `_plain` | the same figure with the labelling removed |
| `_tight` | cropped close to the drawing, for placing at any size |

Transparent backgrounds mean they sit on a white or a dark slide equally well.

---

## Lecture 1 — Symmetry operations

`figures/operations/`

| figure | shows |
|---|---|
| `water_E.gif`, `water_C2.gif`, `water_sigma_v.gif`, `water_sigma_vp.gif` | the four operations of C₂ᵥ |
| `ammonia_C3.gif`, `ammonia_sigma_v.gif` | a threefold axis and a vertical mirror |
| `methane_C3.gif`, `methane_S4.gif` | S₄ in a molecule with no σ perpendicular to that axis |
| `ethane_S6.gif`, `ethane_S6_newman.gif` | improper rotation, 3D and as a Newman projection |
| `benzene_C6.gif`, `benzene_sigma_h.gif` | the sixfold axis and the molecular plane |
| `pf5_*.gif` | trigonal bipyramidal: axial and equatorial behave differently |

The reflections, inversions and improper rotations say **"mapping"** rather than
animating a rotation, because those operations cannot be performed as a
continuous rigid motion in three dimensions. Only proper rotations are genuine
physical motions, and only those are animated as such.

---

## Lecture 2 — Point groups

`figures/operations/` and `figures/conformers/`

| figure | shows |
|---|---|
| `cn_water.gif`, `cn_ammonia.gif`, `cn_IF5.gif` | C₂ᵥ, C₃ᵥ, C₄ᵥ — axis plus vertical planes |
| `cn_boric acid.gif`, `cn_benzenehexol.gif` | C₃ₕ, C₆ₕ — axis plus a horizontal plane |
| `cn_h2o2.gif`, `cn_ph3ch.gif` | C₂, C₃ — an axis and nothing else |
| `cn_s4model.gif` | a pure S₄ group |
| `triphenylmethane_C3.png` | the C₃ propeller, as a still |
| `ethane_conformer_eclipsed/skew/staggered.png` | D₃ₕ → D₃ → D₃d |
| `ethane_conformers.gif` | the rotation, with elements appearing and vanishing |

The ethane set is the real-chemistry worked example for the flowchart: one
molecule, three point groups, and an order that goes 12 → 6 → 12.

---

## Lecture 3 — Group theory and character tables

Mostly tables rather than pictures. The operation GIFs from Lectures 1 and 2 are
the useful illustrations when demonstrating that operations in the same class are
related by a change of viewpoint.

---

## Lecture 4 — Reducible representations

`figures/modes/`

| figure | shows |
|---|---|
| `water_mode_sym_stretch.gif`, `water_mode_bend.gif`, `water_mode_asym_stretch.gif` | the three normal modes of water |
| `water_mode_*_dipole.gif` | the same, with the dipole moment drawn |
| `co2_sym_stretch.png`, `co2_asym_stretch.png`, `co2_bend.png` | CO₂, and where mutual exclusion comes from |
| `water_sym_stretch.png`, `water_bend.png`, `water_asym_stretch.png` | the water modes as stills |

Water uses the course convention throughout: C₂ along z, molecule in the **xz**
plane, so the antisymmetric stretch is B₁.

---

## Lecture 5 — Symmetry and molecular orbitals

`figures/modes/`, `figures/operations/`

The water SALC and MO figures live in the deck itself. The mode figures above are
reused here when the point is made that the hydrogen 1s SALCs and the bond
stretches give the *same* representation.

---

## Lecture 6 — Degenerate orbitals

`figures/modes/`

| figure | shows |
|---|---|
| `ammonia_C3.gif`, `ammonia_sigma_v.gif` | the operations used to build Γ |
| `ammonia_mode_sym_stretch.gif`, `ammonia_mode_umbrella.gif` | the two A₁ modes |
| `ammonia_mode_e_stretch.gif`, `ammonia_mode_e_bend.gif` | one partner of each E pair |
| `ammonia_sym_stretch.png`, `ammonia_umbrella.png`, `ammonia_e_stretch.png`, `ammonia_e_bend.png` | the same four as stills |

These were regenerated after a geometry correction: the bond angle from the C₃
axis is **not** 180° − α. For a C₃ᵥ AX₃ it satisfies cos α = 1.5 cos²β − 0.5,
giving 67.9° for ammonia rather than the 73.3° first used.

---

## Lecture 7 — Walsh diagrams

Figures are built into the deck. Water follows the same xz convention as
Lectures 4 to 6, so its HOMO is **1b₂** throughout.

---

## Lecture 8 — Octahedral complexes

The largest set, in four folders.

### `figures/ml6/` — operations and symmetry elements

| figure | shows |
|---|---|
| `ml6_E/C3/C2p/C4/C2/i/S4/S6/sigma_h/sigma_d.gif` | one per Oh class, with unmoved ligands ringed in green so the contribution to χ can be counted off the figure |
| `ml6_C3_axes.gif`, `ml6_C4_axes.gif` | one axis, then the rest of its class |
| `ml6_C2_class.gif`, `ml6_C2p_class.gif`, `ml6_S4_class.gif`, `ml6_sigma_d_class.gif` | the operation, then the other equivalent elements |
| `ml6_C3_faces.gif` | C₃ through two opposite faces — the staggered-ethane Newman picture |
| `ml6_octahedron_reveal.gif` | the polyhedron and the complex inside it |
| `ml6_mode_a1g/eg_z2/eg_x2y2/t1u_z.gif` | the M–L stretching modes |

### `figures/ml6/cube/` — the cube framing

`cube_setup.png` plus one per class. Putting each ligand at a cube face centre
turns the symmetry elements into features of the cube: face-centre axes for
6C₄/3C₂/6S₄, body diagonals for 8C₃/8S₆, edge-midpoint axes for 6C₂′, and so on.
The exemplary element is orange, the rest of its class faint blue.

### `figures/orbitals/` — SALCs, metal orbitals and MOs

| figure | shows |
|---|---|
| `ml6_basis_sigma.png`, `ml6_basis_pi_set1/set2.png` | the σ and π bases |
| `ml6_salc_a1g/eg_z2/eg_x2y2/t1u_x/t1u_y/t1u_z.png` | the six σ SALCs |
| `ml6_salc_eg_seed_phi1.png`, `ml6_eg_seed_decomposition.png` | why seeding φ₁ gives a valid but unusable e_g function |
| `ml6_ao_4s/4px/4py/4pz.png` | the metal orbitals they pair with |
| `d_*.png`, `d_*_ml6.png` | the five d orbitals, alone and on the ML framework |
| `ml6_mo_*.png` | all fifteen MOs of the σ-only scheme |
| `proj_*.png`, `proj_selfrow.png` | where each class sends the seed orbital |

All of these share one camera and one M–L distance, so any two can be placed side
by side and will register exactly — a d orbital above its SALC partner, for
instance.

### `figures/ml6/vibrations/` — the full set of modes

`ml6_vib_nu1_a1g` through `ml6_vib_nu6_t2u_3`: all fifteen vibrations, being
a₁g + e_g + 2t₁u + t₂g + t₂u. Derived by projection rather than drawn by hand,
and checked to be mutually orthogonal and free of translation and rotation.

### `figures/ml6/descent/` — substituted complexes

`ml6_descent_ML6/ML5Lp/trans/cis.png`: Oh → C₄ᵥ → D₄ₕ → C₂ᵥ. Each marks the
principal axis, σₕ where it survives, and the inversion centre — which is the one
that decides whether IR and Raman bands can coincide.



