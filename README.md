# Synchrotron micro-CT of four crystals

Interactive tomography viewers for four olivine / plagioclase crystals (Fagradalsfjall 2023 and Svartsengi, December 2023 to May 2024), imaged at the SYRMEP beamline (Elettra).

## Open the viewers (click, nothing to download)

- **[Slice viewer](https://albcaracc.github.io/syrmep-crystal-tomography/tomo_4panel_viewer.html)**: four panels in eruption order, each with its own play button and scale bar
- **[3D viewer](https://albcaracc.github.io/syrmep-crystal-tomography/tomo_3d_viewer.html)**: rotate and cut through each crystal; pick the crystal from a drop-down

> Clicking the `.html` files in the file list above shows their source code, because GitHub does not render HTML inside the repository. Use the links here instead (they run the viewers in your browser), or download a file and open it locally.

**You do not need the original scan data to view anything here.** Both viewers are single, self-contained HTML files that embed the images / volumes. The first load takes a few seconds (16 MB and 30 MB).

## Contents

| File | What it is |
|---|---|
| `tomo_4panel_viewer.html` | Four panels in eruption order; each scrolls through its stack with its own play button, slider, slice / z read-out and 500 µm scale bar. Grey levels are auto-contrasted per crystal over the whole stack (0.5-99.8 percentile). |
| `tomo_3d_viewer.html` | WebGL2 volume rendering. Drop-down for Crystal 100 / 128 / 150 / 166; phase opacity sliders (voids, melt, crystal), screen-parallel cut plane, orthographic view with a live scale bar. |
| `code/` | Python that produced the files above (see below). |

| Panel | Crystal | Sample | Mineral |
|---|---|---|---|
| 1 | 100 | Fagradalsfjall 2023 | Plagioclase |
| 2 | 128 | Svartsengi December 2023 | Olivine |
| 3 | 150 | Svartsengi February 2024 | Olivine |
| 4 | 166 | Svartsengi March-May 2024 | Olivine |

## Acquisition and processing

- Reconstructed slices: 2045 x 2045 px, 16-bit, **0.9 µm voxel** (28 keV, propagation distance 100 mm; from the reconstruction parameter files).
- Slice viewer: every 3rd slice, 800 px frames (about 3.6 µm per displayed pixel), WebP.
- 3D viewer: x/y binned and z subsampled to 3.6 µm voxels (4.5 µm for Crystal 100, which is much larger).

### Phases in the 3D viewer (read before interpreting)

Phases are assigned from **X-ray attenuation (grey level) only**, not chemistry:

- **Voids**: enclosed cavities darker than the solid threshold.
- **Melt**: solid voxels below a per-crystal grey limit (`MELT_MAX` in `code/make_tomo_3d.py`), read from each crystal's interior grey-level histogram, away from the outer surface.
- **Crystal**: everything else. Compositional zoning inside the crystal shows as shading, not as a separate phase.

Confidence differs by crystal: good for Crystal 150 (a separate glass-like population forms melt inclusions), fair for Crystal 128 (thin adhering glass film), little resolvable melt in Crystal 100, and for Crystal 166 the "melt" class is the whole groundmass (glass plus microlites). Embayment glass whose attenuation overlaps the Fo-rich core cannot be separated by thresholding. The holder / embedding base was removed from Crystal 166.

## Regenerating the files

Needs the original reconstructed TIFF stacks (not in this repository, about 35 GB) and `numpy`, `scipy`, `Pillow`, `matplotlib`.

```bash
cd code
# set TOMO_DATA_ROOT to the folder containing Eruption_2023, Svartsengi_23D, ...
# set TOMO_OUT_DIR to where the HTML / figures should be written (default: current folder)
python make_tomo_4panel.py      # slice viewer (also writes a static 2x2 figure, not published here)
python make_tomo_3d.py          # 3D viewer (drop-down with the four crystals)
```

Options: `make_tomo_4panel.py --step 2 --out-size 1024 --quality 80` for a bigger, smoother slice viewer; `make_tomo_3d.py --bins 6,5,5,5` for a smaller 3D file.

## Notes

- No licence has been chosen yet; until one is added, all rights are reserved by the owner. Add a `LICENSE` file to state the terms of reuse.
- The 3D viewer needs WebGL2 and a reasonably capable GPU; Crystal 100 and Crystal 166 are the heaviest.
