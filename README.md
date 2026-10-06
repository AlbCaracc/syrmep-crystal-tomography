# Synchrotron micro-CT of four crystals

Micro-CT of four crystals from the 2023-2024 Fagradalsfjall and Svartsengi eruptions, imaged at the SYRMEP beamline (Elettra).

- **[Slice viewer](https://albcaracc.github.io/syrmep-crystal-tomography/tomo_4panel_viewer.html)**: the four crystals in eruption order, each with its own play button and 500 µm scale bar.
- **[3D viewer](https://albcaracc.github.io/syrmep-crystal-tomography/tomo_3d_viewer.html)**: rotate and cut through each crystal; choose the crystal from the drop-down.

| Panel | Crystal | Sample | Mineral |
|---|---|---|---|
| 1 | 100 | Fagradalsfjall 2023 | Plagioclase |
| 2 | 128 | Svartsengi December 2023 | Olivine |
| 3 | 150 | Svartsengi February 2024 | Olivine |
| 4 | 166 | Svartsengi March-May 2024 | Olivine |

## Data

- Voxel size 0.9 µm, 28 keV.
- Slice viewer: every 3rd slice, 800 px frames. Grey levels are auto-contrasted per crystal (0.5-99.8 percentile of the stack).
- 3D viewer: 3.6 µm voxels (4.5 µm for Crystal 100).

## Phases in the 3D viewer

Phases are set from grey level (X-ray attenuation), not chemistry:

- **Voids**: enclosed cavities.
- **Melt**: solid voxels below a grey limit chosen for each crystal (`MELT_MAX` in `code/make_tomo_3d.py`).
- **Crystal**: everything else. Zoning shows as shading.

Melt is best defined in Crystal 150. In Crystal 128 it is a thin glass film, in Crystal 100 there is almost none, and in Crystal 166 the "melt" class is the whole groundmass. Embayment glass with the same grey level as the Fo-rich core is not separated.

## Rebuild

Needs the original TIFF stacks plus `numpy`, `scipy`, `Pillow` and `matplotlib`.

```bash
cd code
# TOMO_DATA_ROOT = folder containing Eruption_2023, Svartsengi_23D, ...
# TOMO_OUT_DIR   = output folder (default: current folder)
python make_tomo_4panel.py   # slice viewer
python make_tomo_3d.py       # 3D viewer
```
