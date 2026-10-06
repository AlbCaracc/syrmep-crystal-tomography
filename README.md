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
- 3D viewer: 3.6 µm voxels (4.5 µm for Crystal 100; its melt is segmented on a 2.7 µm grid).

## Phases in the 3D viewer

Phases are set from grey level (X-ray attenuation), not chemistry:

- **Voids**: enclosed cavities.
- **Melt**: glass-like material.
  - Olivine: solid voxels darker than a grey limit chosen for each crystal (`MELT_MAX` in `code/make_tomo_3d.py`).
  - Plagioclase (Crystal 100): glass is only slightly denser than the host, so a global threshold fails. Melt is taken as blobs brighter than the local host level, found on a finer 2.7 µm grid. Blobs must lie at least 60 µm below the surface and be at least 16 µm thick, which removes the bright surface halo and the thin bright fringes along cracks.
- **Crystal**: everything else. Zoning shows as shading.

Melt is best defined in Crystals 150 and 100, where it forms discrete inclusions (many with a shrinkage bubble). In Crystal 100, inclusions smaller than about 16 µm or within 60 µm of the surface are not picked up. In Crystal 128 melt is a thin glass film, and in Crystal 166 the "melt" class is the whole groundmass. Embayment glass with the same grey level as the Fo-rich core is not separated.

## Rebuild

Needs the original TIFF stacks plus `numpy`, `scipy`, `Pillow` and `matplotlib`.

```bash
cd code
# TOMO_DATA_ROOT = folder containing Eruption_2023, Svartsengi_23D, ...
# TOMO_OUT_DIR   = output folder (default: current folder)
python make_tomo_4panel.py   # slice viewer
python make_tomo_3d.py       # 3D viewer
```
