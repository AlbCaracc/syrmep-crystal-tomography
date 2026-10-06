#!/usr/bin/env python3
"""
Interactive 3D (GPU volume-rendering) viewer of the four micro-CT crystals, with a crystal drop-down
and an "all four, same scale" mode.

    python make_tomo_3d.py                         # all four crystals -> tomo_3d_viewer.html
    python make_tomo_3d.py --samples 3             # only crystal 150
    python make_tomo_3d.py --bins 6,5,5,5          # coarser voxels = smaller / faster file

Pipeline per crystal (the source drive is only read, never written):
  1. read every <bin>-th slice (averaged with its neighbour) and bin x/y by <bin>  -> isotropic volume
  2. smooth, threshold the solid, keep the largest connected body (drops epoxy / holder / ring artefacts),
     close + fill holes, crop to its bounding box
  3. label three phases from grey level (X-ray attenuation):
        voids   enclosed cavities darker than the solid threshold
        melt    solid voxels darker than MELT_MAX[crystal] (glass-like grey level), away from the outer surface
        crystal everything else
  4. pack grey level / void / melt into an RGB8 volume, gzip, embed in one self-contained HTML (WebGL2)

Requires numpy, scipy, Pillow.
"""
import argparse
import base64
import glob
import gzip
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from make_tomo_4panel import OUT_DIR, PIXEL_UM, SAMPLES, crystal_name

# Upper grey limit (16-bit units, after the 0.8-voxel smoothing) of the "melt" class, read off the interior
# grey-level histogram of each crystal: the valley between the glass-like low-attenuation population and the
# main crystal peak. Adjust here if you segment differently.
MELT_MAX = {
    "Xtal_100": 38700,   # plagioclase: one dominant peak at ~40.5k; only a thin low tail -> very little melt
    "Xtal_128": 28500,   # low shoulder 26-28.5k, crystal peak 29.3-35k
    "Xtal_150": 40300,   # glass-like population 37.4-39.7k, valley ~40k, crystal 41-52k (zoning is NOT a phase)
    "Xtal_166": 48600,   # continuum; valley between 46.9k and 49.9k peaks -> lower class includes groundmass
}
DEFAULT_BINS = {"Xtal_100": 5, "Xtal_128": 4, "Xtal_150": 4, "Xtal_166": 4}   # voxel = bin * 0.9 um
MIN_MELT_VOXELS = 20


def load_volume(files, B, cache):
    if os.path.exists(cache):
        return np.load(cache)

    def load(i):
        a = np.asarray(Image.open(files[i])).astype(np.float32)
        if i + 1 < len(files):
            a = 0.5 * (a + np.asarray(Image.open(files[i + 1])).astype(np.float32))
        n = (a.shape[0] // B) * B
        return a[:n, :n].reshape(n // B, B, n // B, B).mean((1, 3)).astype(np.uint16)

    with ThreadPoolExecutor(6) as ex:
        vol = np.stack(list(ex.map(load, range(0, len(files), B))))
    np.save(cache, vol)
    return vol


def otsu(x, bins=256):
    h, e = np.histogram(x, bins=bins)
    c = 0.5 * (e[1:] + e[:-1])
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1)
    m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    return c[np.argmax(w0 * w1 * (m0 - m1) ** 2)]


def build_crystal(s, B, cache_dir, levels=200):
    files = sorted(glob.glob(os.path.join(s["path"], "slice_*.tif")))
    vol = load_volume(files, B, os.path.join(cache_dir, f"{s['xtal']}_bin{B}.npy")).astype(np.float32)
    vox_um = PIXEL_UM * B
    g = ndi.gaussian_filter(vol, 0.8)
    del vol
    sub = g[::2, ::2, ::2]
    lo, hi = np.percentile(sub, [0.5, 99.9])
    solid_thr = otsu(sub[sub > np.percentile(sub, 50)])        # solid vs air / epoxy
    solid = g > solid_thr
    # holder / embedding base: slices where "solid" fills most of the circular field of view are not crystal
    frac = solid.sum((1, 2)) / (np.pi / 4 * g.shape[1] ** 2)
    base = ndi.binary_dilation(frac > 0.6, iterations=2)
    if base.any():
        solid[base] = False
        print(f"{s['xtal']}: dropped {int(base.sum())} base slices (solid fills >60% of the field)", flush=True)
    lab, n = ndi.label(solid)
    sizes = ndi.sum(solid, lab, range(1, n + 1))
    body = lab == 1 + int(np.argmax(sizes))                    # largest body = the crystal (+ attached glass)
    del lab
    if base.any():   # thin remnants of the holder disc still touching the crystal: open (3 vox), keep main body, regrow
        core = ndi.binary_opening(body, iterations=3)
        lab, n = ndi.label(core)
        sizes = ndi.sum(core, lab, range(1, n + 1))
        body = ndi.binary_dilation(lab == 1 + int(np.argmax(sizes)), iterations=3) & body
        del core, lab
    body = ndi.binary_fill_holes(ndi.binary_closing(body, iterations=2))
    mask = ndi.binary_dilation(body, iterations=1)
    zs, ys, xs = ndi.find_objects(mask.astype(np.uint8))[0]
    sl = tuple(slice(max(0, q.start - 2), q.stop + 2) for q in (zs, ys, xs))
    g, body, mask = g[sl], body[sl], mask[sl]

    # --- phases
    depth = ndi.distance_transform_edt(body)                   # voxels from the outer surface
    void = ndi.binary_erosion(body, iterations=2) & (g < solid_thr)
    near_void = ndi.binary_dilation(void, iterations=2)        # partial-volume shell around cavities is not melt
    melt = (g >= solid_thr) & (g < MELT_MAX[s["xtal"]]) & (depth > 2) & ~near_void
    lab, n = ndi.label(melt)
    if n:
        sizes = ndi.sum(melt, lab, range(1, n + 1))
        melt = np.isin(lab, 1 + np.flatnonzero(sizes >= MIN_MELT_VOXELS))
    inner = (g >= solid_thr) & body & ~melt & (depth > 2)
    vv = vox_um ** 3 / 1e9                                     # mm3 per voxel
    tot = body.sum() * vv
    print(f"{s['xtal']}: {B}x binned ({vox_um:.1f} um), cropped {g.shape[::-1]}, solid thr {solid_thr:.0f}, "
          f"melt < {MELT_MAX[s['xtal']]}, body {tot:.3f} mm3 | crystal {inner.sum() * vv:.3f}, "
          f"melt {melt.sum() * vv:.4f}, voids {void.sum() * vv:.4f} mm3", flush=True)

    # --- RGB8: R = grey level (soft sub-voxel edge), G = void, B = melt
    q = np.clip((g - lo) / (hi - lo), 0, 1)
    q8 = 1 + np.round(q * (levels - 1)) * (254 / (levels - 1))
    air = np.median(g[~ndi.binary_dilation(mask, iterations=6)])
    soft = np.clip((g - air) / (solid_thr - air), 0, 1)
    soft = soft * soft * (3 - 2 * soft)
    r8 = np.where(mask, np.maximum(np.round(q8 * np.where(g > solid_thr, 1.0, soft)), 1), 0).astype(np.uint8)
    g8 = np.round(ndi.gaussian_filter(void.astype(np.float32), 0.7) * 255).astype(np.uint8)
    b8 = np.round(ndi.gaussian_filter(melt.astype(np.float32), 0.7) * 255).astype(np.uint8)
    nz, ny, nx = r8.shape
    raw = np.stack([r8, g8, b8], axis=-1).tobytes()
    blob = gzip.compress(raw, 9)
    print(f"   {len(raw) / 1e6:.1f} MB raw RGB8 -> {len(blob) / 1e6:.1f} MB gzip", flush=True)
    return dict(key=s["xtal"], label=s["label"], mineral=s["mineral"], crystal=crystal_name(s["xtal"]),
                nx=nx, ny=ny, nz=nz, vox_um=vox_um, b64=base64.b64encode(blob).decode("ascii"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, nargs="+", default=[1, 2, 3, 4], help="1-4, order of the 4-panel figure")
    ap.add_argument("--bins", default=None, help="comma list of xy binning / z step per sample, e.g. 5,4,4,4")
    ap.add_argument("--cache", default=tempfile.gettempdir(), help="where binned volumes are cached")
    ap.add_argument("--out-dir", default=OUT_DIR)
    a = ap.parse_args()

    bins = [int(b) for b in a.bins.split(",")] if a.bins else [DEFAULT_BINS[s["xtal"]] for s in SAMPLES]
    data = [build_crystal(SAMPLES[i - 1], bins[i - 1], a.cache) for i in a.samples]

    html = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tomo_3d_template.html"),
                encoding="utf-8").read().replace("__DATA__", json.dumps(data))
    out = os.path.join(a.out_dir, "tomo_3d_viewer.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print("written", out, f"({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
