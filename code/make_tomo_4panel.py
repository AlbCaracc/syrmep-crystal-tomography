#!/usr/bin/env python3
"""
Four-panel synchrotron micro-CT viewer (SYRMEP, Elettra) for the Svartsengi MI paper.

Builds, from stacks of reconstructed slices (slice_XXXX.tif, 16-bit):
  * tomo_4panel_viewer.html : self-contained interactive figure, one play button
                              (+ slider, slice read-out, scale bar) per panel
  * tomo_4panel_static.png/.pdf : static 2x2 figure of the mid-stack slices

The source drive is only READ from; every output goes to OUT_DIR.

Requires: numpy, Pillow, matplotlib.
    python make_tomo_4panel.py                 # defaults below
    python make_tomo_4panel.py --step 2 --out-size 1024 --quality 80  # bigger file, smoother scrolling
"""
import argparse
import base64
import glob
import io
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image

# ----------------------------------------------------------------------------
# Samples, in chronological order
# ----------------------------------------------------------------------------
PIXEL_UM = 0.9  # voxel size in um; PHASE_px_size in full_run_params_Xtal_{102,128,150,166}.txt
ROOT = os.environ.get("TOMO_DATA_ROOT", r"D:\20235157_Caracciolo")  # folder with Eruption_2023, Svartsengi_23D, ...
SAMPLES = [
    dict(label="Fagradalsfjall 2023", mineral="Plagioclase", xtal="Xtal_102",
         path=ROOT + r"\Eruption_2023\Xtal_102\output_16"),
    dict(label="Svartsengi December 2023", mineral="Olivine", xtal="Xtal_128",
         path=ROOT + r"\Svartsengi_23D\Xtal_128\output_16"),
    dict(label="Svartsengi February 2024", mineral="Olivine", xtal="Xtal_150",
         path=ROOT + r"\Svartsengi_24F\Xtal_150\output_16"),
    dict(label="Svartsengi March\u2013May 2024", mineral="Olivine", xtal="Xtal_166",
         path=ROOT + r"\Svartsengi_24M\Xtal_166\output_16"),
]

OUT_DIR = os.environ.get("TOMO_OUT_DIR", ".")   # where the HTML / figures are written

def crystal_name(xtal):
    """Xtal_150 -> Crystal 150 (display name)"""
    return xtal.replace("Xtal_", "Crystal ")


SCALEBAR_UM = 500          # length of the scale bar
LOW_PCT, HIGH_PCT = 0.5, 99.8   # auto-contrast percentiles (per crystal, whole stack)


# ----------------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------------
def read_slice(path):
    return np.asarray(Image.open(path))


def circle_mask(n, frac=0.48):
    """Reconstruction FOV is circular; ignore the constant corners when stretching."""
    y, x = np.ogrid[:n, :n]
    c = (n - 1) / 2
    return (x - c) ** 2 + (y - c) ** 2 <= (frac * n) ** 2


def auto_contrast(files, n_sample=24, stride=4):
    """Percentile limits from slices sampled over the whole stack (same limits for every frame,
    so brightness does not flicker while scrolling)."""
    idx = np.linspace(0, len(files) - 1, n_sample).astype(int)
    first = read_slice(files[0])
    mask = circle_mask(first.shape[0])[::stride, ::stride]
    vals = []
    for i in idx:
        a = read_slice(files[i])[::stride, ::stride]
        vals.append(a[mask])
    v = np.concatenate(vals)
    lo, hi = np.percentile(v, [LOW_PCT, HIGH_PCT])
    return float(lo), float(hi)


def to_8bit(a, lo, hi):
    return np.clip((a.astype(np.float32) - lo) / (hi - lo) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def make_frame(path, lo, hi, out_size, quality):
    img = Image.fromarray(to_8bit(read_slice(path), lo, hi), "L")
    if out_size and out_size != img.size[0]:
        img = img.resize((out_size, out_size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="WEBP", quality=quality, method=4)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ----------------------------------------------------------------------------
# Static figure
# ----------------------------------------------------------------------------
def static_figure(samples, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(2, 2, figsize=(10, 10.4), facecolor="white")
    for k, (ax, s) in enumerate(zip(axes.ravel(), samples)):
        n = len(s["files"])
        mid = n // 2
        a = read_slice(s["files"][mid])
        img = to_8bit(a, *s["lims"])
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="lanczos")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"({'abcd'[k]}) {s['label']} \u2013 {s['mineral']}", fontsize=12, loc="left",
                     fontweight="bold")
        ax.text(0.0, -0.015, f"{crystal_name(s['xtal'])}  \u00b7  slice {mid} / {n - 1}  (z = {mid * PIXEL_UM:.0f} \u00b5m)",
                transform=ax.transAxes, va="top", fontsize=9, color="#444")
        # scale bar
        L = SCALEBAR_UM / PIXEL_UM
        h, w = img.shape
        x0, y0, th = 0.04 * w, 0.95 * h, 0.012 * h
        ax.add_patch(Rectangle((x0 - 6, y0 - th - 56), L + 12, th + 80, color="black", alpha=0.45, lw=0))
        ax.add_patch(Rectangle((x0, y0 - th), L, th, color="white", lw=0))
        ax.text(x0 + L / 2, y0 - th - 14, f"{SCALEBAR_UM} \u00b5m", color="white", ha="center",
                va="bottom", fontsize=11, fontweight="bold")
    fig.tight_layout(h_pad=1.6, w_pad=0.6)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"tomo_4panel_static.{ext}"), dpi=200 if ext == "png" else None)
    plt.close(fig)


# ----------------------------------------------------------------------------
# HTML viewer
# ----------------------------------------------------------------------------
HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Synchrotron micro-CT, four crystals</title>
<style>
  :root { --bg:#0e1013; --panel:#161a20; --line:#272d36; --ink:#e8ebef; --mute:#8b94a1; --accent:#f0a64a; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:14px/1.4 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
  header { max-width:1100px; margin:0 auto; padding:20px 16px 8px; display:flex; flex-wrap:wrap;
           gap:10px 24px; align-items:center; justify-content:flex-end; }
  header h1 { font-size:17px; font-weight:600; margin:0; }
  header p { margin:2px 0 0; color:var(--mute); font-size:12.5px; }
  .global { display:flex; gap:10px; align-items:center; color:var(--mute); font-size:13px; }
  .grid { max-width:1100px; margin:0 auto; padding:8px 16px 28px; display:grid;
          grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }
  @media (max-width:640px){ .grid{ grid-template-columns:1fr; } }
  .panel { background:var(--panel); border:1px solid var(--line); border-radius:10px; overflow:hidden; }
  .ph { padding:10px 12px 8px; display:flex; align-items:baseline; gap:8px; }
  .ph .n { color:var(--accent); font-weight:700; }
  .ph .t { font-weight:600; }
  .ph .m { color:var(--mute); font-size:12.5px; margin-left:auto; }
  .view { position:relative; aspect-ratio:1/1; background:#000; }
  .view img { position:absolute; inset:0; width:100%; height:100%; display:block; image-rendering:auto; }
  .sb { position:absolute; left:4%; bottom:4%; width:__SBPCT__%; color:#fff; font-size:12px; font-weight:600;
        text-shadow:0 0 3px #000,0 0 3px #000; pointer-events:none; text-align:center; white-space:nowrap; }
  .sb .bar { height:4px; width:100%; background:#fff; margin-top:3px; box-shadow:0 0 3px #000; }
  .ctl { display:flex; align-items:center; gap:10px; padding:10px 12px; }
  .ctl input[type=range] { flex:1; accent-color:var(--accent); min-width:0; }
  .ctl .z { color:var(--mute); font-variant-numeric:tabular-nums; font-size:12.5px; min-width:150px; text-align:right; }
  button { background:#232a34; color:var(--ink); border:1px solid #333c49; border-radius:7px; height:32px;
           padding:0 12px; font:inherit; cursor:pointer; }
  button:hover { background:#2c3541; } button:focus-visible, select:focus-visible, input:focus-visible
  { outline:2px solid var(--accent); outline-offset:2px; }
  button.play { width:34px; padding:0; font-size:14px; background:var(--accent); color:#1a1205; border-color:var(--accent); }
  button.play:hover { background:#f6b868; }
  select { background:#232a34; color:var(--ink); border:1px solid #333c49; border-radius:7px; height:32px; padding:0 8px; font:inherit; }
  footer { max-width:1100px; margin:0 auto; padding:0 16px 28px; color:var(--mute); font-size:12px; }
</style>
</head>
<body>
<header>
  <div class="global">
    <button id="all">Play all</button>
    <label>Speed <select id="fps">
      <option value="8">8 fps</option><option value="15" selected>15 fps</option>
      <option value="25">25 fps</option><option value="40">40 fps</option></select></label>
  </div>
</header>
<main class="grid" id="grid"></main>
<footer id="foot"></footer>

<script>
const DATA = __DATA__;
const SCALEBAR_UM = __SCALEBAR__;
const grid = document.getElementById('grid');

const panels = DATA.map((d, i) => {
  const urls = d.frames.map(b64 => {
    const bin = atob(b64), u8 = new Uint8Array(bin.length);
    for (let k = 0; k < bin.length; k++) u8[k] = bin.charCodeAt(k);
    return URL.createObjectURL(new Blob([u8], {type:'image/webp'}));
  });
  d.frames = null;
  const el = document.createElement('section');
  el.className = 'panel';
  el.innerHTML = `
    <div class="ph"><span class="n">${i+1}</span><span class="t">${d.label}</span>
      <span class="m">${d.mineral} \u00b7 ${d.crystal}</span></div>
    <div class="view"><img alt="${d.label} ${d.mineral} tomography slice" src="${urls[Math.floor(urls.length/2)]}">
      <div class="sb"><span>${SCALEBAR_UM} \u00b5m</span><div class="bar"></div></div></div>
    <div class="ctl"><button class="play" aria-label="Play ${d.label}">\u25B6</button>
      <input type="range" min="0" max="${urls.length-1}" value="${Math.floor(urls.length/2)}" aria-label="Slice position">
      <span class="z"></span></div>`;
  grid.appendChild(el);
  const p = { d, urls, el, img: el.querySelector('img'), slider: el.querySelector('input'),
              btn: el.querySelector('.play'), zlab: el.querySelector('.z'),
              idx: Math.floor(urls.length/2), playing: false, acc: 0 };
  const show = k => {
    p.idx = k; p.img.src = p.urls[k]; p.slider.value = k;
    const sl = k * d.step;
    p.zlab.textContent = `slice ${sl} / ${d.n_total-1}  \u00b7  z = ${(sl*d.pixel_um).toFixed(0)} \u00b5m`;
  };
  p.show = show;
  p.setPlaying = on => { p.playing = on; p.btn.textContent = on ? '\u275A\u275A' : '\u25B6'; syncAll(); };
  p.btn.onclick = () => p.setPlaying(!p.playing);
  p.slider.oninput = () => show(+p.slider.value);
  show(p.idx);
  return p;
});

const allBtn = document.getElementById('all'), fpsSel = document.getElementById('fps');
function syncAll() { allBtn.textContent = panels.every(p => p.playing) ? 'Pause all' : 'Play all'; }
allBtn.onclick = () => { const on = !panels.every(p => p.playing); panels.forEach(p => p.setPlaying(on)); };

let last = performance.now();
function tick(now) {
  const dt = (now - last) / 1000; last = now;
  const fps = +fpsSel.value;
  panels.forEach(p => {
    if (!p.playing) return;
    p.acc += dt * fps;
    const adv = Math.floor(p.acc);
    if (adv > 0) { p.acc -= adv; p.show((p.idx + adv) % p.urls.length); }
  });
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

document.getElementById('foot').textContent =
  'Beamline SYRMEP (Elettra), 28 keV, voxel size 0.9 \u00b5m. Grey levels stretched per crystal (' +
  '__LOW__\u2013__HIGH__ percentile of the full stack).';
</script>
</body>
</html>
"""


def build_html(samples, out_dir, out_size, quality, step):
    data = []
    for s in samples:
        n = len(s["files"])
        um_per_px = PIXEL_UM * 2045 / out_size
        data.append(dict(label=s["label"], mineral=s["mineral"], xtal=s["xtal"], crystal=crystal_name(s["xtal"]), frames=s["frames"],
                         step=step, n_total=n, pixel_um=PIXEL_UM, um_per_px=um_per_px, out_size=out_size))
    html = (HTML.replace("__DATA__", json.dumps(data))
                .replace("__SCALEBAR__", str(SCALEBAR_UM))
                .replace("__SBPCT__", f"{SCALEBAR_UM / (PIXEL_UM * 2045) * 100:.3f}")
                .replace("__LOW__", str(LOW_PCT)).replace("__HIGH__", str(HIGH_PCT)))
    path = os.path.join(out_dir, "tomo_4panel_viewer.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, default=3, help="use every Nth slice in the animation")
    ap.add_argument("--out-size", type=int, default=800, help="frame size in px (source is 2045)")
    ap.add_argument("--quality", type=int, default=76, help="WebP quality 1-100")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for s in SAMPLES:
        s["files"] = sorted(glob.glob(os.path.join(s["path"], "slice_*.tif")))
        assert s["files"], f"no slices in {s['path']}"
        s["lims"] = auto_contrast(s["files"])
        print(f"{s['xtal']}: {len(s['files'])} slices, grey limits {s['lims'][0]:.0f}-{s['lims'][1]:.0f}", flush=True)

    static_figure(SAMPLES, args.out_dir)
    print("static figure written", flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        for s in SAMPLES:
            sel = s["files"][::args.step]
            s["frames"] = list(ex.map(lambda p: make_frame(p, *s["lims"], args.out_size, args.quality), sel))
            print(f"{s['xtal']}: {len(s['frames'])} frames, {sum(map(len, s['frames'])) / 1e6:.1f} MB (base64)",
                  flush=True)

    path = build_html(SAMPLES, args.out_dir, args.out_size, args.quality, args.step)
    print("written", path, f"({os.path.getsize(path) / 1e6:.1f} MB)")
    json.dump({s["xtal"]: dict(low=s["lims"][0], high=s["lims"][1], n_slices=len(s["files"]),
                               path=s["path"]) for s in SAMPLES},
              open(os.path.join(args.out_dir, "tomo_4panel_settings.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
