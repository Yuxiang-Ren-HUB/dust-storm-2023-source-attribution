#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v2 四框 + 逐日起沙热点叠加图, 三种配色风格供选。

相对 plot_final_envelopes_v2.py 的修正:
  1) 补上北京受体框(原脚本只画了兰州, 图例也只有 LZ)
  2) 底图对比度: 原版把热点直接叠在饱和的地表覆盖上, 红/绿/橙热点和
     地表的绿/橙/棕混在一起分不开; 本版提供三种处理
  3) 框色与热点色分离, 受体框加白色描边确保任何底色上都可见

框(v2 方案, 已与 plot_figure1_final_four_box_v2.py 逐字核对):
  TK (77-90E, 37-42N) / TH (90-94E, 39-45N) / AH (94-103E, 39.5-46.5N) / MG (103-118E, 41-49N)
  台阶式拼接, 六对组合均无重叠, 共边于 90E / 94E / 103E。
"""
import os

SCRATCH = r"C:\Users\Admin\AppData\Local\Temp\claude\f-------Python\db0fd94d-49e5-4aac-afd9-7a176bb37743\scratchpad"
os.environ["ECCODES_DIR"] = os.path.join(SCRATCH, "eccodes_home")
os.environ["PATH"] = os.path.join(SCRATCH, "eccodes_home", "lib") + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import xarray as xr
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap, BoundaryNorm
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from cartopy.feature import ShapelyFeature

SHP = r"D:\shp\SHP\world(WGS1984)\World\world.shp"
FIG_DIR = r"F:\研\科研\Python\atmosphere_revision_figs\AH_investigation"

DAYS = ["2023-03-19", "2023-03-20", "2023-03-21", "2023-03-22"]
DAY_COLORS = {"2023-03-19": "#d81159", "2023-03-20": "#00a878",
              "2023-03-21": "#2e6fdf", "2023-03-22": "#f79824"}

# 正文以两个大源区(系统)组织, 因此框与标注按系统统一配色, 不再逐框区分:
#   System ① = TK + TH (西路, 塔克拉玛干-吐哈)   System ② = AH + MG (北路, 阿拉善河西-蒙古高原)
# 取色直接沿用 plot_figure1_final_four_box_v2.py 的 TK 橙(#c0742d)与 MG 蓝(#1b3a6b),
# 与该图保持一致, 便于论文里两张图并排时读者建立对应
SYS_COLORS = {"1": "#c0742d", "2": "#1b3a6b"}
SYS_LABEL = {"1": "System ① (TK + TH)", "2": "System ② (AH + MG)"}
BOXES = {"TK": ((77.0, 37.0, 90.0, 42.0), SYS_COLORS["1"]),
         "TH": ((90.0, 39.0, 94.0, 45.0), SYS_COLORS["1"]),
         "AH": ((94.0, 39.5, 103.0, 46.5), SYS_COLORS["2"]),
         "MG": ((103.0, 41.0, 118.0, 49.0), SYS_COLORS["2"])}
RECEPTORS = {"Lanzhou": (103.43, 35.65, 104.23, 36.45),
             "Beijing": (115.99, 39.51, 116.79, 40.31)}

LC_GROUP = {10: "Cropland", 11: "Cropland", 12: "Cropland", 20: "Cropland", 30: "Cropland", 40: "Cropland",
            50: "Forest", 60: "Forest", 61: "Forest", 62: "Forest", 70: "Forest", 71: "Forest", 72: "Forest",
            80: "Forest", 81: "Forest", 82: "Forest", 90: "Forest", 100: "Forest",
            110: "Grassland", 130: "Grassland", 140: "Grassland",
            120: "Shrubland", 121: "Shrubland", 122: "Shrubland",
            160: "Wetland", 170: "Wetland", 180: "Wetland",
            150: "Bare land", 151: "Bare land", 152: "Bare land", 153: "Bare land",
            200: "Bare land", 201: "Bare land", 202: "Bare land",
            190: "Artificial", 210: "Water", 220: "Snow/ice"}
CATS = ["Cropland", "Forest", "Grassland", "Shrubland", "Wetland", "Bare land", "Water", "Snow/ice", "Artificial"]
LC_COLORS = {"Cropland": "#e6d8a0", "Forest": "#2f6b3a", "Grassland": "#9bcf6a", "Shrubland": "#c9ab5c",
             "Wetland": "#7fb2d9", "Bare land": "#d99b62", "Water": "#5b9bd5", "Snow/ice": "#f0f0f0",
             "Artificial": "#b03a2e"}

# ---------------- 逐日起沙热点 ----------------
u10 = xr.open_dataset(r"D:\ERA5_WRF\era5_sl_hourly_wind_20230318_20230322.grib", engine="cfgrib",
                      filter_by_keys={"shortName": "10u"})["u10"]
v10 = xr.open_dataset(r"D:\ERA5_WRF\era5_sl_hourly_wind_20230318_20230322.grib", engine="cfgrib",
                      filter_by_keys={"shortName": "10v"})["v10"]
ws = np.sqrt(u10 ** 2 + v10 ** 2)
lon = ws.longitude.values; lat = ws.latitude.values
lat_flip = lat[0] > lat[-1]
if lat_flip:
    lat = lat[::-1]
lon2d, lat2d = np.meshgrid(lon, lat)
domain = (lon2d >= 70) & (lon2d <= 125) & (lat2d >= 25) & (lat2d <= 52)

mosaic = np.load(r"F:\研\科研\Python\atmosphere_revision_figs\cci_lc_mosaic.npy")
lc_lon = np.load(r"F:\研\科研\Python\atmosphere_revision_figs\cci_lc_lon.npy")
lc_lat = np.load(r"F:\研\科研\Python\atmosphere_revision_figs\cci_lc_lat.npy")
is_bare = np.isin(mosaic, [150, 152, 200, 201, 202]).astype(float)
lc_lon2d, lc_lat2d = np.meshgrid(lc_lon, lc_lat)
lon_e = np.concatenate([[lon[0]-(lon[1]-lon[0])/2], (lon[:-1]+lon[1:])/2, [lon[-1]+(lon[-1]-lon[-2])/2]])
lat_e = np.concatenate([[lat[0]-(lat[1]-lat[0])/2], (lat[:-1]+lat[1:])/2, [lat[-1]+(lat[-1]-lat[-2])/2]])
Hb, _, _ = np.histogram2d(lc_lon2d.ravel(), lc_lat2d.ravel(), bins=[lon_e, lat_e], weights=is_bare.ravel())
Ht, _, _ = np.histogram2d(lc_lon2d.ravel(), lc_lat2d.ravel(), bins=[lon_e, lat_e])
bare_mask = (np.divide(Hb, Ht, out=np.zeros_like(Hb), where=Ht > 0).T) >= 0.5

day_masks = {}
for day in DAYS:
    t0 = np.datetime64(f"{day}T00:00"); t1 = np.datetime64(f"{day}T23:00")
    dm = (ws.time.values >= t0) & (ws.time.values <= t1)
    awep = (np.clip(ws.values[dm] - 8.0, 0, None) ** 3).sum(axis=0)
    if lat_flip:
        awep = awep[::-1, :]
    thr = np.percentile(awep[domain][awep[domain] > 0], 90)
    m = (awep >= thr) & bare_mask & domain
    lab, n = ndimage.label(m, structure=np.ones((3, 3)))
    sizes = ndimage.sum(m, lab, range(1, n + 1)) if n > 0 else np.array([])
    day_masks[day] = np.isin(lab, [i + 1 for i, s in enumerate(sizes) if s >= 10])

lut = np.zeros(int(mosaic.max()) + 1, dtype=int)
for code, grp in LC_GROUP.items():
    if code <= mosaic.max():
        lut[code] = CATS.index(grp)
cat_grid = lut[mosaic]
geoms = list(shpreader.Reader(SHP).geometries())

STYLES = {
    "A_greybase": dict(
        title="Style A — plain grey base (maximum hotspot contrast)",
        bg="grey", hotspot_alpha=1.0, box_lw=3.0, border="#3a3a38"),
    "B_greyland": dict(
        title="Style B — greyscale land cover (keeps terrain context)",
        bg="greyland", hotspot_alpha=1.0, box_lw=3.0, border="#2a2a28"),
    "C_mutedland": dict(
        title="Style C — desaturated land cover (keeps land-type colours, muted)",
        bg="mutedland", hotspot_alpha=1.0, box_lw=3.2, border="#2a2a28"),
}

for key, st in STYLES.items():
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(16, 8.2), subplot_kw={"projection": proj}, constrained_layout=True)

    if st["bg"] == "grey":
        ax.set_facecolor("#dcdcdc")
    else:
        if st["bg"] == "greyland":
            cols = []
            for c in CATS:
                r, g, b = mcolors.to_rgb(LC_COLORS[c])
                y = 0.299 * r + 0.587 * g + 0.114 * b
                y = 0.62 + 0.34 * y                       # 压到亮灰区间, 给热点让出对比
                cols.append((y, y, y))
        else:                                             # mutedland: 向白色混合, 保留色相
            cols = [tuple(1 - 0.34 * (1 - np.array(mcolors.to_rgb(LC_COLORS[c])))) for c in CATS]
        cmap = ListedColormap(cols)
        norm = BoundaryNorm(np.arange(-0.5, len(CATS) + 0.5), cmap.N)
        ax.pcolormesh(lc_lon, lc_lat, cat_grid, cmap=cmap, norm=norm, transform=proj,
                      shading="auto", zorder=1, rasterized=True)

    ax.add_feature(ShapelyFeature(geoms, proj, facecolor="none", edgecolor=st["border"], linewidth=0.6), zorder=5)

    for day in DAYS:
        ax.pcolormesh(lon, lat, np.ma.masked_less(day_masks[day].astype(float), 0.5),
                      cmap=ListedColormap([DAY_COLORS[day]]), transform=proj,
                      shading="auto", zorder=3, alpha=st["hotspot_alpha"])

    for name, (bx, color) in BOXES.items():
        x0, y0, x1, y1 = bx
        ln, = ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=color,
                      linewidth=st["box_lw"], transform=proj, zorder=7)
        ln.set_path_effects([pe.withStroke(linewidth=st["box_lw"] + 2.2, foreground="white")])
        ly = y1 - 0.55 if name in ("TK", "TH") else (y0 + y1) / 2
        va = "top" if name in ("TK", "TH") else "center"
        t = ax.text((x0 + x1) / 2, ly, name, color=color, fontsize=15, fontweight="bold",
                    ha="center", va=va, transform=proj, zorder=9)
        t.set_path_effects([pe.withStroke(linewidth=3.4, foreground="white")])

    for name, bx in RECEPTORS.items():
        x0, y0, x1, y1 = bx
        ln, = ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color="#111111",
                      linewidth=2.6, transform=proj, zorder=8)
        ln.set_path_effects([pe.withStroke(linewidth=4.6, foreground="white")])
        t = ax.text(x1 + 0.6, (y0 + y1) / 2, name, color="#111111", fontsize=13, fontweight="bold",
                    ha="left", va="center", transform=proj, zorder=9)
        t.set_path_effects([pe.withStroke(linewidth=3.4, foreground="white")])

    ax.set_extent([70, 124, 33, 51], crs=proj)
    ax.set_title("East Asian dust source regions (v2, AH/MG split at 103°E) vs. day-coloured wind-erosion hotspots\n"
                 "TK 77–90°E,37–42°N  |  TH 90–94°E,39–45°N  |  AH 94–103°E,39.5–46.5°N  |  MG 103–118°E,41–49°N",
                 fontsize=13.5, fontweight="bold", pad=10)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=DAY_COLORS[d], label=d) for d in DAYS]
    handles += [plt.Line2D([0], [0], color=SYS_COLORS[s], lw=3.0, label=SYS_LABEL[s]) for s in ("1", "2")]
    handles += [plt.Line2D([0], [0], color="#111111", lw=2.6, label="LZ / BJ receptor box")]
    ax.legend(handles=handles, loc="lower left", fontsize=10, framealpha=0.94, ncol=1)

    out = fr"{FIG_DIR}\source_boxes_hotspots_v3_{key}.png"
    fig.savefig(out, dpi=200, facecolor="white", bbox_inches="tight")
    print(f"saved [{st['title']}]: {out}")
    plt.close(fig)
