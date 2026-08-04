#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure 1 重绘(v3): 用真实的ESA CCI/C3S 300m土地覆盖分类(2020, LCCS方案)替换此前的裸地比例近似
叠加修正后TK/MG/LZ/BJ框, 延续v2的编辑风格装饰(发光描边/指北针/比例尺/山脉标注)"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap, BoundaryNorm
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from cartopy.feature import ShapelyFeature
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

FIG_DIR = r"F:\研\科研\Python\atmosphere_revision_figs"
SHP = r"D:\shp\SHP\world(WGS1984)\World\world.shp"

mosaic = np.load(fr"{FIG_DIR}\cci_lc_mosaic.npy")  # (lat, lon), uint8 LCCS class codes
lc_lon = np.load(fr"{FIG_DIR}\cci_lc_lon.npy")
lc_lat = np.load(fr"{FIG_DIR}\cci_lc_lat.npy")

# --- ESA CCI/C3S LCCS class code -> 简化类别 映射 ---
CLASS_MAP = {
    10: "Cropland", 11: "Cropland", 12: "Cropland", 20: "Cropland", 30: "Cropland", 40: "Cropland",
    50: "Forest", 60: "Forest", 61: "Forest", 70: "Forest", 80: "Forest", 90: "Forest", 100: "Forest",
    110: "Grassland", 130: "Grassland", 140: "Grassland",
    120: "Shrubland", 121: "Shrubland", 122: "Shrubland",
    160: "Wetland", 170: "Wetland", 180: "Wetland",
    150: "Bare land", 152: "Bare land", 200: "Bare land", 201: "Bare land", 202: "Bare land",
    190: "Artificial surface",
    210: "Water",
    220: "Snow/ice",
}
CATEGORIES = ["Cropland", "Forest", "Grassland", "Shrubland", "Wetland",
              "Bare land", "Water", "Snow/ice", "Artificial surface"]
COLORS = {
    "Cropland": "#e6d8a0",
    "Forest": "#2f6b3a",
    "Grassland": "#9bcf6a",
    "Shrubland": "#c9ab5c",
    "Wetland": "#6fa9c9",
    "Bare land": "#d99a4e",
    "Water": "#5f9bc9",
    "Snow/ice": "#eef3f5",
    "Artificial surface": "#b72e2e",
}

lut = np.zeros(256, dtype=np.uint8)  # class code -> category index (0..8), 未映射(如nodata)保留0
for code, cat in CLASS_MAP.items():
    lut[code] = CATEGORIES.index(cat)
cat_grid = lut[mosaic]  # (lat, lon) int index into CATEGORIES

cmap = ListedColormap([COLORS[c] for c in CATEGORIES])
bounds = np.arange(len(CATEGORIES) + 1) - 0.5
norm = BoundaryNorm(bounds, cmap.N)

TK_BOX = (76.0, 37.0, 90.0, 41.5)
MG_BOX = (97.0, 41.0, 116.0, 49.5)
LZ_BOX = (103.43, 35.65, 104.23, 36.45)
BJ_BOX = (115.99, 39.51, 116.79, 40.31)
ALL_BOXES = {"TK": (TK_BOX, "#b3272d"), "MG": (MG_BOX, "#b3272d"),
             "LZ (Lanzhou)": (LZ_BOX, "#1b4f8a"), "BJ (Beijing)": (BJ_BOX, "#1b4f8a")}
CITIES = {"Lanzhou": (103.83, 36.05), "Beijing": (116.39, 39.91)}
YINSHAN = (112.0, 41.0)
HELANSHAN = (105.9, 38.8)


def load_border_feature(pc_crs):
    geoms = list(shpreader.Reader(SHP).geometries())
    return ShapelyFeature(geoms, pc_crs, facecolor="none", edgecolor="#3a3a38", linewidth=0.5)


proj = ccrs.PlateCarree()
fig = plt.figure(figsize=(13.5, 8.6))
ax = fig.add_axes([0.06, 0.07, 0.88, 0.85], projection=proj)

extent = [68, 135, 20, 55]
ax.set_extent(extent, crs=proj)

im = ax.pcolormesh(lc_lon, lc_lat, cat_grid, cmap=cmap, norm=norm,
                    transform=proj, shading="nearest", zorder=1, rasterized=True)

ax.add_feature(load_border_feature(proj), zorder=3)

for name, (mx, my) in [("Yin Mts", YINSHAN), ("Helan Mts", HELANSHAN)]:
    ax.plot(mx, my, marker="^", color="#3a2f1c", markersize=8, markeredgecolor="white",
             markeredgewidth=0.8, transform=proj, zorder=9)
    txt = ax.text(mx, my - 1.0, name, fontsize=8, transform=proj, zorder=9,
             ha="center", va="top", color="#2a2114", fontstyle="italic")
    txt.set_path_effects([pe.withStroke(linewidth=2.2, foreground="white")])


def draw_box(bx, color, lw=2.6):
    x0, y0, x1, y1 = bx
    shadow, = ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
                        color="black", linewidth=lw + 2.2, alpha=0.20, transform=proj, zorder=6,
                        solid_capstyle="round", solid_joinstyle="round")
    line, = ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
                     color=color, linewidth=lw, transform=proj, zorder=7,
                     solid_capstyle="round", solid_joinstyle="round")
    line.set_path_effects([pe.withStroke(linewidth=lw + 2.0, foreground="white")])


for name, (bx, color) in ALL_BOXES.items():
    lw = 3.4 if name.startswith(("LZ", "BJ")) else 2.6  # 城市尺度小框加粗线宽, 弥补几何尺寸过小的可见度问题
    draw_box(bx, color, lw=lw)

for cname, (cx, cy) in CITIES.items():
    txt = ax.text(cx + 0.9, cy - 0.1, cname, fontsize=11, transform=proj, zorder=8,
             ha="left", va="center", color="#1b1b1a", fontweight="bold", fontfamily="sans-serif")
    txt.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])

for name, (bx, color) in [("TK", ALL_BOXES["TK"]), ("MG", ALL_BOXES["MG"])]:
    cx = (bx[0] + bx[2]) / 2
    cy = bx[3] + 1.0
    txt = ax.text(cx, cy, name, fontsize=15, fontweight="bold", color=color,
             transform=proj, ha="center", zorder=8, fontfamily="sans-serif")
    txt.set_path_effects([pe.withStroke(linewidth=3.5, foreground="white")])

gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="white", alpha=0.6, linestyle="-")
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {"size": 9.5, "color": "#2b2b28"}
gl.ylabel_style = {"size": 9.5, "color": "#2b2b28"}

for spine in ax.spines.values():
    spine.set_edgecolor("#2b2b28")
    spine.set_linewidth(1.1)

# --- 指北针(放在地图区域内部, 右上角, 白色描边替代大背景框) ---
compass_ax_x, compass_ax_y = 0.965, 0.90  # ax.transAxes坐标(地图内部)
arrow = ax.annotate("", xy=(compass_ax_x, compass_ax_y + 0.03), xytext=(compass_ax_x, compass_ax_y - 0.03),
            xycoords=ax.transAxes, textcoords=ax.transAxes, zorder=11,
            arrowprops=dict(arrowstyle="-|>", color="#2b2b28", linewidth=1.8, mutation_scale=13))
arrow.arrow_patch.set_path_effects([pe.withStroke(linewidth=3.2, foreground="white")])
n_txt = ax.text(compass_ax_x, compass_ax_y + 0.045, "N", transform=ax.transAxes, ha="center", va="bottom",
         fontsize=11, fontweight="bold", color="#2b2b28", zorder=11)
n_txt.set_path_effects([pe.withStroke(linewidth=2.8, foreground="white")])

# --- 比例尺(简洁工字形样式: 主线+两端短竖线, 单一居中标注, 按35N纬线附近换算) ---
import numpy as _np

lat_ref = 35.0
km_per_deg_lon = 111.32 * _np.cos(_np.radians(lat_ref))
target_km = 1000.0
frac_width = (target_km / km_per_deg_lon) / (extent[1] - extent[0])

sx0, sy0 = 0.83, 0.04
tick_h = 0.012

main_bar, = ax.plot([sx0, sx0 + frac_width], [sy0, sy0], transform=ax.transAxes,
                     color="#2b2b28", linewidth=1.6, zorder=11, solid_capstyle="butt")
main_bar.set_path_effects([pe.withStroke(linewidth=3.4, foreground="white")])
for x_end in (sx0, sx0 + frac_width):
    tick, = ax.plot([x_end, x_end], [sy0 - tick_h / 2, sy0 + tick_h / 2], transform=ax.transAxes,
                     color="#2b2b28", linewidth=1.6, zorder=11, solid_capstyle="butt")
    tick.set_path_effects([pe.withStroke(linewidth=3.0, foreground="white")])

sb_txt = ax.text(sx0 + frac_width / 2, sy0 + 0.018, f"{target_km:.0f} km", transform=ax.transAxes,
                   ha="center", va="bottom", fontsize=8.5, fontweight="medium", color="#2b2b28", zorder=11)
sb_txt.set_path_effects([pe.withStroke(linewidth=2.6, foreground="white")])

fig.text(0.06, 0.955, "East Asian dust source and receptor regions",
          fontsize=17, fontweight="bold", color="#1b1b1a", fontfamily="sans-serif")
fig.text(0.06, 0.925, "Corrected TK / MG source boxes and city-scale LZ / BJ receptor boxes, "
                       "over ESA CCI/C3S land cover (300 m, 2020)",
          fontsize=10.5, color="#5a5a54", fontstyle="italic")

legend_handles = [Patch(facecolor=COLORS[c], edgecolor="#3a3a38", linewidth=0.4, label=c) for c in CATEGORIES]
legend_handles += [
    Line2D([0], [0], color="#b3272d", lw=2.6, label="TK / MG source box (corrected)"),
    Line2D([0], [0], color="#1b4f8a", lw=3.4, label="LZ (Lanzhou) / BJ (Beijing) receptor box (city-scale)"),
    Line2D([0], [0], marker="^", color="#3a2f1c", markeredgecolor="white", linestyle="None",
           markersize=7, label="Reference mountain range"),
]
leg = ax.legend(handles=legend_handles, loc="lower left", fontsize=8.2, frameon=True,
                  facecolor="white", framealpha=0.94, edgecolor="#c9c8c2", ncol=1,
                  borderpad=0.8, labelspacing=0.55)
leg.set_zorder(10)

out_path = fr"{FIG_DIR}\figure1_landcover_regions_v3.png"
plt.savefig(out_path, dpi=230, facecolor="white")
print("saved:", out_path)
