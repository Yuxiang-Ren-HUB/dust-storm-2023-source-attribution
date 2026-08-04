#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在六宫格PM10图(图2.png)上叠加客观圈定的两个不规则源区轮廓(TK橙/MG蓝虚线,
加粗)+ 贺兰山/阴山标注。

v3修订: v2版假设六个子图严格等间距排布(整图宽高直接三等分/二等分), 结果
发现子图实际间距并不均匀(逐像素检测发现同一列不同行、不同列的边框位置都有
数像素到近百像素的漂移), 导致v2版的框线在不同子图上对不齐/漂移。
v3改为逐子图独立标定: 对每个子图分别用像素级边框检测(扫描"图框灰色细线"
的行/列)找到该子图的绘图区边框, 再在边框外侧扫描坐标轴刻度数字文本块的
质心位置, 对每个子图独立做经度/纬度的线性回归(像素<->度), 完全不假设子图
之间等距。
"""
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import label
from skimage import measure
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, r"F:\研\科研\Python")

SRC = Path(r"F:\研\科研\Python\atmosphere_revision_figs\用到的图\图2.png")
OUT = Path(r"F:\研\科研\Python\atmosphere_revision_figs\用到的图\图2_annotated_v3.png")
MASK_PATH = Path(r"F:\研\科研\Python\atmosphere_revision_figs\reviewer_response\source_region_mask.npy")

NX, NY = 1400, 700
LON_1D = np.linspace(70, 126, NX)
LAT_1D = np.linspace(52, 30, NY)

TK_COLOR = "#c0742d"
MG_COLOR = "#1b3a6b"
SPLIT_LON = 102.5

PEAKS = [
    (105.93, 38.80, "Helan Shan", "#c0392b"),
    (112.0, 41.3, "Yin Shan", "#1f6fb2"),
]

LON_TICKS = [80, 90, 100, 110, 120, 130]
LAT_TICKS = [55, 50, 45, 40, 35, 30, 25, 20]

# rough non-overlapping search windows per panel (col,row), refined from manual inspection
ROUGH = {
    (0, 0): (140, 1050, 20, 650),
    (1, 0): (1100, 2000, 20, 650),
    (2, 0): (2060, 2960, 20, 650),
    (0, 1): (140, 1050, 740, 1330),
    (1, 1): (1100, 2000, 740, 1330),
    (2, 1): (2060, 2960, 740, 1330),
}


def groups_from(mask1d, offset, gap=3):
    idx = np.where(mask1d)[0]
    if len(idx) == 0:
        return []
    groups = []
    cur = [idx[0]]
    for c in idx[1:]:
        if c - cur[-1] <= gap:
            cur.append(c)
        else:
            groups.append((cur[0] + offset, cur[-1] + offset))
            cur = [c]
    groups.append((cur[0] + offset, cur[-1] + offset))
    return groups


def detect_frame(gray, x0, x1, y0, y1, thr=0.7):
    is_frame = (gray > 140) & (gray < 235)
    sub = is_frame[y0:y1, x0:x1]
    cg = groups_from(sub.mean(axis=0) > thr, x0)
    rg = groups_from(sub.mean(axis=1) > thr, y0)
    return cg[0][0], rg[0][0], cg[-1][1], rg[-1][1]   # L, T, R, B


def detect_ticks(gray, L, T, R, B):
    # x ticks: text block centroids in the strip just below the bottom border
    strip = gray[B + 4:B + 42, L - 20:R + 20]
    dark = strip < 140
    col_count = dark.sum(axis=0)
    xg = groups_from(col_count > 0, L - 20, gap=22)
    xg = [g for g in xg if g[1] - g[0] > 5]  # drop 1px noise
    x_centers = [ (g[0]+g[1])/2 for g in xg ]

    # y ticks: text block centroids in the strip just left of the left border,
    # restricted to a narrow column right next to the axis (avoid the rotated
    # "Latitude" title further left)
    strip = gray[T - 10:B + 10, L - 42:L - 2]
    dark = strip < 140
    row_count = dark.sum(axis=1)
    yg = groups_from(row_count > 0, T - 10, gap=16)
    yg = [g for g in yg if g[1] - g[0] > 5]
    y_centers = [ (g[0]+g[1])/2 for g in yg ]
    return x_centers, y_centers


def fit_axis(centers, ticks):
    centers = np.array(centers, dtype=float)
    ticks = np.array(ticks, dtype=float)
    n = min(len(centers), len(ticks))
    if len(centers) != len(ticks):
        # keep the n most central detections if counts mismatch (edgeartifacts)
        centers = centers[:n]
        ticks = ticks[:n]
    A = np.polyfit(ticks, centers, 1)   # pixel = a*value + b
    return A  # (slope, intercept)


def calibrate_all():
    img = Image.open(SRC).convert("RGB")
    arr = np.array(img)
    gray = arr.mean(axis=2)
    calib = {}
    for (col, row), (x0, x1, y0, y1) in ROUGH.items():
        L, T, R, B = detect_frame(gray, x0, x1, y0, y1)
        xc, yc = detect_ticks(gray, L, T, R, B)
        print(f"panel col={col} row={row}: frame=({L},{T},{R},{B}) "
              f"n_xticks={len(xc)} n_yticks={len(yc)}")
        Ax = fit_axis(xc, LON_TICKS[:len(xc)] if len(xc) <= len(LON_TICKS) else LON_TICKS)
        Ay = fit_axis(yc, LAT_TICKS[:len(yc)] if len(yc) <= len(LAT_TICKS) else LAT_TICKS)
        calib[(col, row)] = dict(frame=(L, T, R, B), Ax=Ax, Ay=Ay)
    return img, calib


def to_px(lon, lat, cal):
    ax = cal["Ax"]; ay = cal["Ay"]
    return ax[0] * lon + ax[1], ay[0] * lat + ay[1]


def extract_contours():
    mask = np.load(MASK_PATH)
    lab, n = label(mask)
    contours = []
    labels_out = []
    for i in range(1, n + 1):
        m = lab == i
        ys, xs = np.where(m)
        lon_c = LON_1D[xs].mean()
        color = TK_COLOR if lon_c < SPLIT_LON else MG_COLOR
        name = "TK" if lon_c < SPLIT_LON else "MG"
        for cnt in measure.find_contours(m.astype(float), 0.5):
            rows, cols = cnt[:, 0], cnt[:, 1]
            lons = np.interp(cols, np.arange(NX), LON_1D)
            lats = np.interp(rows, np.arange(NY), LAT_1D)
            if len(lons) > 20:
                contours.append((lons, lats, color))
        # anchor label at the mask pixel closest to the centroid (guaranteed inside the blob)
        lat_c = LAT_1D[ys].mean()
        d2 = (LON_1D[xs] - lon_c) ** 2 + (LAT_1D[ys] - lat_c) ** 2
        j = np.argmin(d2)
        labels_out.append((LON_1D[xs[j]], LAT_1D[ys[j]], name, color))
    return contours, labels_out


def draw_dashed_line(draw, pts, fill, width, dash, gap):
    if len(pts) < 2:
        return
    pts = np.asarray(pts)
    seglen = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    cum = np.concatenate([[0], np.cumsum(seglen)])
    total = cum[-1]
    if total == 0:
        return
    period = dash + gap
    d = 0.0
    while d < total:
        d_end = min(d + dash, total)
        p0 = _interp_along(pts, cum, d)
        p1 = _interp_along(pts, cum, d_end)
        draw.line([p0, p1], fill=fill, width=width, joint="curve")
        d += period


def _interp_along(pts, cum, d):
    i = np.searchsorted(cum, d) - 1
    i = max(0, min(i, len(pts) - 2))
    t = 0.0 if cum[i + 1] == cum[i] else (d - cum[i]) / (cum[i + 1] - cum[i])
    x = pts[i, 0] + t * (pts[i + 1, 0] - pts[i, 0])
    y = pts[i, 1] + t * (pts[i + 1, 1] - pts[i, 1])
    return (x, y)


def draw_solid_line(draw, pts, fill, width):
    if len(pts) < 2:
        return
    draw.line(pts, fill=fill, width=width, joint="curve")


def main():
    img, calib = calibrate_all()
    draw = ImageDraw.Draw(img, "RGBA")
    try:
        font = ImageFont.truetype("arial.ttf", 26)
        font_sm = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font = font_sm = ImageFont.load_default()

    contours, region_labels = extract_contours()
    print(f"extracted {len(contours)} contour path(s)")

    for (col, row), cal in calib.items():
        L, T, R, B = cal["frame"]

        for lons, lats, color in contours:
            pts = [to_px(lo, la, cal) for lo, la in zip(lons, lats)]
            pts = [(x, y) for x, y in pts if L - 40 <= x <= R + 40 and T - 40 <= y <= B + 40]
            if len(pts) > 1:
                draw_solid_line(draw, pts, fill=color, width=6)

        if row == 0 and col == 0:
            for lon0, lat0, name, color in region_labels:
                px, py = to_px(lon0, lat0, cal)
                w = draw.textlength(name, font=font)
                draw.text((px - w / 2, py - 14), name, fill=color, font=font)

        for lon0, lat0, name, color in PEAKS:
            px, py = to_px(lon0, lat0, cal)
            if L <= px <= R and T <= py <= B:
                r = 9
                draw.polygon([(px, py - r), (px - r, py + r * 0.8), (px + r, py + r * 0.8)],
                            fill=color, outline="white")
                if row == 0 and col == 0:
                    draw.text((px + 11, py - 11), name, fill=color, font=font_sm)

    img.save(OUT)
    print("saved:", OUT)


if __name__ == "__main__":
    main()
