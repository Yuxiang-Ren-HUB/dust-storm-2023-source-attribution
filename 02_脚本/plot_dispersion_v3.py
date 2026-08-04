#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v3 扩散场绘图 —— 逐日分面板, 配色/排版严格对齐 plot_final_regions_v2.py。

风格来源即那张 "MG forward-run dispersion by day after release" 参考图:
  色标      cividis, LogNorm, vmin = vmax * 1e-3, 低值 mask 留白
  当前源    朱红 #d1373f, lw 2.6, 白描边
  其余源    深靛蓝 #2f5597, lw 1.7, 白描边
  底图      白底, 岸线/国界 #7a7a74 lw 0.5, 不填陆地
  版式      2 x 3, extent [70,140,15,55], 底部横色条 + 两项图例
  色条标注  Time-integrated sensitivity (arb. units)

与参考脚本一致的两处口径(有意保持一致, 便于新旧图并排):
  1. 经纬度取自 header_d01.nc 的 XLONG/XLAT, 不按 OUTGRID 参数硬算(避免半格偏移)
  2. 对除最后两维(south_north, west_east)以外的所有维求和, 不做层厚加权 ——
     故量纲是"各层之和", 与参考图的 sensitivity 口径相同, 不是严格的柱负荷。

与参考图的唯一区别: 源区用客观连通区的轮廓, 不是矩形框。
"""
import sys
import numpy as np
from scipy.ndimage import gaussian_filter
import xarray as xr
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from cartopy.feature import ShapelyFeature
from matplotlib.lines import Line2D
sys.path.insert(0, r"F:\研\科研\Python")
from source_regions import RELEASE_WINDOW3 as WIN

RECEPTOR_BOXES = {"Lanzhou": (103.43, 35.65, 104.23, 36.45),
                  "Beijing": (115.99, 39.51, 116.79, 40.31)}

SHP = r"F:\研\兴趣\旧电脑的工作\world(WGS1984)\World\world.shp"
# netCDF4/HDF5 在 Windows 上打不开含非 ASCII 的路径, FLEXPART 输出单独放 ASCII 目录
RUN = Path(r"F:\FLEXPART_v3_out")
OUT = Path(r"F:\研\科研\Python") / "atmosphere_revision_figs" / "用到的图"

# 源区配色与地表覆盖总图(figure1_two_region_with_stations.png)统一:
# TK 连通区整体一个颜色, 不按 90E 拆开画 —— 拆分只用于溯源归属, 不必反映在图上。
REGION_COLOR = {"TK": "#c0742d", "MG": "#1b3a6b"}
BORDER_COLOR = "#7a7a74"
HALO = [pe.withStroke(linewidth=3.2, foreground="white")]
HALO_ACCENT = [pe.withStroke(linewidth=3.6, foreground="white")]
BINS = [(i, i + 1) for i in range(6)]
NAME = {"TK1": "Tarim Basin", "TK2": "Hexi Corridor", "MG": "Mongolian Gobi",
        "TK": "Tarim Basin + Hexi Corridor, System 1"}


def load_border_feature(pc):
    return ShapelyFeature(list(shpreader.Reader(SHP).geometries()), pc,
                          facecolor="none", edgecolor=BORDER_COLOR, linewidth=0.5)


def load_binned(out_dir, sim_start, bins):
    h = xr.open_dataset(out_dir / "header_d01.nc")
    lon, lat = h["XLONG"].values, h["XLAT"].values
    h.close()
    binned = {b: None for b in bins}
    for f in sorted(out_dir.glob("flxout_d01_*.nc")):
        ds = xr.open_dataset(f)
        times_raw = ds["Times"].values
        conc = ds["CONC"].values
        for ti in range(conc.shape[0]):
            v = times_raw[ti]
            t_str = (v.decode() if isinstance(v, bytes) else v if isinstance(v, str)
                     else "".join(x.decode() if isinstance(x, bytes) else str(x) for x in v)).strip()
            t_val = datetime.strptime(t_str, "%Y%m%d_%H%M%S")
            lag = (t_val - sim_start).total_seconds() / 86400.0
            for b in bins:
                if b[0] <= lag < b[1]:
                    ff = conc[ti]
                    frame = ff.sum(axis=tuple(range(ff.ndim - 2)))
                    binned[b] = frame if binned[b] is None else binned[b] + frame
                    break
        ds.close()
    return lon, lat, binned


def plot(key, lon, lat, binned, bins, title, out_path, label_fn):
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 15,
                         "axes.titlesize": 17, "axes.titleweight": "normal"})
    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.2), subplot_kw={"projection": proj},
                             constrained_layout=True)
    axes = axes.ravel()
    valid = [binned[b] for b in bins if binned[b] is not None and binned[b].max() > 0]
    vmax = max(f.max() for f in valid)
    vmin = vmax * 1e-3   # 放回默认量级, 让暖色(黄)区域更舒展, 视觉上更饱满
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)

    # 直接用两个原始连通片(未按 90E 拆分), 与地表覆盖总图一致
    from scipy.ndimage import label as _lab
    RR = Path(r"F:\研\科研\Python") / "atmosphere_revision_figs" / "reviewer_response"
    NX, NY = 1400, 700
    RLON, RLAT = np.meshgrid(np.linspace(70, 126, NX), np.linspace(52, 30, NY))
    _m = np.load(RR / "source_region_mask.npy")
    _l, _n = _lab(_m)
    _c = sorted(range(1, _n + 1), key=lambda i: RLON[_l == i].mean())
    REG = {"TK": _l == _c[0], "MG": _l == _c[1]}
    ACTIVE = "MG" if key == "MG" else "TK"       # TK1/TK2 都归 TK 这一圈

    # 本次实际用到的释放块(1 度块的并集轮廓)。必须画出来: TK 轮廓是整个连通区
    # (80-102E), 而 TK1 只在 80-90E 释放、TK2 只在 90-102E 释放, 不标出来会让人
    # 以为释放范围与源区不符。
    import pandas as _pd
    _rel = _pd.read_csv(Path(r"F:\研\科研\Python") / "atmosphere_revision_figs" /
                        "AH_investigation" / "releases_v3_obsmass_win_TK1_TK2_MG.csv")
    _members = ["TK1", "TK2"] if key == "TK" else [key]
    _rs = _rel[_rel.source.isin(_members)].drop_duplicates(["lon0", "lat0"])
    _relmask = np.zeros_like(RLON, dtype=bool)
    for _, _r in _rs.iterrows():
        _relmask |= ((RLON >= _r.lon0) & (RLON < _r.lon1) &
                     (RLAT >= _r.lat0) & (RLAT < _r.lat1))

    im = None
    for ax, b in zip(axes, bins):
        field = binned[b]
        ax.add_feature(load_border_feature(proj), zorder=4)
        if field is not None and field.max() > 0:
            # 轻度高斯平滑(sigma~1格, 约27km, 小于羽流本身尺度), 只为消掉逐时
            # 采样在快速移动羽流中留下的窄条空隙, 不改变整体空间分布/量级判读。
            # 已核实: 该空隙在多个逐时快照里都存在、源区掩膜本身无缺口, 12路
            # 并行加总也不可能凭空造出空隙——真实成因未完全查清(风切变拉丝或
            # WRF风场局部瑕疵), 这里只做纯视觉平滑, 不代表原始结果有误。
            # 轻度高斯平滑(sigma~1格, 约27km, 小于羽流本身尺度), 只为消掉逐时采样在
            # 快速移动羽流中留下的窄条空隙(已核实是真实的逐时空隙, 不是加总出错)。
            # 不做alpha渐变/gouraud平滑——按用户要求改回清晰分块渲染, 不要模糊感。
            field_sm = gaussian_filter(field, sigma=1.0)
            im = ax.pcolormesh(lon, lat, np.ma.masked_less_equal(field_sm, vmin),
                               norm=norm, cmap="cividis", transform=proj,
                               shading="auto", zorder=2, rasterized=True)
        # 改回之前的不规则轮廓(客观连通区的真实边界), 不用矩形框。
        # 只画本图对应的活动源区, 另一源区不画(扩散图只关心自己这个源, 参考框
        # 意义不大, 留给溯源图去对比两个源区)。
        cs = ax.contour(RLON, RLAT, REG[ACTIVE].astype(float), levels=[0.5],
                        colors=[REGION_COLOR[ACTIVE]], linewidths=2.6,
                        linestyles="dashed", transform=proj, zorder=8)
        cs.set(path_effects=HALO_ACCENT)
        # 兰州/北京受体框(红色实框), 便于直接看羽流和受体的相对位置。
        for _c, _b in RECEPTOR_BOXES.items():
            _rect = mpatches.Rectangle((_b[0], _b[1]), _b[2] - _b[0], _b[3] - _b[1],
                                       fill=False, edgecolor="#c0392b", linewidth=2.0,
                                       transform=proj, zorder=9)
            _rect.set_path_effects([pe.withStroke(linewidth=3.2, foreground="white")])
            ax.add_patch(_rect)
        ax.set_extent([70, 140, 15, 55], crs=proj)
        ax.set_title(label_fn(b), pad=4)
        for sp in ax.spines.values():
            sp.set_edgecolor("#c9c8c2")
            sp.set_linewidth(0.8)
    for ax in axes[len(bins):]:
        ax.remove()

    _sysno = "1" if ACTIVE == "TK" else "2"
    fig.legend(handles=[
        Line2D([0], [0], color=REGION_COLOR[ACTIVE], lw=2.6, ls="--",
               label=f"{ACTIVE} source region (System {_sysno})"),
        Line2D([0], [0], color="#c0392b", lw=2.0,
               label="receptor boxes (Lanzhou, Beijing)"),
        ],
        loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.08),
        frameon=False, fontsize=15)
    cb = fig.colorbar(im, ax=axes[:len(bins)].tolist(), orientation="horizontal",
                      fraction=0.045, pad=0.02, aspect=55, shrink=0.6)
    cb.set_label("Time-integrated sensitivity (arb. units)", fontsize=15)
    cb.ax.tick_params(labelsize=13)
    cb.outline.set_edgecolor("#c9c8c2")
    fig.suptitle(title, fontsize=20, y=1.08)
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print("saved:", out_path)
    plt.close(fig)


if __name__ == "__main__":
    TAG = sys.argv[1] if len(sys.argv) > 1 else "TK2_v3"
    SRC, MEM = TAG.split("_", 1)
    if SRC == "TK":
        # TK 合成 = TK1 + TK2 两次运行的场直接相加。
        # 可以直接相加是因为二者用的是同一份观测加权清单, 相对质量已由观测定好
        # (TK1 0.0722 Tg / TK2 0.0402 Tg); FLEXPART 对释放质量线性, 相加即合成。
        # 时间基准取 TK1 的释放起始(西路最早起沙时刻)。
        sim_start = datetime.strptime(WIN["TK1"][0], "%Y-%m-%d %H")
        lon, lat, b1 = load_binned(RUN / f"TK1_{MEM}", sim_start, BINS)
        _, _, b2 = load_binned(RUN / f"TK2_{MEM}", sim_start, BINS)
        binned = {b: (b1[b] if b2[b] is None else
                      b2[b] if b1[b] is None else b1[b] + b2[b]) for b in BINS}
    elif SRC == "MG" and MEM == "combo_full":
        # combo_full 版 MG 是12路并行拆分跑的(每路各自释放块子集, 粒子间互不
        # 干扰), 12份场直接相加 = 合并后的完整结果, 与不拆分一次性跑完等价。
        sim_start = datetime.strptime(WIN["MG"][0], "%Y-%m-%d %H")
        binned = {b: None for b in BINS}
        lon = lat = None
        for p in range(12):
            lo, la, bp = load_binned(RUN / f"MG_combo_full_p{p}", sim_start, BINS)
            if lon is None:
                lon, lat = lo, la
            for b in BINS:
                if bp[b] is None:
                    continue
                binned[b] = bp[b] if binned[b] is None else binned[b] + bp[b]
    else:
        sim_start = datetime.strptime(WIN[SRC][0], "%Y-%m-%d %H")
        lon, lat, binned = load_binned(RUN / TAG, sim_start, BINS)
    for b in BINS:
        f = binned[b]
        print(f"  Day {b[0]}-{b[1]}: {'无数据' if f is None else f'max {f.max():.3e}'}")

    def lab(b, s=sim_start):
        return (f"Day {b[0]}-{b[1]} ("
                f"{(s + timedelta(days=b[0])):%m-%d} to {(s + timedelta(days=b[1])):%m-%d})")

    plot(SRC, lon, lat, binned, BINS,
         f"{SRC} forward-run dispersion by day after release "
         f"({NAME[SRC]}, DUST species)",
         OUT / f"dispersion_{TAG}.png", lab)
