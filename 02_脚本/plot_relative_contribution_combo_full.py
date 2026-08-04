#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""兰州/北京 西路(TK1+TK2)vs MG 相对贡献占比(100%堆叠), combo全粒子版,
L2最优系数(西路23.37x/MG22.07x)。阴影=事件主峰窗口, 虚线=50%参考线。
"""
import sys
import glob
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
matplotlib.rcParams["axes.unicode_minus"] = False
sys.path.insert(0, r"F:\研\科研\Python")
from source_regions import REGION3_COLOR

RUN = Path(r"F:\FLEXPART_v3_out")
FIG = Path(r"F:\研\科研\Python") / "atmosphere_revision_figs"
OUT = FIG / "用到的图" / "combo全粒子_最终结果"
REC = {"Lanzhou": (103.43, 35.65, 104.23, 36.45), "Beijing": (115.99, 39.51, 116.79, 40.31)}
CONV = {"Lanzhou": 1.05e-3, "Beijing": 1.19e-3}
WEST_COEF, MG_COEF = 23.37, 22.07
C_WEST = REGION3_COLOR["TK1"]
C_MG = REGION3_COLOR["MG"]
PEAK_WIN = {"Lanzhou": ("2023-03-20 12", "2023-03-22 00"),
           "Beijing": ("2023-03-21 18", "2023-03-22 12")}
WIN = slice("2023-03-20", "2023-03-28")

h = xr.open_dataset(RUN / "TK1_final" / "header_d01.nc")
LO, LA = h["XLONG"].values, h["XLAT"].values
h.close()
BM = {c: (LO >= b[0]) & (LO <= b[2]) & (LA >= b[1]) & (LA <= b[3]) for c, b in REC.items()}


def series_both(tag):
    ts, v = [], {c: [] for c in REC}
    for f in sorted(glob.glob(str(RUN / tag / "flxout_d01_*.nc"))):
        d = xr.open_dataset(f)
        C, T = d["CONC"].values, d["Times"].values
        for i in range(C.shape[0]):
            raw = T[i]
            s = (raw.decode() if isinstance(raw, bytes)
                 else "".join(x.decode() if isinstance(x, bytes) else str(x) for x in raw)).strip()
            fld = C[i]
            fld = fld.sum(axis=tuple(range(fld.ndim - 3)))[0]
            ts.append(datetime.strptime(s, "%Y%m%d_%H%M%S"))
            for c in REC:
                v[c].append(fld[BM[c]].mean())
        d.close()
    return pd.DataFrame(v, index=pd.DatetimeIndex(ts)).sort_index()


def series_both_sum12(tag_fmt):
    acc = None
    for p in range(12):
        s = series_both(tag_fmt.format(p))
        acc = s if acc is None else acc.add(s, fill_value=0)
    return acc


print("提取 TK1/TK2 combo_full (双受体框)...")
tk1 = series_both("TK1_combo_full")
tk2 = series_both("TK2_combo_full")
print("提取 MG combo_full (12路求和, 双受体框)...")
mg = series_both_sum12("MG_combo_full_p{}")

plt.rcParams.update({"font.family": "sans-serif", "font.size": 13})
fig, axes = plt.subplots(2, 1, figsize=(15, 9), constrained_layout=True)

for ax, c in zip(axes, REC):
    west = (WEST_COEF * (tk1[c] + tk2[c])) * CONV[c]
    north = (MG_COEF * mg[c]) * CONV[c]
    tot = (west + north).loc[WIN]
    w = west.loc[WIN]
    n = north.loc[WIN]
    valid = tot > tot.max() * 0.01
    west_pct = (w / tot * 100).where(valid)
    mg_pct = (n / tot * 100).where(valid)

    t0, t1 = PEAK_WIN[c]
    ax.axvspan(pd.Timestamp(t0), pd.Timestamp(t1), color="gray", alpha=0.15, zorder=1)
    ax.axhline(50, color="white", lw=1.5, ls="--", zorder=5)

    ax.fill_between(w.index, 0, west_pct.values, color=C_WEST, alpha=0.85, lw=0,
                    label="TK contribution", zorder=3)
    ax.fill_between(w.index, west_pct.values, west_pct.values + mg_pct.values, color=C_MG,
                    alpha=0.85, lw=0, label="MG contribution", zorder=2)
    ax.set_ylim(0, 100)
    ax.set_xlim(pd.Timestamp("2023-03-20"), pd.Timestamp("2023-03-28"))
    ax.set_ylabel("Relative contribution (%)", fontsize=13)
    ax.set_title(f"{c}: TK vs MG relative contribution to near-surface dust", fontsize=14, loc="left")
    ax.grid(axis="y", alpha=0.3, color="white")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d 00h"))

axes[0].legend(loc="upper right", ncol=2, frameon=False, fontsize=12)
axes[-1].set_xlabel("Time (UTC)", fontsize=13)
fig.suptitle("Relative TK/MG dust contribution at Lanzhou and Beijing (combo full-particle, "
            "shaded = event peak window)", fontsize=15)
p = OUT / "relative_contribution_combo_full.png"
fig.savefig(p, dpi=190, facecolor="white", bbox_inches="tight")
print("saved:", p)
