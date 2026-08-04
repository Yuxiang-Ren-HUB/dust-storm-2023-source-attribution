#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""兰州: combo配置全粒子版(TK1+TK2, 延长窗口+区域铺散) + L2最优系数(23.37x),
与观测PM10对比。这是全粒子真实结果, 不是1/60噪声测试。
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
OUT = FIG / "用到的图" / "v3生产模拟_图集"
BOX = (103.43, 35.65, 104.23, 36.45)   # Lanzhou
CONV = 1.05e-3
WEST_COEF = 23.37
C_WEST = REGION3_COLOR["TK1"]

h = xr.open_dataset(RUN / "TK1_final" / "header_d01.nc")
LO, LA = h["XLONG"].values, h["XLAT"].values
h.close()
BM = (LO >= BOX[0]) & (LO <= BOX[2]) & (LA >= BOX[1]) & (LA <= BOX[3])


def series(tag):
    ts, v = [], []
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
            v.append(fld[BM].mean())
        d.close()
    return pd.Series(v, index=pd.DatetimeIndex(ts)).sort_index()


tk1, tk2 = series("TK1_combo_full"), series("TK2_combo_full")
west = (WEST_COEF * (tk1 + tk2)) * CONV

pm = pd.read_csv(FIG / "pm10_utc" / "pm10_utc_hourly_by_city_screened.csv",
                 parse_dates=["time_UTC"]).set_index("time_UTC")
obs = pm["Lanzhou_mean_screened"]
obs = (obs - obs.loc["2023-03-17 00":"2023-03-19 00"].median()).clip(lower=0)
idxa = obs.loc["2023-03-19":"2023-03-25"].dropna().index
win = slice("2023-03-19", "2023-03-25")
w = west.loc[win]
o = obs.loc[idxa]
s = w.reindex(o.index)
ok = o.notna() & s.notna()
r2 = 1 - ((s[ok] - o[ok]) ** 2).sum() / ((o[ok] - o[ok].mean()) ** 2).sum()
mass_ratio = s[ok].sum() / o[ok].sum()
peak_ratio = s[ok].max() / o[ok].max()
print(f"兰州(combo全粒子, 西路{WEST_COEF}x): R2={r2:.3f}  总量比={mass_ratio:.2f}  峰值比={peak_ratio:.2f}")

fig, ax = plt.subplots(figsize=(14, 7.2), constrained_layout=True)
ax.plot(idxa, o, color="k", lw=2.6, label="observed", zorder=6)
ax.fill_between(w.index, 0, w.values, color=C_WEST, alpha=0.6, lw=0,
                label=f"TK1+TK2 west (combo full-particle, {WEST_COEF:.1f}x)", zorder=2)
ax.set_ylabel("Lanzhou PM$_{10}$ (µg m$^{-3}$)", fontsize=13)
ax.grid(axis="x", alpha=0.3)
ax.set_ylim(bottom=0)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.xaxis.set_major_locator(mdates.DayLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
ax.set_xlim(pd.Timestamp("2023-03-19"), pd.Timestamp("2023-03-25"))
ax.set_xlabel("March 2023 (UTC)", fontsize=13)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=11,
          bbox_to_anchor=(0.5, 1.0))
fig.suptitle(f"Lanzhou: combo config (wide window + region-spread), FULL-PARTICLE, "
            f"L2-optimal coefficient {WEST_COEF:.2f}x\nR$^2$={r2:.2f}, mass ratio={mass_ratio:.2f}, "
            f"peak ratio={peak_ratio:.2f}", fontsize=14, y=1.13)
p = OUT / "lanzhou_combo_full_l2optimal.png"
fig.savefig(p, dpi=190, facecolor="white", bbox_inches="tight")
print(f"\nsaved: {p}")
