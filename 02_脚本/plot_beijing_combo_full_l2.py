#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""北京: combo配置全粒子版(MG, 12路并行合并) + L2最优系数(22.07x), 与观测PM10
对比。全粒子真实结果, 不是1/60噪声测试。
"""
import sys
import pandas as pd
import numpy as np
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
CONV = 1.19e-3
MG_COEF = 22.07
C_MG = REGION3_COLOR["MG"]

mg = pd.read_csv(RUN / "MG_combo_full_summed.csv", index_col=0, parse_dates=True).iloc[:, 0]
north = (MG_COEF * mg) * CONV

pm = pd.read_csv(FIG / "pm10_utc" / "pm10_utc_hourly_by_city_screened.csv",
                 parse_dates=["time_UTC"]).set_index("time_UTC")
obs = pm["Beijing_mean_screened"]
obs = (obs - obs.loc["2023-03-17 00":"2023-03-19 00"].median()).clip(lower=0)
idxa = obs.loc["2023-03-19":"2023-03-25"].dropna().index
win = slice("2023-03-19", "2023-03-25")
n = north.loc[win]
o = obs.loc[idxa]
s = n.reindex(o.index)
ok = o.notna() & s.notna()
r2 = 1 - ((s[ok] - o[ok]) ** 2).sum() / ((o[ok] - o[ok].mean()) ** 2).sum()
mass_ratio = s[ok].sum() / o[ok].sum()
peak_ratio = s[ok].max() / o[ok].max()
print(f"北京(combo全粒子, MG{MG_COEF}x): R2={r2:.3f}  总量比={mass_ratio:.2f}  峰值比={peak_ratio:.2f}")

fig, ax = plt.subplots(figsize=(14, 7.2), constrained_layout=True)
ax.plot(idxa, o, color="k", lw=2.6, label="observed", zorder=6)
ax.fill_between(n.index, 0, n.values, color=C_MG, alpha=0.6, lw=0,
                label=f"MG (combo full-particle, {MG_COEF:.1f}x)", zorder=2)
ax.set_ylabel("Beijing PM$_{10}$ (µg m$^{-3}$)", fontsize=13)
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
fig.suptitle(f"Beijing: combo config (wide window + region-spread), FULL-PARTICLE, "
            f"L2-optimal coefficient {MG_COEF:.2f}x\nR$^2$={r2:.2f}, mass ratio={mass_ratio:.2f}, "
            f"peak ratio={peak_ratio:.2f}", fontsize=14, y=1.13)
p = OUT / "beijing_combo_full_l2optimal.png"
fig.savefig(p, dpi=190, facecolor="white", bbox_inches="tight")
print(f"\nsaved: {p}")
