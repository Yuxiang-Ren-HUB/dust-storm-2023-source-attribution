#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修正版对比: V3基线 vs final生产版, 与PM10观测的定量吻合度。

两处修正/补充(相对上一版脚本):
  1. 系数反解方法修正: 上一版脚本对每个受体都独立用"总观测峰值"反解西路和MG的
     系数, 导致西路在北京(几乎不贡献)也被强行拉伸去匹配北京总峰值, MG在兰州
     同理, 系数离谱到 380x/3900x, 没有意义。正确做法: 西路只用兰州峰值(西路
     主导受体)反解一次, MG只用北京峰值(MG主导受体)反解一次, 两套系数各自
     固定后, 同时应用到两个受体上评估——这才是可以推广的"一套系数"。
  2. 增加0-2000m近地层输出(排除2000-5000m那层), 更贴近PM10地面暴露的物理意义,
     和全柱求和对比看结论是否稳健。
  3. 散点图(打点)做逐时 obs vs sim 相关, 直接呈现两版拟合优劣, 不只看单一统计量。
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
sys.path.insert(0, r"F:\研\科研\Python")

RUN = Path(r"F:\FLEXPART_v3_out")
FIG = Path(r"F:\研\科研\Python") / "atmosphere_revision_figs"
OUT = FIG / "用到的图" / "v3生产模拟_图集"
REC = {"Lanzhou": (103.43, 35.65, 104.23, 36.45), "Beijing": (115.99, 39.51, 116.79, 40.31)}
CONV = {"Lanzhou": 1.05e-3, "Beijing": 1.19e-3}
WIN_TOTAL = slice("2023-03-19", "2023-03-23")
TAIL = {"Lanzhou": ("2023-03-21 12", "2023-03-23 00"),
       "Beijing": ("2023-03-22 06", "2023-03-23 00")}
DOMINANT_RECEPTOR = {"west": "Lanzhou", "mg": "Beijing"}   # 各自系数反解用哪个受体的峰值

h = xr.open_dataset(RUN / "TK1_final" / "header_d01.nc")
LO, LA = h["XLONG"].values, h["XLAT"].values
h.close()
BM = {c: (LO >= b[0]) & (LO <= b[2]) & (LA >= b[1]) & (LA <= b[3]) for c, b in REC.items()}


def series(tag, levels=None):
    """levels=None -> 全柱(5层)求和; levels=slice(0,4) -> 只取0-2000m(前4层)。"""
    ts, v = [], {c: [] for c in REC}
    for f in sorted(glob.glob(str(RUN / tag / "flxout_d01_*.nc"))):
        d = xr.open_dataset(f)
        C, T = d["CONC"].values, d["Times"].values   # (Time, age, spec, bottom_top, ny, nx)
        for i in range(C.shape[0]):
            raw = T[i]
            s = (raw.decode() if isinstance(raw, bytes)
                 else "".join(x.decode() if isinstance(x, bytes) else str(x) for x in raw)).strip()
            fld = C[i, 0, 0]                          # (bottom_top, ny, nx)
            fld = fld[levels] if levels is not None else fld
            fld = fld.sum(axis=0)                      # (ny, nx)
            ts.append(datetime.strptime(s, "%Y%m%d_%H%M%S"))
            for c in REC:
                v[c].append(fld[BM[c]].mean())
        d.close()
    return pd.DataFrame(v, index=pd.DatetimeIndex(ts)).sort_index()


pm = pd.read_csv(FIG / "pm10_utc" / "pm10_utc_hourly_by_city_screened.csv",
                 parse_dates=["time_UTC"]).set_index("time_UTC")
OBS = {}
for c in REC:
    o = pm[f"{c}_mean_screened"]
    OBS[c] = (o - o.loc["2023-03-17 00":"2023-03-19 00"].median()).clip(lower=0)


def evaluate(name, tk1_tag, tk2_tag, mg_tag, levels, level_label):
    tk1 = series(tk1_tag, levels)
    tk2 = series(tk2_tag, levels)
    mg = series(mg_tag, levels)
    west_raw = {c: (tk1[c] + tk2[c]) * CONV[c] for c in REC}
    mg_raw = {c: mg[c] * CONV[c] for c in REC}

    dom_w, dom_m = DOMINANT_RECEPTOR["west"], DOMINANT_RECEPTOR["mg"]
    west_peak_obs = OBS[dom_w].loc["2023-03-19":"2023-03-23"].dropna().max()
    mg_peak_obs = OBS[dom_m].loc["2023-03-19":"2023-03-23"].dropna().max()
    west_coef = west_peak_obs / west_raw[dom_w].loc[WIN_TOTAL].max()
    mg_coef = mg_peak_obs / mg_raw[dom_m].loc[WIN_TOTAL].max()

    print("=" * 96)
    print(f"【{name} | {level_label}】  西路系数(兰州峰值反解)={west_coef:.2f}x  "
         f"MG系数(北京峰值反解)={mg_coef:.2f}x")
    print("=" * 96)
    results = {}
    for c in REC:
        tot = (west_raw[c] * west_coef + mg_raw[c] * mg_coef).loc[WIN_TOTAL]
        o = OBS[c].loc[WIN_TOTAL].dropna()
        s = tot.reindex(o.index)
        ok = o.notna() & s.notna()
        o2, s2 = o[ok], s[ok]
        peak_ratio = s2.max() / o2.max()
        mass_ratio = s2.sum() / o2.sum()
        r = np.corrcoef(o2.values, s2.values)[0, 1]
        t0, t1 = TAIL[c]
        ot = OBS[c].loc[t0:t1].dropna()
        st = tot.reindex(ot.index)
        okt = ot.notna() & st.notna()
        ot, st = ot[okt], st[okt]
        tail_ratio = st.sum() / ot.sum() if ot.sum() > 0 else np.nan
        print(f"  [{c}]  峰值比值={peak_ratio:.2f}({peak_ratio*100:.0f}%)  "
             f"总量比值={mass_ratio:.2f}({mass_ratio*100:.0f}%)  "
             f"R^2={r**2:.2f}(r={r:.2f})  衰减尾比值={tail_ratio:.2f}({tail_ratio*100:.0f}%)")
        results[c] = (o2, s2, r)
    return results


LEVELS_FULL = None
LEVELS_SFC = slice(0, 4)   # 100/500/1000/2000m, 排除2000-5000m那层

r_v3_full = evaluate("V3基线", "TK1_v3", "TK2_v3", "MG_v3", LEVELS_FULL, "全柱(5层)")
r_final_full = evaluate("final生产版", "TK1_final", "TK2_final", "MG_final", LEVELS_FULL, "全柱(5层)")
r_v3_sfc = evaluate("V3基线", "TK1_v3", "TK2_v3", "MG_v3", LEVELS_SFC, "近地层0-2000m")
r_final_sfc = evaluate("final生产版", "TK1_final", "TK2_final", "MG_final", LEVELS_SFC, "近地层0-2000m")

# ---- 散点图: 逐时 obs vs sim, 近地层口径, 2(城市) x 2(版本) ----
plt.rcParams.update({"font.family": "sans-serif", "font.size": 12})
fig, axes = plt.subplots(2, 2, figsize=(11, 10.5), constrained_layout=True)
for row, c in enumerate(REC):
    for col, (label, res) in enumerate([("V3 baseline", r_v3_sfc), ("final (production)", r_final_sfc)]):
        ax = axes[row, col]
        o2, s2, r = res[c]
        vmax = max(o2.max(), s2.max()) * 1.05
        ax.plot([0, vmax], [0, vmax], color="#999999", lw=1, ls="--", zorder=1)
        ax.scatter(o2.values, s2.values, s=22, alpha=0.6, color="#c0392b" if col == 0 else "#2f5597",
                  edgecolor="none", zorder=3)
        ax.set_xlim(0, vmax)
        ax.set_ylim(0, vmax)
        ax.set_aspect("equal")
        ax.set_title(f"{c} — {label}\nR$^2$={r**2:.2f}  (r={r:.2f}, n={len(o2)})", fontsize=11.5)
        ax.set_xlabel("observed PM$_{10}$ (µg m$^{-3}$)", fontsize=10.5)
        ax.set_ylabel("simulated (µg m$^{-3}$, 0-2000m)", fontsize=10.5)
        ax.grid(alpha=0.25)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
fig.suptitle("Hourly observed vs. simulated PM$_{10}$ (0-2000m column): V3 baseline vs. final production",
            fontsize=14, y=1.03)
p = OUT / "scatter_v3_vs_final_surface.png"
fig.savefig(p, dpi=190, facecolor="white", bbox_inches="tight")
print("\nsaved:", p)
