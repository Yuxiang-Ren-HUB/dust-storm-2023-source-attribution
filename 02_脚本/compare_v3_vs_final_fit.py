#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V3基线 vs final生产版, 同一套指标(峰值比值/事件总量比值/R^2/衰减尾比值)
直接对比, 检验"V3和观测匹配更好"这个直觉印象是否成立。
两版系数都用同样方法(各自曲线自身峰值比观测峰值)反解, 保证公平——不用
之前那组不稳定的2x2精确解。
"""
import sys
import glob
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime
from pathlib import Path
sys.path.insert(0, r"F:\研\科研\Python")

RUN = Path(r"F:\FLEXPART_v3_out")
FIG = Path(r"F:\研\科研\Python") / "atmosphere_revision_figs"
REC = {"Lanzhou": (103.43, 35.65, 104.23, 36.45), "Beijing": (115.99, 39.51, 116.79, 40.31)}
CONV = {"Lanzhou": 1.05e-3, "Beijing": 1.19e-3}
WIN_TOTAL = slice("2023-03-19", "2023-03-23")
TAIL = {"Lanzhou": ("2023-03-21 12", "2023-03-23 00"),
       "Beijing": ("2023-03-22 06", "2023-03-23 00")}
PEAKWIN = {"Lanzhou": ("2023-03-20 09", "2023-03-21 12"),
          "Beijing": ("2023-03-21 22", "2023-03-22 06")}

h = xr.open_dataset(RUN / "TK1_final" / "header_d01.nc")
LO, LA = h["XLONG"].values, h["XLAT"].values
h.close()
BM = {c: (LO >= b[0]) & (LO <= b[2]) & (LA >= b[1]) & (LA <= b[3]) for c, b in REC.items()}


def series(tag):
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


pm = pd.read_csv(FIG / "pm10_utc" / "pm10_utc_hourly_by_city_screened.csv",
                 parse_dates=["time_UTC"]).set_index("time_UTC")


def evaluate(name, tk1_tag, tk2_tag, mg_tag):
    tk1, tk2, mg = series(tk1_tag), series(tk2_tag), series(mg_tag)
    print("=" * 96)
    print(f"【{name}】")
    print("=" * 96)
    for c in REC:
        west_raw = (tk1[c] + tk2[c]) * CONV[c]
        mg_raw = mg[c] * CONV[c]
        obs = pm[f"{c}_mean_screened"]
        obs = (obs - obs.loc["2023-03-17 00":"2023-03-19 00"].median()).clip(lower=0)

        # 各自曲线自身峰值反解系数(与 final 版同方法, 公平对比)
        west_own = west_raw.loc[WIN_TOTAL]
        mg_own = mg_raw.loc[WIN_TOTAL]
        obs_peak_all = obs.loc["2023-03-19":"2023-03-23"].dropna().max()
        # 西路系数: 用西路主导的时段峰值(简单起见直接用总观测峰值近似, 与之前口径一致)
        west_coef = obs_peak_all / west_own.max() if west_own.max() > 0 else np.nan
        mg_coef = obs_peak_all / mg_own.max() if mg_own.max() > 0 else np.nan

        west = west_raw * west_coef
        north = mg_raw * mg_coef
        tot = (west + north).loc[WIN_TOTAL]

        o = obs.loc[WIN_TOTAL].dropna()
        s = tot.reindex(o.index)
        ok = o.notna() & s.notna()
        o2, s2 = o[ok], s[ok]

        peak_ratio = s2.max() / o2.max()
        mass_ratio = s2.sum() / o2.sum()
        r = np.corrcoef(o2.values, s2.values)[0, 1]

        t0, t1 = TAIL[c]
        ot = obs.loc[t0:t1].dropna()
        st = tot.reindex(ot.index)
        okt = ot.notna() & st.notna()
        ot, st = ot[okt], st[okt]
        tail_ratio = st.sum() / ot.sum() if ot.sum() > 0 else np.nan

        print(f"\n  [{c}]  西路系数={west_coef:.1f}x  MG系数={mg_coef:.1f}x")
        print(f"    峰值比值(量级)        : {peak_ratio:.2f}  ({peak_ratio*100:.0f}%)")
        print(f"    事件总量比值(03-19~23): {mass_ratio:.2f}  ({mass_ratio*100:.0f}%)")
        print(f"    逐时相关 R^2          : {r**2:.2f}  (r={r:.2f})")
        print(f"    衰减尾总量比值        : {tail_ratio:.2f}  ({tail_ratio*100:.0f}%)")


evaluate("V3 基线版(默认高度/2.0um粒径/obs质量清单)", "TK1_v3", "TK2_v3", "MG_v3")
evaluate("final 生产版(反演高度/1.3um/葵花d2w质量)", "TK1_final", "TK2_final", "MG_final")
