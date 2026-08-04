#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""定量源贡献 v3 —— 受体 PM10 反演源强 + 三种口径的贡献百分比。

思路(2026-07-22 与用户定稿):
  受体处模拟 PM10(t) = Σ_i w_i · C_i(t), C_i 为第 i 源单位清单的受体浓度响应
  (三源分跑, FLEXPART 对释放质量严格线性)。用兰州+北京实测 PM10 做非负最小
  二乘解 w_i —— 受体观测是完全独立的一套数据, 没参与过源区圈定/清单的任何环节。

三种口径并报:
  A. 观测加权   w=各源观测格点·小时占比(即清单原样) × 补偿系数
  B. DUEM 加权  在 A 基础上把各源质量换成 DUEM 窗内积分(TK1 x1.96, TK2 x0.77, MG x0.52)
  C. NNLS 反演  w 由受体 PM10 定, 不预设
  若 C 落在 A、B 之间 → 两口径各有偏差、结论稳健; 偏向哪端 → 那端更可信。

可辨识性检查: TK1/TK2 到达受体的浓度曲线若高度共线(r>0.95), 则只报
System 级(TK1+TK2 合并)的反演结果, 不硬拆。

背景扣除: PM10 含非沙尘背景, 取事件前 03-17 00Z ~ 03-19 00Z 中位数逐城扣除,
负值截为 0。
"""
import sys
import glob
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from datetime import datetime
from scipy.optimize import nnls
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
matplotlib.rcParams["axes.unicode_minus"] = False
sys.path.insert(0, r"F:\研\科研\Python")

RUN = Path(r"F:\FLEXPART_v3_out")
FIG = Path(r"F:\研\科研\Python") / "atmosphere_revision_figs"
OUT = FIG / "用到的图"
RR = FIG / "reviewer_response"

SRCS = ["TK1", "TK2", "MG"]
COL = {"TK1": "#c0742d", "TK2": "#8c3b12", "MG": "#1b3a6b"}
# 各源"跑进模式的质量"(obsmass 清单, 未乘系数) 与两种目标质量(Tg)
RUN_MASS = {"TK1": 0.0722, "TK2": 0.0402, "MG": 0.1261}
DUEM_MASS = {"TK1": 0.1415, "TK2": 0.0309, "MG": 0.0661}
CALIB = {"TK1": 3.2, "TK2": 3.2, "MG": 3.0}
RECEPTOR = {"Lanzhou": (103.43, 35.65, 104.23, 36.45),
            "Beijing": (115.99, 39.51, 116.79, 40.31)}
WIN = {"Lanzhou": ("2023-03-20 13", "2023-03-22 09"),
       "Beijing": ("2023-03-21 17", "2023-03-22 13")}
BASE_PERIOD = ("2023-03-17 00", "2023-03-19 00")

# ---------------- 1. 受体浓度响应 C_i(t) ----------------
h = xr.open_dataset(RUN / "TK1_v3" / "header_d01.nc")
LON, LAT = h["XLONG"].values, h["XLAT"].values
h.close()
BOXM = {c: (LON >= b[0]) & (LON <= b[2]) & (LAT >= b[1]) & (LAT <= b[3])
        for c, b in RECEPTOR.items()}
for c, m in BOXM.items():
    print(f"{c} 受体框: {int(m.sum())} 个 0.25° 格")


def receptor_series(tag):
    ts, v = [], {c: [] for c in RECEPTOR}
    for f in sorted(glob.glob(str(RUN / tag / "flxout_d01_*.nc"))):
        d = xr.open_dataset(f)
        C, T = d["CONC"].values, d["Times"].values
        for i in range(C.shape[0]):
            raw = T[i]
            s = (raw.decode() if isinstance(raw, bytes)
                 else "".join(x.decode() if isinstance(x, bytes) else str(x) for x in raw)).strip()
            fld = C[i]
            # 对 ageclass/releases/species 求和 -> (bottom_top, y, x); 取最低层(0-100 m)
            fld = fld.sum(axis=tuple(range(fld.ndim - 3)))[0]
            ts.append(datetime.strptime(s, "%Y%m%d_%H%M%S"))
            for c in RECEPTOR:
                v[c].append(fld[BOXM[c]].mean())
        d.close()
    df = pd.DataFrame(v, index=pd.DatetimeIndex(ts)).sort_index()
    return df


print("\n读取三源受体响应 ...")
C = {}
for s in SRCS:
    C[s] = receptor_series(f"{s}_v3")
    print(f"  {s}: {len(C[s])} 时次, LZ 峰值 {C[s]['Lanzhou'].max():.3g} "
          f"({C[s]['Lanzhou'].idxmax():%m-%d %H}Z), BJ 峰值 {C[s]['Beijing'].max():.3g} "
          f"({C[s]['Beijing'].idxmax():%m-%d %H}Z)")
pd.concat({s: C[s] for s in SRCS}, axis=1).to_csv(RR / "receptor_response_v3.csv")

# ---------------- 2. 观测 PM10(扣背景) ----------------
pm = pd.read_csv(FIG / "pm10_utc" / "pm10_utc_hourly_by_city_screened.csv",
                 parse_dates=["time_UTC"]).set_index("time_UTC")
obs = pd.DataFrame({"Lanzhou": pm["Lanzhou_mean_screened"],
                    "Beijing": pm["Beijing_mean_screened"]})
base = obs.loc[BASE_PERIOD[0]:BASE_PERIOD[1]].median()
print(f"\n事件前背景(中位, 扣除): Lanzhou {base['Lanzhou']:.0f}, Beijing {base['Beijing']:.0f} ug/m3")
obs_d = (obs - base).clip(lower=0)

# ---------------- 3. 拟合样本: 两受体窗内逐时 ----------------
rows_X, rows_y, rows_tag = [], [], []
for c in RECEPTOR:
    idx = C["TK1"].loc[WIN[c][0]:WIN[c][1]].index
    idx = idx.intersection(obs_d.dropna(subset=[c]).index)
    X = np.column_stack([C[s].loc[idx, c].values for s in SRCS])
    y = obs_d.loc[idx, c].values
    rows_X.append(X)
    rows_y.append(y)
    rows_tag += [(c, t) for t in idx]
X = np.vstack(rows_X)
y = np.concatenate(rows_y)
print(f"拟合样本: {len(y)} 个受体-小时")

# 可辨识性
cc = np.corrcoef(X.T)
print("\n受体响应互相关(窗内):")
for i in range(3):
    for j in range(i + 1, 3):
        print(f"  {SRCS[i]}-{SRCS[j]}: r = {cc[i, j]:+.2f}")
cond = np.linalg.cond(X)
print(f"条件数: {cond:.1f}")

# ---------------- 4. 三种口径 ----------------
def shares(weights):
    """weights: 对『跑进模式的清单』的乘子; 返回各受体窗内贡献%"""
    out = {}
    for c in RECEPTOR:
        idx = C["TK1"].loc[WIN[c][0]:WIN[c][1]].index
        tot = {s: weights[s] * C[s].loc[idx, c].sum() for s in SRCS}
        z = sum(tot.values())
        out[c] = {s: 100 * tot[s] / z for s in SRCS}
    return out


W_A = {s: CALIB[s] for s in SRCS}                                   # 观测加权(清单原样)
W_B = {s: CALIB[s] * DUEM_MASS[s] / RUN_MASS[s] for s in SRCS}      # DUEM 加权
w_c, res = nnls(X, y)
W_C = {s: w_c[i] for i, s in enumerate(SRCS)}
r2 = 1 - np.sum((X @ w_c - y) ** 2) / np.sum((y - y.mean()) ** 2)
print(f"\nNNLS(三源): w = {', '.join(f'{s} {W_C[s]:.3g}' for s in SRCS)}   R² = {r2:.2f}")

# TK1/TK2 在受体处共线(r=0.88), 三源反演会把其中一个置零 —— 那是伪影。
# 正式口径: 合并为 System 级再反演, 图也只画 System 级。
X2 = np.column_stack([X[:, 0] + X[:, 1], X[:, 2]])
w2, _ = nnls(X2, y)
r2s = 1 - np.sum((X2 @ w2 - y) ** 2) / np.sum((y - y.mean()) ** 2)
W_SYS = {"West": w2[0], "MG": w2[1]}
print(f"NNLS(System 级): West {w2[0]:.3g}, MG {w2[1]:.3g}   R² = {r2s:.2f}")

print("\n" + "=" * 86)
print("贡献百分比(受体窗内, 三种口径)")
print("=" * 86)
print(f"{'口径':<26}{'受体':<10}{'TK1':>8}{'TK2':>8}{'TK1+TK2':>9}{'MG':>8}")
res_tab = []
for nm, W in (("A 观测加权(清单原样)", W_A), ("B DUEM 加权", W_B), ("C NNLS 反演", W_C)):
    sh = shares(W)
    for c in RECEPTOR:
        v = sh[c]
        print(f"{nm:<26}{c:<10}{v['TK1']:>7.1f}%{v['TK2']:>7.1f}%"
              f"{v['TK1'] + v['TK2']:>8.1f}%{v['MG']:>7.1f}%")
        res_tab.append(dict(scheme=nm, receptor=c, **v))
    print()
pd.DataFrame(res_tab).to_csv(RR / "source_contribution_v3.csv", index=False)

# ---------------- 4b. 量级核对(单位换算后) ----------------
# CONC 单位是 'ppt by mass' = ng/kg。换成 ug/m3 要乘空气密度再 x1e-3:
# 兰州海拔 ~1520 m, rho≈1.05; 北京 ~50 m, rho≈1.19。
CONV = {"Lanzhou": 1.05e-3, "Beijing": 1.19e-3}
print("\n" + "=" * 86)
print("量级核对: 口径A(清单 x 3.2/3.0 系数)模拟峰 vs 观测峰")
print("=" * 86)
for c in RECEPTOR:
    sim = (W_A["TK1"] * C["TK1"][c] + W_A["TK2"] * C["TK2"][c]
           + W_A["MG"] * C["MG"][c]) * CONV[c]
    pk_s, pk_o = sim.max(), obs_d[c].max()
    print(f"  {c}: 模拟峰 {pk_s:.0f} ug/m3 ({sim.idxmax():%m-%d %H}Z)  "
          f"观测峰 {pk_o:.0f} ({obs_d[c].idxmax():%m-%d %H}Z)  还差 {pk_o/pk_s:.1f}x")
# 反演 w 隐含的总修正倍数(相对未加系数的清单, 已含单位换算)
print("\nNNLS(System 级)隐含的清单总修正倍数(含 3.2/3.0 在内的全部欠量):")
for nm, wv, cv in (("West", w2[0], CONV["Lanzhou"]), ("MG", w2[1], CONV["Beijing"])):
    print(f"  {nm}: w = {wv:.3g} (ug/m3 per ppt) -> 清单需放大 {wv/cv:.0f}x "
          f"(其中 3.2/3.0 已计入现口径, 额外欠 {wv/cv/3.1:.0f}x)")
print("  ⚠ 此倍数含受体框空间平均、0-100m 层平均对峰值的稀释, 属上限估计。")

# ---------------- 5. 图 ----------------
plt.rcParams.update({"font.family": "sans-serif", "font.size": 12})
fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True, constrained_layout=True)
for ax, c in zip(axes, RECEPTOR):
    idxa = obs_d.loc["2023-03-19":"2023-03-24", c].dropna().index
    ax.plot(idxa, obs_d.loc[idxa, c], color="k", lw=2.2, label="observed PM$_{10}$ (baseline removed)")
    idx = C["TK1"].loc["2023-03-19":"2023-03-24"].index
    # 图只画 System 级: TK1/TK2 在受体处共线不可分, 三源堆叠会出现"看不见的源"
    west = W_SYS["West"] * (C["TK1"].loc[idx, c] + C["TK2"].loc[idx, c]).values
    mg = W_SYS["MG"] * C["MG"].loc[idx, c].values
    ax.fill_between(idx, 0, west, color="#c0742d", alpha=0.78, lw=0,
                    label="System 1  western pathway (TK1+TK2)")
    ax.fill_between(idx, west, west + mg, color="#1b3a6b", alpha=0.78, lw=0,
                    label="System 2  Mongolian Gobi (MG)")
    t0, t1 = pd.Timestamp(WIN[c][0]), pd.Timestamp(WIN[c][1])
    ax.axvspan(t0, t1, color="#888", alpha=0.10, zorder=0)
    ax.set_ylabel(f"{c}  PM$_{{10}}$ (µg m$^{{-3}}$)", fontsize=13)
    ax.legend(loc="upper left", frameon=False, fontsize=11.5)
    ax.grid(axis="x", alpha=0.3)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[-1].xaxis.set_major_locator(mdates.DayLocator())
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
axes[-1].set_xlabel("March 2023 (UTC)", fontsize=13)
fig.suptitle("Source contributions at the receptors: system-level NNLS inversion of observed PM$_{10}$\n"
             f"(TK1/TK2 merged — collinear at receptors, r = {cc[0,1]:.2f}; "
             f"fit window shaded, R² = {r2s:.2f})",
             fontsize=15)
p = OUT / "source_contribution_v3.png"
fig.savefig(p, dpi=185, facecolor="white", bbox_inches="tight")
print(f"\nsaved: {p}")
