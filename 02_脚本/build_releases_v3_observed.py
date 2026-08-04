#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v3 释放清单 —— 观测门控 + MERRA-2 质量。

与 v2(build_releases_v2.py)的口径差别:

  项目        v2                              v3(本脚本)
  ---------   ------------------------------  --------------------------------
  空间单元    四个人工矩形框                    三个客观连通区(TK1/TK2/MG)
  释放位置    DUEM x AWEP 权重取 top-40 个1度块  逐时次观测起沙掩膜聚合到 1 度块
  释放时间    每框一个固定窗                    逐格点逐小时(锋面推进自动包含)
  释放质量    框内 DUEM 积分                    观测门内的 DUEM 积分, 不归一化
  补偿系数    烘进清单                          不烘, 留到跑完再乘(系数为常数)

为什么"门内取 DUEM, 不归一化":
  检验(check_duem_vs_observed_timing.py)显示 DUEM 峰值时刻与观测差 -4~+2 h, 尚可,
  但 DUEM 全程有低值背景排放, 36-74% 的质量落在观测起沙窗之外。FLEXPART 只应释放
  本次事件的沙尘, 故这部分背景应当丢弃 —— 不归一化正是为了丢掉它, 而不是把它
  重新摊回事件期。丢弃量在输出中逐区报告。

为什么系数不烘进清单:
  FLEXPART 对释放质量严格线性, 且 TK/MG 的补偿系数是常数(3.2x / 3.0x)。
  常数乘子等价于把该源的输出浓度整体缩放, 跑完再乘即可, 这样系数改动无需重跑。
  (若系数改成随时间或空间变化, 该等价性不再成立, 必须重跑 —— 见《起沙量级设置方案》第四节。)

粒径口径: DUEM001+002+003 = GOCART bin1-3, 几何直径 <= 6 um, 对应 PM10 上限。
"""
import sys
import glob
import os
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from scipy.interpolate import NearestNDInterpolator
sys.path.insert(0, r"F:\研\科研\Python")
from source_regions import (load_regions3, REGIONS3, CALIB3, SYSTEMS3,
                            RELEASE_WINDOW3 as WIN)
import argparse as _argparse
_ap2 = _argparse.ArgumentParser(add_help=False)
_ap2.add_argument("--extend-end-h", type=float, default=0.0,
                  help="各源释放窗结束时刻延后的小时数(0=不延, 用于延窗敏感性组)。"
                       "不改 source_regions.py 的正式定义, 只在本次运行内叠加。")
_ap2.add_argument("--mg-buffer-km", type=float, default=0.0,
                  help="MG源区掩膜向外膨胀的距离(km), 0=不改动(默认)。"
                       "用于测试扩大释放面积能否改善总量/衰减尾偏低的问题。")
_a2, _ = _ap2.parse_known_args()
if _a2.extend_end_h:
    WIN = {k: (a, (pd.Timestamp(b) + pd.Timedelta(hours=_a2.extend_end_h)).strftime("%Y-%m-%d %H"), pk)
          for k, (a, b, pk) in WIN.items()}
    print(f"释放窗结束时刻统一延后 {_a2.extend_end_h:.0f}h(基于累积起沙 99% 分位, "
          f"各源原延长量: TK1/TK2/MG 均 +2h): {WIN}")

RR = Path(r"F:\研\科研\Python") / "atmosphere_revision_figs" / "reviewer_response"
OUTDIR = Path(r"F:\研\科研\Python") / "atmosphere_revision_figs" / "AH_investigation"
NX, NY = 1400, 700
LON, LAT = np.meshgrid(np.linspace(70, 126, NX), np.linspace(52, 30, NY))
# 观测网格的单元面积(m²), 随纬度收缩
DX_M = (56.0 / NX) * 111320.0
DY_M = (22.0 / NY) * 111320.0
CELL_M2 = DX_M * np.cos(np.radians(LAT)) * DY_M
BLOCK_RES = 1.0
SEC_PER_HOUR = 3600.0
# 口径A(True): 释放窗内, 保留 DUEM 各区各小时的总量, 用观测重分配其空间结构。
# 口径B(False): 只取观测门内的 DUEM, 门外的一律丢弃, 不归一化。
# 两者给出的贡献率相差仅 2.5 个百分点(西路 73.6% vs 71.1%), 结论对此不敏感。
NORMALIZE = True
# 只在释放窗内释放。窗外零星起沙各源仅占 1.9-3.1% 质量, 舍去后每个释放时次
# 都落在一个已声明的窗内, 时间边界干净, 便于向审稿人交代。
WINDOW_ONLY = True
# 质量分配方式:
#   "duem" —— 各区质量按 MERRA-2 DUEM 的窗内积分。
#   "obs"  —— 只从 DUEM 取"三区合计"这一个绝对量级, 其余全部按观测分配:
#             每个被观测到起沙的『格点·小时』释放等量质量。
# 为什么提供 "obs": DUEM 给 TK1 的单位面积通量是 MG 的 2.6 倍, 与观测(MG 的起沙
# 面积·时次是 TK1 的 1.7 倍)方向相反, 而补偿系数救不了 —— TK 3.2x 与 MG 3.0x 几乎
# 相等。机制上讲得通: 补偿系数由柱 AOD 标定, 而柱负荷 = 排放 x 滞留时间。蒙古气旋
# 急流强、沙尘被快速带走, 源区柱 AOD 偏低, 于是 AOD 标定系统性低估 MG 的系数;
# 塔里木是封闭盆地, 滞留久、柱 AOD 高, 方向相反。见《起沙量级设置方案》第六节第一条。
# 代价: "obs" 假设各区单位『格点·小时』的排放强度相同, 丢掉 DUEM 的空间结构。
#   "d2w"  —— 葵花校准: 门控不变, 质量按 -D2 深度加权(0.3-2.0 K 截断:
#             下限=判识阈值, 上限=饱和起点, 见《葵花卫星观测能力检验》第六节)。
#             各源总量固定为 obsmass 成员的值(TK1 0.0722/TK2 0.0402/MG 0.1261 Tg),
#             只改源内时空形状, 受体端差异即可归因于形状。
MASS_MODE = "d2w"
D2W_MASS = {"TK1": 0.0722e9, "TK2": 0.0402e9, "MG": 0.1261e9}    # kg
# 2026-07-23: 云下混沙尘会被BTD云掩膜整体剔除, 导致"逐像元门控"漏掉云覆盖区的
# 释放位置(不是没有, 是没看见)。改为: 该小时只要区域内有任意像元被观测到起沙,
# 就用这些被观测到像元的 D2 权重算出一个"区域平均浓度", 均匀铺到整个源区
# (而不是只铺在被观测到的那几个像元上)。时间窗口、区域面积均不变, 总释放量
# (D2W_MASS)也不变, 只改"同一小时内质量撒在哪"这一项空间权重。
D2W_SPREAD_REGION = True

REGION = load_regions3(mg_buffer_km=_a2.mg_buffer_km)
if _a2.mg_buffer_km:
    print(f"MG源区掩膜膨胀 {_a2.mg_buffer_km:.0f} km, 格点数 "
         f"{int(REGION['MG'].sum())} (原始未膨胀对比需单独跑 --mg-buffer-km 0)")

# ---------------- 观测起沙时空场 ----------------
tm = pd.read_csv(RR / "emit_spacetime_times.csv", parse_dates=["dt"])
A = np.load(RR / "emit_spacetime_ws8.npy")
D2S = np.load(RR / "d2_spacetime_int16.npy") if MASS_MODE == "d2w" else None
nt = len(tm)
print(f"观测时空场 {nt} 个时次: {tm.dt.min():%m-%d %H}Z ~ {tm.dt.max():%m-%d %H}Z")

# ---------------- MERRA-2 DUEM ----------------
fs = sorted(glob.glob(os.path.join("D:", os.sep, "merra2_adg", "*.SUB.nc4")))
d = xr.open_mfdataset(fs, combine="by_coords")
DU = (d["DUEM001"] + d["DUEM002"] + d["DUEM003"])       # kg m-2 s-1
mt = pd.DatetimeIndex(pd.to_datetime(d.time.values)).floor("h")   # tavg1 时标在 HH:30
ml, ma = np.meshgrid(d.lon.values, d.lat.values)
duv = DU.values
d.close()
print(f"MERRA-2 DUEM {len(mt)} 个时次, {DU.shape[-1]}x{DU.shape[-2]} 格点")

# DUEM 粗网格 -> 观测细网格(最近邻)。DUEM 是通量密度, 最近邻取值不破坏量纲。
gi = NearestNDInterpolator(np.column_stack([ml.ravel(), ma.ravel()]),
                           np.arange(ml.size, dtype=float))
FINE_IDX = gi(np.column_stack([LON.ravel(), LAT.ravel()])).astype(np.int64)

BI = np.floor(LON / BLOCK_RES).astype(int)
BJ = np.floor(LAT / BLOCK_RES).astype(int)
BKEY = BI * 1000 + BJ

# ---- MASS_MODE=obs 时先扫一遍, 统计各区窗内的观测格点-小时与 DUEM 总量 ----
CELLHOURS = {k: 0 for k in REGIONS3}
DUEM_WIN_TOTAL = 0.0
W_SUM = {k: 0.0 for k in REGIONS3}
if MASS_MODE == "d2w":
    # 预扫: 与主循环完全相同的透传函数权重, 用于把各源总量归一到 D2W_MASS
    for i in range(nt):
        t = tm.dt.iloc[i]
        e = np.unpackbits(A[i]).reshape(NY, NX).astype(bool)
        d2 = D2S[i].astype(np.float32) / 100.0
        nd2 = np.clip(-d2, 0.3, 2.5)
        nd2[D2S[i] == -32768] = 0.3
        w = 367.0 * nd2 + 1603.0
        for k in REGIONS3:
            if not (pd.Timestamp(WIN[k][0]) <= t <= pd.Timestamp(WIN[k][1])):
                continue
            det = e & REGION[k]
            if not det.any():
                continue
            if D2W_SPREAD_REGION:
                W_SUM[k] += w[det].mean() * REGION[k].sum()   # 均摊到整个区域
            else:
                W_SUM[k] += w[det].sum()                      # 原口径: 只算被探测到的像元
    print("d2w 权重总和:", {k: round(v, 1) for k, v in W_SUM.items()})

if MASS_MODE == "obs":
    for i in range(nt):
        t = tm.dt.iloc[i]
        w = np.where(mt == t)[0]
        if not len(w):
            continue
        mc = duv[w[0]].ravel()[FINE_IDX].reshape(NY, NX) * CELL_M2 * SEC_PER_HOUR
        e = np.unpackbits(A[i]).reshape(NY, NX).astype(bool)
        for k in REGIONS3:
            if not (pd.Timestamp(WIN[k][0]) <= t <= pd.Timestamp(WIN[k][1])):
                continue
            CELLHOURS[k] += int((e & REGION[k]).sum())
            DUEM_WIN_TOTAL += mc[REGION[k]].sum()
    TOT_CH = sum(CELLHOURS.values())
    MASS_PER_CELLHOUR = DUEM_WIN_TOTAL / max(TOT_CH, 1)
    print(f"MASS_MODE=obs: 三区窗内 DUEM 合计 {DUEM_WIN_TOTAL/1e9:.4f} Tg, "
          f"观测格点-小时 {TOT_CH}, 每格点-小时 {MASS_PER_CELLHOUR:.4g} kg")
    for k in REGIONS3:
        print(f"   {k}: {CELLHOURS[k]:>7} 格点-小时 ({100*CELLHOURS[k]/TOT_CH:.1f}%)")

rows = []
stat = {k: dict(hours=0, blocks=0, mass=0.0, gated_out=0.0, tot=0.0,
                renorm_hours=0, dropped_out_win=0.0) for k in REGIONS3}
for i in range(nt):
    t = tm.dt.iloc[i]
    w = np.where(mt == t)[0]
    if not len(w):
        continue
    flux = duv[w[0]].ravel()[FINE_IDX].reshape(NY, NX)      # kg m-2 s-1, 细网格
    mass_cell = flux * CELL_M2 * SEC_PER_HOUR               # kg, 该小时该细格点
    emit = np.unpackbits(A[i]).reshape(NY, NX).astype(bool)
    for k in REGIONS3:
        reg = REGION[k]
        stat[k]["tot"] += mass_cell[reg].sum()              # 全区(不门控)的 DUEM 质量
        in_win = pd.Timestamp(WIN[k][0]) <= t <= pd.Timestamp(WIN[k][1])
        if WINDOW_ONLY and not in_win:
            # 窗外的零星起沙(各源仅占 1.9-3.1% 质量, 且全部不在对方系统的日期上)
            # 一并舍去, 使每个释放时次都落在一个已声明的窗内, 便于交代。
            stat[k]["dropped_out_win"] += mass_cell[emit & reg].sum()
            continue
        det = emit & reg                                    # 观测门控(被探测到的像元)
        if not det.any():
            continue
        stat[k]["gated_out"] += mass_cell[reg & ~emit].sum()
        if MASS_MODE == "d2w" and D2W_SPREAD_REGION:
            # 云下混沙尘会被云掩膜整体剔除, 逐像元门控会漏掉这些位置。该小时只要
            # 区域内有任意像元被观测到, 就用这些像元的平均D2权重铺满整个区域,
            # 而不是只铺在被观测到的那几个像元上。时间窗口/区域面积/总量都不变。
            g = reg
        else:
            g = det
        keys = BKEY[g]
        if MASS_MODE == "obs":
            # 每个观测到起沙的格点-小时等量释放; DUEM 只提供三区合计的绝对量级
            m = np.full(int(g.sum()), MASS_PER_CELLHOUR)
        elif MASS_MODE == "d2w":
            # 葵花校准: 权重 = 地面回归的浓度估计 w = 367·(-D2) + 1603
            # (透传函数取自 530 站配对; -D2 截断 0.3~2.5 K, 无效像元取下限)
            d2 = D2S[i].astype(np.float32) / 100.0
            nd2 = np.clip(-d2, 0.3, 2.5)
            nd2[D2S[i] == -32768] = 0.3
            w_full = 367.0 * nd2 + 1603.0
            if D2W_SPREAD_REGION:
                avg_w = w_full[det].mean()
                m = np.full(int(g.sum()), D2W_MASS[k] * avg_w / W_SUM[k])
            else:
                w_ = w_full[g]
                m = D2W_MASS[k] * w_ / W_SUM[k]
        else:
            m = mass_cell[g]
        if NORMALIZE and in_win and MASS_MODE == "duem":
            # 口径A: 保留 DUEM 该区该小时的总量, 只用观测重新分配其空间结构。
            # 权重仍用 DUEM 通量本身, 故区内相对强弱不变, 变的只是"撒在哪"。
            # 依据: 观测的强项是定时空分布, DUEM 的强项是定总量; 且门控丢掉的
            #       80% 里相当部分是云下起沙(看不见 != 没发生), 不是纯背景。
            s = m.sum()
            if s > 0:
                m = m * (mass_cell[reg].sum() / s)
            stat[k]["renorm_hours"] += 1
        uk, inv = np.unique(keys, return_inverse=True)
        bm = np.bincount(inv, weights=m)
        for u, mv in zip(uk, bm):
            if mv <= 0:
                continue
            bi, bj = u // 1000, u % 1000
            rows.append([k, t.isoformat(), float(bi), float(bj),
                         float(bi + BLOCK_RES), float(bj + BLOCK_RES), float(mv)])
            stat[k]["blocks"] += 1
            stat[k]["mass"] += mv
        stat[k]["hours"] += 1
    if i % 12 == 0:
        print(f"  {t:%m-%d %H}Z  累计条目 {len(rows)}", flush=True)

out = pd.DataFrame(rows, columns=["source", "hour_start_UTC", "lon0", "lat0",
                                  "lon1", "lat1", "mass_kg"])
_ext_suffix = f"_ext{_a2.extend_end_h:.0f}h" if _a2.extend_end_h else ""
tag = (("d2wmass" if MASS_MODE == "d2w" else "obsmass" if MASS_MODE == "obs" else
        ("normA" if NORMALIZE else "gateB")) + ("_win" if WINDOW_ONLY else ""))
p = OUTDIR / f"releases_v3_{tag}{_ext_suffix}_TK1_TK2_MG.csv"
out.to_csv(p, index=False)

print("\n" + "=" * 96)
print(f"v3 释放清单  口径{chr(65) if NORMALIZE else chr(66)}"
      f"{chr(40)}{chr(29992)}观测重分配, 保留 DUEM 窗内总量{chr(41)}" if NORMALIZE
      else "v3 释放清单  口径B (观测门控, 门外丢弃)")
print("=" * 96)
print(f"{'源':<6}{'活跃时次':>9}{'释放块条目':>11}{'释放质量(Tg)':>14}"
      f"{'门外DUEM(Tg)':>14}{'窗外舍去(Tg)':>13}{'待乘系数':>9}")
for k in REGIONS3:
    s = stat[k]
    keep = 100 * s["mass"] / max(s["tot"], 1e-9)
    print(f"{k:<6}{s['hours']:>9}{s['blocks']:>11}{s['mass']/1e9:>13.4f}"
          f"{s['gated_out']/1e9:>14.4f}{s['dropped_out_win']/1e9:>13.4f}{CALIB3[k]:>8.1f}x")
tot_m = sum(stat[k]["mass"] for k in REGIONS3)
tot_c = sum(stat[k]["mass"] * CALIB3[k] for k in REGIONS3)
print(f"\n合计释放 {tot_m/1e9:.4f} Tg;  乘补偿系数后 {tot_c/1e9:.4f} Tg")
for s_, mem in SYSTEMS3.items():
    m = sum(stat[k]["mass"] * CALIB3[k] for k in mem)
    print(f"  System ({s_}) = {'+'.join(mem):<10} {m/1e9:8.4f} Tg  ({100*m/tot_c:.1f}%)")
print(f"\n条目数 {len(out)} 行 -> {p}")
print("\n注: 『门控丢弃』是落在观测起沙区之外的 DUEM 背景排放, 有意丢弃, 不做归一化。")
