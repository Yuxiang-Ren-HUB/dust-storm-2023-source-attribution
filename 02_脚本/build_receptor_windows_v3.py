#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按 PM10 观测重新确定兰州/北京的后向模拟窗口(v3)。

为什么重做:
  v2 的北京窗已修好(旧版曾与事件 0 小时重叠), 但兰州窗止于 03-21 23,
  而观测的显著抬升期延续到 03-22 08 —— 尾部漏掉 9 小时, 而衰减段正是
  之前算出 AH 贡献 83.5% 的那一段, 不能缺。

定窗规则(全部由观测决定, 不引用模式):
  基线   = 事件前(<=03-19 00Z)PM10 的中位数
  抬升期 = PM10 > 3 x 基线 的连续时段
  窗口   = 抬升期前后各留 2 h 余量(捕捉起涨与回落)
  阶段   = 由观测曲线本身的形态划分(见 phases()), 供分相位归因用

同时做一项自诊断:
  新 WRF 气象场从 2023-03-17 00Z 起, 而后向积分要往回追。
  对每个受体小时给出"可用的后向积分时长", 若最短的那个仍 >= 3 天,
  则截断对足迹的影响可忽略(旧结果: 输送时间中位数 0.9-1.2 天, 86% 在 2 天内)。
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

LC = Path(r"F:\研\科研\Python\atmosphere_revision_figs")
OBS = LC / "pm10_utc" / "pm10_utc_hourly_by_city.csv"
OUT = LC / "pm10_utc" / "receptor_release_plan_v3.csv"

MET_START = datetime(2023, 3, 17, 0)     # 新 WRF 起报时刻
MET_END = datetime(2023, 3, 27, 0)
PRE_END = datetime(2023, 3, 19, 0)       # 基线统计截止
THRESH_MULT = 3.0
MARGIN_H = 2

df = pd.read_csv(OBS, parse_dates=["time_UTC"]).set_index("time_UTC")


def phases(city, s, t0, t1, thr):
    """由观测曲线形态划分阶段, 全部数据驱动, 无硬编码时刻。

    兰州的实测形态(见 03-20 12 – 03-22 10 逐小时值):
        03-20 14      1077          起跳
        03-20 15-19   1520-1595     第一峰
        03-20 20-03-21 03  1322-1408 谷(最低 1322 @ 03-20 23)
        03-21 04      1624          全局峰
        03-21 06-08   1362-1491     第二峰
        03-21 09 起   1201 -> 440   单调衰减
    -> pulse1 / trough / pulse2 / decay 四段, 与 v2 的命名一致。

    划分规则:
        onset  = 首个超阈值的小时
        peak2  = 全局峰
        peak1  = onset 到 peak2 之间, 距 peak2 至少 4 h 的那一段里的最大值
        trough = peak1 与 peak2 之间的最小值
        decay  = peak2 之后连续 3 h 下降起, 直到窗口末
    北京是单峰, 用 rise / peak(峰 ±2h) / decay。
    """
    w = s.loc[t0:t1]
    out = {}
    if city == "Lanzhou":
        above = w[w > thr]
        onset = above.index.min()
        peak2 = w.idxmax()
        seg1 = w.loc[onset:peak2 - timedelta(hours=4)]
        peak1 = seg1.idxmax() if len(seg1) else onset
        mid = w.loc[peak1:peak2]
        trough = mid.idxmin() if len(mid) > 2 else peak1
        # 第二脉冲的结束: peak2 之后首次连续 3 h 下降
        after = w.loc[peak2:]
        dec_start = after.index[-1]
        v = after.to_numpy()
        for i in range(len(v) - 3):
            if v[i] > v[i + 1] > v[i + 2] > v[i + 3]:
                dec_start = after.index[i + 1]
                break
        for t in w.index:
            if t < onset:
                out[t] = "pre_event"
            elif t <= peak1 or t < trough:
                out[t] = "pulse1"
            elif t == trough:
                out[t] = "trough"
            elif t < dec_start:
                out[t] = "pulse2"
            else:
                out[t] = "decay"
    else:
        pk = w.idxmax()
        for t in w.index:
            if t < pk - timedelta(hours=2):
                out[t] = "rise"
            elif t <= pk + timedelta(hours=2):
                out[t] = "peak"
            else:
                out[t] = "decay"
    return out


rows = []
print("=" * 100)
print("按 PM10 观测重定后向窗口 (v3)")
print("=" * 100)
for city in ("Lanzhou", "Beijing"):
    s = df[f"{city}_mean"].dropna()
    base = float(s.loc[:PRE_END].median())
    thr = base * THRESH_MULT
    ev = s.loc["2023-03-19":"2023-03-24"]
    hi = ev[ev > thr]
    t0 = hi.index.min() - timedelta(hours=MARGIN_H)
    t1 = hi.index.max() + timedelta(hours=MARGIN_H)
    ph = phases(city, s, t0, t1, thr)
    hrs = pd.date_range(t0, t1, freq="h")

    print(f"\n--- {city} ---")
    print(f"  基线(事件前中位数) {base:.0f} µg/m³   阈值 3x = {thr:.0f}")
    print(f"  抬升期  {hi.index.min():%m-%d %H:%M} – {hi.index.max():%m-%d %H:%M}  ({len(hi)} h)")
    print(f"  窗口(±{MARGIN_H}h)  {t0:%m-%d %H:%M} – {t1:%m-%d %H:%M}  -> {len(hrs)} 次释放")
    print(f"  峰值 {ev.max():.0f} @ {ev.idxmax():%m-%d %H:%M}")
    cnt = {}
    for i, t in enumerate(hrs):
        p = ph.get(t, "decay")
        cnt[p] = cnt.get(p, 0) + 1
        back_h = (t - MET_START).total_seconds() / 3600
        rows.append(dict(city=city, idx=i + 1, t_start_utc=t,
                         t_end_utc=t + timedelta(hours=1),
                         obs_pm10=float(s.get(t, np.nan)), phase=p,
                         backward_hours_available=back_h))
    print(f"  阶段分布  {cnt}")
    bmin = min((t - MET_START).total_seconds() / 3600 for t in hrs)
    bmax = max((t - MET_START).total_seconds() / 3600 for t in hrs)
    print(f"  可用后向积分时长  {bmin/24:.1f} – {bmax/24:.1f} 天  (气象场起于 {MET_START:%m-%d %H:%M})")

plan = pd.DataFrame(rows)
plan.to_csv(OUT, index=False)

print("\n" + "=" * 100)
print("与 v2 方案对比")
print("=" * 100)
old = pd.read_csv(LC / "pm10_utc" / "receptor_release_plan.csv", parse_dates=["t_start_utc"])
print(f"{'城市':<10}{'v2 窗口':<30}{'v2 次数':>8}{'':4}{'v3 窗口':<30}{'v3 次数':>8}")
for city in ("Lanzhou", "Beijing"):
    o = old[old.city == city]; n = plan[plan.city == city]
    print(f"{city:<10}{f'{o.t_start_utc.min():%m-%d %H:%M} – {o.t_start_utc.max():%m-%d %H:%M}':<30}"
          f"{len(o):>8}{'':4}"
          f"{f'{n.t_start_utc.min():%m-%d %H:%M} – {n.t_start_utc.max():%m-%d %H:%M}':<30}{len(n):>8}")

print("\n" + "=" * 100)
print("截断自诊断")
print("=" * 100)
bmin_all = plan.backward_hours_available.min()
print(f"  最短可用后向积分: {bmin_all/24:.1f} 天 (受体 {plan.loc[plan.backward_hours_available.idxmin(),'city']} "
      f"@ {plan.loc[plan.backward_hours_available.idxmin(),'t_start_utc']:%m-%d %H:%M})")
if bmin_all / 24 >= 3:
    print("  ✓ 最短也有 3 天以上。旧结果显示输送时间中位数 0.9–1.2 天、86% 在 2 天内,")
    print("    故 03-17 起报造成的截断对足迹主体无影响。仍会在跑完后检查边界处残余足迹。")
else:
    print("  ⚠ 不足 3 天, 建议把 WRF 往前补跑到 03-15。")
print(f"\nsaved: {OUT}  ({len(plan)} 条)")
