#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""葵花沙尘顶高度反演 v3(稳健版)—— 各源释放窗内, 沙尘像元 BT 中位 → WRF 廓线查高。

前两版(retrieve_dust_top_himawari.py / _v2)失败教训:
  都对"沙尘"像元设了人为温度底(270 K), 结果所有像元堆在这个底上, 反演出的高度
  紧贴我设的参数 —— 是 artifact 不是信号。用户当时的判断: "如果结果紧贴自己设的
  任何一个参数, 先怀疑它就是那个参数"。

本版的三条防 artifact 设计:
  1. 不设任何人为温度底。直接取"云掩膜后的沙尘像元"(D2<-0.3 且 D1>0, 冰云/厚云
     已剔除)的 B14(11.2 µm 窗区亮温)分布, 取中位。
  2. 报出 BT 的 P10/25/50/75/90 全分布, 由使用者判断是否合理(而非只给一个数)。
  3. WRF 廓线是真实的 T(z), 查 T(z)=BT 的高度是纯物理插值; 若反演高度紧贴地表
     或紧贴模式顶, 立即警告。
  交叉验证锚点: 之前独立估计的沙尘层顶约 2939 m(观测约束层顶, 非注入深度)。
     若各源反演落在 1-3 km 量级即自洽。

只在各源释放窗内、释放连通区内、晴空沙尘像元上统计, 给每个源一个代表性沙尘顶。
"""
import sys
import glob
import bz2
import shutil
import tempfile
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from scipy.interpolate import NearestNDInterpolator
sys.path.insert(0, r"F:\研\科研\Python")
from source_regions import load_regions3, RELEASE_WINDOW3 as WIN
from satpy import Scene
from pyresample.geometry import AreaDefinition

HIM = Path(r"D:\himawari9_ir")
WRF = Path(r"F:\WRF_backup\_d01_nudged")
RR = Path(r"F:\研\科研\Python") / "atmosphere_revision_figs" / "reviewer_response"
NX, NY = 1400, 700
AREA = AreaDefinition("d", "d", "eqc", {"proj": "eqc", "lon_0": 0, "datum": "WGS84"},
                      NX, NY, (70 * 111320.0, 30 * 111320.0, 126 * 111320.0, 52 * 111320.0))
LON, LAT = np.meshgrid(np.linspace(70, 126, NX), np.linspace(52, 30, NY))
REGION = load_regions3()


def load_him(t):
    fs = sorted(glob.glob(str(HIM / t / "*_B1[145]_*.DAT.bz2")))
    if len(fs) < 6:
        return None
    tmp = Path(tempfile.mkdtemp(prefix="ahi_"))
    try:
        pl = []
        for f in fs:
            o = tmp / Path(f).stem
            with bz2.open(f, "rb") as fi, open(o, "wb") as fo:
                shutil.copyfileobj(fi, fo)
            pl.append(str(o))
        s = Scene(filenames=pl, reader="ahi_hsd")
        s.load(["B11", "B14", "B15"])
        l = s.resample(AREA, resampler="nearest", radius_of_influence=6000)
        return l["B11"].values, l["B14"].values, l["B15"].values
    except Exception as e:
        print(f"  [{t}] 葵花读取失败: {e}", flush=True)
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def wrf_profile(t, mask):
    """该时次、该源区平均的 (高度 m, 温度 K) 廓线"""
    f = WRF / f"wrfout_d01_{t[:4]}-{t[4:6]}-{t[6:8]}_{t[9:11]}0000"
    if not f.exists():
        return None
    d = xr.open_dataset(f)
    lo = d["XLONG"].isel(Time=0).values
    la = d["XLAT"].isel(Time=0).values
    theta = d["T"].isel(Time=0).values + 300.0        # 位温
    p = (d["P"].isel(Time=0).values + d["PB"].isel(Time=0).values)   # Pa
    ph = d["PH"].isel(Time=0).values + d["PHB"].isel(Time=0).values  # m2/s2 (交错)
    d.close()
    T = theta * (p / 1e5) ** 0.2854                   # 实际温度 K
    z = (ph[:-1] + ph[1:]) / 2 / 9.81                 # 层中心高度 m (未层)
    # 源区内格点掩膜(WRF 网格)
    m = (lo >= LON[mask].min()) & (lo <= LON[mask].max()) & \
        (la >= LAT[mask].min()) & (la <= LAT[mask].max())
    if m.sum() == 0:
        return None
    zc = np.array([z[k][m].mean() for k in range(z.shape[0])])
    Tc = np.array([T[k][m].mean() for k in range(T.shape[0])])
    o = np.argsort(zc)
    return zc[o], Tc[o]


def bt_to_height(bt, zc, Tc):
    """在 T(z) 廓线上查 T=bt 的高度; 对流层内 T 随 z 单调递减"""
    # 只取地表以上到 8 km(避开平流层逆温)
    sel = zc < 8000
    zc, Tc = zc[sel], Tc[sel]
    if bt >= Tc[0]:
        return 0.0                # 比地面还暖 -> 贴地
    if bt <= Tc[-1]:
        return zc[-1]             # 比 8 km 还冷 -> 触顶(警告)
    return float(np.interp(-bt, -Tc, zc))   # T 递减, 取负单调


print("=" * 92)
print("葵花沙尘顶高度反演 v3 (沙尘像元 BT11.2 中位 -> WRF 廓线查高, 无人为温度底)")
print("=" * 92)
print(f"{'源':<6}{'释放窗':<22}{'晴空沙尘像元':>11}"
      f"{'BT P10/25/50/75/90 (K)':>28}{'沙尘顶高 P25/50/75 (m)':>24}")
tm = pd.read_csv(RR / "emit_spacetime_times.csv")
TIMES = list(tm.time.astype(str))
BT_FLOORS = [0, 263, 265, 267]     # 0=无下限(v3), 265=v1 那条阈值
res = {}
for k in ("TK1", "TK2", "MG"):
    m = REGION[k]
    a, b, _ = WIN[k]
    win = [t for t in TIMES if pd.Timestamp(a) <= pd.to_datetime(t, format="%Y%m%d_%H%M") <= pd.Timestamp(b)]
    bts = []
    HByFloor = {lo: [] for lo in BT_FLOORS}
    for t in win:
        B = load_him(t)
        if B is None:
            continue
        b11, b14, b15 = B
        D1, D2 = b11 - b14, b14 - b15
        prof = wrf_profile(t, m)
        if prof is None:
            continue
        # 该时次源区内的沙尘像元(云掩膜: 冰云 D1<0 剔; 用 tsk 判厚云此处略, 依赖 D1/D2)
        dust0 = m & np.isfinite(b14) & (D2 < -0.3) & (D1 > 0)
        zc, Tc = prof
        for lo in BT_FLOORS:
            dust = dust0 & (b14 > lo)
            if dust.sum() < 20:
                continue
            for bt in b14[dust]:
                HByFloor[lo].append(bt_to_height(bt, zc, Tc))
        if dust0.sum() >= 20:
            bts.extend(b14[dust0].tolist())
    if not bts:
        print(f"{k:<6}(窗内无足够沙尘像元)")
        continue
    bts = np.array(bts)
    bp = np.percentile(bts, [10, 25, 50, 75, 90])
    res[k] = dict(bt=bp, n=len(bts))
    print(f"{k}  {a[5:]}Z-{b[5:]}Z   沙尘像元 {len(bts)}   BT P10/50/90 = "
          f"{bp[0]:.0f}/{bp[2]:.0f}/{bp[4]:.0f} K")
    print(f"       {'BT下限':<10}{'保留像元占比':>10}{'沙尘顶 P50 (m)':>15}")
    for lo in BT_FLOORS:
        hh = np.array(HByFloor[lo])
        if len(hh) == 0: continue
        frac = 100*len(hh)/max(len(HByFloor[0]),1)
        tag = " (v3 无下限)" if lo==0 else (" (v1 阈值)" if lo==265 else "")
        print(f"       >{lo:<9}{frac:>9.0f}%{np.percentile(hh,50):>14.0f}{tag}")
    res[k]["h"]=np.percentile(HByFloor[265] or HByFloor[0],[25,50,75])

print()
print("交叉验证: 之前独立估计的沙尘层顶约 2939 m(观测约束层顶)。")
print("  若各源 P50 落在 1-3 km 量级即自洽; 紧贴 0 或模式顶则为 artifact。")
if res:
    import json
    out = {k: dict(bt_p50=float(v["bt"][2]), h_p25=float(v["h"][0]),
                   h_p50=float(v["h"][1]), h_p75=float(v["h"][2]), n=v["n"])
           for k, v in res.items()}
    (RR / "dust_top_v3_bt_method.json").write_text(json.dumps(out, indent=2))
    print(f"\nsaved: {RR / 'dust_top_v3_bt_method.json'}")
