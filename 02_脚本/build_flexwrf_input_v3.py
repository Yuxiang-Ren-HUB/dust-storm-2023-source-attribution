#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""组装 v3 的三个 flexwrf.input(TK1 / TK2 / MG), 正向。

参数全部沿用 FINAL_SIM_v2/01_模式配置/*.v2final1h 那批已跑通的配置, 只改三处:
  1. RELEASES  —— 换成 v3 观测门控清单(releases_v3_obsmass_win_*.csv)
  2. 起止时间  —— 配合 nudged WRF 的可用时段
  3. 输出路径  —— 各源独立目录, 便于事后按源反演质量比

粒子分配:
  按观测『格点·小时』加权, 即与 obsmass 清单的质量成正比, 每块下限 100。
  为什么不按 DUEM 质量分配: 源总质量比要留到跑完由受体 PM10 反演决定,
  若按某一口径分配粒子, 另一口径下该源就会欠采样。观测格点·小时介于两种
  口径之间(TK1 30.3% / TK2 16.8% / MG 52.9%), 哪种口径都不至于欠分辨。

释放块为 12 行, 顺序: ID1 / ID2 / XPOINT1 / YPOINT1 / XPOINT2 / YPOINT2 /
KINDZ / ZPOINT1 / ZPOINT2 / NPART / XMASS / name。
(2026-07 曾按 13 行切片导致 FLEXPART 读完 RELEASES 即崩, 此处务必保持 12 行。)
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
sys.path.insert(0, r"F:\研\科研\Python")
from source_regions import REGIONS3, CALIB3, RELEASE_ZTOP3

AH = Path(r"F:\研\科研\Python") / "atmosphere_revision_figs" / "AH_investigation"
OUTDIR = Path(r"F:\研\科研\Python") / "FINAL_SIM_v3" / "01_模式配置"
OUTDIR.mkdir(parents=True, exist_ok=True)

import argparse
_ap = argparse.ArgumentParser()
_ap.add_argument("--csv", default="releases_v3_obsmass_win_TK1_TK2_MG.csv")
_ap.add_argument("--ztop", type=float, default=None,
                 help="全源统一释放层顶(m)。不给则用 source_regions.RELEASE_ZTOP3 逐源取值"
                      "(葵花沙尘顶反演: TK1 3700/TK2 3500/MG 1900m)。")
_ap.add_argument("--parmean", type=float, default=2.0,
                 help="粒径中值(um)。默认 2.0(GOCART 标称); 1.3 为物种敏感性验证过的"
                      "更优沉降组(MG->北京受体峰值提升 4.16x)。")
_ap.add_argument("--tag", default="v3")
_ap.add_argument("--sfc-option", type=int, default=0, choices=[0, 1],
                 help="SFC_OPTION: 0=FLEXPART自行计算u*/热通量/边界层高度(默认), "
                      "1=直接用WRF模拟的u*/HFX/PBLH。用于测试夜间边界层收缩后"
                      "近地层沙尘是否被过快耗散(衰减尾偏低的敏感性检验)。"
                      "此build要求SFC_OPTION=1时TURB_OPTION必须>=4(V19方案),"
                      "否则readinput.f90会直接stop, 见--turb-option。")
_ap.add_argument("--turb-option", type=int, default=1, choices=[0, 1, 2, 3, 4, 5],
                 help="TURB_OPTION: 0无湍流/1诊断(默认,与flexpart_ecmwf一致)/"
                      "2=tke/3=mytke/4=V19_SDA/5=V19_StepTKE。"
                      "--sfc-option 1 必须搭配 4 或 5, 否则模式启动即报错退出。")
_ap.add_argument("--particle-scale", type=float, default=1.0,
                 help="总粒子数/每块下限粒子数同比例缩放(如0.0167=1/60), 用于快速"
                      "测试跑(结果统计噪声更大, 不作为正式结果)。")
_a = _ap.parse_args()
if _a.sfc_option == 1 and _a.turb_option < 4:
    raise SystemExit(f"--sfc-option 1 需要 --turb-option 4 或 5 (V19方案), "
                     f"当前 --turb-option {_a.turb_option} 会被模式直接拒绝启动。")
CSV = AH / _a.csv
TOTAL_PARTICLES = max(1, round(600_000 * _a.particle_scale))   # 三源合计, 可用 --particle-scale 缩放做快速测试
NPART_FLOOR = max(1, round(100 * _a.particle_scale))            # 每块下限, 同比例缩放
SIM_BEG = "20230319 000000"
SIM_END = "20230327 000000"        # nudged WRF 积分到 03-27, 不可超出
Z0 = 0.0
ZTOP = {k: _a.ztop for k in REGIONS3} if _a.ztop is not None else dict(RELEASE_ZTOP3)
PARMEAN_STR = f"{_a.parmean * 1e-6:.1E}"     # --parmean 以 um 为单位输入, SPECIES 行要求 m

# VM 上的路径。若与实际不符, 改这三行即可。
# 注意: /home/WRF_run 是 nudging 之前的旧输出(全部 07-16 修改, 非整点命名),
# nudged 重跑的输出在 /home/rencheng/WRF_dust(07-22, 整点命名)。用错这个
# 目录会拿到带 16 小时相位误差的那版气象场。
VM_WRF = "/home/rencheng/WRF_dust/"
VM_AVAIL = "/home/rencheng/AVAILABLE_nudged"
VM_OUT = "/home/rencheng/flexpart_runs_v3/{src}_fwd_" + _a.tag + "/output/"

rel = pd.read_csv(CSV, parse_dates=["hour_start_UTC"])
tot_mass = rel.mass_kg.sum()
rel["npart"] = np.maximum(
    NPART_FLOOR, np.round(TOTAL_PARTICLES * rel.mass_kg / tot_mass)).astype(int)

HEAD = """=====================FORMER PATHNAMES FILE===================
{out}
{wrf}
{avail}
=============================================================
=====================FORMER COMMAND FILE=====================
    1                LDIRECT:          1 for forward simulation, -1 for backward simulation
    {beg}  YYYYMMDD HHMISS   beginning date of simulation
    {end}  YYYYMMDD HHMISS   ending date of simulation
    3600             SSSSS  (int)      output every SSSSS seconds
    3600             SSSSS  (int)      time average of output (in SSSSS seconds)
    900              SSSSS  (int)      sampling rate of output (in SSSSS seconds)
    999999999        SSSSS  (int)      time constant for particle splitting (in seconds)
    900              SSSSS  (int)      synchronisation interval of flexpart (in seconds)
    10.              CTL    (real)     factor by which time step must be smaller than tl
    10               IFINE  (int)      decrease of time step for vertical motion by factor ifine
    1                IOUT              1 concentration, 2 mixing ratio, 3 both, 4 plume traject, 5=1+4
    1                IPOUT             particle dump: 0 no, 1 every output interval, 2 only at end
    0                LSUBGRID          subgrid terrain effect parameterization: 1 yes, 0 no
    0                LCONVECTION       convection: 3 yes, 0 no
    3600.            DT_CONV  (real)   time interval to call convection, seconds
    1                LAGESPECTRA       age spectra: 1 yes, 0 no
    0                IPIN              continue simulation with dumped particle data: 1 yes, 0 no
    0                IFLUX             calculate fluxes: 1 yes, 0 no
    0                IOUTPUTFOREACHREL CREATE AN OUPUT FILE FOR EACH RELEASE LOCATION: 1 YES, 0 NO
    0                MDOMAINFILL       domain-filling trajectory option: 1 yes, 0 no, 2 strat. o3 tracer
    1                IND_SOURCE        1=mass unit , 2=mass mixing ratio unit
    2                IND_RECEPTOR      1=mass unit , 2=mass mixing ratio unit
    0                NESTED_OUTPUT     shall nested output be used? 1 yes, 0 no
    0                LINIT_COND   INITIAL COND. FOR BW RUNS: 0=NO,1=MASS UNIT,2=MASS MIXING RATIO UNIT
    {turb_option}                TURB_OPTION       0=no turbulence; 1=diagnosed as in flexpart_ecmwf; 2 and 3=from tke.
    1                LU_OPTION         0=old landuse (IGBP.dat); 1=landuse from WRF
    0                CBL SCHEME        0=no, 1=yes. works if TURB_OPTION=1
    {sfc_option}                SFC_OPTION        0=default computation of u*, hflux, pblh, 1=from wrf
    0                WIND_OPTION       0=snapshot winds, 1=mean winds,2=snapshot eta-dot,-1=w based on divergence
    0                TIME_OPTION       1=correction of time validity for time-average wind,  0=no need
    1                OUTGRID_COORD     0=wrf grid(meters), 1=regular lat/lon grid
    1                RELEASE_COORD     0=wrf grid(meters), 1=regular lat/lon grid
    2                IOUTTYPE          0=default binary, 1=ascii (for particle dump only),2=netcdf
    3                NCTIMEREC (int)   Time frames per output file, only used for netcdf
    0                VERBOSE           VERBOSE MODE,0=minimum, 100=maximum
=====================FORMER AGECLASESS FILE==================
    1                NAGECLASS        number of age classes
    999999           SSSSSS  (int)    age class in SSSSS seconds
=====================FORMER OUTGRID FILE=====================
    70.00            OUTLONLEFT      geographical longitude of lower left corner of output grid
    15.00            OUTLATLOWER     geographical latitude of lower left corner of output grid
    280              NUMXGRID        number of grid points in x direction (= # of cells )
    160              NUMYGRID        number of grid points in y direction (= # of cells )
    0                OUTGRIDDEF      outgrid defined 0=using grid distance, 1=upperright corner coordinate
    0.2500           DXOUTLON        grid distance in x direction or upper right corner of output grid
    0.2500           DYOUTLON        grid distance in y direction or upper right corner of output grid
    5                NUMZGRID        number of vertical levels
    100.0            LEVEL           height of level (upper boundary)
    500.0            LEVEL           height of level (upper boundary)
   1000.0            LEVEL           height of level (upper boundary)
   2000.0            LEVEL           height of level (upper boundary)
   5000.0            LEVEL           height of level (upper boundary)
=====================FORMER RECEPTOR FILE====================
    0                NUMRECEPTOR     number of receptors
=====================FORMER SPECIES FILE=====================
     1               NUMTABLE        number of variable properties. The following lines are fixed format
XXXX|NAME    |decaytime |wetscava  |wetsb|drydif|dryhenry|drya|partrho  |parmean|partsig|dryvelo|weight |
    DUST            -9.9    5.0E-06  0.80   -9.9                 2.6E+03 {parmean} 2.0E+00   -9.90  100.00
=====================FORMER RELEEASES FILE===================
   1                NSPEC           total number of species emitted
   0                EMITVAR         1 for emission variation
   1                LINK            index of species in file SPECIES
   {npoint}                NUMPOINT        number of releases
"""

summary = []
for src in REGIONS3:
    s = rel[rel.source == src].sort_values("hour_start_UTC").reset_index(drop=True)
    z1 = ZTOP[src]
    head_txt = HEAD.format(out=VM_OUT.format(src=src), wrf=VM_WRF, avail=VM_AVAIL,
                           beg=SIM_BEG, end=SIM_END, npoint=len(s),
                           parmean=PARMEAN_STR, sfc_option=_a.sfc_option,
                           turb_option=_a.turb_option).rstrip("\n")
    lines = [head_txt]
    for i, r in s.iterrows():
        t0 = r.hour_start_UTC
        t1 = t0 + pd.Timedelta(hours=1)
        # 12 行, 顺序固定
        lines += [
            f"{t0:%Y%m%d %H%M%S}   ID1, IT1        beginning date and time of release",
            f"{t1:%Y%m%d %H%M%S}   ID2, IT2        ending date and time of release",
            f"{r.lon0:12.4f}         XPOINT1 (real)  longitude [deg] of lower left corner",
            f"{r.lat0:12.4f}         YPOINT1 (real)  latitude [deg] of lower left corner",
            f"{r.lon1:12.4f}         XPOINT2 (real)  longitude [deg] of upper right corner",
            f"{r.lat1:12.4f}         YPOINT2 (real)  latitude [DEG] of upper right corner",
            f"{1:10d}           KINDZ  (int)  1 for m above ground, 2 for m above sea level, 3 pressure",
            f"{Z0:10.3f}         ZPOINT1 (real)  lower z-level",
            f"{z1:10.3f}         ZPOINT2 (real)  upper z-level  (source-specific, Himawari dust-top retrieval)",
            f"{int(r.npart):6d}           NPART (int)     total number of particles to be released",
            f"{r.mass_kg:12.4E}         XMASS (real)    total mass emitted, species 1",
            f"{src}_{t0:%d%H}_b{i+1}",
        ]
    p = OUTDIR / f"{src}_flexwrf.input.{_a.tag}"
    p.write_text("\n".join(lines) + "\n", encoding="ascii", errors="replace")
    summary.append((src, len(s), int(s.npart.sum()), s.mass_kg.sum(),
                    s.hour_start_UTC.min(), s.hour_start_UTC.max(), p))

print("=" * 100)
print("v3 flexwrf.input 组装完成")
print("=" * 100)
print(f"{'源':<6}{'释放块':>7}{'粒子数':>10}{'名义质量(Tg)':>14}{'释放时段':<26}{'层顶(m)':>9}{'待乘系数':>9}")
for src, nb, npx, m, a, b, p in summary:
    print(f"{src:<6}{nb:>7}{npx:>10,}{m/1e9:>13.4f}"
          f"  {f'{a:%m-%d %H}Z - {b:%m-%d %H}Z':<24}{ZTOP[src]:>8.0f}{CALIB3[src]:>8.1f}x")
print(f"\n粒径中值 parmean = {_a.parmean:.2f} um = {PARMEAN_STR} m")
print(f"\n合计 {sum(x[1] for x in summary)} 块, {sum(x[2] for x in summary):,} 粒子")
print("(旧批四源合计 52.8 万粒子, 同量级)")
for src, nb, npx, m, a, b, p in summary:
    print(f"  -> {p}")

print("\n" + "=" * 100)
print("跑之前请核对 VM 上的三个路径")
print("=" * 100)
print(f"  WRF 输出目录   {VM_WRF}")
print(f"  AVAILABLE 文件 {VM_AVAIL}    <- 必须是 nudged 那版")
print(f"  输出目录       {VM_OUT.format(src='<SRC>')}    <- 需先 mkdir -p")
print(f"\n模拟时段 {SIM_BEG} ~ {SIM_END}")
print("  注: nudged WRF 积分到 03-27, 结束时间不可超出, 否则读不到气象场。")

# 自检: 释放块行数必须是 12 的整数倍
for src, nb, npx, m, a, b, p in summary:
    L = p.read_text(encoding="ascii").split("\n")
    i = [j for j, l in enumerate(L) if "NUMPOINT" in l][0]
    body = [l for l in L[i + 1:] if l.strip()]
    assert len(body) == 12 * nb, f"{src}: 释放块行数 {len(body)} != 12*{nb}"
print("\n自检通过: 三个文件的释放块均为 12 行/块。")
