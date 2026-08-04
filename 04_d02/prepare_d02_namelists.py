#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 ndown 三步所需的三份 namelist.input。

为什么用 ndown 而不是重跑双层嵌套:
  d01(27 km, ERA5 nudging)已经跑完并通过两项验收, 重跑双层要把 d01 一起重算,
  白白多花 8 小时且可能得到与已验收版本不完全一致的父域。ndown 是单向嵌套,
  直接拿现成的 wrfout_d01 作为 d02 的初始与边界, 父域结果完全不变。

必须修正的一处不一致:
  namelist.input 里的 d02 参数(e_we=190, e_sn=136, i_parent_start=82, j_parent_start=78)
  是西移 10 度之前的旧值, 与 namelist.wps 和实际生成的 geo_em.d02.nc(462x252,
  i_parent_start=27, j_parent_start=71)对不上。照旧值跑 real.exe 会失败或错位。

三步与三份 namelist:
  step1_real  max_dom=2, 用 met_em 生成 wrfinput_d01/d02 + wrfbdy_d01
              -> 把 wrfinput_d02 改名为 wrfndi_d02
  step2_ndown max_dom=2, interval_seconds=3600(wrfout_d01 的输出间隔),
              io_form_auxinput2=2; ndown.exe 读 wrfout_d01 + wrfndi_d02
              -> 生成 wrfinput_d02 / wrfbdy_d02, 再改名为 _d01
  step3_wrf   max_dom=1, 把 d02 的网格参数提到第 1 位, time_step 150->50
              (9 km 需要 1/3 的步长), 关掉 nudging(d02 不 nudge)

时段: 03-18 00Z ~ 03-24 00Z(6 天)。覆盖三个源的全部释放窗(03-19 08Z~03-21 13Z)
      与到两个受体的输送, 前面留 1 天 spin-up。全程 10 天要多花 3.4 小时, 不值。
"""
import re
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "namelist.input.d01_current"
txt = SRC.read_text(encoding="utf-8", errors="replace")

# d02 的真实网格参数(与 namelist.wps / geo_em.d02.nc 一致)
D02 = dict(e_we=463, e_sn=253, dx=9000, dy=9000,
           i_parent_start=27, j_parent_start=71)
# 03-19 00Z ~ 03-23 00Z (4 天)。TK1 首次释放 03-19 08Z, 前留 8 h spin-up;
# MG 释放止于 03-21 13Z, 输送到北京 03-22 12Z, 03-23 00Z 足够覆盖。
BEG = dict(year=2023, month=3, day=19, hour=0)
END = dict(year=2023, month=3, day=23, hour=0)
RUN_DAYS = 4


def setval(t, key, val, dom2=None):
    """替换 namelist 中某项的值; dom2 给出则写成 'v1, v2,' 双域形式"""
    v = f"{val}, {dom2}," if dom2 is not None else f"{val},"
    pat = re.compile(rf"^(\s*{key}\s*=\s*)([^!\n]*)(.*)$", re.M)
    if not pat.search(t):
        raise SystemExit(f"未找到 {key}")
    return pat.sub(lambda m: f"{m.group(1)}{v:<24}{m.group(3)}", t, count=1)


def base(t):
    """三步共用的修改: 时段、d02 真实网格参数"""
    t = setval(t, "run_days", RUN_DAYS)
    t = setval(t, "run_hours", 0)
    for k, v in BEG.items():
        t = setval(t, f"start_{k}", f"{v:02d}" if k != "year" else v,
                   f"{v:02d}" if k != "year" else v)
    for k, v in END.items():
        t = setval(t, f"end_{k}", f"{v:02d}" if k != "year" else v,
                   f"{v:02d}" if k != "year" else v)
    # d02 网格参数改成真实值
    t = setval(t, "e_we", 240, D02["e_we"])
    t = setval(t, "e_sn", 176, D02["e_sn"])
    t = setval(t, "i_parent_start", 1, D02["i_parent_start"])
    t = setval(t, "j_parent_start", 1, D02["j_parent_start"])
    return t


# ---------------- step 1: real.exe ----------------
s1 = base(txt)
s1 = setval(s1, "max_dom", 2)
(HERE / "namelist.input.step1_real").write_text(s1, encoding="utf-8")

# ---------------- step 2: ndown.exe ----------------
s2 = base(txt)
s2 = setval(s2, "max_dom", 2)
# ndown 读的是 wrfout_d01(逐时), 不是 met_em(6 小时)
s2 = setval(s2, "interval_seconds", 3600)
if "io_form_auxinput2" in s2:
    s2 = setval(s2, "io_form_auxinput2", 2)
else:
    s2 = s2.replace(" &time_control", " &time_control\n io_form_auxinput2                   = 2,", 1)
(HERE / "namelist.input.step2_ndown").write_text(s2, encoding="utf-8")

# ---------------- step 3: wrf.exe (d02 单域) ----------------
s3 = base(txt)
s3 = setval(s3, "max_dom", 1)
# 把 d02 的参数提到第 1 位
s3 = setval(s3, "e_we", D02["e_we"])
s3 = setval(s3, "e_sn", D02["e_sn"])
s3 = setval(s3, "dx", D02["dx"])
s3 = setval(s3, "dy", D02["dy"])
s3 = setval(s3, "i_parent_start", 1)
s3 = setval(s3, "j_parent_start", 1)
s3 = setval(s3, "grid_id", 1)
s3 = setval(s3, "parent_id", 0)
s3 = setval(s3, "parent_grid_ratio", 1)
s3 = setval(s3, "parent_time_step_ratio", 1)
s3 = setval(s3, "time_step", 50)          # 9 km 需 1/3 步长; 3600/50=72 整除
s3 = setval(s3, "interval_seconds", 3600)
# d02 不做 nudging: 正确相位已由 d01 边界带入, 内域 nudging 会压制
# 我们正要分辨的地形强迫垂直运动。
s3 = setval(s3, "grid_fdda", 0)
(HERE / "namelist.input.step3_wrf").write_text(s3, encoding="utf-8")

print("已生成三份 namelist:")
for f in ("step1_real", "step2_ndown", "step3_wrf"):
    p = HERE / f"namelist.input.{f}"
    print(f"  {p.name}")
print()
print("核对 d02 参数:")
s = (HERE / "namelist.input.step3_wrf").read_text(encoding="utf-8")
for k in ("max_dom", "e_we", "e_sn", "dx", "time_step", "grid_fdda",
          "start_day", "end_day", "run_days", "interval_seconds"):
    m = re.search(rf"^\s*{k}\s*=\s*([^!\n]*)", s, re.M)
    print(f"  {k:<22} {m.group(1).strip() if m else '?'}")
