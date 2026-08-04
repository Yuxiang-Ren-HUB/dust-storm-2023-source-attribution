#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把"最后一次模拟"(v3 combo_full 方案)的全部内容归到一个文件夹, 仿照
FINAL_SIM_v2/02_脚本/archive_final_sim.py 的做法。

v3 相对 v2 的核心变化(由 source_regions.py 的历史注释与各脚本 docstring 归纳,
非作者本人复核, 细节以 00_说明/README.md 中的免责声明为准):
    - WRF 用 FDDA nudging 重跑, 修正了 v2 版气象场约 16 小时的相位误差
    - 源区从人工划定的四框(TK/TH/AH/MG)改为由观测起沙频次客观导出的三区
      (TK1 塔里木盆地 / TK2 河西走廊 / MG 蒙古戈壁), 依据见 source_regions.py
    - 粒子分配从"按 DUEM 排放质量"改为"按观测格点·小时"加权, 避免与源强标定
      的循环论证
    - 释放层顶改用葵花卫星沙尘顶反演(而非固定值)
    - 经过 v3 -> s1z2000 -> s2duem -> s3d2w -> final -> final_sfcpbl ->
      final_mgwin -> ... -> combo_test60 -> combo_full 的物理参数敏感性排查链,
      combo_full 是全粒子版终点, 用 L2 最小二乘法重新反解了 West/MG 两个标定系数

原始 flxout NetCDF 留在 F:\\FLEXPART_v3_out, 体积较大, 不复制, 只记清单;
其余(脚本、结果表、图)全部复制进来。
"""
import shutil
from pathlib import Path
from datetime import datetime

PY = Path(r"F:\研\科研\Python")
LC = PY / "atmosphere_revision_figs"
DST = PY / "FINAL_SIM_v3"

SCRIPTS = [
    "source_regions.py",
    "build_flexwrf_input_v3.py", "build_releases_v3_observed.py",
    "build_receptor_windows_v3.py",
    "analyze_source_contribution_v3.py",
    "compare_v3_vs_final_fit.py", "compare_v3_final_surface_scatter.py",
    "retrieve_dust_top_v3.py",
    "plot_dispersion_v3.py", "plot_source_boxes_hotspots_v3.py",
    "plot_figure1_landcover_regions_v3.py", "plot_merra2_compare_v3styles.py",
    "annotate_pm10_sixpanel_v3.py",
    "plot_lanzhou_combo_full_l2.py", "plot_beijing_combo_full_l2.py",
    "plot_relative_contribution_combo_full.py",
    "archive_final_sim_v3.py",
]

FIG_DIRS = [
    LC / "用到的图" / "v3生产模拟_图集",
    LC / "用到的图" / "combo全粒子_最终结果",
]
FIG_LOOSE = [
    LC / "用到的图" / "figure1_landcover_regions_v3.png",
    LC / "用到的图" / "figure1_two_region_with_stations_v3.png",
    LC / "用到的图" / "图2_annotated_v3.png",
    LC / "AH_investigation" / "source_boxes_hotspots_v3_A_greybase.png",
    LC / "AH_investigation" / "source_boxes_hotspots_v3_B_greyland.png",
    LC / "AH_investigation" / "source_boxes_hotspots_v3_C_mutedland.png",
]

TABLES = [
    LC / "reviewer_response" / "dust_top_v3.csv",
    LC / "reviewer_response" / "dust_top_v3_bt_method.json",
    LC / "reviewer_response" / "receptor_response_v3.csv",
    LC / "reviewer_response" / "source_contribution_v3.csv",
    LC / "AH_investigation" / "releases_v3_observed_TK1_TK2_MG.csv",
    LC / "AH_investigation" / "releases_v3_normA_TK1_TK2_MG.csv",
    LC / "AH_investigation" / "releases_v3_obsmass_win_TK1_TK2_MG.csv",
    LC / "AH_investigation" / "releases_v3_normA_win_TK1_TK2_MG.csv",
    LC / "AH_investigation" / "releases_v3_d2wmass_win_TK1_TK2_MG.csv",
]

OBS = [
    LC / "pm10_utc" / "receptor_release_plan_v3.csv",
]

DIRS = ["00_说明", "02_脚本", "03_结果表", "05_图", "06_观测数据"]
for d in DIRS:
    (DST / d).mkdir(parents=True, exist_ok=True)

log = []


def cp(src, dstdir, rename=None):
    src = Path(src)
    if not src.exists():
        log.append(f"  [缺] {src}")
        return False
    t = Path(dstdir) / (rename or src.name)
    shutil.copy2(src, t)
    log.append(f"  [OK] {t.relative_to(DST)}  ({src.stat().st_size / 1024:.0f} KB)")
    return True


print("=" * 88)
print("① 脚本")
print("=" * 88)
for s in SCRIPTS:
    cp(PY / s, DST / "02_脚本")
print("\n".join(log))

print("\n" + "=" * 88)
print("② 结果表")
print("=" * 88)
log.clear()
for f in TABLES:
    cp(f, DST / "03_结果表")
print("\n".join(log))

print("\n" + "=" * 88)
print("③ 图")
print("=" * 88)
log.clear()
for d in FIG_DIRS:
    for f in sorted(d.glob("*.png")):
        cp(f, DST / "05_图")
for f in FIG_LOOSE:
    cp(f, DST / "05_图")
print("\n".join(log))

print("\n" + "=" * 88)
print("④ 观测数据")
print("=" * 88)
log.clear()
for f in OBS:
    cp(f, DST / "06_观测数据")
print("\n".join(log))

n = sum(1 for _ in DST.rglob("*") if _.is_file())
sz = sum(f.stat().st_size for f in DST.rglob("*") if f.is_file())
print("\n" + "=" * 88)
print(f"归档完成: {DST}")
print(f"  {n} 个文件, {sz / 1e6:.1f} MB")
for d in sorted(DST.iterdir()):
    if d.is_dir():
        c = sum(1 for _ in d.rglob("*") if _.is_file())
        print(f"    {d.name:<16} {c:>4} 个文件")
print("=" * 88)
