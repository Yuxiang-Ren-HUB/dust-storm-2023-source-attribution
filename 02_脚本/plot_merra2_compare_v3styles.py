#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MERRA-2 DUSMASS vs PM10 相关性: 三种画法供选择"""
import numpy as np
import xarray as xr
import openpyxl
import csv
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime, timedelta

STATION_LIST = r"D:\Chapter3.1\data\中国空气质量\_站点列表\站点列表-2022.02.13起.xlsx"
PM10_DIR = Path(r"D:\Chapter3.1\data\中国空气质量\站点_20150101-20251231\站点_20230101-20231231")
MERRA_DIR = Path(r"D:\merra2_aer")

LZ_BOX_TIGHT = (103.43, 35.65, 104.23, 36.45)
BJ_BOX_TIGHT = (115.99, 39.51, 116.79, 40.31)
LZ_BOX_ORIG = (101.0, 33.5, 106.5, 39.0)
BJ_BOX_ORIG = (114.0, 37.0, 120.0, 42.5)

CITIES = {
    "Lanzhou": dict(center=(103.83, 36.05), tight=LZ_BOX_TIGHT, orig=LZ_BOX_ORIG, color="#c0742d"),
    "Beijing": dict(center=(116.39, 39.91), tight=BJ_BOX_TIGHT, orig=BJ_BOX_ORIG, color="#3f6fae"),
}
PM10_DATES = [f"202303{d:02d}" for d in range(18, 25)]
MERRA_DATES = [f"202303{d:02d}" for d in range(17, 25)]

REF_COLOR = "#9a9a94"   # 大框(原方案), 中性灰
ACCENT_MAP = {"Lanzhou": "#c0742d", "Beijing": "#3f6fae"}  # 小框(最终方案), 主色


def stations_in_box(rows, box):
    x0, y0, x1, y1 = box
    codes = []
    for r in rows:
        code, lon, lat = r[0], r[3], r[4]
        try:
            lon = float(lon); lat = float(lat)
        except (TypeError, ValueError):
            continue
        if x0 <= lon <= x1 and y0 <= lat <= y1:
            codes.append(code)
    return codes


def load_pm10_series(dates, codes):
    result = {}
    for date in dates:
        fn = PM10_DIR / f"china_sites_{date}.csv"
        if not fn.exists():
            continue
        with open(fn, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("type") != "PM10":
                    continue
                hour = int(row["hour"])
                t_bjt = datetime.strptime(row["date"], "%Y%m%d") + timedelta(hours=hour)
                t_utc = t_bjt - timedelta(hours=8)
                vals = []
                for c in codes:
                    v = row.get(c, "")
                    if v not in ("", None):
                        try:
                            vals.append(float(v))
                        except ValueError:
                            pass
                if vals:
                    result[t_utc] = np.mean(vals)
    return result


def load_merra_dusmass_series(cx, cy):
    series = {}
    for date in MERRA_DATES:
        fn = MERRA_DIR / f"MERRA2_400.tavg1_2d_aer_Nx.{date}.nc4"
        ds = xr.open_dataset(fn)
        lat = ds["lat"].values
        lon = ds["lon"].values
        j = np.argmin(np.abs(lat - cy))
        i = np.argmin(np.abs(lon - cx))
        dusmass = ds["DUSMASS"].isel(lat=j, lon=i).values * 1e9
        times = ds["time"].values
        for ti in range(len(times)):
            t_val = datetime.utcfromtimestamp(times[ti].astype("datetime64[s]").astype(int))
            t_floor = t_val.replace(minute=0, second=0, microsecond=0)
            series[t_floor] = float(dusmass[ti])
        ds.close()
    return series


def stats(obs, model):
    r = np.corrcoef(obs, model)[0, 1]
    bias = np.mean(model - obs)
    rmse = np.sqrt(np.mean((model - obs) ** 2))
    return r, bias, rmse


wb = openpyxl.load_workbook(STATION_LIST, read_only=True)
ws = wb.active
station_rows = list(ws.iter_rows(values_only=True))[1:]

# --- 预计算所有需要的数据 ---
data = {}
for city, info in CITIES.items():
    model_series = load_merra_dusmass_series(*info["center"])
    data[city] = {"model": model_series}
    for box_key, box in [("orig", info["orig"]), ("tight", info["tight"])]:
        codes = stations_in_box(station_rows, box)
        pm10_series = load_pm10_series(PM10_DATES, codes)
        common_t = sorted(set(pm10_series.keys()) & set(model_series.keys()))
        obs_arr = np.array([pm10_series[t] for t in common_t])
        model_arr = np.array([model_series[t] for t in common_t])
        r, bias, rmse = stats(obs_arr, model_arr)
        data[city][box_key] = dict(t=common_t, obs=obs_arr, model=model_arr,
                                     n=len(common_t), n_stations=len(codes), r=r, bias=bias, rmse=rmse)

print("数据准备完毕")
for city in CITIES:
    for bk in ["orig", "tight"]:
        d = data[city][bk]
        print(f"{city} {bk}: n={d['n']} r={d['r']:.3f} bias={d['bias']:.1f} rmse={d['rmse']:.1f}")


# ============================================================
# 方案A: 改良版log-log散点图 (大框/小框同框对比)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), constrained_layout=True)
for ax, city in zip(axes, CITIES):
    d_orig = data[city]["orig"]
    d_tight = data[city]["tight"]
    valid_o = (d_orig["obs"] > 0) & (d_orig["model"] > 0)
    valid_t = (d_tight["obs"] > 0) & (d_tight["model"] > 0)

    ax.scatter(d_orig["obs"][valid_o], d_orig["model"][valid_o], c=REF_COLOR, s=22, alpha=0.65,
               edgecolor="none", zorder=2, label=f"original box ({d_orig['n_stations']} stn)")
    ax.scatter(d_tight["obs"][valid_t], d_tight["model"][valid_t], c=ACCENT_MAP[city], s=22, alpha=0.8,
               edgecolor="black", linewidth=0.3, zorder=3, label=f"tight box ({d_tight['n_stations']} stn)")

    all_obs = np.concatenate([d_orig["obs"][valid_o], d_tight["obs"][valid_t]])
    all_model = np.concatenate([d_orig["model"][valid_o], d_tight["model"][valid_t]])
    lims = [min(all_obs.min(), all_model.min()) / 1.3, max(all_obs.max(), all_model.max()) * 1.3]
    ax.plot(lims, lims, color="#4a4a48", linewidth=1.0, linestyle="--", zorder=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Observed PM10 (ug/m3)")
    ax.set_ylabel("MERRA-2 DUSMASS (ug/m3)")
    ax.set_title(city, fontsize=12)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(color="#e8e8e5", linewidth=0.5, which="both")
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)

    stat_text = (f"orig: r={d_orig['r']:.2f}, RMSE={d_orig['rmse']:.0f}\n"
                 f"tight: r={d_tight['r']:.2f}, RMSE={d_tight['rmse']:.0f}")
    ax.text(0.98, 0.03, stat_text, transform=ax.transAxes, fontsize=8.5, ha="right", va="bottom",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="#c9c8c2", alpha=0.9))

fig.suptitle("Method A: scatter comparison (original vs. tight receptor box, overlaid)", fontsize=12)
plt.savefig(r"F:\研\科研\Python\atmosphere_revision_figs\merra2_compare_methodA_scatter.png",
            dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("方案A已保存")

# ============================================================
# 方案B: 归一化时间序列叠加图
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
for ax, city in zip(axes, CITIES):
    d = data[city]["tight"]
    obs_norm = d["obs"] / d["obs"].max() * 100
    model_norm = d["model"] / d["model"].max() * 100
    ax.plot(d["t"], obs_norm, color="#4a4a48", linewidth=1.4, label="Observed PM10 (norm.)")
    ax.plot(d["t"], model_norm, color=ACCENT_MAP[city], linewidth=1.4, label="MERRA-2 DUSMASS (norm.)")
    ax.set_ylabel("% of own peak value")
    ax.set_title(f"{city} (tight box, r={d['r']:.2f})", fontsize=11)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(color="#e8e8e5", linewidth=0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.set_ylim(-5, 110)

fig.suptitle("Method B: normalized time-series overlay (each series scaled to its own peak = 100%)", fontsize=12)
plt.savefig(r"F:\研\科研\Python\atmosphere_revision_figs\merra2_compare_methodB_normalized.png",
            dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("方案B已保存")

# ============================================================
# 方案C: 纯统计指标柱状图
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
labels = ["Lanzhou\n(original)", "Lanzhou\n(tight)", "Beijing\n(original)", "Beijing\n(tight)"]
colors = [REF_COLOR, ACCENT_MAP["Lanzhou"], REF_COLOR, ACCENT_MAP["Beijing"]]
r_vals = [data["Lanzhou"]["orig"]["r"], data["Lanzhou"]["tight"]["r"],
          data["Beijing"]["orig"]["r"], data["Beijing"]["tight"]["r"]]
bias_vals = [data["Lanzhou"]["orig"]["bias"], data["Lanzhou"]["tight"]["bias"],
             data["Beijing"]["orig"]["bias"], data["Beijing"]["tight"]["bias"]]
rmse_vals = [data["Lanzhou"]["orig"]["rmse"], data["Lanzhou"]["tight"]["rmse"],
             data["Beijing"]["orig"]["rmse"], data["Beijing"]["tight"]["rmse"]]

x = np.arange(4)
for ax, vals, title, ylabel in zip(
        axes, [r_vals, bias_vals, rmse_vals],
        ["Correlation (r)", "Bias (model - obs)", "RMSE"],
        ["r", "ug/m3", "ug/m3"]):
    bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.4)
    for xi, v in zip(x, vals):
        ax.text(xi, v + (0.02 * max(abs(min(vals)), abs(max(vals)))) * np.sign(v if v != 0 else 1),
                f"{v:.2f}" if title.startswith("Correlation") else f"{v:.0f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_title(title, fontsize=11)
    ax.set_ylabel(ylabel)
    ax.axhline(0, color="#333333", linewidth=0.6)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#e8e8e5", linewidth=0.5)

fig.suptitle("Method C: summary statistics bar chart (gray = original box, color = tight box)", fontsize=12)
plt.savefig(r"F:\研\科研\Python\atmosphere_revision_figs\merra2_compare_methodC_barstats.png",
            dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("方案C已保存")
