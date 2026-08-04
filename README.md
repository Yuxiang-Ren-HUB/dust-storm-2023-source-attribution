# Dust Storm 2023 — Source Attribution

Code and results supporting:

**"Dual-Source Transport, Vertical Evolution, and Topographic Modulation of the
March 2023 East Asian Dust Storm in the Context of 2000–2024 Spring Dust Variability"**
(manuscript atmosphere-4373208)

FLEXPART-WRF forward/backward trajectory simulation with FDDA-nudged meteorology,
objectively-derived TK1 (Tarim Basin) / TK2 (Hexi Corridor) / MG (Mongolian Gobi)
source regions, and observation-weighted particle release, converging on the
full-particle `combo_full` configuration with L2-calibrated source-contribution
coefficients.

## Structure

| Folder | Contents |
|---|---|
| `00_说明/` | Full methodology write-up, version-history notes, known open items |
| `01_模式配置/` | FLEXPART `flexwrf.input` files across the parameter-sensitivity chain (v3 → combo_full) |
| `02_脚本/` | Source-region definitions, release construction, calibration, plotting, diagnostics |
| `03_结果表/` | Release manifests, source-contribution and receptor-response tables, dust-top retrievals |
| `04_d02/` | WRF two-way nested-domain (27 km → 9 km) setup for terrain-resolution testing |
| `05_图/` | Figures from the v3 methodology and the `combo_full` production simulation |
| `06_观测数据/` | Receptor release plan / observation bookkeeping |

Raw FLEXPART NetCDF output is not included here (large binary files); `00_说明/README.md`
records where the source data lives and how to regenerate it from the scripts.

**Start here:** [`00_说明/README.md`](00_说明/README.md) for the full methodology, what changed
relative to the earlier four-box scheme, and a documented open item worth checking before
citing calibration coefficients from `source_regions.py`.
