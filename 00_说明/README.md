# 最终模拟 v3(combo_full 方案)

支撑论文: **"Dual-Source Transport, Vertical Evolution, and Topographic Modulation of the
March 2023 East Asian Dust Storm in the Context of 2000–2024 Spring Dust Variability"**
(atmosphere-4373208)

生成日期: 2026-08-05

论文标题里的三个关键词与本目录的对应关系:
- **Dual-Source**(双源)→ 西路(TK1+TK2)vs 北路(MG)的二源结构,见第二节 `source_regions.py`
- **Vertical Evolution**(垂直演变)→ 葵花沙尘顶反演定的释放层顶(TK1 3700m/TK2 3500m/MG 1900m)
- **Topographic Modulation**(地形调制)→ `04_d02/` 的 WRF 27km→9km 双重嵌套地形分辨率检验

> **免责声明**:本文档由我(Claude)读取各脚本的 docstring、注释和 `source_regions.py`
> 里的历史修订记录整理而成,不是作者本人撰写的方法学陈述。发布/引用前请通读核实,
> 尤其是"三、发现的一处不一致"一节——这处我没有替你下判断,需要你自己确认。

---

## 一、v3 相对 v2 的核心变化

| 项目 | v2(`FINAL_SIM_v2`) | v3(本目录) |
|---|---|---|
| 气象场 | WRF 原始积分 | WRF **FDDA nudging 重跑**,修正约16小时相位误差(见 `build_flexwrf_input_v3.py` 注释) |
| 源区划分 | 人工四框 TK/TH/AH/MG(103E 东西分界) | 由观测起沙频次**客观导出**的三区 TK1(塔里木盆地)/ TK2(河西走廊)/ MG(蒙古戈壁),依据见 `source_regions.py` 中 2026-07-22 的多条一致性检验(空间连续性、起沙时序、锋面过境时序、后向轨迹归因) |
| 粒子分配 | 按 DUEM 排放质量 | 按**观测"格点·小时"**加权,避免与源强标定形成循环论证 |
| 释放层顶 | 固定值 | 葵花卫星沙尘顶反演(`retrieve_dust_top_v3.py`):TK1 3700 m / TK2 3500 m / MG 1900 m |
| 参数敏感性 | 8 维度 OAT,12 个成员 | 一条更长的排查链:`v3 → s1z2000 → s2duem → s3d2w → final → final_sfcpbl → final_mgwin → final_mgwin_test60 → final_sfcpbl2 → newwin_test60 → d2wspread_test60 → combo_test60 → combo_full`(+ `combo_full_p0`–`p11` 全粒子系综) |
| 标定方法 | 源区质量比订正 | 受体端 **L2 最小二乘**反解 West(TK1+TK2 合并)/ MG 两个系数,`combo_full` 完成后用同样方法重新反解过一次 |

`01_模式配置/` 里那一长串 `flexwrf.input.*` 后缀,就是上面这条排查链每一步对应的配置文件。

## 二、combo_full 是什么

`combo_full` = 延长释放窗口(TK1 04–20Z / TK2 15Z–次日16Z / MG 00–21Z)+ 区域平均铺散
(`D2W_SPREAD_REGION`,填补云下漏检)的组合配置,**全粒子版**(区别于之前用于快速试错的
1/60 粒子噪声测试 `combo_test60`)。

`plot_lanzhou_combo_full_l2.py` / `plot_beijing_combo_full_l2.py` 的脚本内注释明确写着
"这是全粒子真实结果, 不是1/60噪声测试",并给出 L2 最优系数:**West(TK1+TK2)= 23.37x,
MG = 22.07x**(见 `05_图/relative_contribution_combo_full.png` 及两张 combo_full_l2optimal 图)。

## 三、发现的一处不一致(需要你确认)

`02_脚本/source_regions.py` 第 152–166 行有一个常量 `CALIB_COMBO_PRELIM = {"TK1": 14.23,
"TK2": 14.23, "MG": 15.77}`,上方注释写着:

> "!! 暂定值,来自1/60粒子噪声测试(combo_test60),全粒子版(combo_full)跑完后必须用
> 同样方法重新反解一遍并更新此处,不要在论文里直接引用这组数字。"

但实际用于最终 `combo_full` 图的系数是 **23.37 / 22.07**(见上一节),和这个常量里的
14.23 / 15.77 对不上——也就是说 `combo_full` 跑完后,`plot_*_combo_full_l2.py` 里确实
重新反解了,但 `source_regions.py` 里的 `CALIB_COMBO_PRELIM` 常量本身**没有同步更新**。

如果之后还有代码会 `from source_regions import CALIB_COMBO_PRELIM`,拿到的会是过期的
14.23/15.77 而不是实际用的 23.37/22.07。建议你确认一下这是不是需要修的遗留问题
(比如加一个 `CALIB_COMBO_FULL = {"TK1": 23.37, "TK2": 23.37, "MG": 22.07}` 之类的新常量,
并把 `CALIB_COMBO_PRELIM` 标记为已废弃)。这一步我没有替你改,因为不确定 23.37/22.07
是否就是你想保留的最终值,还是还有更新的版本我没找到。

## 四、目录结构

```
FINAL_SIM_v3/
├── 00_说明/          本文件
├── 01_模式配置/      flexwrf.input 参数扫描链(v3 → combo_full,共 51 个文件)
├── 02_脚本/          源区定义(source_regions.py)、释放清单构建、L2 标定、绘图、诊断
├── 03_结果表/        释放清单 CSV、源贡献、受体响应、沙尘顶反演结果
├── 04_d02/           WRF 双重嵌套(ndown)配置与运行脚本,用于地形分辨率检验
├── 05_图/            全部 v3/combo_full 相关图(73 张,含前几版对比图)
└── 06_观测数据/      v3 受体释放计划
```

原始 FLEXPART 输出(`flxout_*.nc`)留在 `F:\FLEXPART_v3_out`,体积较大,未复制进本目录。

## 五、尚未确认/待你补充的部分

- 上面第三节的系数不一致问题
- `combo_full` 是否已经是最终定稿,还是后面还有 `combo_full_p0`–`p11` 系综的进一步处理步骤
- 04_d02(WRF 27km→9km 双重嵌套)的检验结论是否已经完成并写入正文
