#!/bin/bash
# d02 (9 km) 单向嵌套运行手册 —— ndown 三步
#
# 目的: 回应审稿意见"地形对沙尘传输的主动调制定量分析不足"。
#       27 km 分辨不出地形强迫的垂直运动与辐合, 9 km 可以。
#       需要的诊断量: W(垂直速度)、U/V(辐合)、HGT(地形剖面)、上下游浓度对比。
#
# 为什么用 ndown 不重跑双层:
#   d01 已跑完并通过两项验收(相位误差 16h->2h, 输送速度 21.4 m/s)。
#   ndown 单向嵌套直接用现成 wrfout_d01 作 d02 的初始与边界, 父域结果一字不动。
#
# ★★ 必须在独立目录运行 ★★
#   d02 作为单域跑, 输出命名仍是 wrfout_d01_*。若在 /home/rencheng/WRF_dust 里跑,
#   会覆盖掉 nudged d01 的 03-18~03-24 输出 —— 那正是 FLEXPART 用的数据, 重跑要 8 小时。
#   故本脚本在 $D02 下建独立运行目录, 用符号链接引入可执行文件与静态数据。
#
# 时段 03-18 00Z ~ 03-24 00Z (6 天), 预计约 5 小时。

set -e
SRC=/home/rencheng/WRF_dust          # d01 的运行目录(只读取, 不写入)
D02=/home/rencheng/WRF_d02_run       # d02 的独立运行目录
CFG=/home/rencheng/d02_cfg
NP=10                                # 与 d01 那次一致(rsl.out 有 10 个)
# wrf.exe 链接的是 /root/wrf_libs 里的自建 OpenMPI 4.1.6, 不是系统的
# /usr/lib64/openmpi —— 用系统那个 mpirun 会因 ABI 不匹配而失败。
MPI=/root/wrf_libs/install/bin/mpirun
export LD_LIBRARY_PATH=/root/wrf_libs/install/lib:$LD_LIBRARY_PATH

echo "=== 0. 前置检查 ==="
[ $(pgrep -c flexwrf33 || true) -eq 0 ] || { echo "FLEXPART 仍在跑, 先等它结束"; exit 1; }
[ $(pgrep -c wrf.exe   || true) -eq 0 ] || { echo "已有 wrf.exe 在跑"; exit 1; }
mkdir -p $D02 && cd $D02

# 可执行文件与运行期静态数据: 全部软链, 不复制(WRF 的 run 目录有上百个表)
for f in $SRC/*.exe $SRC/*.TBL $SRC/*.tbl $SRC/*_DATA* $SRC/RRTM* $SRC/CAM* \
         $SRC/ozone* $SRC/aerosol* $SRC/*.formatted $SRC/*.asc $SRC/*.bin \
         $SRC/co2_trans $SRC/*.TXT $SRC/CCN* $SRC/CLM* $SRC/capacity* \
         $SRC/bulk* $SRC/grib2map* $SRC/gribmap* $SRC/tr* $SRC/eclipse*; do
  [ -e "$f" ] && ln -sf "$f" . 2>/dev/null || true
done
ln -sf /home/rencheng/WPS_dust/met_em.d0[12].2023-03-1[89]* . 2>/dev/null || true
ln -sf /home/rencheng/WPS_dust/met_em.d0[12].2023-03-2[0-4]* . 2>/dev/null || true
echo "  运行目录 $D02"
echo "  可执行:   $(ls -l real.exe ndown.exe wrf.exe 2>/dev/null | wc -l)/3"
echo "  met_em:   $(ls met_em.d01.* 2>/dev/null | wc -l) (d01) + $(ls met_em.d02.* 2>/dev/null | wc -l) (d02)"
echo "  d01 输出目录 $SRC 全程只读, 不会被写入"

echo "=== 1. real.exe (max_dom=2) 生成 wrfinput_d02 ==="
cp $CFG/namelist.input.step1_real namelist.input
$MPI -np $NP ./real.exe
tail -3 rsl.error.0000
[ -f wrfinput_d02 ] || { echo "real.exe 未生成 wrfinput_d02"; tail -20 rsl.error.0000; exit 1; }
mv wrfinput_d02 wrfndi_d02
echo "  wrfinput_d02 -> wrfndi_d02"

echo "=== 2. ndown.exe 由 wrfout_d01 生成 d02 的初始与边界 ==="
# ndown 要读 d01 的 wrfout。软链进来(只读)。
ln -sf $SRC/wrfout_d01_2023-03-1[89]_*:00:00 . 2>/dev/null || true
ln -sf $SRC/wrfout_d01_2023-03-2[0-4]_*:00:00 . 2>/dev/null || true
echo "  链入 wrfout_d01: $(ls wrfout_d01_* 2>/dev/null | wc -l) 个"
cp $CFG/namelist.input.step2_ndown namelist.input
$MPI -np $NP ./ndown.exe
tail -3 rsl.error.0000
[ -f wrfinput_d02 ] && [ -f wrfbdy_d02 ] || { echo "ndown.exe 输出不全"; tail -20 rsl.error.0000; exit 1; }
mv wrfinput_d02 wrfinput_d01_ndown
mv wrfbdy_d02   wrfbdy_d01_ndown
# 移走 d01 的 wrfout 软链, 否则第 3 步的输出会与之同名冲突
rm -f wrfout_d01_2023-03-*
mv wrfinput_d01_ndown wrfinput_d01
mv wrfbdy_d01_ndown   wrfbdy_d01
echo "  已生成 d02 的 wrfinput_d01 / wrfbdy_d01(单域运行用的命名)"

echo "=== 3. wrf.exe 跑 d02 (max_dom=1, 9 km, dt=50s, 不 nudge) ==="
cp $CFG/namelist.input.step3_wrf namelist.input
nohup $MPI -np $NP ./wrf.exe > run_d02.log 2>&1 &
echo "  已后台启动 PID=$!"
echo "  日志 $D02/run_d02.log"
echo "  输出 $D02/wrfout_d01_2023-03-*  (实为 9 km 的 d02)"
echo
echo "进度: ls $D02/wrfout_d01_* | wc -l   (共应 145 个)"
