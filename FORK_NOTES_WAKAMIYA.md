# wakamiya0315/openmx4.0_gpu: fork notes

This fork adds small changes for generating machine-learning Hamiltonian (ML-H) training data
with OpenMX. Upstream: https://github.com/dc1394/openmx4.0_gpu (GPL-3.0).

## ブランチの使い分け

| ブランチ | 内容 | 扱い |
|---|---|---|
| `main` | 上流（dc1394）の `main` と同じ | **変更しない**（上流の追従専用） |
| `main-wakamiya` | このフォークの統合ブランチ。土台は上流のタグ `v1.1.1`（コミット e227507f） | 作業ブランチをここへマージする |
| `ml-scfout` など | 個別の改造 | `main-wakamiya` へマージ |

土台を `v1.1.1` にした理由: TSUBAME4 で、上流の `main`（v4.0 系）は CPU 版の起動時の SIGSEGV と CUDA 版のビルドの
失敗があり、`v1.1.1` は同じバイナリで GPU（gpu_h）と CPU（`scf.eigen.lib elpa2`）の両方が動いた（2026-11-17）。

## 改造の一覧

### `HS.fileout.ml`（ブランチ `ml-scfout`）

```
HS.fileout     on
HS.fileout.ml  on    # 既定は off（元の動作）
```

- scfout から、位置演算子（OLPpo）と運動量演算子（OLPmo）の行列の計算を省く。これらは構造と基底だけで決まり
  （SCF の結果を含まない）、ML-H の学習には使わない。実空間の格子での積分は並列化されておらず、GPU 版（1 MPI）で遅い。
- scfout の形式は保つ: `order_max = 0` を書く（OLPpo のブロックはなくなる）、OLPmo は 0 を書く。H（Hks）、H の虚部
  （iHks）、S（OLP）、DM、iDM、構造・近接リスト・化学ポテンシャルなどは元と同じ。既存の読み取り（`read_scfout`、
  HamGNN の `read_openmx`）はそのまま読める（要確認の結果は catalyst-ml-hamiltonian の記録を参照）。
- scfout に埋め込まれる入力ファイルに `HS.fileout.ml on` が残るので、OLPmo が 0 の出力を区別できる。
- 標準出力に `<SCF2File> time: OLPpo …, OLPmo …, write …` を出す（両方のモードで）。

### `GEMMUL8=off`（ブランチ `gemmul8-optional`）

GEMMul8 は CUDA 12.8 以降の cuBLASLt の機能を要するので、古い CUDA ではビルドできない。`make GEMMUL8=off` で、
GEMMul8 の橋渡しを cuBLAS の倍精度の行列積だけの代わり（`gemmul8_bridge_stub.cu`）に置き換える。
`scf.gemmul8.enable off` で実行するのと同じ。既定（`GEMMUL8=on`）は変わらない。

## GPU 実行の注意: `OPENMX_BAND_GPU_SWITCH_NUM`

上流の既定 800 では、行列の次元が 800 未満のバンド計算は CPU の ELPA に退避する。TSUBAME の gpu_h（1 MPI × 4 スレッド）
では退避の方がはるかに遅かった（Pt(111)+CO、次元 442: 846 s → GPU で 76 s。バルク Pt、次元 26: 72 s → 28 s）。
GPU で回すときは `OPENMX_BAND_GPU_SWITCH_NUM=1`（または `OPENMX_BAND_GPU_DIAG=1`）を `mpirun -x` で渡す。

## TSUBAME4 でのビルド（NVHPC 26.1 / CUDA 13.1）

```bash
module load gcc/14.2.0 nvhpc/26.1_cuda13.1
sdk=/apps/t4/rhel9/isv/nvidia/hpc_sdk/Linux_x86_64/26.1
export NVCOMPILER_COMM_LIBS_HOME=$sdk/comm_libs/13.1
cd source
make -j4 NVHPC_ROOT=$sdk NVHPC_CUDA_VERSION=13.1 NVHPC_CUDA_HOME=$sdk/cuda/13.1 \
  NVHPC_MPI_BIN=$sdk/comm_libs/13.1/hpcx/hpcx-2.25.1/ompi/bin \
  NVCC_HOST=/apps/t4/rhel9/free/gcc/14.2.0/bin/g++ GEMMUL8_GPU_ARCH=90 openmx
```

FFTW 3.3.11 と GEMMul8 は make が取得する（計算ノードで通信できない場合は、ログインノードで先に取得する）。
実行は `mpirun -np 1 --bind-to none openmx openmx.dat -nt 4`（gpu_h）。CPU では入力に `scf.eigen.lib elpa2` を足し、
`mpirun -np 4 ... -nt 1` など。数値の再現性のため `scf.gemmul8.enable off` を推奨。

## LRZ Linux Cluster（CPU のみ、GPU なし）でのビルドと実行（2026-11-17 確認）

LRZ の NVHPC は 24.9（CUDA 12.6）までなので `GEMMUL8=off` が要る。GPU も CUDA のドライバもない計算ノードで、
CPU 経路（`scf.eigen.lib elpa2`）が動き、TSUBAME の結果と H 2.7e-10 Ha・S 1e-13・DM 6e-10 以内で一致した。

```bash
# module purge はしない（nvhpc などを提供する stack のモジュールまで外れる）
module load nvhpc/24.9 gcc/13.3.0
N=/dss/lrzsys/sys/spack/release/24.4.0/opt/x86_64/nvhpc/24.9-gcc-geugw7j/Linux_x86_64/24.9
cd source
# FFTW と GEMMul8 の取得（git・curl）はログインノードで先に済ませる（計算ノードに git がない）
make -j12 NVHPC_ROOT=$N NVHPC_CUDA_VERSION=12.6 NVHPC_CUDA_HOME=$N/cuda/12.6 \
  NVHPC_MPI_BIN=$N/comm_libs/12.6/hpcx/hpcx-2.20/ompi/bin NVCC_HOST=$(which g++) GEMMUL8_GPU_ARCH=90 \
  GEMMUL8=off HOST_CC=$(which gcc) \
  PATH="$(dirname $(which gcc)):$N/compilers/bin:$N/comm_libs/12.6/hpcx/hpcx-2.20/ompi/bin:$N/cuda/12.6/bin:/usr/bin:/bin" openmx
```

- Makefile は PATH を上書きするが、LRZ の計算ノードには `/usr/bin/gcc` がない（`force_rpath` と NVHPC の `nvdd` が gcc を要る）
  ので、gcc のモジュールの場所を含む PATH を `make` に渡す。
- 実行は Slurm で `--ntasks` を MPI のプロセス数に合わせ、NVHPC の `mpirun`（hpcx）で起動する。`--ntasks=1` だと
  `mpirun -np 4` は slot 不足で起動しない。
