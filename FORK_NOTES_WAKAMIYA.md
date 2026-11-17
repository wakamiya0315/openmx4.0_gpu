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
