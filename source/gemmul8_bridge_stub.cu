/* GEMMul8 を使わないビルド（make GEMMUL8=off）用の代わりの実装。
   gemmul8_bridge.cu と同じ 6 つの関数を、cuBLAS の倍精度の行列積（GEMMul8 なし）で提供する。
   GEMMul8 は新しい cuBLASLt（CUDA 12.8 以降）の機能を要するので、古い CUDA（例: LRZ の NVHPC 24.9 /
   CUDA 12.6）ではビルドできない。scf.gemmul8.enable off で実行するのと同じ結果になる。

   Stub for builds without GEMMul8 (make GEMMUL8=off): plain cuBLAS FP64 GEMMs,
   equivalent to running with scf.gemmul8.enable off. */
#include <cublas_v2.h>
#include <cuComplex.h>
#include <cstddef>
#include <cstdio>

extern "C" cublasStatus_t openmx_gemmul8Dgemm(cublasHandle_t handle, cublasOperation_t transa,
                                               cublasOperation_t transb, int m, int n, int k,
                                               const double *alpha, const double *A, int lda,
                                               const double *B, int ldb, const double *beta,
                                               double *C, int ldc)
{
    if (m <= 0 || n <= 0 || k <= 0) return CUBLAS_STATUS_SUCCESS;
    return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

extern "C" cublasStatus_t openmx_gemmul8Zgemm(cublasHandle_t handle, cublasOperation_t transa,
                                               cublasOperation_t transb, int m, int n, int k,
                                               const cuDoubleComplex *alpha, const cuDoubleComplex *A,
                                               int lda, const cuDoubleComplex *B, int ldb,
                                               const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc)
{
    if (m <= 0 || n <= 0 || k <= 0) return CUBLAS_STATUS_SUCCESS;
    return cublasZgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

extern "C" void openmx_gemmul8ReleaseWorkspaces(void) {}

extern "C" void openmx_gemmul8SetEnabled(int enabled)
{
    static int announced = 0;
    if (enabled && !announced) {
        std::printf("<GEMMul8> this executable was built with GEMMUL8=off; plain cuBLAS FP64 GEMMs are used.\n");
        std::fflush(stdout);
        announced = 1;
    }
}

extern "C" size_t openmx_gemmul8ZWorkspaceSize(int, int, int) { return 0; }
extern "C" size_t openmx_gemmul8DWorkspaceSize(int, int, int) { return 0; }
