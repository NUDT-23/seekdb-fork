/* sk_exact_knn.hip - exact kNN graph builder on GPU (AMD HIP).
 *
 * Pipeline for the AMD hipVS CAGRA challenge (600072):
 *   1. sk_norms_kernel  : xn[i] = |x_i|^2
 *   2. sk_gemm_kernel   : scores = X_chunk . X^T   (raw dots, row-chunked)
 *   3. sk_topk_kernel   : per row, exact top-K by d2 = qn + xn[c] - 2*dot
 *                         (256-element segments; threshold-reject, compact,
 *                          bitonic sort, rank merge into a sorted bag;
 *                          self excluded; rows end sorted ascending)
 *   4. sk_verify_kernel : brute-force exact top-16 on sampled rows; the
 *                         bridge falls back to cuvsCagraBuild if any fails
 *
 * The graph (uint32 [n][K], device) and the dataset copy (float [n][dim],
 * device) are handed to cuvsCagraIndexFromArgs by the bridge - zero copy,
 * both buffers must outlive the index.
 *
 * FLOP budget: n^2*dim*2 = 1.6e13 (SIFT-128) / 3.2e13 (hidden-256).  The
 * deliberately simple tiled kernel targets >=6 TFLOPS => ~3-6 s on a
 * W7900D-class part vs the ~20-30 s NN-descent baseline; the score formula
 * clamps build speedup at 3.0x, so elegance is traded for desk-checkability.
 *
 * Device memory: X (n*dim*4) + graph (n*K*4) + norms (n*4) + scores
 * (chunk*n*4, default 2048*250000*4 ~ 1.9 GB, auto-halved down to 256 rows
 * if allocation fails).  Peak ~2.3 GB - far under the vram gates.
 *
 * Determinism: compaction slot order is nondeterministic (atomicAdd), but
 * every segment is fully sorted before merging and (dist,id) is a total
 * order, so the merged result does not depend on slot order.
 */
#include <hip/hip_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define SK_THREADS   256     /* threads per CTA everywhere        */
#define SK_MAXK      256     /* compile-time cap on K             */
#define SK_TY        64      /* gemm rows per CTA tile            */
#define SK_TX        128     /* gemm cols per CTA tile            */
#define SK_TK        32      /* gemm k-tile                       */

/* ------------------------------------------------------------------ */
/* helpers                                                             */
/* ------------------------------------------------------------------ */

/* total order on (dist, id): smaller dist first, ties by smaller id.
 * ids are unique within a row, so the order is strict and total. */
__device__ __forceinline__ bool sk_better(float av, unsigned ai, float bv, unsigned bi)
{
  if (av != bv) return av < bv;
  return ai < bi;
}

/* lower_bound: number of elements in vv[0..len) strictly better than (v,id) */
__device__ __forceinline__ int sk_lb(int len, const float* vv, const unsigned* ii,
                                     float v, unsigned id)
{
  int lo = 0, hi = len;
  while (lo < hi) {
    int mid = (lo + hi) >> 1;
    if (sk_better(vv[mid], ii[mid], v, id)) lo = mid + 1; else hi = mid;
  }
  return lo;
}

/* bitonic sort of 256 (v,id) pairs, ascending by sk_better.
 * Must be entered by all 256 threads of the CTA. */
__device__ __forceinline__ void sk_bitonic256(float* v, unsigned* id)
{
  const int t = threadIdx.x;
  for (int size = 2; size <= 256; size <<= 1) {
    for (int st = size >> 1; st > 0; st >>= 1) {
      __syncthreads();
      int p = t ^ st;
      if (p > t) {
        bool asc = ((t & size) == 0);
        bool pb  = sk_better(v[p], id[p], v[t], id[t]);
        if (asc == pb) {              /* asc & partner better, or desc & partner worse */
          float tv = v[t]; v[t] = v[p]; v[p] = tv;
          unsigned ti = id[t]; id[t] = id[p]; id[p] = ti;
        }
      }
    }
  }
}

/* ------------------------------------------------------------------ */
/* kernel 1: squared norms                                             */
/* ------------------------------------------------------------------ */

__global__ void sk_norms_kernel(const float* __restrict__ X, float* __restrict__ xn,
                                long n, int dim)
{
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float* r = X + i * dim;
  float acc = 0.f;
  int k = 0;
  for (; k + 4 <= dim; k += 4) {
    float4 v = *reinterpret_cast<const float4*>(r + k);
    acc += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
  }
  for (; k < dim; k++) acc += r[k] * r[k];
  xn[i] = acc;
}

/* ------------------------------------------------------------------ */
/* kernel 2: chunked GEMM  scores[lr][c] = sum_k X[row][k] * X[c][k]   */
/*   grid = (ceil(n/SK_TX), ceil(rows/SK_TY)), block = 256             */
/*   thread (ty16 = t/16, tx16 = t%16) owns rows ty16*4 .. +4 and      */
/*   cols tx16*8 .. +8 (4x8 register tile, 32 accumulators)            */
/*   LDS: As[SK_TK][SK_TY] (k-major) + Bs[SK_TX][SK_TK+1] (col-major,  */
/*   +1 padding: load phase contiguous, compute phase 4-way max)       */
/* ------------------------------------------------------------------ */

__global__ void sk_gemm_kernel(const float* __restrict__ X, float* __restrict__ scores,
                               long chunk_start, long rows, long n, int dim)
{
  __shared__ float As[SK_TK][SK_TY];
  __shared__ float Bs[SK_TX][SK_TK + 1];

  const int t    = threadIdx.x;                 /* 0..255 */
  const int ty16 = t / 16;                      /* 0..15  */
  const int tx16 = t % 16;                      /* 0..15  */
  const long row0 = chunk_start + (long)blockIdx.y * SK_TY;
  const long col0 = (long)blockIdx.x * SK_TX;

  float acc[4][8];
#pragma unroll
  for (int i = 0; i < 4; i++)
#pragma unroll
    for (int j = 0; j < 8; j++) acc[i][j] = 0.f;

  const int ktiles = (dim + SK_TK - 1) / SK_TK;
  for (int kt = 0; kt < ktiles; kt++) {
    const int k0 = kt * SK_TK;
    /* A tile: 64 rows x 32 k -> As[kk][m]; consecutive t -> consecutive m */
    for (int idx = t; idx < SK_TY * SK_TK; idx += SK_THREADS) {
      int kk = idx / SK_TY, m = idx % SK_TY;
      long row = row0 + m;
      int  k   = k0 + kk;
      As[kk][m] = (row < n && k < dim) ? X[row * dim + k] : 0.f;
    }
    /* B tile: 128 cols x 32 k -> Bs[col][kk]; consecutive t -> consecutive kk
     * (coalesced 128 B global runs along each X row, contiguous LDS writes) */
    for (int idx = t; idx < SK_TX * SK_TK; idx += SK_THREADS) {
      int col = idx / SK_TK, kk = idx % SK_TK;
      long c  = col0 + col;
      int  k  = k0 + kk;
      Bs[col][kk] = (c < n && k < dim) ? X[c * dim + k] : 0.f;
    }
    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < SK_TK; kk++) {
      float a[4], b[8];
#pragma unroll
      for (int i = 0; i < 4; i++) a[i] = As[kk][ty16 * 4 + i];
#pragma unroll
      for (int j = 0; j < 8; j++) b[j] = Bs[tx16 * 8 + j][kk];
#pragma unroll
      for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 8; j++) acc[i][j] += a[i] * b[j];
    }
    __syncthreads();
  }

  /* write: 4x8 per thread; a row-group's 16 threads cover 128 consecutive
   * columns of one row => fully coalesced rows */
#pragma unroll
  for (int i = 0; i < 4; i++) {
    long lr = (long)blockIdx.y * SK_TY + ty16 * 4 + i;   /* row local to chunk */
    if (lr >= rows) continue;
#pragma unroll
    for (int j = 0; j < 8; j++) {
      long c = col0 + tx16 * 8 + j;
      if (c < n) scores[lr * n + c] = acc[i][j];
    }
  }
}

/* ------------------------------------------------------------------ */
/* kernel 3: exact top-K per row (one CTA per row)                     */
/* ------------------------------------------------------------------ */

__global__ void sk_topk_kernel(const float* __restrict__ scores,
                               const float* __restrict__ xn,
                               uint32_t* __restrict__ graph,
                               long chunk_start, long n, int K)
{
  __shared__ float    kv[SK_MAXK];
  __shared__ unsigned ki[SK_MAXK];
  __shared__ float    nv[SK_MAXK];
  __shared__ unsigned ni[SK_MAXK];
  __shared__ float    cv[SK_THREADS];
  __shared__ unsigned ci[SK_THREADS];
  __shared__ int      cnt;

  const int   t   = threadIdx.x;
  const long  row = chunk_start + blockIdx.x;
  const float* srow = scores + (long)blockIdx.x * n;
  const float qn  = xn[row];

  bool init = false;
  for (long seg = 0; seg < n; seg += SK_THREADS) {
    long c = seg + t;
    float v = FLT_MAX;
    unsigned id = 0xFFFFFFFFu;
    if (c < n && c != row) {              /* self-exclusion by index */
      v  = qn + xn[c] - 2.f * srow[c];    /* squared L2 up to monotone fold */
      id = (unsigned)c;
    }
    cv[t] = v; ci[t] = id;
    __syncthreads();

    if (init) {
      const float    mn = kv[K - 1];
      const unsigned mi = ki[K - 1];
      const float    ov = cv[t];
      const unsigned oi = ci[t];
      if (t == 0) cnt = 0;
      __syncthreads();
      /* Collect survivors into slots [0, cnt) via atomics.  Non-survivors
       * keep their original slots: every survivor is strictly better than
       * (mn, mi) = the bag's worst, every non-survivor is not, so after the
       * sort below all survivors sort before all non-survivors and the merge
       * can simply use cv[0..cnt).  No sentinel overwrite needed (which
       * would race with the atomic collection writes). */
      if (sk_better(ov, oi, mn, mi)) {
        int s = atomicAdd(&cnt, 1);
        cv[s] = ov; ci[s] = oi;
      }
      __syncthreads();
      if (cnt == 0) continue;              /* uniform branch: shared read after sync */
    }

    sk_bitonic256(cv, ci);                 /* ascending; FLT_MAX padding sinks to end */
    __syncthreads();

    if (!init) {
      if (t < K) { kv[t] = cv[t]; ki[t] = ci[t]; }
      init = true;
      __syncthreads();
      continue;
    }

    /* rank-merge sorted kv[0..K) with sorted cv[0..cnt) into nv/ni.
     * Every output rank r < K gets exactly one writer (total order). */
    if (t < K) {
      int r = t + sk_lb(cnt, cv, ci, kv[t], ki[t]);
      if (r < K) { nv[r] = kv[t]; ni[r] = ki[t]; }
    }
    {
      int r = t + sk_lb(K, kv, ki, cv[t], ci[t]);
      if (r < K) { nv[r] = cv[t]; ni[r] = ci[t]; }
    }
    __syncthreads();
    if (t < K) { kv[t] = nv[t]; ki[t] = ni[t]; }
    __syncthreads();
  }

  if (t < K) graph[row * K + t] = (uint32_t)ki[t];
}

/* ------------------------------------------------------------------ */
/* kernel 4: brute-force verification on sampled rows (exact top-16).  */
/*   Each thread keeps its stripe's top-16 in registers; the union     */
/*   provably contains the row's global top-16 (any global-top-16      */
/*   element is among the 16 smallest of its own stripe).  16 block-   */
/*   wide argmin extractions recover the global top-16; the row        */
/*   passes when >= 15 of them appear in the graph row's first 16      */
/*   entries (graph rows are sorted ascending by distance).            */
/* ------------------------------------------------------------------ */

__device__ __forceinline__ unsigned long long sk_key(float v, unsigned id)
{
  unsigned u = __float_as_uint(v);
  u = (u & 0x80000000u) ? ~u : (u ^ 0x80000000u);   /* monotonic float->uint */
  return ((unsigned long long)u << 32)
       | (unsigned long long)(0xFFFFFFFFu - id);    /* smaller key = better */
}

__global__ void sk_verify_kernel(const float* __restrict__ X, const float* __restrict__ xn,
                                 const uint32_t* __restrict__ graph, long n, int dim, int K,
                                 int* __restrict__ d_pass, const long* __restrict__ rows)
{
  extern __shared__ float sh[];                 /* 256*16 floats + 256*16 ids */
  float*    vv = sh;
  unsigned* ii = reinterpret_cast<unsigned*>(sh + 256 * 16);
  __shared__ unsigned long long bmin;
  __shared__ unsigned ext_id[16];

  const int   t   = threadIdx.x;
  const long  row = rows[blockIdx.x];
  const float* qr = X + row * dim;

  float    tv[16];
  unsigned ti[16];
#pragma unroll
  for (int j = 0; j < 16; j++) { tv[j] = FLT_MAX; ti[j] = 0xFFFFFFFFu; }

  for (long c = t; c < n; c += SK_THREADS) {
    if (c == row) continue;
    const float* xr = X + c * dim;
    float acc = 0.f;
    int k = 0;
    for (; k + 4 <= dim; k += 4) {
      float4 q = *reinterpret_cast<const float4*>(qr + k);
      float4 x = *reinterpret_cast<const float4*>(xr + k);
      acc += q.x * x.x + q.y * x.y + q.z * x.z + q.w * x.w;
    }
    for (; k < dim; k++) acc += qr[k] * xr[k];
    float d2 = xn[row] + xn[c] - 2.f * acc;
    if (sk_better(d2, (unsigned)c, tv[15], ti[15])) {
      int p = 15;
      while (p > 0 && sk_better(d2, (unsigned)c, tv[p - 1], ti[p - 1])) p--;
#pragma unroll
      for (int q = 15; q > p; q--) { tv[q] = tv[q - 1]; ti[q] = ti[q - 1]; }
      tv[p] = d2; ti[p] = (unsigned)c;
    }
  }
#pragma unroll
  for (int j = 0; j < 16; j++) { vv[t * 16 + j] = tv[j]; ii[t * 16 + j] = ti[j]; }
  __syncthreads();

  for (int e = 0; e < 16; e++) {
    if (t == 0) bmin = 0xFFFFFFFFFFFFFFFFull;
    __syncthreads();
    unsigned long long best = 0xFFFFFFFFFFFFFFFFull;
#pragma unroll
    for (int j = 0; j < 16; j++) {
      unsigned long long key = sk_key(vv[t * 16 + j], ii[t * 16 + j]);
      if (key < best) best = key;
    }
    atomicMin(&bmin, best);
    __syncthreads();
    if (t == 0) ext_id[e] = 0xFFFFFFFFu - (unsigned)(bmin & 0xFFFFFFFFu);
    __syncthreads();
    unsigned eid = ext_id[e];
    if (eid != 0xFFFFFFFFu) {
#pragma unroll
      for (int j = 0; j < 16; j++)          /* ids unique: at most one slot matches */
        if (ii[t * 16 + j] == eid) { vv[t * 16 + j] = FLT_MAX; ii[t * 16 + j] = 0xFFFFFFFFu; }
    }
    __syncthreads();
  }

  if (t == 0) {
    int hit = 0;
    for (int e = 0; e < 16; e++) {
      unsigned want = ext_id[e];
      if (want == 0xFFFFFFFFu) continue;
      for (int g = 0; g < 16 && g < K; g++)
        if (graph[row * K + g] == want) { hit++; break; }
    }
    if (hit >= 15) atomicAdd(d_pass, 1);
  }
}

/* ------------------------------------------------------------------ */
/* host orchestration                                                  */
/* ------------------------------------------------------------------ */

extern "C" long sk_exact_graph_build(const float* h_base, long n, int dim, int k,
                                     long chunk_rows, void** dev_x_out, void** dev_graph_out)
{
  *dev_x_out = NULL; *dev_graph_out = NULL;
  if (!h_base || n <= 0 || dim <= 0 || k <= 0 || k > SK_MAXK || n < (long)k + 1) return 1;

  float*    X  = NULL;
  uint32_t* G  = NULL;
  float*    xn = NULL;
  float*    SC = NULL;

  if (chunk_rows <= 0) chunk_rows = 2048;
  chunk_rows = (chunk_rows + SK_TY - 1) / SK_TY * SK_TY;

  if (hipMalloc(&X,  sizeof(float)    * (size_t)n * dim) != hipSuccess) return 2;
  if (hipMalloc(&G,  sizeof(uint32_t) * (size_t)n * k)   != hipSuccess) { hipFree(X); return 2; }
  if (hipMalloc(&xn, sizeof(float)    * (size_t)n)       != hipSuccess) { hipFree(X); hipFree(G); return 2; }
  if (hipMemcpy(X, h_base, sizeof(float) * (size_t)n * dim, hipMemcpyHostToDevice) != hipSuccess)
    { hipFree(X); hipFree(G); hipFree(xn); return 2; }

  {
    int blocks = (int)((n + SK_THREADS - 1) / SK_THREADS);
    hipLaunchKernelGGL(sk_norms_kernel, dim3(blocks), dim3(SK_THREADS), 0, 0,
                       X, xn, n, dim);
    if (hipGetLastError() != hipSuccess || hipDeviceSynchronize() != hipSuccess)
      { fprintf(stderr, "[sk_knn] norms failed\n"); hipFree(X); hipFree(G); hipFree(xn); return 3; }
  }

  /* score buffer: shrink until it allocates (floor 256 rows) */
  long alloc_rows = chunk_rows;
  while (alloc_rows >= 256) {
    if (hipMalloc(&SC, sizeof(float) * (size_t)alloc_rows * (size_t)n) == hipSuccess) break;
    SC = NULL;
    alloc_rows /= 2;
  }
  if (!SC) { fprintf(stderr, "[sk_knn] cannot allocate score buffer\n"); hipFree(X); hipFree(G); hipFree(xn); return 2; }

  for (long cs = 0; cs < n; cs += alloc_rows) {
    long rows = (cs + alloc_rows <= n) ? alloc_rows : (n - cs);
    dim3 grid((unsigned)((n + SK_TX - 1) / SK_TX), (unsigned)((rows + SK_TY - 1) / SK_TY));
    hipLaunchKernelGGL(sk_gemm_kernel, grid, dim3(SK_THREADS), 0, 0,
                       X, SC, cs, rows, n, dim);
    if (hipGetLastError() != hipSuccess) { fprintf(stderr, "[sk_knn] gemm launch failed\n"); goto fail_gpu; }
    hipLaunchKernelGGL(sk_topk_kernel, dim3((unsigned)rows), dim3(SK_THREADS), 0, 0,
                       SC, xn, G, cs, n, k);
    if (hipGetLastError() != hipSuccess) { fprintf(stderr, "[sk_knn] topk launch failed\n"); goto fail_gpu; }
  }
  if (hipDeviceSynchronize() != hipSuccess) { fprintf(stderr, "[sk_knn] device sync failed\n"); goto fail_gpu; }

  /* verification: 24 sampled rows, exact top-16 vs graph row prefix */
  {
    const int S = 24;
    long hrows[24];
    for (int i = 0; i < S; i++) hrows[i] = (long)(((unsigned long long)(i + 1) * 2654435761ull) % (unsigned long long)n);
    long* d_rows = NULL; int* d_pass = NULL;
    if (hipMalloc(&d_rows, sizeof(long) * S) != hipSuccess) goto fail_gpu;
    if (hipMalloc(&d_pass, sizeof(int))     != hipSuccess) { hipFree(d_rows); goto fail_gpu; }
    if (hipMemcpy(d_rows, hrows, sizeof(long) * S, hipMemcpyHostToDevice) != hipSuccess)
      { hipFree(d_rows); hipFree(d_pass); goto fail_gpu; }
    hipMemset(d_pass, 0, sizeof(int));
    size_t shm = sizeof(float) * 256 * 16 + sizeof(unsigned) * 256 * 16;
    hipLaunchKernelGGL(sk_verify_kernel, dim3(S), dim3(SK_THREADS), shm, 0,
                       X, xn, G, n, dim, k, d_pass, d_rows);
    hipError_t ve = hipGetLastError();
    int pass = -1;
    if (ve == hipSuccess) hipMemcpy(&pass, d_pass, sizeof(int), hipMemcpyDeviceToHost);
    hipFree(d_rows); hipFree(d_pass);
    if (ve != hipSuccess || pass != S) {
      fprintf(stderr, "[sk_knn] verify failed (%d/%d, err=%d) - caller falls back\n", pass, S, (int)ve);
      goto fail_gpu;
    }
    if (getenv("SK_KNN_TRACE") && getenv("SK_KNN_TRACE")[0] == '1')
      fprintf(stderr, "[sk_knn] verify ok (%d/%d)\n", pass, S);
  }

  hipFree(SC);
  hipFree(xn);
  *dev_x_out = X;          /* viewed by cuvsCagraIndexFromArgs; freed by the bridge */
  *dev_graph_out = G;
  return 0;

fail_gpu:
  hipFree(SC); hipFree(xn); hipFree(G); hipFree(X);
  return 4;
}

extern "C" void sk_free_exact(void* p)
{
  if (p) hipFree(p);
}
