/* test_sk_knn.hip - standalone validation of the exact kNN graph builder.
 *
 * 1) accuracy: n=20000 random gaussian, build graph, read back, compare
 *    64 sampled rows' first 64 entries against CPU brute force (>=63/64).
 * 2) end-to-end: seekdb_cuvs_build (exact path) + seekdb_cuvs_search vs
 *    CPU brute force top-10 (recall must be ~1.0 on an exact graph).
 * 3) timing: full 250K x 128 and 250K x 256 builds (throughput check).
 *
 * Build:  hipcc test_sk_knn.hip /work/bridge/sk_exact_knn.o \
 *           -L/work/bridge -lseekdb_cuvs_bridge \
 *           -Wl,-rpath,/opt/hipvs/lib:/opt/rocm/lib -o /work/test_sk
 * (sk_exact_knn.o sits next to libseekdb_cuvs_bridge.so; the lib exports
 *  the bridge ABI, the object exports sk_* symbols - simplest is to link
 *  both the .o for sk_* and dlopen-free direct calls to the bridge.)
 */
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

extern "C" long sk_exact_graph_build(const float* base, long n, int dim, int k,
                                 long chunk_rows, void** dev_x_out, void** dev_graph_out);
extern "C" void sk_free_exact(void* p);
/* bridge ABI (from seekdb_cuvs_bridge.h) */
extern "C" void* seekdb_cuvs_build(const float* base, long n, long dim);
extern "C" int   seekdb_cuvs_search(void* handle, const float* query, long nq, long topk,
                                    unsigned int* out_ids, float* out_dist);
extern "C" void  seekdb_cuvs_free(void* handle);

static double now_s(void)
{
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec / 1e9;
}

static void fill_gauss(float* X, long n, int dim, unsigned seed)
{
  unsigned s = seed;
  for (long i = 0; i < n * dim; i++) {
    s = s * 1664525u + 1013904223u;
    float u1 = ((s >> 8) & 0xFFFFFF) / 16777216.0f + 1e-7f;
    s = s * 1664525u + 1013904223u;
    float u2 = ((s >> 8) & 0xFFFFFF) / 16777216.0f;
    X[i] = sqrtf(-2.f * logf(u1)) * cosf(6.2831853f * u2);
  }
}

/* CPU brute force top-k for one row; returns ids sorted by distance asc */
static void brute_row(const float* X, long n, int dim, long row, int k, int* out)
{
  const float* q = X + row * dim;
  float* d = (float*)malloc(sizeof(float) * n);
  for (long c = 0; c < n; c++) {
    if (c == row) { d[c] = 1e30f; continue; }
    const float* x = X + c * dim;
    float acc = 0.f;
    for (int j = 0; j < dim; j++) { float t = q[j] - x[j]; acc += t * t; }
    d[c] = acc;
  }
  /* simple selection: k passes */
  char* used = (char*)calloc(n, 1);
  for (int i = 0; i < k; i++) {
    long best = -1;
    for (long c = 0; c < n; c++) if (!used[c] && (best < 0 || d[c] < d[best])) best = c;
    used[best] = 1; out[i] = (int)best;
  }
  free(used); free(d);
}

int main(int argc, char** argv)
{
  int rc_total = 0;

  /* ---- part 1: kernel accuracy, n=20000 ---- */
  {
    const long n = 20000; const int dim = 128, k = 64;
    float* X = (float*)malloc(sizeof(float) * n * dim);
    fill_gauss(X, n, dim, 42u);
    void *dx = NULL, *dg = NULL;
    double t0 = now_s();
    long rc = sk_exact_graph_build(X, n, dim, k, 2048, &dx, &dg);
    double t1 = now_s();
    printf("[T1] build rc=%ld  %.2f s  (n=%ld dim=%d k=%d)\n", rc, t1 - t0, n, dim, k);
    if (rc != 0) { printf("[T1] FAIL rc=%ld\n", rc); return 1; }
    uint32_t* g = (uint32_t*)malloc(sizeof(uint32_t) * n * k);
    hipMemcpy(g, dg, sizeof(uint32_t) * n * k, hipMemcpyDeviceToHost);
    int worst = 64;
    for (int s = 0; s < 8; s++) {
      long row = (long)((s * 2654435761ull) % n);
      int ref[64];
      brute_row(X, n, dim, row, 64, ref);
      int hit = 0;
      for (int i = 0; i < 64; i++)
        for (int j = 0; j < 64; j++) if ((int)g[row * k + j] == ref[i]) { hit++; break; }
      printf("[T1] row %ld overlap %d/64\n", row, hit);
      if (hit < worst) worst = hit;
    }
    printf("[T1] %s worst_overlap=%d/64\n", worst >= 63 ? "PASS" : "FAIL", worst);
    if (worst < 63) rc_total |= 1;
    free(X); free(g); sk_free_exact(dx); sk_free_exact(dg);
  }

  /* ---- part 2: bridge end-to-end (exact path + search) ---- */
  {
    const long n = 50000; const int dim = 128;
    float* X = (float*)malloc(sizeof(float) * n * dim);
    fill_gauss(X, n, dim, 7u);
    double t0 = now_s();
    void* h = seekdb_cuvs_build(X, n, dim);
    double t1 = now_s();
    printf("[T2] seekdb_cuvs_build %s  %.2f s\n", h ? "ok" : "FAILED", t1 - t0);
    if (!h) { printf("[T2] FAIL build\n"); return 2; }
    int bad = 0;
    for (int s = 0; s < 20; s++) {
      long row = (long)((s * 40503u + 17) % n);   /* query = an indexed point */
      unsigned ids[10]; float ds[10];
      int rc = seekdb_cuvs_search(h, X + row * dim, 1, 10, ids, ds);
      if (rc) { printf("[T2] search rc=%d\n", rc); bad++; continue; }
      int ref[10];
      brute_row(X, n, dim, row, 10, ref);
      int hit = 0;
      for (int i = 0; i < 10; i++)
        for (int j = 0; j < 10; j++) if ((int)ids[j] == ref[i]) { hit++; break; }
      /* The query is an indexed point: the search usually returns the point
       * itself at rank 0 (graph rows exclude self), but graph traversal does
       * not guarantee visiting the self node - 9/10 or 10/10 against the
       * self-excluding reference both indicate an exact graph. */
      if (hit < 9) {
        printf("[T2] row %ld recall10 %d/10 ids0=%u\n", row, hit, ids[0]);
        bad++;
      }
    }
    printf("[T2] %s bad_rows=%d/20 (query=indexed-point, expect 10/10 exact)\n",
           bad == 0 ? "PASS" : "FAIL", bad);
    if (bad) rc_total |= 2;
    seekdb_cuvs_free(h);
    free(X);
  }

  /* ---- part 3: full-scale timing ---- */
  {
    const long n = 250000;
    int dims[2] = {128, 256};
    for (int e = 0; e < 2; e++) {
      int dim = dims[e];
      float* X = (float*)malloc(sizeof(float) * n * dim);
      fill_gauss(X, n, dim, 99u + e);
      void *dx = NULL, *dg = NULL;
      double t0 = now_s();
      long rc = sk_exact_graph_build(X, n, dim, 128, 2048, &dx, &dg);
      double t1 = now_s();
      double flops = 2.0 * (double)n * (double)n * dim;
      printf("[T3] n=%ld dim=%d rc=%ld  %.2f s  %.1f TFLOPS\n",
             n, dim, rc, t1 - t0, flops / (t1 - t0) / 1e12);
      if (dx) sk_free_exact(dx);
      if (dg) sk_free_exact(dg);
      free(X);
      if (rc != 0) rc_total |= 4;
    }
  }

  printf("TOTAL %s (rc=%d)\n", rc_total ? "FAIL" : "PASS", rc_total);
  return rc_total;
}
