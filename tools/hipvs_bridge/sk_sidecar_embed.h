/* Embedded exact-kNN sidecar payload - PLACEHOLDER (no blob).
 *
 * Regenerate after building libseekdb_sk_exact.so on a gfx1100 instance:
 *   python3 tools/hipvs_bridge/embed_gen.py \
 *     /path/to/libseekdb_sk_exact.so \
 *     tools/hipvs_bridge/sk_sidecar_embed.h
 *
 * While this placeholder is in place the bridge compiles and behaves exactly
 * as before: the embedded channel compiles away to a no-op and the exact
 * kernels are resolved through the file/dlsym channels only. */
#define SK_SIDECAR_B64_LEN 0
static const char SK_SIDECAR_B64[] __attribute__((unused)) = "";
