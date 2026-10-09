// SPDX-License-Identifier: Apache-2.0
// Opt-in small-allreduce policy for one eight-GPU, dual-NVLink-quad node.
#include "nccl/tuner.h"

typedef struct { int eligible; } Context;

static ncclResult_t init(size_t ranks, size_t nodes, ncclDebugLogger_t log, void **ctx) {
  Context *state = malloc(sizeof(*state));
  if (!state) return ncclSystemError;
  state->eligible = ranks == 8 && nodes == 1;
  *ctx = state;
  if (log) log(NCCL_LOG_INFO, NCCL_TUNING, __FILE__, __LINE__,
               "V100 quad tuner: ranks=%zu nodes=%zu small-allreduce=%d", ranks, nodes, state->eligible);
  return ncclSuccess;
}

static ncclResult_t choose(void *ctx, ncclFunc_t op, size_t bytes, int pipes,
                          float **costs, int algorithms, int protocols,
                          int registered, int *channels) {
  (void)pipes; (void)registered; (void)channels;
  if (!ctx || !((Context*)ctx)->eligible || op != ncclFuncAllReduce ||
      bytes < 8192 || bytes > 262144 || protocols != NCCL_NUM_PROTOCOLS ||
      algorithms <= NCCL_ALGO_TREE) return ncclSuccess;
  float (*table)[NCCL_NUM_PROTOCOLS] = (float (*)[NCCL_NUM_PROTOCOLS]) costs;
  if (table[NCCL_ALGO_TREE][NCCL_PROTO_LL] == NCCL_ALGO_PROTO_IGNORE) return ncclSuccess;
  for (int a = 0; a < algorithms; ++a)
    for (int p = 0; p < protocols; ++p)
      if (table[a][p] == 0) table[a][p] = 1;
  table[NCCL_ALGO_TREE][NCCL_PROTO_LL] = 0;
  return ncclSuccess;
}

static ncclResult_t destroy(void *ctx) { free(ctx); return ncclSuccess; }

const ncclTuner_v4_t ncclTunerPlugin_v4 = {
  .name = "V100QuadSmallAllReduce", .init = init, .getCollInfo = choose, .destroy = destroy
};
