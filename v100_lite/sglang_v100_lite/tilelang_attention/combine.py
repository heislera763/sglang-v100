"""TileLang fallback combining QSA split-KV attention."""

import tilelang
import tilelang.language as T
from .config import pass_configs


@tilelang.jit(out_idx=[-1], pass_configs=pass_configs)
def _decode_combine_kernel(
    batch: int,
    heads: int,
    dim: int,
    max_splits: int,
    threads: int,
    min_tokens_per_split: int,
    selected_tokens: int = 0,
):
    @T.prim_func
    def main(
        PartialO: T.Tensor([batch, max_splits, heads, dim], T.float16),
        PartialLSE: T.Tensor([batch, max_splits, heads], T.float32),
        SeqLens: T.Tensor([batch], T.int32),
        Output: T.Tensor([batch, heads, dim], T.float16),
    ):
        with T.Kernel(heads, batch, threads=threads) as (head, batch_id):
            lse = T.alloc_shared([max_splits], T.float32)
            max_lse = T.alloc_fragment([1], T.float32)
            sum_lse = T.alloc_fragment([1], T.float32)
            output = T.alloc_fragment([dim], T.float32)
            context = (
                T.min(SeqLens[batch_id], selected_tokens)
                if selected_tokens > 0
                else SeqLens[batch_id]
            )
            active_splits = T.min(
                max_splits,
                T.max(
                    1,
                    T.ceildiv(context, min_tokens_per_split),
                ),
            )
            for split in T.Parallel(max_splits):
                lse[split] = T.if_then_else(
                    split < active_splits,
                    PartialLSE[batch_id, split, head],
                    -(2**30),
                )
            T.fill(max_lse, -(2**30))
            for split in T.serial(max_splits):
                max_lse[0] = T.max(max_lse[0], lse[split])
            T.fill(sum_lse, 0)
            for split in T.serial(max_splits):
                if split < active_splits:
                    sum_lse[0] += T.exp2(lse[split] - max_lse[0])
            T.fill(output, 0)
            for split in T.serial(max_splits):
                if split < active_splits:
                    weight = T.exp2(lse[split] - max_lse[0]) / sum_lse[0]
                    for d in T.Parallel(dim):
                        output[d] += weight * T.cast(
                            PartialO[batch_id, split, head, d],
                            T.float32,
                        )
            for d in T.Parallel(dim):
                Output[batch_id, head, d] = T.cast(output[d], T.float16)

    return main
