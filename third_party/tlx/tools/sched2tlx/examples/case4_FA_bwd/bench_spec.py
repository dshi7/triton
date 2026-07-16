"""Benchmark spec for case4 (FA backward, 5-MMA dK/dV kernel) consumed by
examples/testing/perf_regression/perf_harness.py.

Launch logic mirrors perf_generated.py (generated `fa_bwd_dkdv_5mma`, config
HEAD_DIM=64 / BLOCK_M=64 / BLOCK_N=128) and its hand-written WS ground truth
(`handwritten.fa_bwd_dkdv_ws`). Following the single-output contract of the
other cases, the harness asserts dV (deterministic, tile-owned — dQ is a
TMA-reduce accumulator and dK is transitively covered by run_generated.py's
full three-gradient check at small shapes). dQ must start zeroed, so both
calls zero their own dq buffer inside the timed region — symmetric on the
gen and hw sides, matching perf_generated.py's methodology.

The dV reference is the direct fp32 formula dV = softmax(qkᵀ)ᵀ @ dO (q comes
pre-scaled by 1/sqrt(HEAD_DIM)) rather than fp16 autograd: it stays accurate
at the large-N_CTX sweep shapes where fp16 softmax error would eat the
tolerance. TOL matches run_generated.py's 3e-2 for fp16 FA-bwd outputs.
"""

from __future__ import annotations

import math

import torch

LOG2E = 1.4426950408889634
HEAD_DIM, BLOCK_M, BLOCK_N = 64, 64, 128
NUM_BUFFERS = 2  # handwritten kernel's SMEM ring depth
TOL = 3e-2

# (BH, N_CTX): batch*heads and context length; N_CTX must divide by BLOCK_N.
SHAPES = [
    (2, 1024),
    (8, 4096),
    (8, 8192),
    (8, 16384),
]


def make_inputs(shape):
    BH, N_CTX = shape
    sm = 1.0 / math.sqrt(HEAD_DIM)
    q = torch.randn(BH, N_CTX, HEAD_DIM, device="cuda", dtype=torch.float16) * sm
    k = torch.randn(BH, N_CTX, HEAD_DIM, device="cuda", dtype=torch.float16)
    v = torch.randn(BH, N_CTX, HEAD_DIM, device="cuda", dtype=torch.float16)
    do = torch.randn(BH, N_CTX, HEAD_DIM, device="cuda", dtype=torch.float16)

    # Forward statistics the kernel consumes (M = logsumexp in log2 domain,
    # D = rowsum(dO*O)) and the fp32 dV reference, all from one fp32 pass.
    s = torch.matmul(q.float(), k.float().transpose(-1, -2))
    p = torch.softmax(s, dim=-1)
    m = (torch.logsumexp(s, dim=-1) * LOG2E).contiguous()
    del s
    o = torch.matmul(p, v.float())
    D = (do.float() * o).sum(-1).contiguous()
    del o
    ref_dv = torch.matmul(p.transpose(-1, -2), do.float())
    del p

    q, k, v, do = q.contiguous(), k.contiguous(), v.contiguous(), do.contiguous()
    # Separate gradient buffers per side so gen and hw results stay comparable.
    return {
        "q": q, "k": k, "v": v, "do": do, "m": m, "D": D, "ref_dv": ref_dv,
        "dq": torch.zeros_like(q), "dk": torch.empty_like(k), "dv": torch.empty_like(v),
        "dq_hw": torch.zeros_like(q), "dk_hw": torch.empty_like(k), "dv_hw": torch.empty_like(v),
        "grid": (N_CTX // BLOCK_N, BH), "N_CTX": N_CTX,
    }


def gen_call(generated, inputs):
    inputs["dq"].zero_()  # dQ accumulates via TMA reduce; must start zeroed
    generated.fa_bwd_dkdv_5mma[inputs["grid"]](
        inputs["q"], inputs["k"], inputs["v"], inputs["do"],
        inputs["dq"], inputs["dk"], inputs["dv"],
        inputs["m"], inputs["D"],
        HEAD_DIM, HEAD_DIM, inputs["N_CTX"],
        num_warps=4, num_ctas=1, num_stages=1,
    )
    return inputs["dv"]


def hw_call(handwritten, inputs):
    inputs["dq_hw"].zero_()
    handwritten.fa_bwd_dkdv_ws[inputs["grid"]](
        inputs["q"], inputs["k"], inputs["v"], inputs["do"],
        inputs["dq_hw"], inputs["dk_hw"], inputs["dv_hw"],
        inputs["m"], inputs["D"],
        HEAD_DIM, HEAD_DIM, inputs["N_CTX"],
        BLOCK_M, BLOCK_N, HEAD_DIM, NUM_BUFFERS,
        num_warps=4, num_ctas=1, num_stages=1,
    )
    return inputs["dv_hw"]


def metric(shape):
    BH, N_CTX = shape
    return (5 * 2 * BH * N_CTX * N_CTX * HEAD_DIM, 1e12, "TFLOPS")  # 5 MMAs


def reference(inputs):
    return inputs["ref_dv"]
