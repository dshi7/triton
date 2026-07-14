// RUN: triton-opt %s -split-input-file --triton-nvidia-gpu-insert-tmem-alias-war-barrier | FileCheck %s

// Cross-warp overlap: an f32 qk read reused as an aliased f16 P store (different
// frame -> row-level footprint overlap across warps) with no barrier between.
// A task-scoped ttg.barrier must be inserted between the read and the store.

#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 64]], block = []}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @aliased_f32_read_f16_store
  // CHECK: ttng.tmem_load
  // CHECK: ttg.barrier all
  // CHECK: ttng.tmem_store
  tt.func @aliased_f32_read_f16_store(%val: tensor<128x128xf16, #linear>, %pred: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %root = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable>
    %f32v = ttg.memdesc_reinterpret %root : !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %f16v = ttg.memdesc_reinterpret %root : !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<2x128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %rd = ttg.memdesc_index %f32v[%c0] : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %wr = ttg.memdesc_index %f16v[%c1] : !ttg.memdesc<2x128x128xf16, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %qk = ttng.tmem_load %rd : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
    ttng.tmem_store %val, %wr, %pred : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

// Same-frame accumulator read-modify-write (f32 read + f32 store, same memdesc):
// compared at (row, col) cell granularity; each warp only rewrites its own
// cells -> no cross-warp overlap -> NO barrier inserted.

#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 64]], block = []}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @accumulator_rmw_no_barrier
  // CHECK-NOT: ttg.barrier
  tt.func @accumulator_rmw_no_barrier(%val: tensor<128x128xf32, #linear>, %pred: i1) {
    %root = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %acc = ttng.tmem_load %root : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
    ttng.tmem_store %val, %root, %pred : tensor<128x128xf32, #linear> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}

// -----

// Already guarded by a barrier -> idempotent: exactly one barrier, no extra.

#linear = #ttg.linear<{register = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [0, 32]], lane = [[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp = [[32, 0], [64, 0], [0, 64]], block = []}>
#tmem = #ttng.tensor_memory_encoding<blockM = 128, blockN = 128, colStride = 1>
module attributes {"ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "cuda:100"} {
  // CHECK-LABEL: @already_guarded_idempotent
  // CHECK: ttng.tmem_load
  // CHECK: ttg.barrier all
  // CHECK: ttng.tmem_store
  // CHECK-NOT: ttg.barrier
  tt.func @already_guarded_idempotent(%val: tensor<128x128xf16, #linear>, %pred: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %root = ttng.tmem_alloc : () -> !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable>
    %f32v = ttg.memdesc_reinterpret %root : !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %f16v = ttg.memdesc_reinterpret %root : !ttg.memdesc<128x128xi32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<2x128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %rd = ttg.memdesc_index %f32v[%c0] : !ttg.memdesc<1x128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable>
    %wr = ttg.memdesc_index %f16v[%c1] : !ttg.memdesc<2x128x128xf16, #tmem, #ttng.tensor_memory, mutable> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    %qk = ttng.tmem_load %rd : !ttg.memdesc<128x128xf32, #tmem, #ttng.tensor_memory, mutable> -> tensor<128x128xf32, #linear>
    ttg.barrier all
    ttng.tmem_store %val, %wr, %pred : tensor<128x128xf16, #linear> -> !ttg.memdesc<128x128xf16, #tmem, #ttng.tensor_memory, mutable>
    tt.return
  }
}
