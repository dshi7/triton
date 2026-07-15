#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/ADT/DenseSet.h"

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#define GEN_PASS_DEF_TRITONNVIDIAGPUINSERTTMEMALIASWARBARRIERPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

namespace {

// Trace a TMEM memdesc value back to its root allocation, following
// metadata-only view ops (memdesc_reinterpret / _index / _subslice / _trans)
// and warp_specialize partition captures. Two accesses whose roots are the
// same SSA value refer to the same physical TMEM (e.g. aliased qk/P or dp/dsT
// declared via storage_alias and lowered to memdesc_reinterpret of one alloc).
static Value getTmemRoot(Value v) {
  while (v) {
    if (auto ba = dyn_cast<BlockArgument>(v)) {
      Operation *parent = ba.getOwner()->getParentOp();
      if (auto part = dyn_cast<ttg::WarpSpecializePartitionsOp>(parent)) {
        v = part->getOperand(ba.getArgNumber());
        continue;
      }
      break;
    }
    Operation *def = v.getDefiningOp();
    if (def && def->hasTrait<OpTrait::MemDescViewTrait>()) {
      v = def->getOperand(0);
      continue;
    }
    break;
  }
  return v;
}

// A full-partition execution barrier (bar.sync over every warp of the task)
// that orders all the partition's warps against each other. We deliberately do
// NOT treat ttng::NamedBarrierWaitOp as ordering: a named barrier may
// synchronize only a subset of threads (e.g. `wait_barrier_named ..., 128` =
// 4 warps), which would not order warp N against warp N+4 sharing a TMEM
// lane-group. Trusting a partial named barrier could clear a pending read and
// suppress a required insertion -- a false negative that reintroduces the WAR.
// mbarrier arrive/wait (ttng.*_barrier) are async producer/consumer signals and
// likewise do not order warps within a task. Being conservative here can at
// worst emit a redundant (measured-free) barrier, never miss a needed one.
// TODO: could treat a named barrier as ordering once we can prove its thread
// count covers all warps participating in the pending TMEM access.
static bool isOrderingBarrier(Operation *op) { return isa<ttg::BarrierOp>(op); }

// Per-warp footprint of a tcgen05 access in physical TMEM.
// Phi = regLayout.invertAndCompose(tmemLayout) maps (register, lane, warp[,
// block]) -> (row, col). In cellMode the footprint is the set of (row, col)
// cells (encoded row*nCol+col); otherwise it is the set of rows only. Rows are
// physical TMEM lanes (0-127) and share a common frame across element types
// (dtype only repacks columns), so row sets are comparable even when the two
// accesses have different dtypes; columns are only comparable when the two
// accesses share the same memdesc frame, which is what cellMode gates on.
static LogicalResult
computeWarpFootprint(RankedTensorType regTy, ttg::MemDescType tmemTy,
                     bool cellMode,
                     SmallVectorImpl<llvm::DenseSet<int64_t>> &warpFp) {
  MLIRContext *ctx = regTy.getContext();
  LinearLayout reg = ttg::toLinearLayout(regTy);
  LinearLayout tmem = ttg::toLinearLayout(tmemTy);
  StringAttr kWarp = StringAttr::get(ctx, "warp");
  StringAttr kReg = StringAttr::get(ctx, "register");
  StringAttr kLane = StringAttr::get(ctx, "lane");
  StringAttr kRow = StringAttr::get(ctx, "row");
  StringAttr kCol = StringAttr::get(ctx, "col");
  if (!llvm::is_contained(reg.getInDimNames(), kWarp) ||
      !llvm::is_contained(reg.getInDimNames(), kReg) ||
      !llvm::is_contained(reg.getInDimNames(), kLane) ||
      !llvm::is_contained(tmem.getInDimNames(), kRow))
    return failure();
  LinearLayout phi = reg.invertAndCompose(tmem);
  int nWarp = phi.getInDimSize(kWarp);
  int nReg = phi.getInDimSize(kReg);
  int nLane = phi.getInDimSize(kLane);
  int64_t nCol = phi.getOutDimSize(kCol);
  warpFp.assign(nWarp, {});
  SmallVector<std::pair<StringAttr, int32_t>> pt;
  for (StringAttr d : phi.getInDimNames())
    pt.push_back({d, 0});
  for (int w = 0; w < nWarp; ++w)
    for (int r = 0; r < nReg; ++r)
      for (int l = 0; l < nLane; ++l) {
        for (auto &p : pt)
          p.second = (p.first == kReg)    ? r
                     : (p.first == kLane) ? l
                     : (p.first == kWarp) ? w
                                          : 0;
        int64_t row = 0, col = 0;
        for (auto &o : phi.apply(pt)) {
          if (o.first == kRow)
            row = o.second;
          else if (o.first == kCol)
            col = o.second;
        }
        warpFp[w].insert(cellMode ? row * nCol + col : row);
      }
  return success();
}

// A cross-warp write-after-read exists iff some warp's store footprint overlaps
// a *different* warp's read footprint. If every warp only overwrites what it
// itself read, the reuse is warp-local and safe. Same-frame accesses (identical
// memdesc type) are compared at (row, col) cell granularity; different-frame
// aliases (e.g. an f32 qk region reused as an f16 P region, whose column frames
// differ by an unmodeled offset) fall back to a conservative row comparison.
static bool hasCrossWarpOverlap(ttng::TMEMLoadOp rd, ttng::TMEMStoreOp st) {
  bool sameFrame = rd.getSrc().getType() == st.getDst().getType();
  SmallVector<llvm::DenseSet<int64_t>> readFp, writeFp;
  if (failed(computeWarpFootprint(rd.getResult().getType(),
                                  rd.getSrc().getType(), sameFrame, readFp)) ||
      failed(computeWarpFootprint(st.getSrc().getType(), st.getDst().getType(),
                                  sameFrame, writeFp)))
    return false; // can't prove a hazard -> don't insert
  int n = std::min(readFp.size(), writeFp.size());
  for (int a = 0; a < n; ++a)
    for (int b = 0; b < n; ++b) {
      if (a == b)
        continue;
      for (int64_t cell : writeFp[a])
        if (readFp[b].contains(cell))
          return true;
    }
  return false;
}

// Scan a block in program order. For a TMEM store that overwrites a root with a
// still-live, cross-warp-overlapping read and no intervening barrier, insert a
// task-scoped ttg.barrier before the store. Idempotent: an ordering barrier
// clears the pending read, so already-guarded pairs are not re-inserted.
static unsigned processBlock(Block &block) {
  unsigned inserted = 0;
  DenseMap<Value, Operation *> liveReadOp;
  for (Operation &opRef : llvm::make_early_inc_range(block)) {
    Operation *op = &opRef;
    if (auto ld = dyn_cast<ttng::TMEMLoadOp>(op)) {
      liveReadOp[getTmemRoot(ld.getSrc())] = op;
    } else if (isOrderingBarrier(op)) {
      liveReadOp.clear();
    } else if (auto st = dyn_cast<ttng::TMEMStoreOp>(op)) {
      auto it = liveReadOp.find(getTmemRoot(st.getDst()));
      if (it != liveReadOp.end()) {
        if (hasCrossWarpOverlap(cast<ttng::TMEMLoadOp>(it->second), st)) {
          OpBuilder b(op);
          ttg::BarrierOp::create(b, op->getLoc(), ttg::AddrSpace::All);
          ++inserted;
        }
        liveReadOp.erase(it);
      }
    }
    for (Region &region : op->getRegions())
      for (Block &nested : region)
        inserted += processBlock(nested);
  }
  return inserted;
}

} // namespace

class TritonNvidiaGPUInsertTMemAliasWARBarrierPass
    : public impl::TritonNvidiaGPUInsertTMemAliasWARBarrierPassBase<
          TritonNvidiaGPUInsertTMemAliasWARBarrierPass> {
public:
  using TritonNvidiaGPUInsertTMemAliasWARBarrierPassBase::
      TritonNvidiaGPUInsertTMemAliasWARBarrierPassBase;

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    for (Region &region : mod->getRegions())
      for (Block &block : region)
        processBlock(block);
  }
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
