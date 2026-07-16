"""Unit tests for _derive_persistent_grid_strides (persistent-loop stride
rewrite: baked NUM_SMS constant -> tl.num_programs).

These pin the behavior the corpus fixtures cannot: the rewrite is a no-op at
the baked grid (num_programs == the baked constant on the dev fleet, so every
launch in the repo exercises it as an identity), so its structural
correctness — which constants it does and does NOT swap — is only observable
here. Each test builds a minimal ScheduleGraph and asserts the rewrite touches
exactly the loop-stride carriers and nothing else, plus a byte-identical
re-emission of the three shipped persistent fixtures.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from sched2tlx.schedule_graph import (
    ConstRef,
    IterArgRef,
    IvRef,
    Kernel,
    Loop,
    Op,
    OpRef,
    ScheduleGraph,
    ScheduleLoop,
)
from sched2tlx.emitter import _derive_persistent_grid_strides as rewrite

ok = True


def check(cond, msg):
    global ok
    print(("  PASS" if cond else "  FAIL") + f": {msg}")
    ok = ok and cond


def _op(oid, kind, operands, scope="function"):
    return Op(oid, kind, scope, list(operands), ["i32"], {})


def _graph(ops, step, *, lb_kind="tt.get_program_id", inits=(), loop_id=1):
    """Minimal one-outer-loop graph. ops: dict; step: OperandRef; inits: the
    scf.for iter_arg init operands (operands[3:])."""
    all_ops = {"pid": _op("pid", lb_kind, [])}
    all_ops.update(ops)
    for_operands = [OpRef(op_id="pid"), OpRef(op_id="hi"), step, *inits]
    all_ops["hi"] = _op("hi", "arith.muli", [])
    all_ops["scf_for"] = _op("scf_for", "scf.for", for_operands, scope="function")
    sched = ScheduleLoop(
        id=loop_id, II=1, max_stage=0, prologue_latency=0, trip_count=1,
        trip_count_estimated=False, induction_var_name="tile_id",
        induction_var_type="i32", lower_bound=OpRef(op_id="pid"),
        upper_bound=OpRef(op_id="hi"), step=step, buffers=[], nodes=[], edges=[],
    )
    loop = Loop(loop_id=loop_id, is_outer=True, warp_groups=[], schedule=sched)
    return ScheduleGraph("1", Kernel("k", []), all_ops, [loop])


def _swapped(op):
    return any(isinstance(o, OpRef) and "nprog" in o.op_id for o in op.operands)


def _has_const(op, c):
    return any(isinstance(o, ConstRef) and o.value == c for o in op.operands)


def test_step_and_scffor_rewritten_together():
    print("== step + scf.for step operand rewritten together ==")
    g = _graph({}, ConstRef(value=148, type="i32"))
    rewrite(g)
    check(isinstance(g.loops[0].schedule.step, OpRef), "schedule.step -> OpRef(nprog)")
    check(isinstance(g.ops["scf_for"].operands[2], OpRef), "scf.for step operand -> OpRef(nprog)")
    check(g.loops[0].schedule.step == g.ops["scf_for"].operands[2], "the two stay equal")


def test_genuine_seed_and_carrier_rewritten():
    print("== genuine stride seed + carrier are rewritten ==")
    g = _graph(
        {
            "seed": _op("seed", "arith.subi", [OpRef(op_id="pid"), ConstRef(value=148, type="i32")]),
            "carrier": _op("carrier", "arith.addi",
                           [IterArgRef(loop_id=1, idx=0), ConstRef(value=148, type="i32")], scope="loop:1"),
        },
        ConstRef(value=148, type="i32"),
        inits=(OpRef(op_id="seed"),),  # iter_arg idx 0 seeded by pid-148
    )
    rewrite(g)
    check(_swapped(g.ops["seed"]), "seed `pid - 148` -> `pid - nprog`")
    check(_swapped(g.ops["carrier"]), "carrier `iter_arg[0] + 148` -> `+ nprog`")


def test_unrelated_recurrence_left_alone():
    print("== M1: unrelated recurrence whose increment equals the grid size is NOT swapped ==")
    # iter_arg idx 1 is NOT stride-seeded (its init is a plain constant, not pid+-C).
    g = _graph(
        {
            "seed": _op("seed", "arith.subi", [OpRef(op_id="pid"), ConstRef(value=148, type="i32")]),
            "cnt_init": _op("cnt_init", "arith.constant", []),
            "counter": _op("counter", "arith.addi",
                           [IterArgRef(loop_id=1, idx=1), ConstRef(value=148, type="i32")], scope="loop:1"),
        },
        ConstRef(value=148, type="i32"),
        inits=(OpRef(op_id="seed"), OpRef(op_id="cnt_init")),
    )
    rewrite(g)
    check(_swapped(g.ops["seed"]), "the genuine seed still rewritten")
    check(_has_const(g.ops["counter"], 148) and not _swapped(g.ops["counter"]),
          "unrelated `counter + 148` (idx 1, not stride-seeded) stays baked")


def test_ivref_stride_use_rewritten():
    print("== M2: a stride use on the induction variable (tile_id + C) is rewritten ==")
    g = _graph(
        {"prefetch": _op("prefetch", "arith.addi",
                         [IvRef(loop_id=1), ConstRef(value=148, type="i32")], scope="loop:1")},
        ConstRef(value=148, type="i32"),
    )
    rewrite(g)
    check(_swapped(g.ops["prefetch"]), "`iv + 148` -> `iv + nprog`")


def test_seed_via_intermediate_op_rewritten():
    print("== M4: a stride seed reaching pid through an intermediate op is rewritten ==")
    g = _graph(
        {
            "mid": _op("mid", "arith.addi", [OpRef(op_id="pid"), ConstRef(value=0, type="i32")]),
            "seed": _op("seed", "arith.subi", [OpRef(op_id="mid"), ConstRef(value=148, type="i32")]),
            "carrier": _op("carrier", "arith.addi",
                           [IterArgRef(loop_id=1, idx=0), ConstRef(value=148, type="i32")], scope="loop:1"),
        },
        ConstRef(value=148, type="i32"),
        inits=(OpRef(op_id="seed"),),
    )
    rewrite(g)
    check(_swapped(g.ops["seed"]), "seed `(pid+0) - 148` -> `- nprog` (transitive provenance)")
    check(_swapped(g.ops["carrier"]), "its carrier rewritten too")


def test_forop_unresolved_no_partial_rewrite():
    print("== M7: when the scf.for cannot be matched, the loop is left wholly baked ==")
    g = _graph({}, ConstRef(value=148, type="i32"))
    g.ops["scf_for"].operands[2] = ConstRef(value=149, type="i32")  # break (lo,hi,step) match
    rewrite(g)
    check(isinstance(g.loops[0].schedule.step, ConstRef) and g.loops[0].schedule.step.value == 148,
          "schedule.step left baked (no half-rewrite vs. the unmatched scf.for)")


def test_non_grid_steps_skipped():
    print("== M5/M6: step of 1 / 0 / negative is an ordinary loop, not a CTA stride ==")
    for c in (1, 0, -148):
        g = _graph(
            {"counter": _op("counter", "arith.addi",
                            [IterArgRef(loop_id=1, idx=0), ConstRef(value=c, type="i32")], scope="loop:1")},
            ConstRef(value=c, type="i32"),
            inits=(OpRef(op_id="counter"),),
        )
        rewrite(g)
        check(isinstance(g.loops[0].schedule.step, ConstRef) and g.loops[0].schedule.step.value == c,
              f"step={c}: loop skipped, `+{c}` counter untouched")


def test_idempotent():
    print("== rewrite is idempotent ==")
    g = _graph({}, ConstRef(value=148, type="i32"))
    rewrite(g)
    keys = list(g.ops.keys())
    rewrite(g)
    check(list(g.ops.keys()) == keys, "second pass adds no new synth op and re-swaps nothing")


def test_shipped_fixtures_reemit_identically():
    print("== the three shipped persistent fixtures re-emit byte-identically ==")
    root = Path(__file__).resolve().parent
    for case in ("case2_persistent_gemm", "case5_addmm_bias", "case9_scaled_mm/blockwise"):
        d = root / "examples" / case
        proc = subprocess.run(
            [sys.executable, "-m", "sched2tlx", str(d / "schedule_graph.json")],
            cwd=str(root), capture_output=True, text=True,
        )
        got = proc.stdout
        want = (d / "generated.py").read_text()
        check(proc.returncode == 0 and got == want, f"{case} re-emits byte-identically")


def main():
    test_step_and_scffor_rewritten_together()
    test_genuine_seed_and_carrier_rewritten()
    test_unrelated_recurrence_left_alone()
    test_ivref_stride_use_rewritten()
    test_seed_via_intermediate_op_rewritten()
    test_forop_unresolved_no_partial_rewrite()
    test_non_grid_steps_skipped()
    test_idempotent()
    test_shipped_fixtures_reemit_identically()
    print("\n=== ALL PASS ===" if ok else "\n=== FAILURES ===")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
