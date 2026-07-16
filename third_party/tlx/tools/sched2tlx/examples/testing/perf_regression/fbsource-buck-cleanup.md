# fbsource 侧 sched2tlx buck target 清理清单

> 背景：GitHub 仓库（facebookexperimental/triton）里的
> `third_party/tlx/tools/sched2tlx/examples/testing/perf_regression/perf_harness.py`
> 已于 2026-07-16 精简为只保留 `compare` 子命令（`bench` / `regression` / `e2e` /
> `e2e-worker` 已删除，可从 git 历史找回）。fbsource 有两个 buck target 以该模块为
> `main_module`，同步落地后它们**构建不红、但运行必挂**（打出
> `usage: perf_harness.py {compare}` 后退出 2），需要一并清理。
>
> 本清单依据旧代码与旧文档（`.claude/sched2tlx/e2e_perf_harness.md`，同批删除）中的
> 实证信息整理；`BUCK.template` 的准确内容以 fbsource 侧为准。

## A. 修改 `third-party/triton/beta/BUCK.template`

两个 target 均定义于此模板（改完需重新生成 BUCK 文件）：

- [ ] 删除 **`sched2tlx_regression`**（`python_binary`）
  - main_module = perf_harness；原调用方式：
    `buck2 run @fbcode//mode/dev-nosan //third-party/triton/beta/triton:sched2tlx_regression -- regression`
- [ ] 删除 **`sched2tlx_e2e`**（`python_binary` / par）
  - main_module = perf_harness；原由 e2e orchestrator 以
    `buck2 run @mode/opt -m ovr_config//triton:beta -c fbcode.nvcc_arch=... -c fbcode.platform010_cuda_version=... fbsource//third-party/triton/beta/triton:sched2tlx_e2e -- e2e-worker ...` 调用
  - 连同其特有依赖一起删：
    - [ ] 把 `:triton-opt` 作为 **par resource** 打包的 wiring（旧代码经
      `libfb.py.parutil.get_file_path("triton-opt")` 读取）
    - [ ] torch / triton-py 依赖（worker 在 par 内跑 GPU bench）
    - [ ] `libfb.py.parutil` 依赖
- [ ] 若模板里存在只供这两个 binary 使用的辅助 `python_library`
  （如包 `perf_harness.py` 的 srcs 条目），一并删除。
  **`perf_harness.py` 文件本身保留**——`compare` 仍在使用。

## B. 全库引用搜索（在 devserver 上执行）

```bash
fbgs sched2tlx_regression
fbgs sched2tlx_e2e
fbgs "e2e-worker"
fbgs "perf_harness.py regression"
```

- [ ] contbuild / CI / testinfra 配置中对这两个 target 的引用（是否存在待确认）
- [ ] 内部 wiki / Runbook 中的 `buck2 run` 命令示例
- [ ] 任何调用脚本、cron、oncall 工具

## C. 不要误伤

- **`:triton-opt` target 本身要保留**——ddg 生成流程
  （`.claude/sched2tlx/generating_ddg_json.md`）仍在用
  `buck2 build fbsource//third-party/triton/beta/triton:triton-opt --show-full-output`；
  只删 e2e par 对它的 resource 引用。
- **`src_hash.txt` genrule 要保留**——它是 triton-py 构建的一部分
  （旧文档提到它只是因为 argv 长度上限的坑），与本次清理无关。

## D. 落地顺序

同步落地后 buck **构建不会红**（模块仍存在，`main_module` 解析正常），只有运行时
传 `regression` / `e2e-worker` 才失败。因此清理 diff 可与同步同一个 diff 或紧随其后，
无原子性压力。

## 参考（GitHub 仓库侧已完成的对应改动，2026-07-16，暂未提交，位于
`triton-beta-2-mispricing-wt` 工作树）

- `perf_harness.py` 精简为 compare-only（四列输出 + 提升百分比、fork 隔离测量、
  递归发现 bench_spec、branch 列变更加粗）
- `README.md` / `.claude/skills/sched2tlx-perf-testing/SKILL.md` 重写为 compare-only
- `.claude/sched2tlx/e2e_perf_harness.md` 删除（只描述已删代码）
- 新增 `case4_FA_bwd/bench_spec.py`、`case8_multiphase_gemm/bench_spec.py`
- 现存树中对两个 buck target 的引用仅剩新 README "History" 一节的一句历史说明（有意保留）
