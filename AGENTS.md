# AGENTS.md

纯 Python 实现的对称欧氏 TSP 实验框架：多算法求解、统一对比、可复现实验与绘图。
对外契约只有两个：`TSPSolver.solve(instance) -> TSPSolution`，以及 `compare(instance, solvers)`。

---
## 1. 环境与命令

解释器固定用 `.venv-plot/Scripts/python.exe`（已装 numpy 与 matplotlib），不要用系统 `python`。

```powershell
.\.venv-plot\Scripts\python.exe -m pytest tests -q    # 全量测试
.\.venv-plot\Scripts\python.exe example.py            # 示例，会弹图
.\.venv-plot\Scripts\python.exe tool/bench_after.py   # 性能对照，图写进 .workbuddy/logs/
```

numpy / matplotlib 只是可选加速。改 `models.py`、`candidates.py`、`local_search.py` 的 numpy 分支时，把 numpy 从 `sys.modules` 摘掉再跑同一组用例。

## 2. 硬不变量

违反即 bug，不要以"顺手优化"为理由改动：

1. `route` 是 **1 起始**、末尾重复起点的闭环；长度由 `tsp.metrics.route_length` 计算，不要另写一份。
2. `tsp/` 下的核心模块不得裸 import 第三方包；numpy / matplotlib 必须 try-import 或延迟导入，缺失时降级为纯 Python。
3. 稀疏模式（n ≥ `large_scale_threshold`）下 `metadata["lower_bound"]` 只是候选图上松弛问题的界，**可能高于真实最优**，不可用于最优性判断。
4. 列生成未收敛时，受限 LP 从下方逼近松弛，**用它剪枝不合法**；离开 `_NodeLP` 的界必须来自完整松弛。
5. `TSPInstance` / `TSPSolution` 的字段属于对外契约，变更前先确认。

## 3. 模块地图与复用要求

仓库根目录只有 `example.py`，它是示例脚本，不承担模块职责。真正可复用的是 `tsp/` 包根层的模块：

| 模块 | 提供 | 现有调用方 |
| --- | --- | --- |
| `models.py` | `TSPInstance` / `TSPSolution`、距离矩阵策略（预计算 / 按需） | 全部 |
| `metrics.py` | `route_length` | 全部 |
| `construction.py` | `nearest_neighbor_route` 等构造 | 7 个 solver |
| `perturbations.py` | `double_bridge` 双桥扰动 | 4 个 solver |
| `local_search.py` | `tour_length`、候选表 + don't-look bits 的 `two_opt` | `two_opt`：LKH / Hybrid / ILS / 分支切割；`tour_length`：全部启发式 |
| `candidates.py` | `nearest_neighbours`、`alpha_values_for_city`、`alpha_candidate_sets` | LKH / 分支切割 / Hybrid / ILS |
| `one_tree.py` | `minimum_one_tree`、`held_karp_ascent`、下界 | LKH |
| `spatial.py` | `nearest_neighbour_graph`（网格加速的精确 kNN） | LKH |
| `cuts.py` | 子回路分离、comb 不等式 | 分支切割 |
| `lp.py` | `LPModel` 增量单纯形 | 分支切割 |
| `compare.py` | `compare` / `format_results` | `example.py`、`tool/` |

**新增求解器时的强制顺序：**

1. 先读现有求解器的组装方式，`tsp/solvers/hybrid.py` 与 `tsp/solvers/lkh.py` 是最完整的两个参考。
2. 在上表里找现成组件，直接 `from ..xxx import`。
3. 只有确实缺失的能力才写新代码，并且写成 `tsp/<能力>.py` 的独立模块供所有算法复用，**不要**写进 `tsp/solvers/<你的算法>.py`。
4. 把构造、扰动、局部搜索的逻辑复制粘贴进新 solver 视为错误，必须抽到共享模块后再调用。

`tsp/solvers/*.py` 是组装层：只做参数编排、调用共享组件、填 `metadata`，不承载算法本体。

## 4. 产物与文档存放

仓库根目录保持干净，只放 `README.md`、`AGENTS.md`、`example.py`、`requirements.txt`、`.gitignore` 与代码目录（`tsp/`、`tests/`、`tool/`、`data/`）。

- 未经明确要求，**不要**在根目录新建测试报告、日志、实验分析、临时 markdown 等文档。
- 日志与实验输出 → `.workbuddy/logs/<主题>/`
- 分析报告、基准记录 → `.workbuddy/docs/`
- 一次性草稿、中间文件 → `.workbuddy/tmp/`
- **不要新建** `output/`、`results/`、`logs/` 之类的顶层产物目录。
- 测试与实验产出的图片不入库，与对应日志放在一起；不要为了入库把图片改成 `.jpg` / `.svg` 绕过 `*.png` 规则。
- `.workbuddy/` 整体已在 `.gitignore` 中忽略，放进去的内容天然不会进入提交。

## 5. Git 提交规范

一律用英文 Conventional Commits：`type(scope): imperative subject`，整行 ≤ 72 字符，小写开头、不加句号；正文写**为什么**，性能改动带 before → after 数字。一次提交只做一件事，不跳 hook，不 amend 已推送的提交。

`type` 取 `feat` `fix` `perf` `refactor` `test` `docs` `chore` `bench`；`scope` 用模块名，如 `lkh` `bc` `local-search` `tool` `docs`。

```
perf(local-search): cut 2-opt convergence from O(n^3) to near O(nK)

Candidate lists plus don't-look bits bring n=1000 convergence from 42.7s to 1.6s.
```

## 6. 改动流程

1. 定位：按第 3 节判断改的是共享组件还是组装层。
2. 改核心算法：先在 `tests/` 加或改用例。`tests/test_lkh.py` 里的 `brute_one_tree_cost` 是暴力对拍写法，照这个风格写。
3. 跑全量 pytest。
4. 碰性能路径：跑 `tool/bench_after.py`，把数字更新进 README 对应表格。
5. 提交。

## 7. 不要做

- 不引入 LKH / Concorde 官方二进制、C 扩展、scipy、numba。
- 不为让测试跑快而放宽断言、调低迭代次数或改阈值。
- 不提交 `data/*`、`*.png`、`__pycache__`、`.workbuddy/*`。
- 不重排 README 的章节顺序，只增补；性能数字变化时同步更新表格。
- 不在 `tsp/` 里新增与已存在模块功能重叠的文件。

## 8. 先问人再动

- 改 `TSPInstance` / `TSPSolution` 的字段或 route 索引约定。
- 改输入格式（`x y` 坐标对、`output` 后的参考路径）。
- 新增任何第三方依赖。
- 删除或重命名 `tsp/` 下的公开模块与 `__init__.py` 导出。
