# TSP Course Experiment Framework

这是一个用于 TSP 课程实验的轻量 Python 框架。它支持从多行 txt 文件读取实例、通过统一 solver 接口运行多个算法、比较路线长度和运行时间，并预留 RL 环境接口。

## 输入格式

每行一个实例。坐标按 `x y` 成对出现，`output` 后可以放参考闭环路径：

```text
0.1 0.2 0.3 0.4 0.5 0.6 output 2 3 1 2
```

没有 `output` 时仍然可以求解和比较。

## 按规模分类数据

使用 `tool/classify_data.py` 将一个数据文件中的实例按点数拆分。默认在输入文件所在目录生成分类文件，文件名由原文件名和规模类别组成；没有实例的类别不会生成文件：

```powershell
.\.venv-plot\Scripts\python.exe tool/classify_data.py data/combined_train.txt
```

上述命令会按需生成 `combined_train_small.txt`、`combined_train_medium.txt`、`combined_train_large.txt` 和 `combined_train_ultra_large.txt`。分类范围为：小规模少于 100 个点，中规模 100 至 999 个点，大规模 1000 至 10000 个点，超大规模多于 10000 个点。可以使用 `-o` 指定其他输出目录：

```powershell
.\.venv-plot\Scripts\python.exe tool/classify_data.py data/combined_train.txt -o output
```

## 运行示例

当前使用 Windows Python 3.13 虚拟环境，解释器位于 `.venv-plot/Scripts/python.exe`，其中已安装 Matplotlib：

```powershell
.\.venv-plot\Scripts\python.exe example.py
```

算法通过 `TSPSolver` 接口实现，多个算法可以传给 `compare()` 统一运行：

```python
from tsp.compare import compare
from tsp.parser import load_instances
from tsp.solvers import NearestNeighborSolver

instance = load_instances("data/sample.txt")[0]
results = compare(instance, [NearestNeighborSolver()])
```

可在构造实例或读取数据时指定距离矩阵阈值。例如：

```python
instance = load_instances("data/sample.txt", distance_matrix_threshold=1000)[0]
```

城市数大于阈值时，`instance.distance_matrix[i][j]` 会实时计算且不缓存距离；等于或小于阈值时仍预计算。未指定阈值则保持原有的预计算行为。

绘图功能是可选的，需要安装 `requirements.txt` 中的 Matplotlib。核心解析、指标和算法对比不依赖第三方包。

## 迭代局部搜索（ILS）

`ILSSolver` 使用最近邻构造初始解，每轮执行双桥扰动和候选表 2-opt，并始终保留历史最优解：

```python
from tsp.solvers import ILSSolver

solver = ILSSolver(
    iterations=100,
    candidate_count=15,
    perturbation_strength=1,
    acceptance="better",  # 接受更优或等长解；也可设为 "always"
    seed=42,
)
solution = solver.solve(instance)
```

`metadata["history"]` 记录每轮结束后的历史最优长度，`accepted_iterations` 记录接受新 incumbent 的次数。最近邻构造、双桥扰动和闭环长度计算分别由 `tsp.construction`、`tsp.perturbations` 和 `tsp.local_search` 提供，可供其他算法复用。

## Lin-Kernighan-Helsgaun（LKH）

`LKHSolver` 是纯 Python 的 LKH-2 核心实现，面向本项目的对称欧氏 TSP，不下载或调用官方 LKH 程序。它包含：

- **Held–Karp ascent**：以城市罚值和最小 1-tree 提高下界，对应公式为 `w(π) = L(Tπ) - 2Σπ`；
- **α-nearness 候选集**：逐边计算“强制该边进入最小 1-tree”所增加的成本，不再用普通 K 近邻冒充 LKH 候选集；
- **LK 顺序 k-opt**：从两个方向搜索起始边，遵循累计正增益、边不重复和单环可行性约束，并通过回溯寻找深层交换；
- **多 run 与 trials/kicks**：每次 trial 在双桥扰动后重新执行 LK，始终保留历史最优闭环。

可复用能力按职责拆分：`tsp/one_tree.py` 提供 1-tree 与下界，`tsp/candidates.py` 提供普通 K 近邻和 α 候选集，`tsp/spatial.py` 从坐标构建近邻图，`tsp/local_search.py`、`tsp/perturbations.py` 分别提供 2-opt 和双桥扰动；`tsp/solvers/lkh.py` 只负责组装求解流程。

### 1000 点以上的大规模模式

`n >= large_scale_threshold`（默认 1000，即 1000 点及以上）时自动切换为近邻图模式，也可用 `sparse=True/False` 强制开关：

- `tsp/spatial.py` 用均匀网格（退化数据改用分块 numpy）生成精确 k 近邻图，时间 O(nK)、内存 O(nK)，不生成 n² 距离表；
- 最小 1-tree、Held–Karp ascent 和 α 值都只在这张 O(nK) 的候选图上计算：`minimum_one_tree(..., candidates=graph)`、`held_karp_ascent(..., graph)`、`alpha_candidate_sets(..., graph)`；α 值的树上路径最大边改用倍增 LCA，避免每个城市各做一次 O(n) DFS；
- LK 收敛使用 don't-look 队列和增量维护的无向邻接数组，改进后只重新检查受影响的城市，不再重建整表或扫描全部起点。

```python
solver = LKHSolver(
    large_scale_threshold=1000,   # 超过该规模自动启用近邻图
    sparse=None,                  # True / False 可强制开关
    candidate_graph_width=None,   # 默认 max(3 * candidate_count, 12)
)
solution = solver.solve(instance)
```

`metadata["mode"]` 为 `"sparse"` 或 `"complete"`，`metadata["bound_scope"]` 说明下界的作用域。**注意**：稀疏模式下 1-tree 建在受限候选图上，`metadata["lower_bound"]` 是该图松弛问题的界，可能高于真实最优值，不能再当作 TSP 的严格下界或最优性证明；完整模式下仍是严格下界。

大实例建议用 `distance_matrix_threshold` 打开按需距离，避免生成 n² 距离矩阵：

```python
instance = load_instances("data/large.txt", distance_matrix_threshold=1000)[0]
```

```python
from tsp.solvers import LKHSolver

solver = LKHSolver(
    candidate_count=5,   # 每个城市保留的 α 候选数
    runs=5,             # 独立初始解数量
    max_depth=5,        # 顺序交换的最大删边数
    kick_count=1,       # 允许的连续无改进 kick 数
    max_trials=800,     # 每次 run 的 trial 上限（含未扰动的首次 trial）
    ascent_iterations=50,
    seed=42,
    # 下面三个参数只影响 n > large_scale_threshold 的大规模模式
    large_scale_threshold=1000,
    sparse=None,
    candidate_graph_width=None,
)
solution = solver.solve(instance)
print(solution.metadata["lower_bound"])
```

原有构造参数和 `solve(instance)` 接口均保留，路径仍为 1 起始、末尾重复起点的闭环。`max_trials` 现在表示 LKH 的 trial 上限，不再是递归节点预算；`ascent_iterations=None` 时按实例规模自动选择预算。`metadata["candidate_type"]` 为 `"alpha"`，`metadata["trials"]` 记录实际 trial 数。

该实现不声称与官方 C 版逐项等价：它不包含 LKH-3 的其他问题类型、POPMUSIC、大规模 Delaunay/k-d tree 候选集以及完整的非顺序交换工程优化。

## 性能相关实现（n 接近 1000 时必读）

- `tsp/models.py` 在装有 numpy 时按行向量化构建距离矩阵：n ≤ 2000 返回 list of lists（标量索引更快），更大时返回 numpy 数组以节省内存；n 上万时可用 `distance_matrix_threshold` 改为按需计算，把这部分内存降到 O(n)。
- `tsp/candidates.py` 提供三类可复用候选集：普通 K 近邻（numpy `argpartition`，无 numpy 时退回 `heapq`）、完整图上的 α 候选集，以及候选图上的 α 候选集（传入 `candidates=graph`，复杂度 O(nK log K)）。
- `tsp/one_tree.py` 完整图模式用 O(n²) 的 Prim；传入 `candidates` 后改用堆式稀疏 Prim，每轮 O(nK)。`tsp/spatial.py` 的网格近邻图是 O(nK) 时间与内存，实测 n=10000 的图约 15 MB。
- `tsp/local_search.py` 提供候选表 + don't-look bits + 位置数组的 2-opt，单次收敛从 O(n³) 降到接近 O(nK)。`HybridSolver` 与 `LKHSolver` 都调用它；`HybridSolver` 的 `candidate_count` 默认 15，`LKHSolver` 默认使用 5 个 α 候选。
- `LKHSolver` 的 LK 收敛改用增量邻接数组 + don't-look 队列后，不再随每次改进重建 O(n) 的边表或扫描全部起点。

`distance_matrix_threshold` 在 n≈1000 时不建议使用：按需距离会在每次 `d[i][j]` 访问时重算 `hypot`，实测比预计算慢约 3 倍，而 n=1000 的完整矩阵只占约 8 MB。n ≥ 5000 时内存收益超过这点开销，建议开启；完整矩阵在 n=10000 时约 800 MB。

单 run、无 kick、`ascent_iterations=10`、随机均匀点的实测（旧完整图实现 → 大规模模式）：

| 城市数 | 旧完整图 LKH | 大规模模式 | 加速 |
| --- | --- | --- | --- |
| 1000 | 约 42.7 s | 约 1.6 s | 约 26 倍 |
| 2000 | 约 203 s | 约 5.4 s | 约 38 倍 |
| 5000 | — | 约 42 s | — |
| 10000 | — | 约 156 s | 按需距离、`ascent_iterations=5` |

前三种规模下两种模式得到的路径长度完全一致。上述时间不含距离矩阵构建；n≥10000 时请配合按需距离使用。

默认的 `runs=5` 与 kicks 会按 trial 数成倍增加耗时，万点实例建议显式给出预算，例如：

```python
solver = LKHSolver(runs=1, kick_count=0, max_trials=1, ascent_iterations=5, seed=42)
```

## Concorde 式分支切割精确求解

`tsp/solvers/branch_and_cut.py` 把 Concorde 的算法骨架用纯 Python 重写了一遍，不依赖任何原生二进制：

- **松弛**：DFJ 模型 —— 度数等式 + 子回路消除约束 `x(δ(S)) ≥ 2`，再叠加 comb 不等式。
- **分离**（`tsp/cuts.py`）：整数解取支撑图的连通分量；分数解用 Stoer-Wagner 全局最小割找最便宜的违约 `δ(S)`。两者都是精确分离：没有割可加时，当前 `x` 满足全部子回路约束。comb 分离在子回路收敛后进行。
- **分支**：在分数边上分支 `x_e = 0` / `x_e = 1`，best-bound 优先，用 incumbent 剪枝。
- **LP**（`tsp/lp.py`）：自带两阶段单纯形，增量热启动（见下），变量上界 `x_e ≤ 1` 作为懒惰约束按需加入。
- **上界**：多起点最近邻 + 2-opt + 双桥踢动，保证根节点一开始就有不错的 incumbent。

```python
from tsp.solvers import BranchAndCutSolver

solver = BranchAndCutSolver(time_limit=120)
solution = solver.solve(instance)
solution.metadata["optimal"]      # True 表示已证明最优
solution.metadata["lower_bound"]  # 未证明最优时的下界
solution.metadata["nodes"], solution.metadata["cuts"], solution.metadata["comb_cuts"]
```

### 三件加速关键件

**1. comb 不等式**（`comb_inequalities`）

```
x(δ(H)) + Σ_i x(δ(T_i)) ≥ 3k + 1      k 为奇数（≥3），齿两两不交且都与 handle 相交
```

因为每一项在可行点上至少是 2，被违反的 comb 只能是「齿很紧」的那一类，所以齿按**边界收缩**的方向构造：先取跨越 `δ(H)` 的 `x = 1` 连通分量（Padberg–Rinaldi 的经典取法），再用离开 handle 的边做种子、沿降低 `δ(T)` 的方向生长。handle 候选包括 `x = 1` 分量、相邻分量并集、分量加邻点、最小割一侧，以及**由 0.5 边组成的三角形**——半整数顶点上的 comb 往往正是被这种三角形 handle 切出来的，只用整块紧分量永远构造不出 `δ(H) = 3`。所有候选在返回前都会重算左端项，只返回真正违约的。

实测（随机欧氏实例，根节点）：

| n | 关 comb 下界 | 开 comb 下界 | 根节点 gap |
| --- | --- | --- | --- |
| 40 | 536.404 | 539.618 | 1.46% → 0.85% |
| 50 | 584.511 | 585.764 | 2.28% → 2.06% |
| 60 | 617.590 | 619.046 | 0.46% → 0.23% |
| 80 | 708.138 | 711.235 | 1.49% → 1.04% |

**2. LP 热启动**（`LPModel`）

`minimize()` 每次调用都重建 tableau、跑阶段一。分支切割节点则反复微扰同一个 LP：先解一次，再加一条割、再加一条。`LPModel` 因此把 tableau `T = B⁻¹A`、基 `basis`、基逆 `B⁻¹`、reduced cost `rc` 全部留住：

- 加一行：`row ← row − (row)_B · B⁻¹A`，一次 Gauss-Jordan 扫过；RHS 变负时走对偶单纯形。基逆按 `B' = [[B,0],[r,1]] ⟹ B'⁻¹ = [[B⁻¹,0],[−rB⁻¹,1]]` 扩展。
- 加一列：`α' = B⁻¹α`，`rc = c − c_B B⁻¹ α`，一个矩阵向量乘。
- `add_row` 以调用方的系数编号为准，行内符号（`>=` 行会被取反以拿到 `+1` 的松弛列）由 `_signs` 记录并回填到 `duals`，所以 `rc(e) = c(e) − Σ_r duals[r] · coef(r, e)` 直接成立。

不变式 `T = B⁻¹A`、`B·B⁻¹ = I`、reduced cost 自洽在测试里逐步校验；增量结果与冷启动 `minimize()` 在 300 次随机求解上完全一致（numpy 与纯 Python 两条路径各 300/300）。

**3. 列生成 pricing**

节点 LP 起步只放候选边（每城 k 近邻 + incumbent 边 + 割池涉及的边），解完后用对偶价 `prices[e] = Σ_r duals[r] · coef(r,e)` 给未激活边定价，reduced cost 为负的追加进来再解。pricing 最多 6 轮；若仍未收敛，就把剩余列全部激活重解——**受限 LP 从上方逼近松弛，用它剪枝是不合法的**，所以离开 `_NodeLP` 的界一定来自完整松弛。

实测根节点：n=40 用 366 列（全列 780）得到与全列**完全相同**的 3604.4472；n=60 用 542 列（全列 1770），快 3–6 倍。

### 一个来自退化行的实测加速

分支到一定深度后，某城市可能已有两条固定边，此时「其余边必须为 0」被写成 `Σ x = 0` 的等式行，阶段一在这个极度退化的顶点上要磨掉几百次旋转。把这些必然为 0 的边直接从列集合里删掉后，n=40 的节点吞吐从 73 节点/10 s 提升到 1173 节点/10 s，旋转数从约 700/节点降到约 60/节点，同一时限下的下界也从 530.06 升到 534.86。

主要参数：`node_limit`（默认 2000）、`time_limit`（默认 None）、`cut_rounds`（根节点分离轮数）、`deep_cut_rounds`（浅层子节点分离轮数）、`separation_depth`（此深度以下只用已有割池）、`comb_cuts` / `comb_depth` / `comb_rounds`（comb 开关、生效深度、轮数）、`pricing`（列生成开关）、`max_cuts`、`polish_kicks`。

规模建议：

| 规模 | 表现 |
| --- | --- |
| n ≤ 20 | 根节点即证明最优（DFJ 松弛在这里通常已是整数解），0.0 s |
| n ≈ 30–40 | 多数实例在根节点或少量节点内证明最优；少数需要更多节点 |
| n ≈ 50–60 | 下界与启发式解差距 0.2%–2%，证明最优通常超出 `node_limit` |
| n ≥ 100 | 不适合：请用 `HybridSolver` / `LKHSolver`，或用真正的 Concorde 二进制做精确解 |

与真正的 Concorde 相比，仍然缺：blossom / domino / local cuts、跨节点的基继承（目前每个节点仍重建一次 LP，只在节点内做热启动）、预处理的边删减、稀疏形式的 revised simplex、并行搜索 —— 这些正是 Concorde 能把精确解推到上万城市的差距所在。

```powershell
# 用精确解反过来标定启发式的真实差距
.\.venv-plot\Scripts\python.exe example_branch_and_cut.py
```

## 扩展算法

在 `tsp/solvers/` 新增类并实现：

```python
def solve(self, instance: TSPInstance) -> TSPSolution:
    ...
```

RL 推理算法也可以通过同一个 `TSPSolver` 接口接入；训练环境遵循 `tsp.env.TSPEnvProtocol` 的 `reset()` / `step()` 约定。

