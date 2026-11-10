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

## 扩展算法

在 `tsp/solvers/` 新增类并实现：

```python
def solve(self, instance: TSPInstance) -> TSPSolution:
    ...
```

RL 推理算法也可以通过同一个 `TSPSolver` 接口接入；训练环境遵循 `tsp.env.TSPEnvProtocol` 的 `reset()` / `step()` 约定。

